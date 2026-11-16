#!/usr/bin/env python3
"""Create content-addressed Python environments for CI jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

CPU_TORCH_VERSION = os.environ.get("CI_CPU_TORCH_VERSION", "2.10.0")
READY_FILE = ".tt-ready.json"
INSTALL_SCHEMA = "2"


class VenvDefinition(Protocol):
    name: str
    python: str
    requirements: tuple[str, ...]
    overrides: tuple[str, ...]
    extends: str | None
    constraints: str


@dataclass(frozen=True)
class Venv:
    path: Path
    python: Path
    spec: VenvDefinition

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update({
            "VIRTUAL_ENV": str(self.path),
            "PYTHONNOUSERSITE": "1",
            "PATH": f"{self.path / 'bin'}:{env.get('PATH', '')}",
        })
        return env


class VenvManager:
    """Build or reuse one venv per (env definition, device) pair."""

    def __init__(self, root: Path, cache_root: Path, device: str,
                 log_path: Path, specs: Mapping[str, VenvDefinition]) -> None:
        self.root = root
        self.cache_root = cache_root
        self.device = device
        self.log_path = log_path
        self.specs = dict(specs)
        self._memory: dict[str, Venv] = {}

    def ensure(self, env_name: str) -> Venv:
        if env_name in self._memory:
            return self._memory[env_name]
        spec = self.specs.get(env_name)
        if spec is None:
            raise ValueError(f"unknown environment: {env_name}")
        chain = self._environment_chain(spec)
        digest = self._profile_digest(spec, chain)
        target = self.cache_root / f"{env_name}-{digest}"
        ready = self._read_ready(target)
        if ready is not None and not (target / "bin" / "python").is_file():
            ready = None
        if ready is None:
            if target.exists():
                # A cancelled install leaves no marker. Start a fresh directory
                # instead of deleting a shared cache entry.
                target = self.cache_root / f"{env_name}-{digest}-retry-{time.time_ns()}"
            self._install(target, spec, chain)
        venv = Venv(path=target, python=target / "bin" / "python", spec=spec)
        self._memory[env_name] = venv
        return venv

    def _environment_chain(self, spec: VenvDefinition) -> list[VenvDefinition]:
        chain: list[VenvDefinition] = []
        seen: set[str] = set()
        current: VenvDefinition | None = spec
        while current is not None:
            if current.name in seen:
                raise ValueError(f"environment cycle at {current.name}")
            seen.add(current.name)
            chain.append(current)
            current = self.specs.get(current.extends) if current.extends else None
        return list(reversed(chain))

    def _profile_digest(self, spec: VenvDefinition,
                        chain: list[VenvDefinition]) -> str:
        digest = hashlib.sha256()
        digest.update(f"schema={INSTALL_SCHEMA}\n".encode())
        digest.update(f"device={self.device}\n".encode())
        digest.update(f"python={spec.python}\n".encode())
        if self.device == "cpu":
            digest.update(f"torch={CPU_TORCH_VERSION}\n".encode())
        # Installer changes must invalidate cached profiles as well as
        # dependency changes.
        for relative in (
            Path("pyproject.toml"),
            Path("scripts/ci/consumer/runtime/venv.py"),
            Path("scripts/ci/consumer/runtime/env.py"),
        ):
            path = self.root / relative
            digest.update(f"file={relative.as_posix()}\n".encode())
            digest.update(path.read_bytes())
        for item in chain:
            digest.update(f"env={item.name}\n".encode())
            digest.update(f"env-python={item.python}\n".encode())
            for requirements in _requirement_paths(item):
                for path in _requirement_files(self.root / requirements):
                    name = path.relative_to(self.root).as_posix()
                    digest.update(f"file={name}\n".encode())
                    digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    def _read_ready(self, path: Path) -> dict | None:
        marker = path / READY_FILE
        if not marker.is_file():
            return None
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _install(self, target: Path, spec: VenvDefinition,
                 chain: list[VenvDefinition]) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        base_python = _base_python(spec.python)
        print(f"[venv] creating {target.name} with {base_python}", flush=True)
        command = [base_python, "-m", "venv"]
        if self.device != "cpu":
            command.append("--system-site-packages")
        command.append(str(target))
        self._run(command, cwd=self.root, environment=os.environ.copy())

        venv = Venv(path=target, python=target / "bin" / "python", spec=spec)
        environment = venv.environment()
        if self.device == "cpu":
            self._pip(venv, ["install", f"torch=={CPU_TORCH_VERSION}"])
        self._run(
            [str(venv.python), str(self.root / "scripts/ci/consumer/runtime/env.py"),
             "probe", "--device", self.device,
             "--file", str(target / "torch-before.json")],
            cwd=self.root, environment=environment,
        )

        constraints = target / "ci-constraints.txt"
        constraints.write_text(self._constraints(venv, environment), encoding="utf-8")
        for item in chain:
            for requirements in _requirement_paths(item):
                self._pip(venv, [
                    "install", "-r", str(self.root / requirements),
                    "-c", str(constraints),
                ])

        self._pip(venv, ["install", "--no-build-isolation", "--no-deps",
                         "-e", str(self.root)])
        self._run(
            [str(venv.python), str(self.root / "scripts/ci/consumer/runtime/env.py"),
             "check", "--device", self.device,
             "--file", str(target / "torch-before.json")],
            cwd=self.root, environment=environment,
        )
        ready = {
            "env": spec.name,
            "device": self.device,
            "python": str(venv.python),
            "torch": CPU_TORCH_VERSION if self.device == "cpu" else "image",
        }
        (target / READY_FILE).write_text(
            json.dumps(ready, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[venv] ready: {target}", flush=True)

    def _constraints(self, venv: Venv, environment: dict[str, str]) -> str:
        packages = ["torch", "numpy"] if self.device != "cpu" else ["torch"]
        code = (
            "import importlib.metadata as m, sys\n"
            "names = sys.argv[1:]\n"
            "for name in names:\n"
            "    try: print(f'{name}=={m.version(name)}')\n"
            "    except m.PackageNotFoundError: pass\n"
        )
        result = subprocess.run(
            [str(venv.python), "-c", code, *packages],
            cwd=self.root, env=environment, capture_output=True, text=True, check=True,
        )
        return result.stdout

    def _pip(self, venv: Venv, args: list[str]) -> None:
        self._run(
            [str(venv.python), "-m", "pip", *args],
            cwd=self.root, environment=venv.environment(),
        )

    def _run(self, command: list[str], cwd: Path, environment: dict[str, str]) -> None:
        print("+", " ".join(command), flush=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"command failed with rc={rc}: {' '.join(command)}")


def _base_python(version: str) -> str:
    major_minor = ".".join(version.split(".")[:2])
    if f"{sys.version_info.major}.{sys.version_info.minor}" == major_minor:
        return sys.executable
    found = shutil.which(f"python{major_minor}")
    if found:
        return found
    raise RuntimeError(f"Python {version} is required; python{major_minor} was not found")


def _requirement_files(path: Path) -> list[Path]:
    """Return a requirement file and nested `-r` includes in stable order."""
    files: list[Path] = []
    seen: set[Path] = set()
    pending = [path]
    while pending:
        current = pending.pop(0).resolve()
        if current in seen:
            continue
        seen.add(current)
        files.append(current)
        for line in current.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value.startswith("-r "):
                pending.append(current.parent / value[3:].strip())
    return files


def _requirement_paths(spec: VenvDefinition) -> tuple[str, ...]:
    return (*spec.requirements, *spec.overrides)
