#!/usr/bin/env python3
"""Run generic linters and repository-specific CI contracts."""

from __future__ import annotations

import argparse
import ast
import configparser
import os
import platform
import re
import sys
import time
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
WORKFLOWS = ROOT / ".github" / "workflows"
CI = ROOT / "scripts" / "ci"
TOOL_DIR = Path(os.environ.get("TORCHTURBO_LINT_TOOLS", "/tmp/torchturbo-lint-tools"))

ACTIONLINT_VERSION = "1.7.12"
HADOLINT_VERSION = "2.15.1"
SHELLCHECK_VERSION = "0.11.0"
# ruff and PyYAML are not self-bootstrapped; their pinned versions live in
# requirements-lint.txt alone, so this file cannot drift from the workflow.
REQUIREMENTS = "requirements-lint.txt"

# These are build-policy choices, not accidental lint debt.
HADOLINT_IGNORES = ("DL3041", "DL3042", "DL3003", "DL3059")

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # reported by the checks that need it
    yaml = None


@dataclass(frozen=True)
class Problem:
    path: str
    message: str
    line: int = 0

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    function: Callable[[], list[Problem]]


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    raise RuntimeError(f"unsupported architecture: {machine}")



PROBE = {"actionlint": "-version", "hadolint": "--version", "shellcheck": "--version"}


def _tool_path(name: str, version: str | None = None) -> Path | None:
    candidates = [
        os.environ.get(f"{name.upper()}_BIN"),
        TOOL_DIR / name,
        Path("/tmp") / name,
        shutil.which(name),
    ]
    for index, candidate in enumerate(candidates):
        if not candidate or not Path(candidate).is_file():
            continue
        if not os.access(candidate, os.X_OK):
            continue
        # An explicit *_BIN override is honoured as-is; everything else must
        # report the pinned version or the download below supplies it instead.
        if index == 0 or version is None:
            return Path(candidate)
        try:
            probe = run([str(candidate), PROBE[name]], timeout=30)
            reported = probe.stdout + probe.stderr
        except (OSError, subprocess.SubprocessError):
            reported = ""
        if version in reported:
            return Path(candidate)
    return None


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "torchturbo-ci-lint"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(temporary, "wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"download produced no file: {url}")
    destination.chmod(0o755)


def _extract_binary(archive: Path, target: Path, member_name: str) -> Path:
    with tarfile.open(archive, "r:*") as bundle:
        member = next((m for m in bundle.getmembers()
                       if Path(m.name).name == member_name), None)
        if member is None:
            raise RuntimeError(f"{archive.name} has no {member_name} binary")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"cannot read {member_name} from {archive.name}")
        temporary = target.with_suffix(".part")
        with open(temporary, "wb") as output:
            shutil.copyfileobj(source, output)
    temporary.replace(target)
    target.chmod(0o755)
    return target


def _ensure_actionlint() -> Path:
    found = _tool_path("actionlint", ACTIONLINT_VERSION)
    if found:
        return found
    arch = _arch()
    version = ACTIONLINT_VERSION
    url = (f"https://github.com/rhysd/actionlint/releases/download/v{version}/"
           f"actionlint_{version}_linux_{arch}.tar.gz")
    archive = TOOL_DIR / f"actionlint-{version}-{arch}.tar.gz"
    _download(url, archive)
    return _extract_binary(archive, TOOL_DIR / "actionlint", "actionlint")


def _ensure_hadolint() -> Path:
    found = _tool_path("hadolint", HADOLINT_VERSION)
    if found:
        return found
    arch = "arm64" if _arch() == "arm64" else "x86_64"
    url = (f"https://github.com/hadolint/hadolint/releases/download/"
           f"v{HADOLINT_VERSION}/hadolint-linux-{arch}")
    target = TOOL_DIR / "hadolint"
    _download(url, target)
    return target


def _ensure_shellcheck() -> Path:
    found = _tool_path("shellcheck", SHELLCHECK_VERSION)
    if found:
        return found
    arch = "aarch64" if _arch() == "arm64" else "x86_64"
    version = SHELLCHECK_VERSION
    url = (f"https://github.com/koalaman/shellcheck/releases/download/v{version}/"
           f"shellcheck-v{version}.linux.{arch}.tar.xz")
    archive = TOOL_DIR / f"shellcheck-{version}-{arch}.tar.xz"
    _download(url, archive)
    return _extract_binary(archive, TOOL_DIR / "shellcheck", "shellcheck")


def _ensure_ruff() -> Path:
    found = _tool_path("ruff")
    if found:
        return found
    raise RuntimeError(f"ruff is not installed (pip install -r {REQUIREMENTS})")


def _diagnostics(cmd: list[str], fallback: str, timeout: int = 120) -> list[Problem]:
    """Run a tool and parse `path:line[:column]: message` diagnostics."""
    proc = run(cmd, timeout=timeout)
    if proc.returncode == 0:
        return []
    problems = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^(.*?):(\d+):(?:\d+: )?(.*)$", line)
        if match:
            problems.append(Problem(relative(Path(match.group(1))),
                                    match.group(3), int(match.group(2))))
    if not problems:
        detail = (proc.stdout + proc.stderr).strip() or f"exit {proc.returncode}"
        problems.append(Problem(fallback, f"{Path(cmd[0]).name} failed: {detail.splitlines()[-1]}"))
    return problems


def check_ruff() -> list[Problem]:
    """Run Ruff over repository Python, with stronger rules on CI code."""
    try:
        tool = _ensure_ruff()
    except Exception as exc:
        return [Problem("scripts/ci", str(exc))]
    problems = _diagnostics(
        [str(tool), "check", "--no-cache", "--output-format", "concise",
         "--select", "E9", "src", "scripts", "tests"],
        "scripts/ci",
    )
    problems += _diagnostics(
        [str(tool), "check", "--no-cache", "--output-format", "concise",
         "--select", "E4,E7,F", "--ignore", "E501", "scripts/ci"],
        "scripts/ci",
    )
    return problems


def check_shellcheck() -> list[Problem]:
    """Shellcheck every shell script and bin wrapper under scripts/ci."""
    scripts = {str(path) for path in CI.rglob("*.sh")}
    scripts.update(str(path) for path in (CI / "consumer" / "runtime" / "bin").glob("*") if path.is_file())
    scripts = sorted(scripts)
    if not scripts:
        return []
    try:
        tool = _ensure_shellcheck()
    except Exception as exc:
        return [Problem("scripts/ci", f"cannot obtain shellcheck: {exc}")]
    return _diagnostics([str(tool), "-x", "-f", "gcc", *scripts], "scripts/ci")


def check_actionlint() -> list[Problem]:
    """Lint workflow files with a pinned actionlint."""
    try:
        tool = _ensure_actionlint()
    except Exception as exc:
        return [Problem(".github/workflows", f"cannot obtain actionlint: {exc}")]
    return _diagnostics([str(tool), "-no-color", "-oneline"], ".github/workflows", timeout=180)


def check_hadolint() -> list[Problem]:
    """Lint Dockerfiles, failing on new warnings after the explicit allowlist."""
    try:
        tool = _ensure_hadolint()
    except Exception as exc:
        return [Problem("docker", f"cannot obtain hadolint: {exc}")]
    problems = []
    for dockerfile in sorted((ROOT / "docker").rglob("Dockerfile*")):
        cmd = [str(tool), "--format", "gnu", "--failure-threshold", "warning"]
        for rule in HADOLINT_IGNORES:
            cmd += ["--ignore", rule]
        cmd.append(str(dockerfile))
        problems += _diagnostics(cmd, relative(dockerfile))
    return problems

BUILTIN_MARKERS = frozenset({
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "usefixtures",
    "xfail",
})

def _yaml_document(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed "
                           "(pip install -r requirements-lint.txt)")

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate key {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    doc = yaml.load(text(path), Loader=UniqueKeyLoader)
    if not isinstance(doc, dict):
        raise ValueError("top level is not a mapping")
    return doc


def _job_needs(job: dict) -> list[str]:
    needs = (job or {}).get("needs") or []
    if isinstance(needs, str):
        return [needs]
    return list(needs)


def _job_run_text(job: dict) -> str:
    chunks = []
    for step in (job or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("run"):
            chunks.append(str(step["run"]))
    return "\n".join(chunks)


def _job_closure(jobs: dict[str, dict], name: str) -> set[str]:
    """Return every job reachable from `name` through `needs` edges."""
    seen: set[str] = set()
    pending = list(_job_needs(jobs.get(name, {})))
    while pending:
        current = pending.pop()
        if current in seen or current not in jobs:
            continue
        seen.add(current)
        pending.extend(_job_needs(jobs[current]))
    return seen


def check_contract() -> list[Problem]:
    """Cross-job wiring that no YAML parser or actionlint can see."""
    if yaml is None:
        return [Problem(".github/workflows", "PyYAML is required for contract checks")]
    problems = []
    docs = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        rel = relative(path)
        try:
            docs[path] = _yaml_document(path)
        except Exception as exc:
            problems.append(Problem(rel, f"cannot inspect contract: {exc}"))
    if problems:
        return problems

    for path, doc in docs.items():
        rel = relative(path)
        jobs = {str(name): job for name, job in (doc.get("jobs") or {}).items()}
        plan_jobs = [name for name, job in jobs.items()
                     if "scripts.ci.producer.planner" in _job_run_text(job)]
        if plan_jobs:
            problems.extend(_check_planner_v2(rel, jobs, yaml.safe_dump(doc, allow_unicode=True)))
            continue
    return problems


def _check_planner_v2(rel: str, jobs: dict, dumped: str) -> list[Problem]:
    """Validate the Planner -> Executor -> Gate topology."""
    problems: list[Problem] = []
    planners = [name for name, job in jobs.items()
                if "scripts.ci.producer.planner" in _job_run_text(job)]
    if len(planners) != 1:
        return [Problem(rel, f"expected one Planner job; found {sorted(planners)}")]
    planner_name = planners[0]
    planner = jobs[planner_name]
    planner_body = _job_run_text(planner)
    if "scripts/ci/tools/lint.py" not in planner_body:
        problems.append(Problem(rel, "Planner job must also run scripts/ci/tools/lint.py"))
    declared = set((planner.get("outputs") or {}).keys())
    output_pattern = rf"needs\.{re.escape(planner_name)}\.outputs\.([A-Za-z0-9_-]+)"
    used = set(re.findall(output_pattern, dumped))
    for name in sorted(used - declared):
        problems.append(Problem(
            rel,
            f"reads needs.{planner_name}.outputs.{name} but the Planner does not declare it",
        ))
    for name in sorted(declared - used):
        problems.append(Problem(rel, f"Planner output {name!r} has no consumer"))

    needed = {dependency for job in jobs.values() for dependency in _job_needs(job)}
    terminals = sorted(set(jobs) - needed)
    lanes = sorted({
        name for name, job in jobs.items()
        if planner_name in _job_needs(job)
    } - set(terminals))
    if not lanes:
        problems.append(Problem(rel, "Planner must feed at least one Executor lane"))
    for name in lanes:
        body = _job_run_text(jobs[name])
        if "scripts.ci.consumer.executor" not in body:
            problems.append(Problem(rel, f"lane {name!r} does not invoke the executor"))
    if len(terminals) != 1:
        problems.append(Problem(
            rel, f"Planner workflow must have exactly one terminal job; found {terminals}"
        ))
        return problems
    terminal = terminals[0]
    terminal_body = _job_run_text(jobs[terminal])
    if "scripts.ci.consumer.gate" not in terminal_body:
        problems.append(Problem(rel, f"terminal job {terminal!r} does not invoke the gate"))
    if "download-artifact" not in yaml.safe_dump(jobs[terminal], allow_unicode=True):
        problems.append(Problem(rel, f"terminal job {terminal!r} does not download results"))
    reachable = _job_closure(jobs, terminal)
    missing = sorted({planner_name, *lanes} - reachable)
    if missing:
        problems.append(Problem(rel, f"terminal job {terminal!r} does not reach {missing}"))
    return problems


def check_catalog() -> list[Problem]:
    """Validate catalog.yml and all facts derived from the repository."""
    from ..producer.catalog import CatalogError, load

    try:
        load(ROOT)
    except CatalogError as exc:
        return [Problem(issue.path, issue.message) for issue in exc.issues]
    return []


def _test_marker_names() -> set[str]:
    """Collect custom pytest markers used under tests/."""
    names: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        try:
            tree = ast.parse(text(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            base = node.value
            if (isinstance(base, ast.Attribute) and base.attr == "mark"
                    and isinstance(base.value, ast.Name) and base.value.id == "pytest"):
                names.add(node.attr)
    return names - BUILTIN_MARKERS


def check_markers() -> list[Problem]:
    """Every marker used by tests is registered in pytest.ini."""
    ini = ROOT / "pytest.ini"
    pyproject = ROOT / "pyproject.toml"
    problems = []

    if "[tool.pytest.ini_options]" in text(pyproject):
        problems.append(Problem("pyproject.toml", "pytest config belongs in pytest.ini"))
    if not ini.exists():
        return problems + [Problem("pytest.ini", "missing test configuration")]

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ini, encoding="utf-8")
    configured = set()
    for line in (parser.get("pytest", "markers", fallback="") or "").splitlines():
        if ":" in line:
            configured.add(line.split(":", 1)[0].strip())

    from ..producer.scan import REQUIRED_MARKERS

    markers = _test_marker_names()
    for name in sorted(markers - configured):
        problems.append(Problem("pytest.ini", f"marker {name!r} is used but not registered"))
    for name in sorted(configured - set(REQUIRED_MARKERS)):
        problems.append(Problem("pytest.ini", f"obsolete marker {name!r}"))
    return problems

CHECKS = (
    Check("ruff", "Python syntax and CI-side Ruff rules", check_ruff),
    Check("shellcheck", "shell scripts under scripts/ci", check_shellcheck),
    Check("actionlint", "GitHub Actions workflow syntax", check_actionlint),
    Check("hadolint", "Dockerfile warnings", check_hadolint),
    Check("contract", "planner/lint/lane topology and output contracts", check_contract),
    Check("catalog", "catalog, test facts, and E2E case contract", check_catalog),
    Check("markers", "registered pytest marker contract", check_markers),
)
CHECK_BY_NAME = {check.name: check for check in CHECKS}

def _selected(names: list[str]) -> list[Check]:
    if not names:
        return list(CHECKS)
    unknown = [name for name in names if name not in CHECK_BY_NAME]
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)}")
    return [CHECK_BY_NAME[name] for name in names]


def _summary(rows: list[tuple[str, str, int]]) -> str:
    lines = ["## Lint", "", "| check | result | problems |", "|---|---|---:|"]
    for name, result, count in rows:
        lines.append(f"| `{name}` | {result} | {count} |")
    return "\n".join(lines) + "\n"


def run_checks(checks: Iterable[Check], summary_path: str | None) -> int:
    started = time.monotonic()
    total = 0
    rows = []
    failed = []
    for check in checks:
        try:
            problems = check.function()
        except Exception as exc:
            problems = [Problem("<check>", f"check crashed: {type(exc).__name__}: {exc}")]
        total += len(problems)
        rows.append((check.name, "fail" if problems else "ok", len(problems)))
        if not problems:
            print(f"{check.name}: ok")
            continue
        failed.append(check.name)
        print(f"{check.name}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem.render()}")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                fields = [f"title={check.name}", f"file={problem.path}"]
                if problem.line:
                    fields.append(f"line={problem.line}")
                message = problem.message.replace("\n", " ")
                print(f"::error {','.join(fields)}::{message}")

    elapsed = time.monotonic() - started
    destination = summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with open(destination, "a", encoding="utf-8") as output:
            output.write(_summary(rows))

    if total:
        print(f"\nFAILED: {', '.join(failed)} ({total} problem(s), {elapsed:.1f}s)")
        return 1
    print(f"\nLint OK ({len(rows)} checks, {elapsed:.1f}s)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run TorchTurbo static checks")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run one check (repeatable)")
    parser.add_argument("--list", action="store_true", help="list check names and exit")
    parser.add_argument("--summary", default=None,
                        help="write a Markdown summary to this file")
    args = parser.parse_args(argv)

    if args.list:
        for check in CHECKS:
            print(f"{check.name}\t{check.description}")
        return 0
    try:
        checks = _selected(args.only)
    except ValueError as exc:
        parser.error(str(exc))
    return run_checks(checks, args.summary)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
