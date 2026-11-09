"""Compile author routing kernels; no device computation or SHMEM initialization."""
from __future__ import annotations

import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def package_root(name: str) -> Path:
    """Locate a package without importing its native backend."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Install {name} in the active Python environment first")
    return Path(spec.origin).parent


def main() -> None:
    """Rebuild the changed allocator before workers import the component."""
    package = package_root("tt_megamoe")
    target = package / "moonep/_a3_allocator.so"
    # A failed rebuild must never silently expose an older allocator ABI.
    target.unlink(missing_ok=True)
    receipt = package / "component_metadata/allocator_build.json"
    receipt.unlink(missing_ok=True)
    if platform.machine() != "aarch64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("The supplied compute libraries require Linux aarch64 and Python 3.11")
    versions = {name: version(name) for name in ("torch", "torch-npu")}
    if versions["torch"].split("+")[0] != "2.10.0" or versions["torch-npu"] != "2.10.0.post4":
        raise RuntimeError(f"Expected torch 2.10.0 and torch-npu 2.10.0.post4; found {versions}")
    value = os.environ.get("ASCEND_HOME_PATH")
    if not value or not Path(value).is_dir():
        raise RuntimeError("Source CANN 9.1 set_env.sh first (ASCEND_HOME_PATH is required)")
    cann = Path(value).resolve()
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("bisheng is missing; source the CANN development toolkit set_env.sh")
    abi = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(int(torch.compiled_with_cxx11_abi()))"],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}, text=True,
    ).strip()
    if abi != "1":
        raise RuntimeError(f"The supplied CANN 9.1 compute libraries require CXX11 ABI=1; found {abi!r}")
    sources = package / "native_sources/moonep"
    release = json.loads((package / "component_metadata/moonep_release.json").read_text())
    for name, expected in release["routing_sources"].items():
        if hashlib.sha256((sources / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Routing source hash mismatch: {name}")
    torch_root, npu_root = package_root("torch"), package_root("torch_npu")
    includes = [torch_root / "include", torch_root / "include/torch/csrc/api/include",
                npu_root / "include", Path(sysconfig.get_path("include")), sources,
                cann / "include", cann / "include/runtime", cann / "pkg_inc",
                cann / "pkg_inc/runtime", cann / "aarch64-linux/include",
                cann / "aarch64-linux/pkg_inc", cann / "aarch64-linux/pkg_inc/runtime",
                cann / "aarch64-linux/include/op_common",
                cann / "aarch64-linux/include/op_common/op_host"]
    opdev = sorted(p for p in cann.rglob("bfloat16.h") if p.parent.name == "opdev")
    if not opdev:
        raise RuntimeError("The selected CANN toolkit has no opdev/bfloat16.h")
    includes.append(opdev[0].parent.parent)
    with tempfile.TemporaryDirectory(prefix="allocator-build-", dir=target.parent) as directory:
        output = Path(directory) / target.name
        command = [compiler, "-x", "asc", "--npu-arch=dav-2201", "--cce-auto-sync=off",
                   "-shared", "-fPIC", "-O2", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1", "-O3",
                   "-DTORCH_EXTENSION_NAME=_a3_allocator", str(sources / "a3_allocator.asc"),
                   "-o", str(output)]
        command += [f"-I{path}" for path in includes if path.is_dir()]
        command += [f"-L{path}" for path in (torch_root / "lib", npu_root / "lib", cann / "lib64")]
        command += ["-ltorch_npu", "-ltorch", "-lc10", "-ltorch_python"]
        print("ALLOCATOR_BUILD " + shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        output.replace(target)
    receipt.write_text(json.dumps({
        "source_commit": release["source_commit"], "routing_sources": release["routing_sources"],
        "cann": str(cann), "compiler": compiler, "versions": versions,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(f"ALLOCATOR_BUILD_PASS source={release['source_commit']} target={target}", flush=True)


if __name__ == "__main__":
    main()
