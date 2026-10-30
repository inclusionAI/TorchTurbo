#!/usr/bin/env python3
"""Build TorchTurbo MegaMoE + MoonEP against the official cann-shmem SDK."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig

from importlib.metadata import version


def run(*args: object) -> None:
    command = [str(arg) for arg in args]
    print("BUILD_COMMAND " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Install {name} in the active Python environment first")
    return Path(spec.origin).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shmem-source", type=Path, help="Official SHMEM Git checkout for offline header extraction")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parent
    native = root / "native"
    build = (args.build_dir or root / "build").resolve()
    cann_value = os.environ.get("ASCEND_HOME_PATH")
    if not cann_value or not Path(cann_value).is_dir():
        parser.error("Source the CANN set_env.sh before building (ASCEND_HOME_PATH is required)")
    cann = Path(cann_value).resolve()
    for tool in ("cmake", "bisheng", "g++", "readelf"):
        if shutil.which(tool) is None:
            parser.error(f"Required build tool not found: {tool}")
    torch_root, npu_root = package_root("torch"), package_root("torch_npu")
    abi = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(int(torch.compiled_with_cxx11_abi()))"],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}, text=True,
    ).strip()
    if abi not in ("0", "1"):
        raise RuntimeError(f"Unexpected Torch ABI: {abi!r}")
    build.mkdir(parents=True, exist_ok=True)
    package = root / "runtime"
    sdk = package_root("shmem")
    if version("cann-shmem") != "1.6.0":
        raise RuntimeError("This build requires the official cann-shmem==1.6.0 headers and runtime")
    sdk_lib = sdk / "backends/910"
    if not (sdk_lib / "libshmem.so").is_file():
        raise RuntimeError("The installed cann-shmem package has no Ascend 910 runtime")

    # The 1.6.0 wheel omits src/host_device headers required by its device SDK.
    # Fetch the exact commit recorded in version.info; never build or patch SHMEM.
    commit = "aee374a396b389626f68fab74ec8c555e18978a7"
    if commit not in (sdk / "version.info").read_text():
        raise RuntimeError("cann-shmem binary provenance does not match the pinned SDK headers")
    checkout = args.shmem_source or build / "shmem-upstream"
    if not checkout.exists():
        run("git", "init", checkout)
        run("git", "-C", checkout, "remote", "add", "origin", "https://gitcode.com/cann/shmem.git")
    exists = subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", f"{commit}^{{commit}}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not exists:
        run("git", "-C", checkout, "fetch", "--depth", "1", "origin", commit)
    headers = build / "shmem-headers"
    headers.mkdir(exist_ok=True)
    archive = build / "shmem-headers.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(checkout), "archive", commit, "include", "src"], stdout=stream, check=True)
    run("tar", "xf", archive, "-C", headers)

    def cmake(source: Path, name: str, *options: str, prefix: Path | None = None,
              binary: bool = False) -> Path:
        directory = build / name
        run("cmake", "-S", source, "-B", directory, "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_LINK_DEPENDS_USE_LINKER=FALSE", *options)
        run("cmake", "--build", directory, "--parallel", args.jobs)
        if binary:
            run("cmake", "--build", directory, "--parallel", args.jobs, "--target", "binary")
        command = ["cmake", "--install", directory]
        if prefix is not None:
            command.extend(["--prefix", str(prefix)])
        run(*command)
        return directory

    print("BUILD_PHASE mega_moe_kernels", flush=True)
    vendor_stage = build / "vendor-stage"
    cmake(native / "cmake/vendor", "vendor", f"-DASCEND_CANN_PACKAGE_PATH={cann}",
          "-DASCEND_COMPUTE_UNIT=ascend910_93", f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
          f"-DMEGA_MOE_SOURCE_ROOT={native / 'operators'}", f"-DSHMEM_SDK_ROOT={headers}",
          f"-DVENDOR_STAGE_ROOT={vendor_stage}", binary=True)
    vendor = vendor_stage / "packages/vendors/mega_moe"
    shutil.copytree(vendor, package / "lib/vendors/mega_moe", dirs_exist_ok=True)
    print("BUILD_PHASE mega_moe_torch", flush=True)
    cmake(native / "torch", "torch-bindings", f"-DMEGA_MOE_VENDOR_ROOT={vendor}",
          f"-DPython3_EXECUTABLE={sys.executable}", prefix=package / "lib/framework/torch")

    print("BUILD_COMPLETE " + str(package), flush=True)


if __name__ == "__main__":
    main()
