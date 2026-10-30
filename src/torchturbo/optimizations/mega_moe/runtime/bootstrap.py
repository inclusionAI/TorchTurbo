"""Activate this component's native loader paths before importing Torch."""
import os
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys


NATIVE_LIBRARIES: tuple[str, ...] = (
    "lib/vendors/mega_moe/op_api/lib/libcust_opapi.so",
    "lib/framework/torch/libmega_moe_torch.so",
    "moonep/libmoonep_shared_kernels.so",
    "moonep/_moonep_shared_bridge.so",
    "moonep/_a3_allocator.so",
)


def activate() -> bool:
    package = Path(__file__).resolve().parent
    vendor = package / "lib/vendors/mega_moe"
    missing = [name for name in NATIVE_LIBRARIES if not (package / name).is_file()]
    if missing:
        raise RuntimeError(
            f"MegaMoE native libraries are missing: {missing}. "
            "Run bash scripts/runs/deps/mega_moe.sh once in the training environment."
        )
    spec = importlib.util.find_spec("shmem")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "MegaMoE requires official cann-shmem==1.6.0. "
            "Run bash scripts/runs/deps/mega_moe.sh once in the training environment."
        )
    if version("cann-shmem") != "1.6.0":
        raise RuntimeError("MegaMoE requires cann-shmem==1.6.0; run bash scripts/runs/deps/mega_moe.sh")
    shmem_lib = Path(spec.origin).resolve().parent / "backends/910"
    if not all((shmem_lib / name).is_file() for name in ("libshmem.so", "libshmem_utils.so")):
        raise RuntimeError("The installed cann-shmem package has no complete Ascend 910 runtime")
    paths = {
        "ASCEND_CUSTOM_OPP_PATH": [vendor],
        "LD_LIBRARY_PATH": [vendor / "op_api/lib", shmem_lib],
    }
    missing = {}
    for variable, entries in paths.items():
        current = os.environ.get(variable, "")
        registered = {Path(value).resolve() for value in current.split(os.pathsep) if value}
        pending = [str(path) for path in entries if path.resolve() not in registered]
        if pending:
            missing[variable] = os.pathsep.join([*pending, current]).rstrip(os.pathsep)
    if not missing:
        return True
    os.environ.update(missing)
    # The dynamic loader reads LD_LIBRARY_PATH when the process starts.
    os.execv(sys.executable, [sys.executable, *sys.argv])
    return True
