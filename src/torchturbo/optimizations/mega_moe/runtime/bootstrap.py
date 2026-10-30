"""Activate this component's native loader paths before importing Torch."""
import os
import importlib.util
from pathlib import Path
import sys


NATIVE_LIBRARIES: tuple[str, ...] = (
    "lib/vendors/mega_moe/op_api/lib/libcust_opapi.so",
    "lib/framework/torch/libmega_moe_torch.so",
)

def activate() -> bool:
    package = Path(__file__).resolve().parent
    vendor = package / "lib/vendors/mega_moe"
    if not vendor.is_dir():
        raise RuntimeError(f"MegaMoE native payload is missing: {vendor}")
    spec = importlib.util.find_spec("shmem")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install official cann-shmem==1.6.0 before enabling mega_moe")
    shmem_lib = Path(spec.origin).resolve().parent / "backends/910"
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
