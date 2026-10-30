"""Load MoonEP kernels against the component-owned SHMEM runtime."""
from functools import lru_cache
import importlib
import subprocess
import sys
from types import ModuleType

import torch


@lru_cache(maxsize=None)
def _device_ub_bytes(device_index: int) -> int:
    """Query CANN for this device without changing the caller's compiler SOC."""
    soc = torch.npu.get_device_name(device_index)
    script = (
        "import sys; from tbe.common import platform; "
        "platform.set_current_compile_soc_info(sys.argv[1]); "
        "print(platform.get_soc_spec('UB_SIZE'))"
    )
    size = int(subprocess.check_output([sys.executable, "-c", script, soc], text=True, timeout=30).strip())
    if size <= 0:
        raise RuntimeError(f"CANN returned an invalid UB size for {soc}: {size}")
    return size


@lru_cache(maxsize=1)
def load_allocator() -> ModuleType:
    """Load the device allocator without initializing another SHMEM runtime."""
    module = importlib.import_module(__name__ + '._a3_allocator')
    native_allocate = module.allocate

    def allocate(counts: torch.Tensor, capacity: int) -> list[torch.Tensor]:
        return native_allocate(counts, capacity, _device_ub_bytes(counts.device.index))

    module.allocate = allocate
    return module


@lru_cache(maxsize=1)
def load_bridge() -> ModuleType:
    """Use the same SHMEM library as the fused MegaMoE operators."""
    return importlib.import_module(__name__ + '._moonep_shared_bridge')
