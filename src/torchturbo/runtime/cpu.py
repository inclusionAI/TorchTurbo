from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from .base import BaseProfiler, DeviceRuntime, TurboDevice


class CpuRuntime(DeviceRuntime):
    turbo_device = TurboDevice.CPU
    dist_backend = "gloo"

    def available(self) -> bool:
        return True

    def initialize(self) -> None:
        return None

    def current_device(self):
        return 0

    def set_device(self, device: int) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def memory_info(self) -> tuple[int, int]:
        try:
            import psutil  # type: ignore
        except ImportError:
            psutil = None

        if psutil is not None:
            vm = psutil.virtual_memory()
            return int(vm.available), int(vm.total)

        if hasattr(os, "sysconf"):
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                phys_pages = os.sysconf("SC_PHYS_PAGES")
                avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            except (AttributeError, OSError, ValueError, TypeError):
                pass
            else:
                return int(page_size * avail_pages), int(page_size * phys_pages)

        raise NotImplementedError(
            f"{type(self).__name__} does not implement memory_info()."
        )

    def stream(self, stream=None):
        return nullcontext()

    def current_stream(self):
        return None

    def new_stream(self, priority: int = 0):
        return None

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        record_shapes: bool,
        with_stack: bool,
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement profiler()."
        )
