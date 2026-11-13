"""Defer automatic cyclic GC to optimizer step boundaries."""

from __future__ import annotations

from contextlib import AbstractContextManager
import gc
from types import TracebackType

from ..runtime import TurboDevice, get_device_runtime
from ..runtime.cpu import CpuRuntime


def _under_memory_pressure() -> bool:
    free, total = CpuRuntime().memory_info()
    if free < total * 0.1:
        return True

    runtime = get_device_runtime()
    if runtime.turbo_device == TurboDevice.CPU:
        return False
    free, total = runtime.memory_info()
    # Allocator cache can satisfy tensor allocations without driver free memory.
    reusable = runtime.memory_reserved() - runtime.memory_allocated()
    return free + reusable < total * 0.1


class GarbageCollector(AbstractContextManager["GarbageCollector"]):
    """Pause cyclic GC during a step and restore its caller-owned state.

    Call ``begin_step`` before the first accumulation micro-batch and
    ``end_step`` after the optimizer update. Between steps, automatic GC
    resumes with unchanged thresholds. Less than 10% available host or device
    memory triggers a full collection, including old-generation cycles.
    Reference counting remains active throughout. An enclosing scope restores
    GC even when training exits before ``end_step``. Scopes must be nested
    because Python's GC state is process-wide.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._step_gc_enabled: bool | None = None

    def __enter__(self) -> GarbageCollector:
        self._depth += 1
        return self

    def begin_step(self) -> None:
        """Pause automatic GC for one complete optimizer step."""
        if self._depth == 0:
            raise RuntimeError("begin_step() requires an active GarbageCollector scope")
        if self._step_gc_enabled is not None:
            return
        self._step_gc_enabled = gc.isenabled()
        gc.disable()

    def end_step(self) -> None:
        """Restore automatic GC and collect fully when memory is constrained."""
        if self._restore_gc() and _under_memory_pressure():
            gc.collect()

    def _restore_gc(self) -> bool:
        enabled = self._step_gc_enabled
        self._step_gc_enabled = None
        if enabled is None:
            return False
        if enabled:
            gc.enable()
        else:
            gc.disable()
        return enabled

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._restore_gc()
