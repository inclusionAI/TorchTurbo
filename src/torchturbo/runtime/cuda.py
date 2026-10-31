from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from .base import BaseProfiler, DeviceRuntime, TurboDevice


class _CudaProfiler(BaseProfiler):
    def __init__(
        self,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        record_shapes: bool,
        aic_metrics: bool,
        with_stack: bool,
        profiling_step: int | None,
        memory_snapshot_steps: int | None,
    ) -> None:
        # aic_metrics is NPU-only; CUDA has no equivalent knob.
        del aic_metrics
        self._profiler = None
        if profiling_enabled:
            # Training step N (1-based) runs inside the ProfilerStep#(N-1)
            # record_function window, so profiling_step=N maps to
            # skip_first=N-1. With no profiling_step, default to step 1.
            skip_first = (
                profiling_step - 1 if profiling_step is not None else 0
            )
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1, skip_first=skip_first
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_dir, worker_name=worker_name
                ),
                record_shapes=record_shapes,
                profile_memory=False,
                with_stack=with_stack,
                with_modules=False,
                with_flops=False,
            )
        self._snapshot_path = os.path.join(
            os.path.dirname(output_dir),
            "memory_snapshot",
            f"{worker_name}.pickle",
        )
        self._memory_snapshot_steps = memory_snapshot_steps
        self._memory_snapshot_active = False
        self._step = 0

    def start(self) -> None:
        if self._profiler is not None:
            self._profiler.start()
        if self._memory_snapshot_steps is not None:
            torch.cuda.memory._record_memory_history()
            self._memory_snapshot_active = True

    def step(self) -> None:
        self._step += 1
        if self._memory_snapshot_active and self._step == self._memory_snapshot_steps:
            self._memory_snapshot_active = False
            try:
                torch.cuda.memory._dump_snapshot(self._snapshot_path)
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.step()

    def stop(self) -> None:
        if self._memory_snapshot_active:
            self._memory_snapshot_active = False
            torch.cuda.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.stop()


class CudaRuntime(DeviceRuntime):
    turbo_device = TurboDevice.CUDA
    dist_backend = "nccl"

    @property
    def supports_async_streams(self) -> bool:
        return True

    def use_deterministic_algorithms(self) -> None:
        super().use_deterministic_algorithms()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def initialize(self) -> None:
        return None

    def available(self) -> bool:
        return torch.cuda.is_available()

    def current_device(self):
        return torch.cuda.current_device()

    def set_device(self, device: int) -> None:
        torch.cuda.set_device(device)

    def synchronize(self) -> None:
        if self.available():
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        if self.available():
            torch.cuda.empty_cache()

    def memory_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info()

    def memory_reserved(self) -> int:
        return torch.cuda.memory_reserved()

    def memory_allocated(self) -> int:
        return torch.cuda.memory_allocated()

    def stream(self, stream=None):
        if stream is None:
            return nullcontext()
        return torch.cuda.stream(stream)

    def current_stream(self):
        if not self.available():
            return None
        return torch.cuda.current_stream()

    def new_stream(self, priority: int = 0):
        return torch.cuda.Stream(priority=priority)

    def new_event(self):
        return torch.cuda.Event()

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        record_shapes: bool,
        aic_metrics: bool,
        with_stack: bool,
        profiling_step: int | None,
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        return _CudaProfiler(
            output_dir,
            worker_name,
            profiling_enabled,
            record_shapes,
            aic_metrics,
            with_stack,
            profiling_step,
            memory_snapshot_steps,
        )
