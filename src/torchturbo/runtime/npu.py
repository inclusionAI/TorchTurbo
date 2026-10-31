from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from .base import BaseProfiler, DeviceRuntime, TurboDevice


# Ascend environment variable reference:
# https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_001052.html
_NPU_ENV_DEFAULTS = {
    "HCCL_CONNECT_TIMEOUT": "7200",
    "HCCL_EXEC_TIMEOUT": "7200",
    "HCCL_EVENT_TIMEOUT": "14500",
    "ACL_DEVICE_SYNC_TIMEOUT": "14400",
    # AI_CPU is default for A3; AIV does not support multi-comm-domain parallelism.
    "HCCL_OP_EXPANSION_MODE": "AI_CPU",
    "ASCEND_GLOBAL_LOG_LEVEL": "3",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_ASYNC_ERROR_HANDLING": "0",
}

_NPU_DETERMINISTIC_ENV = {
    "HCCL_DETERMINISTIC": "True",
    "LCCL_DETERMINISTIC": "1",
    "CLOSE_MATMUL_K_SHIFT": "1",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


class _NpuProfiler(BaseProfiler):
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
        self._profiler = None
        if profiling_enabled:
            import torch_npu

            # Level0 omits input shapes from kernel_details.csv and drops the
            # AiCore utilization columns; Level1 emits both. aic_metrics needs
            # a non-None value for the utilization counters to be collected.
            detailed_kernel_info = record_shapes or aic_metrics
            profiler_level = (
                torch_npu.profiler.ProfilerLevel.Level1
                if detailed_kernel_info
                else torch_npu.profiler.ProfilerLevel.Level0
            )
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=profiler_level,
                mstx=False,
                aic_metrics=(
                    torch_npu.profiler.AiCMetrics.PipeUtilization
                    if aic_metrics
                    else torch_npu.profiler.AiCMetrics.AiCoreNone
                ),
                l2_cache=False,
                op_attr=False,
                data_simplification=True,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            # Training step N (1-based) runs inside the ProfilerStep#(N-1)
            # record_function window, so profiling_step=N maps to
            # skip_first=N-1. With no profiling_step, default to step 1.
            skip_first = (
                profiling_step - 1 if profiling_step is not None else 0
            )
            self._profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1, skip_first=skip_first
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir, worker_name=worker_name
                ),
                record_shapes=record_shapes,
                profile_memory=False,
                with_stack=with_stack,
                with_modules=False,
                with_flops=False,
                experimental_config=experimental_config,
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
            torch.npu.memory._record_memory_history()
            self._memory_snapshot_active = True

    def step(self) -> None:
        self._step += 1
        if self._memory_snapshot_active and self._step == self._memory_snapshot_steps:
            self._memory_snapshot_active = False
            try:
                torch.npu.memory._dump_snapshot(self._snapshot_path)
            finally:
                torch.npu.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.step()

    def stop(self) -> None:
        if self._memory_snapshot_active:
            self._memory_snapshot_active = False
            torch.npu.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.stop()


class NpuRuntime(DeviceRuntime):
    turbo_device = TurboDevice.NPU
    dist_backend = "hccl"

    def __init__(self) -> None:
        self._initialized = False

    @property
    def supports_async_streams(self) -> bool:
        return True

    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return hasattr(torch, "npu") and torch.npu.is_available()

    def initialize(self) -> None:
        if self._initialized:
            return
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu as _  # noqa: F401
        for name, value in _NPU_ENV_DEFAULTS.items():
            os.environ.setdefault(name, value)

        self._initialized = True

    def use_deterministic_algorithms(self) -> None:
        super().use_deterministic_algorithms()
        for name, value in _NPU_DETERMINISTIC_ENV.items():
            os.environ[name] = value

    @property
    def _npu(self):
        return torch.npu

    def current_device(self):
        return self._npu.current_device()

    def set_device(self, device: int) -> None:
        self._npu.set_device(device)

    def synchronize(self) -> None:
        if self.available():
            self._npu.synchronize()

    def empty_cache(self) -> None:
        if self.available():
            self._npu.empty_cache()

    def memory_info(self) -> tuple[int, int]:
        if not hasattr(self._npu, "mem_get_info"):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement memory_info()."
            )
        return self._npu.mem_get_info()

    def memory_reserved(self) -> int:
        return self._npu.memory_reserved()

    def memory_allocated(self) -> int:
        return self._npu.memory_allocated()

    def stream(self, stream=None):
        if stream is None:
            return nullcontext()
        return self._npu.stream(stream)

    def current_stream(self):
        if not self.available():
            return None
        return self._npu.current_stream()

    def new_stream(self, priority: int = 0):
        try:
            return self._npu.Stream(priority=priority)
        except TypeError:
            return self._npu.Stream()

    def new_event(self):
        return self._npu.Event()

    def manual_seed_all(self, seed: int) -> None:
        self._npu.manual_seed_all(seed)

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
        return _NpuProfiler(
            output_dir,
            worker_name,
            profiling_enabled,
            record_shapes,
            aic_metrics,
            with_stack,
            profiling_step,
            memory_snapshot_steps,
        )
