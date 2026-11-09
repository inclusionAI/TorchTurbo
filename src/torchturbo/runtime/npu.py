from __future__ import annotations

import ctypes
import os
from contextlib import nullcontext
from functools import lru_cache
from typing import ClassVar

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


class _ShmemHandle:
    """Keep the symmetric allocation, EP team, and device pointer table alive."""

    def __init__(self, symmetric_memory, team: int, device: torch.device) -> None:
        self.symmetric_memory = symmetric_memory
        self.team = team
        self.peer_ptrs = torch.tensor(symmetric_memory.buffer_ptrs, dtype=torch.int64, device=device)

    @property
    def world_size(self) -> int:
        return self.symmetric_memory.world_size

    @property
    def buffer_ptrs(self) -> list[int]:
        return self.symmetric_memory.buffer_ptrs

    def get_buffer(self, rank: int, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        return self.symmetric_memory.get_buffer(rank, shape, dtype)

    def barrier(self) -> None:
        """Order prior peer accesses and wait for this EP team on the current stream."""
        NpuRuntime.shmem_library().aclshmemx_barrier_on_stream(
            self.team, torch.npu.current_stream().npu_stream,
        )


class _NpuProfiler(BaseProfiler):
    def __init__(
        self,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> None:
        self._profiler = None
        if profiling_enabled:
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                mstx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            self._profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=10, active=1, repeat=1, skip_first=1
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir, worker_name=worker_name
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
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
    _shmem_teams: ClassVar[dict[tuple[int, ...], int]] = {}

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

    @staticmethod
    @lru_cache(maxsize=1)
    def shmem_library() -> ctypes.CDLL:
        """Load the Ascend SHMEM library and configure its team and barrier APIs."""
        library = ctypes.CDLL("libshmem.so")
        library.aclshmem_team_split_strided.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_int)]
        library.aclshmem_team_split_strided.restype = ctypes.c_int
        library.aclshmemx_barrier_on_stream.argtypes = [ctypes.c_int, ctypes.c_void_p]
        library.aclshmemx_barrier_on_stream.restype = None
        return library

    @classmethod
    def _initialize_shmem_teams(cls, groups: list[tuple[int, ...]], world_size: int) -> None:
        """Create EP teams in the same order on every WORLD rank."""
        library = cls.shmem_library()
        world = tuple(range(world_size))
        cls._shmem_teams[world] = 0
        for members in sorted(set(groups)):
            if members in cls._shmem_teams:
                continue
            stride = members[1] - members[0] if len(members) > 1 else 1
            if members != tuple(range(members[0], members[-1] + 1, stride)):
                raise ValueError("NPU SHMEM EP groups must have evenly spaced WORLD ranks.")
            team = ctypes.c_int(-1)
            result = library.aclshmem_team_split_strided(
                0, members[0], stride, len(members), ctypes.byref(team),
            )
            if result != 0:
                raise RuntimeError(f"SHMEM EP team creation failed with error {result}.")
            cls._shmem_teams[members] = team.value

    def allocate_communication_buffer(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: torch.distributed.ProcessGroup | None,
    ) -> tuple[torch.Tensor, object]:
        """Allocate SHMEM on single-host WORLD and map peers in the given group.

        Every WORLD rank must allocate the same layouts in the same order.
        The installed NPUSHMEM allocator initializes WORLD even for EP subgroups.
        """
        import socket

        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        import torch_npu

        layouts = [None] * dist.get_world_size()
        ep_group = dist.group.WORLD if group is None else group
        members = tuple(dist.get_process_group_ranks(ep_group))
        layout = (socket.gethostname(), numel, dtype, members)
        dist.all_gather_object(layouts, layout)
        if any(peer[:3] != layout[:3] for peer in layouts):
            raise RuntimeError(
                "NPU SHMEM requires a single-host WORLD with identical buffer sizes and dtypes "
                "on every rank."
            )
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        # NPUSHMEM alloc rejects a group argument; rendezvous selects the EP peers.
        allocation = symm_mem.empty(numel, dtype=dtype, device=device)
        symmetric_memory = symm_mem.rendezvous(allocation, ep_group)
        self._initialize_shmem_teams([peer[3] for peer in layouts], len(layouts))
        handle = _ShmemHandle(symmetric_memory, self._shmem_teams[members], device)
        # PyTorch from_blob creates generic storage, which HCCL cannot inspect.
        # Borrow the same address through NPUStorageImpl and initialize its ND
        # descriptor with set_. The handle owns the underlying SHMEM allocation.
        storage = torch_npu._C._construct_storage_from_data_pointer(
            allocation.data_ptr(), device, allocation.numel() * allocation.element_size(),
        )
        arena = torch.empty(0, dtype=dtype, device=device).set_(storage, 0, (numel,), (1,))
        return arena, handle

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
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        return _NpuProfiler(
            output_dir,
            worker_name,
            profiling_enabled,
            memory_snapshot_steps,
        )
