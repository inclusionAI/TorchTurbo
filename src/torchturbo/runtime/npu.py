from __future__ import annotations

import ctypes
import os
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from functools import cached_property, partial
from math import ceil, prod
from threading import Lock

import torch
import torch.distributed as dist
from torch.distributed.collective_utils import all_gather
from torch.multiprocessing.reductions import StorageWeakRef

from .base import BaseProfiler, DeviceRuntime, ShmemHandle, TorchShmem, TurboDevice


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

    def __init__(self) -> None:
        self.shmem = NpuShmem()
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
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        return _NpuProfiler(
            output_dir,
            worker_name,
            profiling_enabled,
            memory_snapshot_steps,
        )


class _NpuShmemHandle:
    """Peer addresses and a stream-ordered fence for one tensor in a process group."""

    def __init__(self, tensor: torch.Tensor, memory: ShmemHandle, barrier: Callable[[], None]) -> None:
        self._tensor = tensor
        self.barrier = barrier
        self.buffer_ptrs = memory.buffer_ptrs
        self.world_size = memory.world_size
        # post7 exposes the peer addresses only on the host.
        self._device_ptrs = torch.tensor(self.buffer_ptrs, dtype=torch.int64, device=tensor.device)

    @property
    def buffer_ptrs_dev(self) -> int:
        self._device_ptrs.record_stream(torch.npu.current_stream(self._tensor.device))
        return self._device_ptrs.data_ptr()

    @staticmethod
    def _view(ptr: int, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, owner) -> torch.Tensor:
        import torch_npu

        # PyTorch's from_blob storage lacks the NPU descriptor required by HCCL.
        storage = torch_npu._C._construct_storage_from_data_pointer(ptr, device, prod(shape) * dtype.itemsize)
        storage._torchturbo_shmem_owner = owner
        return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, shape)

    def get_buffer(self, rank: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        size = prod(shape) * dtype.itemsize
        if not 0 <= rank < self.world_size or any(n < 0 for n in shape) or size > self._tensor.nbytes:
            raise ValueError("Requested SHMEM view is outside its window.")
        return self._view(self.buffer_ptrs[rank], shape, dtype, self._tensor.device, self)


class NpuShmem(TorchShmem):
    """Allocate symmetric buffers on WORLD and map peers in the caller's process group."""

    def __init__(self) -> None:
        self._world: weakref.ReferenceType[dist.ProcessGroup] | None = None
        self._capacity: int | None = None
        self._buffers: dict[torch.Tensor, StorageWeakRef | None] = {}
        self._team_barriers: dict[tuple[int, ...], Callable[[], None] | None] = {}
        self._failed = False

    @cached_property
    def _library(self) -> ctypes.CDLL:
        library = ctypes.CDLL("libshmem.so")
        # torch-npu resolves these symbols only when its allocator first runs.
        for name in (
            "aclshmemx_set_conf_store_tls", "aclshmemx_get_uniqueid", "aclshmemx_set_attr_uniqueid_args",
            "aclshmemx_init_attr", "aclshmem_malloc", "aclshmem_free", "aclshmem_ptr",
        ):
            getattr(library, name)
        library.aclshmem_team_split_strided.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_int)]
        library.aclshmem_team_split_strided.restype = ctypes.c_int
        library.aclshmemx_barrier_on_stream.argtypes = [ctypes.c_int, ctypes.c_void_p]
        # 1.8 introduced sync_on_stream and changed the barrier return type from void to int.
        library.aclshmemx_barrier_on_stream.restype = (
            ctypes.c_int if getattr(library, "aclshmemx_sync_on_stream", None) is not None else None
        )
        return library

    def available(self) -> bool:
        try:
            import torch_npu
            from packaging.version import Version
            from torch.distributed._symmetric_memory import empty, rendezvous  # noqa: F401

            # Older allocators retain rendezvous handles after the tensor is freed.
            return Version(torch_npu.__version__) >= Version("2.10.0.post4") and bool(self._library)
        except (ImportError, OSError, AttributeError):
            return False

    def prepare_heap(self, buffer_sizes: list[int | None]) -> int:
        """Compare allocation requests across WORLD, including ranks with no peer buffers."""
        return self._prepare_heap(buffer_sizes, dtype=None)

    def empty(
        self, numel: int, *, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Allocate on WORLD and prepare the group's stream barrier after the allocator bootstraps SHMEM."""
        import torch.distributed._symmetric_memory as symm_mem

        capacity = self._prepare_heap([numel * dtype.itemsize], dtype)
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.npu.current_device())

        with torch.npu.device(device):
            try:
                if self._world is None:
                    world = dist.group.WORLD
                    symm_mem.enable_symm_mem_for_group(world.group_name)
                    self._world = weakref.ref(world)
                raw = super().empty(max(1, numel), dtype=dtype, device=device)
            except Exception:
                # Peers may still be in native bootstrap and cannot enter another collective.
                self._failed = True
                raise

            # Pin before constructing the local view; native free is WORLD-collective.
            self._buffers[raw] = None
            error = None
            try:
                tensor = _NpuShmemHandle._view(raw.data_ptr(), (numel,), dtype, device, raw)
                self._buffers[raw] = StorageWeakRef(tensor.untyped_storage())
            except Exception as exc:
                error = str(exc)
        group = dist.group.WORLD if group is None else group
        members = tuple(dist.get_process_group_ranks(group))
        requests = all_gather((members, error), pg=dist.group.WORLD)
        for rank, (_, failure) in enumerate(requests):
            if failure is not None:
                self._failed = True
                raise RuntimeError(f"SHMEM allocation failed on rank {rank}: {failure}")
        self._capacity = capacity
        with torch.npu.device(device):
            self._prepare_teams([ranks for ranks, _ in requests], device)
        return tensor

    def _make_team_barrier(self, team: int, device: torch.device) -> Callable[[], None]:
        # All windows in a team share one SHMEM barrier counter.
        launch = self._library.aclshmemx_barrier_on_stream
        done = torch.npu.Event()
        lock = Lock()

        def barrier() -> None:
            with lock, torch.npu.device(device):
                stream = torch.npu.current_stream(device)
                stream.wait_event(done)
                status = launch(team, stream.npu_stream)
                if status:
                    raise RuntimeError(f"SHMEM team {team} barrier failed with status {status}.")
                done.record(stream)

        return barrier

    def _prepare_teams(self, groups: list[tuple[int, ...]], device: torch.device) -> None:
        # Split is collective on its parent. All WORLD ranks visit the same memberships once.
        # Keep team IDs for this WORLD: SHMEM's barrier counters outlive individual windows.
        for ranks in sorted(set(groups).difference(self._team_barriers)):
            if len(ranks) == len(groups):
                self._team_barriers[ranks] = self._make_team_barrier(0, device)
                continue
            if len(ranks) == 1 or ranks != tuple(range(ranks[0], ranks[-1] + 1, ranks[1] - ranks[0])):
                self._team_barriers[ranks] = None
                continue
            team = ctypes.c_int(-1)
            status = self._library.aclshmem_team_split_strided(
                0, ranks[0], ranks[1] - ranks[0], len(ranks), ctypes.byref(team),
            )
            results = all_gather((status, team.value), pg=dist.group.WORLD)
            member_ids = {results[rank][1] for rank in ranks}
            if any(code for code, _ in results) or len(member_ids) != 1 or -1 in member_ids:
                self._failed = True
                raise RuntimeError(f"Could not create a common SHMEM team for ranks {ranks}: {results}")
            self._team_barriers[ranks] = self._make_team_barrier(team.value, device) if team.value >= 0 else None

    def rendezvous(self, tensor: torch.Tensor, group: dist.ProcessGroup | None) -> _NpuShmemHandle:
        group = dist.group.WORLD if group is None else group
        memory = super().rendezvous(tensor, group)
        barrier = self._team_barriers.get(tuple(dist.get_process_group_ranks(group)))
        if memory.world_size == 1:
            barrier = lambda: None
        elif barrier is None:
            # A raw tensor may be mapped later by an unrelated subgroup without WORLD participation.
            fence = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            barrier = partial(dist.all_reduce, fence, group=group)
        return _NpuShmemHandle(tensor, memory, barrier)

    def release_unused(self) -> None:
        """Free buffers in WORLD order once all local and peer views are gone."""
        if self._world is not None and self._world() is not dist.group.WORLD:
            raise RuntimeError("Native SHMEM retains its original WORLD. Restart the processes after recreating WORLD.")
        if self._failed:
            raise RuntimeError("An earlier SHMEM allocation failed. Restart the affected processes.")
        if not self._buffers:
            return
        entries = list(self._buffers)
        unused = {i for i, storage in enumerate(self._buffers.values()) if storage.expired()}
        reclaim = set.intersection(*all_gather(unused, pg=dist.group.WORLD))
        if reclaim:
            torch.npu.synchronize(entries[0].device)
            # Drop the final owning references in the same order on every rank.
            for i in sorted(reclaim, reverse=True):
                del self._buffers[entries[i]]
                entries[i] = None

    def _heap_capacity(self, sizes: list[int]) -> int:
        if any(size < 0 for size in sizes):
            raise ValueError("SHMEM buffer sizes must be nonnegative.")
        sizes = [raw.nbytes for raw in self._buffers] + sizes
        # Ascend SHMEM rounds each allocation up to 16 bytes.
        required = sum(max(16, (size + 15) // 16 * 16) for size in sizes)
        page = 2 << 20
        suggested = (required + page - 1) // page * page
        capacity = self._capacity
        if capacity is None:
            setting = os.environ.get("NPU_SHMEM_SYMMETRIC_SIZE")
            if setting is None:
                return max(1 << 30, suggested)
            scales = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
            suffix = setting[-1:].upper()
            number = setting[:-1] if suffix in scales else setting
            try:
                capacity = ceil(float(number) * scales.get(suffix, 1))
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid NPU_SHMEM_SYMMETRIC_SIZE={setting!r}.") from exc
        if capacity < required:
            raise ValueError(f"SHMEM needs {required} bytes per rank. Set NPU_SHMEM_SYMMETRIC_SIZE={suggested} or larger.")
        return capacity

    def _prepare_heap(self, sizes: list[int | None], dtype: torch.dtype | None) -> int:
        self.release_unused()
        # One exchange checks the requested layout and any rank-local sizing error before malloc.
        capacity = 0
        error = None
        try:
            allocations = [size for size in sizes if size is not None]
            if allocations:
                capacity = self._heap_capacity(allocations)
        except ValueError as exc:
            error = str(exc)
        layout = sizes, dtype, capacity, error
        layouts = all_gather(layout, pg=dist.group.WORLD)
        for _, _, _, failure in layouts:
            if failure is not None:
                raise ValueError(failure)
        if any(peer != layout for peer in layouts):
            details = "; ".join(
                f"rank {rank}: {buffers}, dtype={peer_dtype}, capacity={peer_capacity}"
                for rank, (buffers, peer_dtype, peer_capacity, _) in enumerate(layouts)
            )
            raise RuntimeError(f"SHMEM buffer sizes, dtypes and heap capacities must match across WORLD. {details}")
        if self._capacity is None and capacity > 1 << 30:
            os.environ.setdefault("NPU_SHMEM_SYMMETRIC_SIZE", str(capacity))
        return capacity
