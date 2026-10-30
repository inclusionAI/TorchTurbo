"""Collective tensor ownership on the official cann-shmem runtime."""

from __future__ import annotations

import math
import threading
from typing import Any

import torch
import torch.distributed as dist
import shmem as cann

_lock = threading.RLock()
_users = 0
_group: Any = None
_members: tuple[int, ...] = ()
_heap_bytes = 0
_allocations: dict[int, tuple[torch.Tensor, int]] = {}


def acquire(root_group: Any = None, *, heap_size_bytes: int | None = None) -> None:
    """Initialize one EP root or retain its existing fixed symmetric heap."""
    global _users, _group, _members, _heap_bytes
    with _lock:
        members = tuple(dist.get_process_group_ranks(root_group or dist.group.WORLD))
        if _users:
            if members != _members:
                raise RuntimeError("MoonEP requires one active EP group per process")
            if heap_size_bytes is not None and heap_size_bytes > _heap_bytes:
                raise RuntimeError("The active MoonEP heap is smaller than the requested layout")
            _users += 1
            return
        if not isinstance(heap_size_bytes, int) or heap_size_bytes <= 0:
            raise ValueError("The first MoonEP owner must specify a positive heap size")
        rank = dist.get_rank(root_group)
        payload: list[Any] = [None, None]
        if rank == 0:
            try:
                payload[0] = cann.aclshmem_get_unique_id()
            except Exception as error:
                payload[1] = str(error)
        dist.broadcast_object_list(payload, src=members[0], group=root_group)
        if payload[1] is not None:
            raise RuntimeError(f"cann-shmem bootstrap failed: {payload[1]}")
        status = cann.aclshmem_init_using_unique_id(rank, len(members), heap_size_bytes, payload[0])
        if status != 0:
            raise RuntimeError(f"cann-shmem initialization failed: {status}")
        _group, _members, _heap_bytes, _users = root_group, members, heap_size_bytes, 1


def host_barrier() -> None:
    """Complete current-stream work before every EP rank reaches the barrier."""
    if not _users:
        raise RuntimeError("MoonEP SHMEM is not initialized")
    dist.barrier(group=_group)


def empty(shape: tuple[int, ...], *, dtype: torch.dtype, alignment: int = 512) -> torch.Tensor:
    """Allocate a symmetric tensor; the owning backend frees it collectively."""
    with _lock:
        if not _users:
            raise RuntimeError("MoonEP SHMEM is not initialized")
        if any(size < 0 for size in shape):
            raise ValueError("Symmetric tensor dimensions must be nonnegative")
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        pointer = cann.aclshmem_align(alignment, max(size, 1))
        if not pointer:
            raise RuntimeError(f"cann-shmem could not allocate {size} bytes")
        try:
            tensor = cann.construct_tensor_from_ptr(pointer, shape, dtype, torch.device("npu", torch.npu.current_device()))
        except Exception:
            cann.aclshmem_free(pointer)
            raise
        _allocations[pointer] = (tensor, size)
        return tensor


def free(tensor: torch.Tensor) -> None:
    """Free an original allocation after the caller has synchronized all users."""
    with _lock:
        pointer = tensor.data_ptr()
        record = _allocations.get(pointer)
        if record is None or record[0] is not tensor:
            raise ValueError("Only an original live MoonEP allocation can be freed")
        cann.aclshmem_free(pointer)
        del _allocations[pointer]


def release() -> None:
    """Finalize the official runtime after the final owner releases all tensors."""
    global _users, _group, _members, _heap_bytes
    with _lock:
        if _users <= 0:
            raise RuntimeError("MoonEP SHMEM reference was already released")
        if _users > 1:
            _users -= 1
            return
        torch.npu.synchronize()
        counts = [None] * len(_members)
        dist.all_gather_object(counts, len(_allocations), group=_group)
        if any(counts):
            raise RuntimeError(f"MoonEP still owns symmetric tensors: {counts}")
        host_barrier()
        status = cann.aclshmem_finalize()
        if status != 0:
            raise RuntimeError(f"cann-shmem finalization failed: {status}")
        _users, _group, _members, _heap_bytes = 0, None, (), 0
