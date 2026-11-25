"""Check WORLD allocation and EP rendezvous with PyTorch tensors and Gloo groups."""

import os
import sys
import weakref
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.runtime.npu import NpuRuntime


def _check_windows(rank, directory):
    import torch.distributed._symmetric_memory as symm_mem

    timeout = timedelta(seconds=30)
    shmem = NpuRuntime().shmem
    dist.init_process_group("gloo", init_method=f"file://{directory}/first", rank=rank, world_size=2, timeout=timeout)
    try:
        ep = [dist.new_group([peer]) for peer in range(2)][rank]
        allocations = []
        device = torch.device("cpu", rank)

        def empty(numel, *, dtype, device):
            tensor = torch.empty(numel, dtype=dtype, device=device)
            allocations.append(weakref.ref(tensor))
            return tensor

        def rendezvous(tensor, group):
            assert group is ep
            assert tensor.untyped_storage().data_ptr() == allocations[-1]().untyped_storage().data_ptr()
            return SimpleNamespace(buffer_ptrs=[tensor.data_ptr()], world_size=dist.get_world_size(group))

        def storage(ptr, device, size):
            return torch._C._construct_storage_from_data_pointer(ptr, torch.device("cpu"), size)

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("NPU_SHMEM_SYMMETRIC_SIZE", "2M")
            patch.setattr(torch, "npu", SimpleNamespace(
                device=nullcontext, synchronize=lambda device: None, Event=torch.Event,
            ), raising=False)
            patch.setattr(type(shmem), "_library", SimpleNamespace(aclshmemx_barrier_on_stream=pytest.fail))
            patch.setitem(sys.modules, "torch_npu", SimpleNamespace(
                _C=SimpleNamespace(_construct_storage_from_data_pointer=storage),
            ))
            patch.setattr(symm_mem, "enable_symm_mem_for_group", lambda name: None)
            patch.setattr(symm_mem, "empty", empty)
            patch.setattr(symm_mem, "rendezvous", rendezvous)

            for layouts in (
                ([None, None, None], [32, 64, 64]),
                ([None, 64, None], [None, None, 64]),
                ([None], [0]),
                ([], [64]),
            ):
                with pytest.raises(RuntimeError, match="must match across WORLD") as mismatch:
                    shmem.prepare_heap(layouts[rank])
                for peer, layout in enumerate(layouts):
                    assert f"rank {peer}: {layout}" in str(mismatch.value)
                assert not allocations
            assert shmem.prepare_heap([None, None, None]) == 0
            with pytest.MonkeyPatch.context() as env:
                env.delenv("NPU_SHMEM_SYMMETRIC_SIZE")
                with pytest.raises(RuntimeError, match="must match across WORLD"):
                    shmem.prepare_heap([(2 + rank) << 30])
                assert "NPU_SHMEM_SYMMETRIC_SIZE" not in os.environ
                assert shmem.prepare_heap([2 << 30]) == 2 << 30
                assert int(os.environ["NPU_SHMEM_SYMMETRIC_SIZE"]) == 2 << 30

            with pytest.raises(RuntimeError, match="must match across WORLD"):
                shmem.empty(7 + rank, dtype=torch.float32, device=device)
            assert not allocations

            arena = shmem.empty(7, dtype=torch.float32, device=device)
            local_view = arena.view(-1) if rank == 0 else None
            del arena
            shmem.release_unused()
            assert allocations[0]() is not None
            if local_view is not None:
                local_view.fill_(7)
                assert allocations[0]().tolist() == [7] * 7
            del local_view
            shmem.release_unused()
            assert allocations[0]() is None

            arena = shmem.empty(7, dtype=torch.float32, device=device)
            # Only the EP members must enter rendezvous after the WORLD allocation.
            if rank == 0:
                handle = shmem.rendezvous(arena, ep)
            dist.barrier()
            if rank == 1:
                handle = shmem.rendezvous(arena, ep)
            assert list(handle.buffer_ptrs) == [arena.data_ptr()]
            alias = handle.get_buffer(0, arena.shape, arena.dtype).view(-1) if rank else None
            del arena, handle
            shmem.release_unused()
            assert allocations[-1]() is not None
            del alias
            shmem.release_unused()
            assert allocations[-1]() is None
    finally:
        dist.destroy_process_group()

    dist.init_process_group("gloo", init_method=f"file://{directory}/second", rank=rank, world_size=2, timeout=timeout)
    try:
        with pytest.raises(RuntimeError, match="original WORLD"):
            shmem.release_unused()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_ep_window_owners_follow_world_lifetime(tmp_path):
    mp.spawn(_check_windows, args=(str(tmp_path),), nprocs=2)


def _check_allocation_errors(rank, directory):
    import torch.distributed._symmetric_memory as symm_mem
    from torchturbo.runtime import npu

    device = torch.device("cpu", rank)
    dist.init_process_group("gloo", init_method=f"file://{directory}/errors", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "npu", SimpleNamespace(device=nullcontext), raising=False)
            patch.setattr(symm_mem, "enable_symm_mem_for_group", lambda name: None)
            patch.setattr(symm_mem, "empty", torch.empty)
            for operation in ("enable_symm_mem_for_group", "empty"):
                shmem = NpuRuntime().shmem
                failure = RuntimeError(f"{operation} failed")
                with pytest.MonkeyPatch.context() as inject:
                    inject.setattr(symm_mem, operation, Mock(side_effect=failure))
                    gather = Mock(wraps=npu.all_gather)
                    inject.setattr(npu, "all_gather", gather)
                    with pytest.raises(RuntimeError) as caught:
                        shmem.empty(7, dtype=torch.float32, device=device)
                    assert caught.value is failure
                    gather.assert_called_once()  # Only the pre-allocation layout check.
                    with pytest.raises(RuntimeError, match="earlier SHMEM allocation failed"):
                        shmem.release_unused()

            def view(ptr, shape, dtype, device, owner):
                if rank == 1:
                    raise ValueError("local view failed")
                return owner.view(shape)

            patch.setattr(npu._NpuShmemHandle, "_view", staticmethod(view))
            shmem = NpuRuntime().shmem
            with pytest.raises(RuntimeError, match="SHMEM allocation failed on rank 1: local view failed"):
                shmem.empty(7, dtype=torch.float32, device=device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_native_and_local_shmem_allocation_errors(tmp_path):
    mp.spawn(_check_allocation_errors, args=(str(tmp_path),), nprocs=2)


def test_prepare_sizes_the_heap_before_its_first_allocation(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("NPU_SHMEM_SYMMETRIC_SIZE", raising=False)
    shmem = NpuRuntime().shmem
    sizes = [32768, 720 << 20, 1440 << 20]
    assert shmem.prepare_heap(sizes) == 2162 << 20
    assert int(os.environ["NPU_SHMEM_SYMMETRIC_SIZE"]) == 2162 << 20
    monkeypatch.setenv("NPU_SHMEM_SYMMETRIC_SIZE", "2G")
    with pytest.raises(ValueError, match="NPU_SHMEM_SYMMETRIC_SIZE"):
        shmem.prepare_heap(sizes)
    monkeypatch.setenv("NPU_SHMEM_SYMMETRIC_SIZE", "2.5G")
    shmem.prepare_heap(sizes)
    assert os.environ["NPU_SHMEM_SYMMETRIC_SIZE"] == "2.5G"
    monkeypatch.setenv("NPU_SHMEM_SYMMETRIC_SIZE", "unused")
    assert shmem.prepare_heap([None]) == 0
