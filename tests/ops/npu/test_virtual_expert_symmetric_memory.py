"""Run MoonEP's PyTorch windows in a fresh torchrun process per rank."""

import os
import subprocess
import sys
import time
import weakref
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from tests.models.test_moon_ep_distributed import _check_moon
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import get_device_runtime, is_npu_available


def _check_registered_windows(manager, runtime, group):
    import torch.distributed._symmetric_memory as symm_mem
    import torch_npu

    empty, rendezvous = symm_mem.empty, symm_mem.rendezvous
    allocations, windows = {}, {}

    def record_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        allocations[tensor.data_ptr()] = weakref.ref(tensor)
        return tensor

    def record_window(tensor, group):
        window = rendezvous(tensor, group)
        windows[tensor.data_ptr()] = (group, list(window.buffer_ptrs))
        return window

    # Track raw owners without retaining their storage.
    with patch.object(symm_mem, "empty", record_empty), patch.object(symm_mem, "rendezvous", record_window):
        manager.wrap_layer_input(torch.empty(1, 2, 12, dtype=torch.bfloat16, device=runtime.device))
    planner, storage = manager.planner, manager.storage
    arenas = ((planner.gathered_counts, planner.histogram_handle),
              (storage.weight_arena, storage.weight_handle), (storage.grad_arena, storage.grad_handle))
    assert len({arena.data_ptr() for arena, _ in arenas}) == 3
    for arena, handle in arenas:
        assert arena.data_ptr() in allocations
        target, pointers = windows[arena.data_ptr()]
        assert target in (group, group.group_name)
        assert len(pointers) == handle.world_size == dist.get_world_size(group)
        assert pointers[dist.get_rank(group)] == arena.data_ptr()
        table = torch.empty(0, dtype=torch.int64, device=arena.device)
        storage = torch_npu._C._construct_storage_from_data_pointer(
            handle.buffer_ptrs_dev, arena.device, len(pointers) * table.element_size(),
        )
        assert pointers == list(handle.buffer_ptrs) == table.set_(storage, 0, (len(pointers),)).cpu().tolist()
    return allocations


def _check_token_exchange_during_weight_barrier(handle, weight_stream, group, device, store):
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    release_key = f"moon-ep-weight-barrier/{dist.get_global_rank(group, 0)}"
    if rank == 0:
        store.delete_key(release_key)
    splits = [1] * size
    tokens = torch.arange(size, dtype=torch.int32, device=device) + rank * size
    received = torch.empty_like(tokens)
    # Initialize both communication paths before deliberately delaying the peers.
    dist.all_to_all_single(received, tokens, splits, splits, group=group)
    received.fill_(-1)
    torch.npu.current_stream(device).synchronize()
    with torch.npu.stream(weight_stream), patch.object(
        dist, "all_reduce", side_effect=AssertionError("SHMEM barriers must run on the weight stream"),
    ):
        handle.barrier()
    weight_stream.synchronize()

    if rank == 0:
        weight_done = torch.npu.Event()
        with torch.npu.stream(weight_stream):
            handle.barrier()
            weight_done.record()
    exchange = dist.all_to_all_single(received, tokens, splits, splits, group=group, async_op=True)
    if rank == 0:
        failure = ""
        try:
            deadline = time.monotonic() + 30
            while not exchange.is_completed():
                assert time.monotonic() < deadline, "Token exchange waited for the weight barrier"
                time.sleep(0.01)
            assert not weight_done.query(), "Weight barrier finished before its peers entered"
        except AssertionError as exc:
            failure = str(exc)
        finally:
            store.set(release_key, failure)
    else:
        try:
            store.wait([release_key], timedelta(seconds=60))
        finally:
            with torch.npu.stream(weight_stream):
                handle.barrier()
    exchange.wait()
    weight_stream.synchronize()
    failure = store.get(release_key).decode()
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    assert not any(failures), failures
    expected = torch.arange(size, dtype=tokens.dtype) * size + rank
    torch.testing.assert_close(received.cpu(), expected)


def _check_model_windows(runtime, mesh, store):
    group = mesh["ep"].get_group()
    rank = dist.get_rank()
    ep_rank, ep_size = dist.get_rank(group), dist.get_world_size(group)
    config = MoEConfig(
        hidden_size=12, moe_ffn_hidden_size=10, num_moe_experts=2 * ep_size,
        expert_model_parallel_size=ep_size, moe_router_topk=2, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, moe_token_dispatcher_type="flex",
        moe_virtual_expert_load_balance=True, params_dtype=torch.bfloat16,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    layer = MoELayer(config, group, submodules=submodules).to(runtime.device)
    fully_shard(layer.experts, mesh=mesh["edp"])
    manager = layer.token_dispatcher._comm_manager
    owners = _check_registered_windows(manager, runtime, group)

    storage = manager.storage
    arena, handle = storage.weight_arena, storage.weight_handle
    _check_token_exchange_during_weight_barrier(handle, manager.weight_streams[0], group, arena.device, store)
    members = dist.get_process_group_ranks(group)
    peer = (ep_rank + 1) % ep_size
    sender = members[(ep_rank - 1) % ep_size]
    remote = handle.get_buffer(peer, arena.shape, arena.dtype)
    assert remote.data_ptr() == handle.buffer_ptrs[peer]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        arena.fill_(rank + 1)
        handle.barrier()
        initial = remote.clone()
        handle.barrier()
        remote.fill_(rank + 11)
        handle.barrier()
        written = arena.clone()
        dist.all_reduce(arena, group=group)
        handle.barrier()
        reduced = remote.clone()
    stream.synchronize()
    total = sum(member + 11 for member in members)
    for value, expected in ((initial, members[peer] + 1), (written, sender + 11), (reduced, total)):
        torch.testing.assert_close(value.cpu(), torch.full(arena.shape, expected, dtype=arena.dtype))

    # One remote alias on one rank must keep the WORLD allocation alive everywhere.
    weight_owner = owners[arena.data_ptr()]
    retained = remote.view(-1) if rank == dist.get_world_size() - 1 else None
    del arena, handle, remote
    VirtualExpertLoadBalancer.finalize()
    assert weight_owner() is not None
    if retained is not None:
        torch.testing.assert_close(retained.cpu(), reduced.cpu().view(-1))
    del retained
    VirtualExpertLoadBalancer.finalize()
    assert all(owner() is None for owner in owners.values())


def _run_worker():
    runtime = get_device_runtime()
    runtime.set_device(int(os.environ["LOCAL_RANK"]))
    timeout = timedelta(minutes=5)
    store, rank, size = next(dist.rendezvous("env://", timeout=timeout))
    # The parent pytest process can have used this torchrun store for other groups.
    dist.init_process_group(
        "hccl", store=dist.PrefixStore("moon-ep-pytorch-windows", store),
        rank=rank, world_size=size, timeout=timeout,
    )
    try:
        ep_size = 2 if size == 2 else size // 2
        mesh = init_device_mesh("npu", (size // ep_size, ep_size), mesh_dim_names=("edp", "ep"))
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE, overrides={
            "plan_virtual_expert_routes": "npu_triton_plan_virtual_expert_routes",
            "virtual_expert_weight_exchange": "npu_shmem_virtual_expert_weight_exchange",
            "virtual_expert_grad_reduce": "npu_shmem_virtual_expert_grad_reduce",
        })
        _check_model_windows(runtime, mesh, store)
        _check_moon(
            mesh["ep"].get_group(), runtime=runtime, fsdp_mesh=mesh, mixed_precision=True, cast_inputs=True,
        )
        VirtualExpertLoadBalancer.finalize()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")
def test_pytorch_ep_windows_order_peer_writes_on_a_side_stream():
    if int(os.environ.get("LOCAL_WORLD_SIZE", 1)) % 2:
        pytest.skip("Run with torchrun and pairs of NPU ranks per host")
    # torch-npu's NPUSHMEM bootstrap persists after its last tensor is freed.
    # A fresh worker keeps this test's WORLD separate from earlier torchrun fixtures.
    subprocess.run([sys.executable, "-m", __name__], check=True, timeout=600)


if __name__ == "__main__":
    _run_worker()
