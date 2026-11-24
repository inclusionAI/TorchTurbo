"""Run the weight contract with direct SHMEM and prohibit native weight exchange."""

import gc
import weakref
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.ops.npu.test_virtual_expert_weight_exchange import (
    weight_buffers,
    weight_env as native_weight_env,
    test_weight_values_and_owner_gradients,
    test_prefetched_frozen_and_unused_outputs,
    test_two_plans_before_backward_and_repeated_stream_reuse,
    test_noncontiguous_ep_subgroups_with_one_empty_group,
    test_prefetched_consumer_does_not_exchange_again,
    test_replica_only_loss_materializes_input_probability_zero_gradients,
    _assert_equal_on_all_ranks,
)
from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.providers.native.virtual_expert_weight_exchange import build_weight_exchange_context
from torchturbo.ops.registry import registry
from torchturbo.runtime import is_npu_available


pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(scope="module")
def weight_env(native_weight_env):
    native, group = native_weight_env
    dispatcher = OpDispatcher(
        device_runtime=native.device_runtime, registry=registry,
        default_policy=OpPolicy.NATIVE,
        overrides={"virtual_expert_weight_exchange": "npu_shmem_virtual_expert_weight_exchange"},
    )
    with patch(
        "torchturbo.ops.providers.native.virtual_expert_weight_exchange.ep_all_to_all",
        side_effect=AssertionError("Direct SHMEM must not use native weight all-to-all"),
    ):
        yield dispatcher, group


def test_reclaims_arena_after_last_peer_view(weight_env):
    import torch.distributed._symmetric_memory as symm_mem

    dispatcher, group = weight_env
    if dist.get_world_size(group) < 2:
        pytest.skip("Requires two NPU ranks")
    runtime = dispatcher.device_runtime
    rank = dist.get_rank(group)
    empty = symm_mem.empty
    owners = []

    def record_owner(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        owners.append(weakref.ref(tensor))
        return tensor

    with patch.object(symm_mem, "empty", record_owner):
        arena = runtime.shmem.empty(16, dtype=torch.float32, device=runtime.device)
    owner, = owners
    handle = runtime.shmem.rendezvous(arena, group)
    arena.fill_(rank + 1)
    runtime.synchronize()
    dist.barrier(group=group)
    peer = handle.get_buffer(0, arena.shape, arena.dtype).view(-1) if rank == 1 else None
    del arena, handle
    gc.collect()
    runtime.shmem.release_unused()
    assert owner() is not None
    expected = torch.ones(16)
    _assert_equal_on_all_ranks(peer.cpu() if peer is not None else expected, expected, group)
    del peer
    gc.collect()
    runtime.shmem.release_unused()
    assert owner() is None


def test_weight_push_reuses_caller_slots(weight_env, weight_buffers):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    weights = [torch.full(shape, rank + 1., device="npu") for shape in ((3, 6, 9), (3, 9, 4))]
    buffers = weight_buffers(weights, group)
    copies = torch.full((size, 3), -1, dtype=torch.int64)
    copies[:, :2] = torch.tensor([0, 2])
    context = build_weight_exchange_context(copies, group)
    with torch.no_grad(), patch.object(
        dispatcher.device_runtime.shmem, "empty", side_effect=AssertionError("Unexpected SHMEM allocation"),
    ):
        replicas = dispatcher.virtual_expert_weight_exchange(*weights, None, None, context, **buffers)[:2]
        for weight in weights:
            weight.add_(1)
        dispatcher.virtual_expert_weight_exchange(*weights, None, None, context, **buffers)
    for replica, slot in zip(replicas, buffers["out"], strict=True):
        assert replica.data_ptr() == slot.data_ptr()
        _assert_equal_on_all_ranks(replica, torch.full_like(slot[:2], 2), group)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tail", [False, True])
def test_large_tiles_and_many_replicas(weight_env, weight_buffers, dtype, tail):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    count = 5
    shapes = ((count, 65, 129), (count, 129, 33)) if tail else ((count, 64, 128), (count, 128, 32))
    generator = torch.Generator().manual_seed(8301)
    owner_weights = [torch.randn(shape, generator=generator).to(dtype) for shape in shapes]
    weights = [(value + rank).to("npu").requires_grad_() for value in owner_weights]
    copies = torch.arange(count).repeat(size, 1)
    context = build_weight_exchange_context(copies, group)
    buffers = weight_buffers(weights, group)
    result = dispatcher.virtual_expert_weight_exchange(*weights, None, None, context, **buffers)[:2]
    for actual, expected in zip(result, owner_weights):
        _assert_equal_on_all_ranks(actual.cpu(), expected, group)
    gradient_inputs = [
        [torch.randn(shape, generator=generator).to(dtype) for shape in shapes]
        for _ in range(size)
    ]
    torch.autograd.backward(result, tuple(value.to("npu") for value in gradient_inputs[rank]))
    for fc, weight in enumerate(weights):
        expected = torch.zeros(shapes[fc], dtype=torch.float32)
        if rank == 0:
            for peer in range(size):
                expected.add_(gradient_inputs[peer][fc].float())
        _assert_equal_on_all_ranks(weight.grad.cpu(), expected.to(dtype), group)


def test_bf16_weights_use_fp32_gradient_peer_storage(weight_env, weight_buffers):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    shapes = ((3, 4, 5), (3, 5, 2))
    weights = [torch.full(shape, rank + 1., dtype=torch.bfloat16, device="npu", requires_grad=True)
               for shape in shapes]
    buffers = weight_buffers(weights, group)
    assert buffers["grad_buffer"].dtype == torch.float32
    copies = torch.full((size, 3), -1, dtype=torch.int64)
    copies[:, 0] = 0
    context = build_weight_exchange_context(copies, group)
    replicas = dispatcher.virtual_expert_weight_exchange(*weights, None, None, context, **buffers)[:2]
    torch.autograd.backward(replicas, tuple(torch.full_like(value, (rank + 1) / 32) for value in replicas))
    for weight in weights:
        expected = torch.zeros_like(weight)
        if rank == 0:
            expected[0] = size * (size + 1) / 64
        _assert_equal_on_all_ranks(weight.grad, expected, group)
