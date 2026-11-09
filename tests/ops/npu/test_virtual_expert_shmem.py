"""Run the weight contract with direct SHMEM and prohibit native weight exchange."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.ops.npu.test_virtual_expert_weight_exchange import (
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
from torchturbo.runtime.npu import NpuRuntime


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


def test_standalone_reuses_shmem_and_preserves_live_outputs(weight_env, monkeypatch):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    allocations = []
    allocate = NpuRuntime.allocate_communication_buffer

    def track_allocation(*args, **kwargs):
        tensor, handle = allocate(*args, **kwargs)
        allocations.append((tensor.data_ptr(), tensor.numel() * tensor.element_size()))
        return tensor, handle

    monkeypatch.setattr(NpuRuntime, "allocate_communication_buffer", track_allocation)
    copies = torch.full((size, 3), -1, dtype=torch.int64)
    copies[:, :2] = torch.tensor([0, 2])
    context = build_weight_exchange_context(copies, group)
    streams = (torch.npu.Stream(), torch.npu.Stream())
    retained = []
    for generation in range(8):
        weights = [
            torch.full(
                shape, 100 * rank + generation + fc + 1,
                dtype=torch.float32, device="npu", requires_grad=True,
            )
            for fc, shape in enumerate(((3, 6, 9), (3, 9, 4)))
        ]
        stream = streams[generation % 2]
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            prefetched = None
            if generation % 2:
                with torch.no_grad():
                    prefetched = dispatcher.virtual_expert_weight_exchange(
                        *weights, None, None, context,
                    )[:2]
            result = dispatcher.virtual_expert_weight_exchange(
                *weights, None, None, context, prefetched=prefetched,
            )[:2]
        torch.npu.current_stream().wait_stream(stream)
        retained.append((weights, result))
    empty = build_weight_exchange_context(torch.full_like(copies, -1), group)
    dispatcher.virtual_expert_weight_exchange(*weights, None, None, empty)
    assert len(allocations) == 2, "A layout must reuse one weight arena and one gradient arena"

    for generation in reversed(range(len(retained))):
        weights, result = retained[generation]
        for fc, value in enumerate(result):
            _assert_equal_on_all_ranks(value, torch.full_like(value, generation + fc + 1), group)
        sum((value * value).sum() for value in result).backward()
        for fc, weight in enumerate(weights):
            expected = torch.zeros_like(weight)
            if rank == 0:
                expected[0] = expected[2] = 2 * (generation + fc + 1) * size
            _assert_equal_on_all_ranks(weight.grad, expected, group)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tail", [False, True])
def test_large_tiles_and_many_replicas(weight_env, dtype, tail):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    count = 5
    shapes = ((count, 65, 129), (count, 129, 33)) if tail else ((count, 64, 128), (count, 128, 32))
    generator = torch.Generator().manual_seed(8301)
    owner_weights = [torch.randn(shape, generator=generator).to(dtype) for shape in shapes]
    weights = [(value + rank).to("npu").requires_grad_() for value in owner_weights]
    copies = torch.arange(count).repeat(size, 1)
    context = build_weight_exchange_context(copies, group)
    result = dispatcher.virtual_expert_weight_exchange(*weights, None, None, context)[:2]
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


def test_bf16_weights_use_fp32_gradient_peer_storage(weight_env):
    dispatcher, group = weight_env
    runtime = dispatcher.device_runtime
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    shapes = ((3, 4, 5), (3, 5, 2))
    weights = [torch.full(shape, rank + 1., dtype=torch.bfloat16, device="npu", requires_grad=True)
               for shape in shapes]
    numels = [weight.numel() for weight in weights]
    arena, weight_handle = runtime.allocate_communication_buffer(
        sum(numels), dtype=torch.bfloat16, device=weights[0].device, group=group,
    )
    grad_arena, grad_handle = runtime.allocate_communication_buffer(
        sum(numels), dtype=torch.float32, device=weights[0].device, group=group,
    )
    slots = tuple(section.view(shape) for section, shape in zip(arena.split(numels), shapes))
    copies = torch.full((size, 3), -1, dtype=torch.int64)
    copies[:, 0] = 0
    context = build_weight_exchange_context(copies, group)
    replicas = dispatcher.virtual_expert_weight_exchange(
        *weights, None, None, context, out=slots, grad_buffer=grad_arena,
        buffer_handles=(weight_handle, grad_handle),
    )[:2]
    torch.autograd.backward(replicas, tuple(torch.full_like(value, (rank + 1) / 32) for value in replicas))
    for weight in weights:
        expected = torch.zeros_like(weight)
        if rank == 0:
            expected[0] = size * (size + 1) / 64
        _assert_equal_on_all_ranks(weight.grad, expected, group)
