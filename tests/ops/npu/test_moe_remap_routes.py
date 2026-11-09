"""Ascend remapping parity, exact quotas, strided inputs and probabilities."""

import itertools

import pytest
import torch

from torchturbo.ops.providers.native.plan_virtual_expert_routes import (
    plan_placement, count_routes, build_remap_table,
)
from torchturbo.ops import OpPolicy, get_ops, get_or_create_ops
from torchturbo.ops.providers.native.moe_remap_routes import TorchMoERemapRoutes
from torchturbo.ops.providers.native.moe_route_metadata import TorchMoERouteMetadata
from torchturbo.runtime import TurboDevice, is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")



@pytest.fixture(autouse=True)
def npu_ops():
    if is_npu_available():
        get_or_create_ops(device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)


def _cpu_remap(indices, counts, allocations, copies, rank):
    """Run the PyTorch metadata and remap providers on CPU as the reference."""
    positions, _ = TorchMoERouteMetadata()(indices, counts.shape[1])
    table = build_remap_table(counts, allocations, copies, rank)
    return TorchMoERemapRoutes()(indices, positions, table)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("num_ranks,num_experts,topk,num_tokens,pattern", [
    (1, 8, 1, 17, "random"), (2, 8, 2, 0, "empty"),
    (4, 16, 3, 35, "random"), (4, 16, 4, 35, "random"),
    (4, 16, 5, 35, "random"), (4, 16, 7, 35, "random"),
    (4, 32, 16, 17, "random"), (3, 12, 6, 37, "skew"),
    (8, 32, 8, 35, "random"), (4, 16, 2, 17, "duplicates"),
    (2, 16, 0, 3, "empty"), (4, 64, 32, 9, "random"),
    (2, 64, 64, 5, "random"),
    (7, 28, 4, 9, "random"), (16, 64, 8, 35, "random"),
])
def test_remap_matches_native_routes_counts_and_probability_gradients(
    dtype, strided, num_ranks, num_experts, topk, num_tokens, pattern,
):
    dispatcher = get_ops()
    selected = dispatcher._candidate_impl("moe_remap_routes", TurboDevice.NPU)
    assert selected.op_spec().impl_name == "npu_triton_moe_remap_routes"
    generator = torch.Generator().manual_seed(1729)
    active = max(1, num_experts // 4) if pattern == "skew" else num_experts
    inputs = [torch.randint(active, (num_tokens, topk), generator=generator, dtype=dtype)
              for _ in range(num_ranks)]
    if pattern == "duplicates":
        inputs = [torch.zeros_like(x) for x in inputs]
    elif num_ranks > 1 and num_tokens and pattern == "random":
        inputs[-1] = inputs[-1][:0]
    counts = torch.stack([torch.bincount(x.flatten().long(), minlength=num_experts) for x in inputs])
    allocations, copies = plan_placement(counts)
    mapped_sources = []
    for rank, indices in enumerate(inputs):
        expected, expected_order = _cpu_remap(indices, counts, allocations, copies, rank)
        device_indices = indices.to("npu")
        positions, _ = dispatcher.moe_route_metadata(device_indices, num_experts)
        table = build_remap_table(counts, allocations, copies, rank).to("npu")
        if strided:
            device_indices = device_indices.T.contiguous().T
            positions = positions.T.contiguous().T
            table = table.T.contiguous().T
        mapped, order = dispatcher.moe_remap_routes(device_indices, positions, table)
        native, native_order = dispatcher.moe_remap_routes(
            device_indices, positions, table, impl="torch_moe_remap_routes",
        )
        torch.testing.assert_close(mapped.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(order.cpu(), expected_order, rtol=0, atol=0)
        torch.testing.assert_close(native, mapped, rtol=0, atol=0)
        torch.testing.assert_close(native_order, order, rtol=0, atol=0)
        probs = torch.randn(indices.shape, generator=generator).to("npu").requires_grad_()
        gradient = torch.randn(indices.shape, generator=generator).to("npu")
        actual_probs = probs.gather(1, order)
        expected_probs = probs.gather(1, expected_order.to("npu"))
        torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual_probs, probs, gradient)[0],
            torch.autograd.grad(expected_probs, probs, gradient)[0], rtol=0, atol=0,
        )
        mapped_sources.append(mapped.cpu())
    local = num_experts // num_ranks
    for destination in range(num_ranks):
        _, received = count_routes(counts, allocations, copies, destination)
        for source, mapped in enumerate(mapped_sources):
            histogram = torch.bincount(mapped.flatten().long(), minlength=2 * num_experts)
            start = destination * 2 * local
            torch.testing.assert_close(histogram[start:start + 2 * local], received[source])


@pytest.mark.parametrize("num_experts", [128, 512])
def test_remap_matches_native_at_training_shape(num_experts):
    generator = torch.Generator().manual_seed(591)
    inputs = [torch.randint(8, (8192, 8), generator=generator) for _ in range(8)]
    counts = torch.stack([torch.bincount(x.flatten(), minlength=num_experts) for x in inputs])
    allocations, copies = plan_placement(counts)
    indices = inputs[0]
    expected, expected_order = _cpu_remap(indices, counts, allocations, copies, 0)
    device_indices = indices.to("npu")
    positions, _ = get_ops().moe_route_metadata(device_indices, num_experts)
    table = build_remap_table(counts, allocations, copies, 0).to("npu")
    mapped, order = get_ops().moe_remap_routes(device_indices, positions, table)
    torch.testing.assert_close(mapped.cpu(), expected)
    torch.testing.assert_close(order.cpu(), expected_order)


def test_remap_retains_int64_offsets_with_int32_indices():
    table = torch.tensor([
        [2**32, 2**32, 2**32 + 2, 0, 2],
        [0, 0, 0, 1, -1], [0, 0, 0, -1, 0], [0, 0, 0, -1, 1],
    ], device="npu")
    mapped, order = get_ops().moe_remap_routes(
        torch.tensor([[0, 0]], device="npu", dtype=torch.int32),
        torch.tensor([[0, 1]], device="npu", dtype=torch.int32), table,
    )
    torch.testing.assert_close(mapped.cpu(), torch.tensor([[6, 6]], dtype=torch.int32))
    torch.testing.assert_close(order.cpu(), torch.tensor([[0, 1]]))


@pytest.mark.parametrize("offset", [0, 2**40])
@pytest.mark.parametrize("num_ranks", [1, 2, 3, 5, 8])
def test_upper_bound_with_all_small_quotas(offset, num_ranks):
    # Every quota vector over {0, 1, 2}, including repeated and zero boundaries.
    quotas = torch.tensor(list(itertools.product(range(3), repeat=num_ranks)))
    num_experts = (quotas.shape[0] + num_ranks - 1) // num_ranks * num_ranks
    num_local_experts = num_experts // num_ranks
    ends = quotas.cumsum(1) + offset
    ordinals = torch.arange(2 * num_ranks).expand(quotas.shape[0], -1)
    active = ordinals < quotas.sum(1, keepdim=True)
    experts = torch.arange(quotas.shape[0])[:, None].expand_as(ordinals)[active]
    positions = ordinals[active]
    table = torch.zeros((num_experts, 1 + 2 * num_ranks), dtype=torch.int64)
    table[:quotas.shape[0], 0] = offset
    table[:quotas.shape[0], 1:1 + num_ranks] = ends
    table[:, 1 + num_ranks:] = torch.arange(num_experts)[:, None] % num_local_experts
    destinations = ((positions[:, None] + offset) >= ends[experts]).sum(1)
    expected = destinations * (2 * num_local_experts) + experts % num_local_experts
    mapped, order = get_ops().moe_remap_routes(
        experts[:, None].to(device="npu", dtype=torch.int32),
        positions[:, None].to(device="npu", dtype=torch.int32), table.to("npu"),
    )
    torch.testing.assert_close(mapped.cpu(), expected[:, None].int(), rtol=0, atol=0)
    torch.testing.assert_close(order.cpu(), torch.zeros_like(order.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("num_experts", [2**27, 2**27 + 1])
def test_remap_preserves_stable_order_at_int32_key_boundary(num_experts):
    dispatcher = get_ops()
    selected = dispatcher._candidate_impl("moe_remap_routes", TurboDevice.NPU)
    assert selected.op_spec().impl_name == "npu_triton_moe_remap_routes"
    # Exercise actual key narrowing and fallback without a huge native oracle.
    table = torch.zeros((num_experts, 3), device="npu", dtype=torch.int64)
    table[-1] = torch.tensor([0, 8, 2 * num_experts - 1], device="npu")
    indices = torch.full((1, 8), num_experts - 1, device="npu", dtype=torch.int32)
    positions = torch.arange(8, device="npu", dtype=torch.int32).reshape(1, 8)
    mapped, order = dispatcher.moe_remap_routes(indices, positions, table)
    torch.testing.assert_close(
        mapped.cpu(), torch.full((1, 8), 2 * num_experts - 1, dtype=torch.int32),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(order.cpu(), torch.arange(8).reshape(1, 8), rtol=0, atol=0)
    table[-2:] = torch.tensor([
        [0, 4, 2 * num_experts - 2], [0, 4, 2 * num_experts - 1],
    ], device="npu")
    indices = torch.tensor(
        [[num_experts - 1, num_experts - 2] * 4], device="npu", dtype=torch.int32,
    )
    positions = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], device="npu", dtype=torch.int32)
    mapped, order = dispatcher.moe_remap_routes(indices, positions, table)
    expected = torch.tensor(
        [[2 * num_experts - 2] * 4 + [2 * num_experts - 1] * 4], dtype=torch.int32,
    )
    torch.testing.assert_close(mapped.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(order.cpu(), torch.tensor([[1, 3, 5, 7, 0, 2, 4, 6]]), rtol=0, atol=0)
