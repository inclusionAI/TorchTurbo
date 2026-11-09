"""Stable NPU route metadata contracts."""

import pytest
import torch

from torchturbo.ops import OpPolicy, get_ops, get_or_create_ops
from torchturbo.runtime import is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(autouse=True)
def npu_ops():
    if is_npu_available():
        get_or_create_ops(device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_metadata_handles_stable_ties_noncontiguous_and_empty(dtype, empty):
    indices = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 2]], dtype=dtype)
    expected = torch.tensor([[4, 0], [2, 5], [1, 3], [6, 7]], dtype=torch.int32)
    counts = torch.tensor([2, 2, 4, 0])
    if empty:
        indices, expected = indices[:0], expected[:0]
        counts.zero_()
    device_indices = indices.to("npu").T.contiguous().T
    positions, actual_counts = get_ops().moe_route_metadata(device_indices, 4)
    torch.testing.assert_close(positions.cpu(), expected)
    torch.testing.assert_close(actual_counts.cpu(), counts)
    torch.testing.assert_close(device_indices.cpu(), indices)


@pytest.mark.parametrize("num_experts", [128, 512])
def test_metadata_matches_stable_cpu_sort_at_training_size(num_experts):
    indices = torch.randint(num_experts, (8192, 8), generator=torch.Generator().manual_seed(39))
    order = indices.flatten().argsort(stable=True)
    expected = torch.empty_like(order, dtype=torch.int32)
    expected[order] = torch.arange(order.numel(), dtype=torch.int32)
    positions, counts = get_ops().moe_route_metadata(indices.to("npu"), num_experts)
    torch.testing.assert_close(positions.cpu().flatten(), expected)
    torch.testing.assert_close(counts.cpu(), torch.bincount(indices.flatten(), minlength=num_experts))
