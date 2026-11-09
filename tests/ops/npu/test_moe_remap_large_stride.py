"""Large storage strides must retain exact 64-bit element addresses."""

import pytest
import torch

from torchturbo.runtime import is_npu_available

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.mark.parametrize("operand", ["indices", "positions", "table"])
def test_remap_large_stride_uses_correct_storage_offset(operand):
    from torchturbo.ops.providers.npu._kernels.moe_remap_routes import remap_routes

    device = torch.device("npu:0")
    # Offset the view so a wrapped int32 address still lands inside the allocation.
    # The red run then reads a wrong sentinel instead of an illegal device address.
    stride = 2**30
    storage = torch.empty(2**32 + 32, dtype=torch.int32, device=device)
    storage[:32].zero_()
    indices = torch.tensor([[0], [1], [2]], dtype=torch.int32, device=device)
    positions = torch.zeros((3, 1), dtype=torch.int32, device=device)
    table = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=torch.int64, device=device)
    if operand in ("indices", "positions"):
        view = storage.as_strided((3, 1), (stride, 1), storage_offset=2**31)
        if operand == "indices":
            view.copy_(indices)
            indices = view
        else:
            # Earlier source occurrences plus this occurrence select rank 1.
            table = torch.tensor([[0, 1, 2, 0, 2], [0, 1, 2, 1, 3],
                                  [0, 1, 2, 2, 0], [0, 1, 2, 3, 1]],
                                 dtype=torch.int64, device=device)
            view.fill_(1)
            positions = view
    else:
        del storage
        storage = torch.empty(2**32 + 32, dtype=torch.int64, device=device)
        storage[:32].zero_()
        view = storage.as_strided((3, 3), (stride, 1), storage_offset=2**31)
        view.copy_(table)
        table = view
    from torchturbo.ops.providers.native.moe_remap_routes import TorchMoERemapRoutes
    reference = TorchMoERemapRoutes()(indices.cpu(), positions.cpu(), table.cpu())
    actual = remap_routes(indices, positions, table)
    torch.npu.synchronize()
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
