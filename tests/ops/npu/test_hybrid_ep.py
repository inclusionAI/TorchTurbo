"""NPU compact dispatch keeps dense-path values and probability gradients."""

import importlib

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import TurboDevice, is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


class _CompactPacking(TorchDispatchMode):
    def __init__(self, shape):
        self.shape = shape

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.sort.default, torch.ops.aten.sort.stable):
            assert args[0].device.type != "npu", f"Compact NPU packing called a generic sort through {func}."
        result = func(*args, **(kwargs or {}))
        values = result if isinstance(result, (tuple, list)) else (result,)
        for value in values:
            assert not isinstance(value, torch.Tensor) or tuple(value.shape) != self.shape, (
                f"Compact source packing allocated dense routing through {func}."
            )
        return result


@pytest.fixture
def dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    torch.npu.set_device(0)
    dist.init_process_group("hccl", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        yield get_or_create_ops(device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("frozen_tokens", [False, True])
def test_compact_dispatch_without_dense_routing(dispatcher, empty, dtype, frozen_tokens):
    length = 0 if empty else 7
    ids = torch.tensor([[15, 0, 4, 8], [4, 8, 15, 0], [8, 15, 0, 4]], device="npu").repeat(3, 1)[:length]
    x = (torch.arange(length * 16, device="npu").reshape(length, 16) % 7 - 3).to(dtype)
    x = x[:, ::2].requires_grad_(not frozen_tokens)
    weights = (torch.arange(length * 8, device="npu").reshape(length, 8) % 4 / 8)[:, ::2].requires_grad_()
    ref_x = x.detach().clone().requires_grad_(not frozen_tokens)
    ref_weights = weights.detach().clone().requires_grad_()
    dense = torch.zeros(length, 16, device="npu").scatter(1, ids, ref_weights)
    mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, ids, True)
    assert dispatcher._candidate_impl("hybrid_ep_dispatch", TurboDevice.NPU).op_spec().impl_name == "npu_hybrid_ep_dispatch"
    with _CompactPacking((length, 16)):
        compact = dispatcher.hybrid_ep_dispatch(x, ids, weights, dist.group.WORLD, 16, 4)
    reference = dispatcher.hybrid_ep_dispatch(
        ref_x, mask, dense, dist.group.WORLD, 16, 4, impl="torch_hybrid_ep_dispatch",
    )
    for actual, expected in zip(compact[:3], reference[:3], strict=True):
        torch.testing.assert_close(actual, expected)
    output = dispatcher.hybrid_ep_combine((compact[0] * compact[1][:, None]).to(dtype), compact[3])
    expected = dispatcher.hybrid_ep_combine(
        (reference[0] * reference[1][:, None]).to(dtype), reference[3], impl="torch_hybrid_ep_combine",
    )
    torch.testing.assert_close(output, expected)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    if not frozen_tokens:
        torch.testing.assert_close(x.grad, ref_x.grad)


@pytest.mark.parametrize("dispatch,combine", [("npu", "npu"), ("torch", "npu"), ("npu", "torch")])
@pytest.mark.parametrize("all_padding", [False, True])
def test_padded_compact_routes_preserve_order_and_gradients(dispatcher, dispatch, combine, all_padding):
    ids = torch.tensor([[2, 2], [-1, -1], [1, -1], [-1, 2]], device="npu", dtype=torch.int32)
    if all_padding:
        ids.fill_(-1)
    x = torch.arange(1, 5, device="npu", dtype=torch.float32)[:, None].expand(-1, 128).contiguous().requires_grad_()
    gates = torch.tensor([[1., 2.], [9., 8.], [0., 7.], [6., 3.]], device="npu", requires_grad=True)
    packed, probs, counts, handle = dispatcher.hybrid_ep_dispatch(
        x, ids, gates, dist.group.WORLD, 4, 2, impl=f"{dispatch}_hybrid_ep_dispatch",
    )
    rows = [] if all_padding else [2, 0, 0, 3]
    expected_gates = [] if all_padding else [0., 1., 2., 3.]
    if rows:
        # A view sliced from a fully padded activation still retains its storage.
        assert packed.untyped_storage().nbytes() == packed.nbytes
    torch.testing.assert_close(packed, x[torch.tensor(rows, device="npu", dtype=torch.long)])
    torch.testing.assert_close(probs, gates.new_tensor(expected_gates))
    torch.testing.assert_close(counts.cpu(), torch.tensor([0, 0, 0, 0] if all_padding else [0, 1, 3, 0]))
    # A mixed combine also reuses the dispatch ordering without sorting the routes again.
    with _CompactPacking((ids.shape[0], 4)):
        output = dispatcher.hybrid_ep_combine(packed * probs[:, None], handle, impl=f"{combine}_hybrid_ep_combine")
    ref_x = x.detach().clone().requires_grad_()
    ref_gates = gates.detach().clone().requires_grad_()
    reference = ref_x * ref_gates.masked_fill(ids < 0, 0).sum(1, keepdim=True)
    torch.testing.assert_close(output, reference)
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(gates.grad, ref_gates.grad)
