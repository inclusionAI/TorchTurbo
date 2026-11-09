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


class _NoDenseRouting(TorchDispatchMode):
    def __init__(self, shape):
        self.shape = shape

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
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
    ids = torch.tensor([[15, 0, 4, 8]], device="npu").expand(length, -1)
    x = torch.randn(length, 16, device="npu", dtype=dtype)[:, ::2].requires_grad_(not frozen_tokens)
    weights = torch.randn(length, 8, device="npu")[:, ::2].requires_grad_()
    ref_x = x.detach().clone().requires_grad_(not frozen_tokens)
    ref_weights = weights.detach().clone().requires_grad_()
    dense = torch.zeros(length, 16, device="npu").scatter(1, ids, ref_weights)
    mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, ids, True)
    assert dispatcher._candidate_impl("hybrid_ep_dispatch", TurboDevice.NPU).op_spec().impl_name == "npu_hybrid_ep_dispatch"
    with _NoDenseRouting((length, 16)):
        compact = dispatcher.hybrid_ep_dispatch(x, ids, weights, dist.group.WORLD, 16, 4)
    reference = dispatcher.hybrid_ep_dispatch(ref_x, mask, dense, dist.group.WORLD, 16, 4)
    for actual, expected in zip(compact[:3], reference[:3], strict=True):
        torch.testing.assert_close(actual, expected)
    output = dispatcher.hybrid_ep_combine((compact[0] * compact[1][:, None]).to(dtype), compact[3])
    expected = dispatcher.hybrid_ep_combine((reference[0] * reference[1][:, None]).to(dtype), reference[3])
    torch.testing.assert_close(output, expected)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    if not frozen_tokens:
        torch.testing.assert_close(x.grad, ref_x.grad)
