"""NPU fused gated activation output, gradient and dispatch contracts."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from torchturbo.ops import OpDispatcher, OpPolicy, registry
from torchturbo.runtime.npu import NpuRuntime


@pytest.fixture(scope="module")
def dispatcher():
    return OpDispatcher(device_runtime=NpuRuntime(), registry=registry)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [F.silu, F.gelu], ids=["silu", "gelu"])
@pytest.mark.parametrize("limit", [0.0, 4.0])
@pytest.mark.parametrize("after", [False, True])
def test_output_and_gradient_match_native(dispatcher, dtype, activation, limit, after):
    torch.manual_seed(42)
    values = torch.randn(7, 130, device="npu") * 5
    values[0, :8] = torch.tensor([-5, -4, -1, 0, 1, 4, 5, 8], device="npu")
    values[0, 65:73] = torch.tensor([-5, -4, -1, 0, 1, 4, 5, 8], device="npu")
    x = values.to(dtype).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    kwargs = dict(activation_func=activation, limit=limit, clamp_gate_after_activation=after)
    expected = dispatcher.gated_activation(reference, impl="torch_gated_activation", **kwargs)
    actual = dispatcher.gated_activation(x, impl="triton_gated_activation", **kwargs)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    tolerance = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x.grad, reference.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape", [(0, 1024), (1, 2), (3, 10, 1024), (4099, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_auto_selection_and_noncontiguous_input(dispatcher, shape, dtype):
    x = torch.randn(*shape[:-1], shape[-1] * 2, device="npu", dtype=dtype)[..., ::2]
    x = x.detach().requires_grad_()
    ref = x.detach().clone().requires_grad_()
    actual = dispatcher.gated_activation(x, activation_func=F.silu, limit=4, clamp_gate_after_activation=True)
    expected = dispatcher.gated_activation(ref, activation_func=F.silu, limit=4, clamp_gate_after_activation=True, impl="torch_gated_activation")
    actual.sum().backward()
    expected.sum().backward()
    tolerance = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)
    selected = dispatcher._candidate_impl_cache[("gated_activation", "npu", None)]
    assert selected.spec.impl_name == "triton_gated_activation"


def test_invalid_activation_and_negative_limit(dispatcher):
    x = torch.randn(2, 8, device="npu")
    with pytest.raises(ValueError, match="silu.*gelu"):
        dispatcher.gated_activation(x, activation_func=F.relu)
    with pytest.raises(ValueError, match="non-negative"):
        dispatcher.gated_activation(x, activation_func=F.silu, limit=-1)


def test_missing_triton_falls_back_to_native(monkeypatch, tmp_path):
    from torchturbo.ops.providers.npu import gated_activation

    original_find_spec = gated_activation.find_spec
    monkeypatch.setenv("TORCHTURBO_ASCENDC_LIB_DIR", str(tmp_path))
    monkeypatch.setattr(
        gated_activation, "find_spec",
        lambda name: None if name == "triton" else original_find_spec(name),
    )
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    x = torch.randn(2, 8, device="npu")
    actual = dispatcher.gated_activation(x, activation_func=F.silu)
    expected = dispatcher.gated_activation(x, activation_func=F.silu, impl="torch_gated_activation")
    torch.testing.assert_close(actual, expected)
    selected = dispatcher._candidate_impl_cache[("gated_activation", "npu", None)]
    assert selected.spec.impl_name == "torch_gated_activation"
    with pytest.raises(RuntimeError):
        dispatcher.gated_activation(x, activation_func=F.silu, impl="triton_gated_activation")
    custom = OpDispatcher(
        device_runtime=NpuRuntime(), registry=registry, default_policy=OpPolicy.CUSTOM,
    )
    with pytest.raises(RuntimeError, match="No implementation.*policy=custom"):
        custom.gated_activation(x, activation_func=F.silu)
