"""Numerical and loading contracts for the ahead-of-time AscendC provider."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from torchturbo.ops import OpDispatcher, registry
from torchturbo.runtime.npu import NpuRuntime


@pytest.fixture(scope="module")
def dispatcher():
    return OpDispatcher(device_runtime=NpuRuntime(), registry=registry)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [F.silu, F.gelu], ids=["silu", "gelu"])
@pytest.mark.parametrize("limit,after", [(0., False), (4., False), (4., True)])
@pytest.mark.parametrize("shape", [(7, 130), (33, 1024), (2, 3, 4096), (43, 384), (3, 8258)])
def test_forward_backward(dispatcher, dtype, activation, limit, after, shape):
    torch.manual_seed(42)
    x = (torch.randn(shape, device="npu") * 5).to(dtype).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    kwargs = dict(activation_func=activation, limit=limit, clamp_gate_after_activation=after)
    actual = dispatcher.gated_activation(x, impl="ascendc_gated_activation", **kwargs)
    expected = dispatcher.gated_activation(ref, impl="torch_gated_activation", **kwargs)
    dy = torch.randn_like(expected)
    actual.backward(dy)
    expected.backward(dy)
    tol = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(x.grad, ref.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("after", [False, True])
def test_clamp_boundaries(dispatcher, after):
    gate = torch.tensor([-20., -4., -1., 0., 1., 4., 4.001, 20.], device="npu")
    up = torch.tensor([-4.001, -4., -3.999, 0., 3.999, 4., 4.001, 20.], device="npu")
    x = torch.cat((gate, up)).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    kwargs = dict(activation_func=F.silu, limit=4., clamp_gate_after_activation=after)
    actual = dispatcher.gated_activation(x, impl="ascendc_gated_activation", **kwargs)
    expected = dispatcher.gated_activation(ref, impl="torch_gated_activation", **kwargs)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad, ref.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(0, 1024), (5, 2), (3, 6), (4099, 1024)])
def test_noncontiguous_and_dynamic_shapes(dispatcher, shape):
    for rows in [shape[0], shape[0] + 1, shape[0]]:
        x = torch.randn(rows, shape[1] * 2, device="npu", dtype=torch.bfloat16)[:, ::2].detach().requires_grad_()
        ref = x.detach().clone().requires_grad_()
        kwargs = dict(activation_func=F.silu, limit=4., clamp_gate_after_activation=True)
        y = dispatcher.gated_activation(x, impl="ascendc_gated_activation", **kwargs)
        expected = dispatcher.gated_activation(ref, impl="torch_gated_activation", **kwargs)
        dy = torch.randn(y.shape[-1], rows, device="npu", dtype=y.dtype).t()
        y.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(y, expected, atol=.03, rtol=.03)
        torch.testing.assert_close(x.grad, ref.grad, atol=.03, rtol=.03)


def test_missing_library_is_not_compiled_or_silently_replaced(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTURBO_ASCENDC_LIB_DIR", str(tmp_path))
    ops = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    with pytest.raises(RuntimeError):
        ops.gated_activation(torch.ones(1, 8, device="npu"), activation_func=F.silu,
                             impl="ascendc_gated_activation")


def test_prebuilt_kernel_on_nondefault_stream(dispatcher, monkeypatch):
    from torchturbo.ops.providers.native import gated_activation
    from torchturbo.ops.providers.npu.ascendc._library import library_path

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Supported inputs must execute the AscendC kernel")

    monkeypatch.setattr(gated_activation, "_torch_gated_activation", unexpected_fallback)
    path = library_path("gated_activation")
    before = path.stat().st_mtime_ns
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.randn(257, 1024, device="npu", dtype=torch.bfloat16).requires_grad_()
        y = dispatcher.gated_activation(x, activation_func=F.silu, limit=4.,
                                       impl="ascendc_gated_activation")
        ref = x.detach().clone().requires_grad_()
        g, u = ref.chunk(2, -1)
        expected = (F.silu(g.float().clamp(max=4.)) * u.float().clamp(-4., 4.)).to(ref.dtype)
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
    stream.synchronize()
    torch.testing.assert_close(y, expected, atol=.03, rtol=.03)
    torch.testing.assert_close(x.grad, ref.grad, atol=.03, rtol=.03)
    assert path.stat().st_mtime_ns == before
    assert str(path.resolve()) in torch.ops.loaded_libraries


@pytest.mark.parametrize("shape,dtype", [((3, 3), torch.float32), ((2, 10), torch.float64)])
def test_native_contract_fallback(dispatcher, shape, dtype):
    x = torch.randn(shape, device="npu", dtype=dtype).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    kwargs = dict(activation_func=F.silu, limit=4.)
    y = dispatcher.gated_activation(x, impl="ascendc_gated_activation", **kwargs)
    expected = dispatcher.gated_activation(ref, impl="torch_gated_activation", **kwargs)
    y.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, ref.grad, atol=0, rtol=0)


def test_invalid_options(dispatcher):
    x = torch.ones(1, 8, device="npu")
    with pytest.raises(ValueError, match="non-negative"):
        dispatcher.gated_activation(x, activation_func=F.silu, limit=-1., impl="ascendc_gated_activation")
    with pytest.raises(ValueError, match="silu.*gelu"):
        dispatcher.gated_activation(x, activation_func=F.relu, impl="ascendc_gated_activation")


@pytest.mark.parametrize("after", [False, True])
def test_nonfinite_clamp_and_gradient(dispatcher, after):
    x = torch.tensor([[float("nan"), float("inf"), -float("inf"), 5., 5., 1., 1., 1.,
                       1., 1., 1., 1., 1., float("nan"), float("inf"), -float("inf")]],
                     device="npu").requires_grad_()
    ref = x.detach().clone().requires_grad_()
    dy = torch.tensor([[1., 1., 1., float("nan"), float("inf"), 1., 1., 1.]], device="npu")
    kwargs = dict(activation_func=F.silu, limit=4., clamp_gate_after_activation=after)
    actual = dispatcher.gated_activation(x, impl="ascendc_gated_activation", **kwargs)
    expected = dispatcher.gated_activation(ref, impl="torch_gated_activation", **kwargs)
    actual.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5, equal_nan=True)
    torch.testing.assert_close(x.grad, ref.grad, atol=2e-5, rtol=2e-5, equal_nan=True)
