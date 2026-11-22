"""Numerical integration with an unchanged external torch_npu callable."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchturbo.ops import OpDispatcher, OpRegistry
from torchturbo.runtime.npu import NpuRuntime


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_external_npu_swiglu_forward_and_backward(dtype, noncontiguous):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for numerical plugin integration.")
    ops = OpDispatcher(
        device_runtime=NpuRuntime(), registry=OpRegistry(),
        plugins=[{"name": "external_npu", "bindings": [{
            "op_name": "plugin_swiglu", "impl_name": "external_npu::swiglu",
            "target": "torch_npu:npu_swiglu", "devices": ["npu"],
        }]}],
    )
    source = torch.randn(8, 4, dtype=dtype)
    if noncontiguous:
        source = source.t()
    reference_input = source.detach().requires_grad_()
    actual_input = source.to("npu").detach().requires_grad_()
    gate, up = reference_input.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    actual = ops.plugin_swiglu(actual_input)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient.to("npu"))
    tolerance = {torch.float32: 1e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_input.grad.cpu(), reference_input.grad, atol=tolerance, rtol=tolerance)
