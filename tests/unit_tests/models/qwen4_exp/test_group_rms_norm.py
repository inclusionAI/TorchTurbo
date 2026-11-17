from types import MethodType

import pytest
import torch

from torchturbo.models.qwen4_exp import model as model_module
from torchturbo.models.qwen4_exp.model import _group_rms_norm
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm
from torchturbo.ops import OpDispatcher, OpPolicy, registry
from torchturbo.runtime.cpu import CpuRuntime


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("groups", [2, 4])
def test_group_norm_keeps_per_group_statistics_and_channel_weights(
    monkeypatch, dtype, noncontiguous, groups
):
    dispatcher = OpDispatcher(
        device_runtime=CpuRuntime(), registry=registry, default_policy=OpPolicy.NATIVE
    )
    monkeypatch.setattr(model_module, "ops", dispatcher)
    torch.manual_seed(761)
    module = Qwen4ExpTextRMSNorm(groups * 16, group_size=16).to(dtype)
    with torch.no_grad():
        module.weight.uniform_(-0.5, 0.5)
    value = torch.randn(2, 3, groups * 16, dtype=dtype)
    if noncontiguous:
        value = value.transpose(0, 1)
    value = value.requires_grad_()
    gradient = torch.randn_like(value)
    expected = module(value)
    expected_grads = torch.autograd.grad(expected, (value, module.weight), gradient)
    module._norm = MethodType(_group_rms_norm, module)
    actual = module(value)
    actual_grads = torch.autograd.grad(actual, (value, module.weight), gradient)
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
