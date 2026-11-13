"""TE expert outputs and gradients when some or all local experts receive no tokens."""

import importlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchturbo.models.modules.moe import experts as expert_module
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import is_cuda_available
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime


class _CpuGroupedLinear(nn.Module):
    """CPU substitute for TE's separate expert weights and split-based linear API."""

    def __init__(
        self, num_gemms, in_features, out_features, *, bias, params_dtype, device,
    ):
        super().__init__()
        assert not bias
        for index in range(num_gemms):
            self.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.empty(out_features, in_features, dtype=params_dtype)),
            )

    def forward(self, tokens, m_splits):
        return torch.cat([
            F.linear(chunk, self.get_parameter(f"weight{index}"))
            for index, chunk in enumerate(tokens.split(m_splits))
        ])


@pytest.fixture(params=["cpu", "cuda"])
def te_device(request, monkeypatch):
    if request.param == "cuda":
        if not is_cuda_available():
            pytest.skip("CUDA is unavailable")
        te = pytest.importorskip("transformer_engine.pytorch")
        monkeypatch.setattr(expert_module, "GroupedLinear", te.GroupedLinear)
        runtime = CudaRuntime()
    else:
        monkeypatch.setattr(expert_module, "GroupedLinear", _CpuGroupedLinear)
        runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
    return runtime.device


@pytest.mark.parametrize("counts", [(0, 0, 0), (2, 0, 1)], ids=["all-empty", "one-empty"])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True], ids=["plain", "non-reentrant", "reentrant"])
@pytest.mark.parametrize("on_input", [False, True])
def test_te_experts_preserve_outputs_and_gradients(te_device, counts, checkpoint_mode, on_input):
    torch.manual_seed(14)
    dtype = torch.bfloat16 if te_device.type == "cuda" else torch.float32
    experts = expert_module.GroupedMLP(
        num_local_experts=3, use_te=True,
        config=MoEConfig(
            hidden_size=16, moe_ffn_hidden_size=16, gated_linear_unit=True,
            activation_func=F.silu, params_dtype=dtype,
            moe_apply_probs_on_input=on_input, moe_router_topk=1 if on_input else 2,
        ),
    )
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.normal_(std=0.1)
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in experts.named_parameters()
    }
    tokens = torch.randn(sum(counts), 16, device=te_device, dtype=dtype, requires_grad=True)
    probs = torch.rand(sum(counts), device=te_device, requires_grad=True)
    ref_tokens = tokens.detach().clone().requires_grad_()
    ref_probs = probs.detach().clone().requires_grad_()
    tokens_per_expert = torch.tensor(counts)

    if checkpoint_mode is None:
        output, bias = experts(tokens, tokens_per_expert, probs)
    else:
        output, bias = checkpoint(
            experts, tokens, tokens_per_expert, probs, use_reentrant=checkpoint_mode,
        )

    contributions = []
    chunks = zip(ref_tokens.split(counts), ref_probs.split(counts))
    for index, (expert_tokens, expert_probs) in enumerate(chunks):
        expert_probs = expert_probs.unsqueeze(-1)
        if on_input:
            expert_tokens = (expert_tokens * expert_probs).to(dtype)
        hidden = F.linear(expert_tokens, reference_parameters[f"fc1_linear.weight{index}"])
        gate, up = hidden.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        if not on_input:
            activation = (activation * expert_probs).to(dtype)
        contributions.append(F.linear(activation, reference_parameters[f"fc2_linear.weight{index}"]))
    expected = torch.cat(contributions)

    assert bias is None
    assert output.shape == tokens.shape
    torch.testing.assert_close(output, expected)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient)
    assert tokens.grad is not None
    assert probs.grad is not None
    torch.testing.assert_close(tokens.grad, ref_tokens.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    for name, parameter in experts.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, msg=name)
    for index, count in enumerate(counts):
        if count == 0:
            for linear in (experts.fc1_linear, experts.fc2_linear):
                assert torch.count_nonzero(linear.get_parameter(f"weight{index}").grad) == 0
