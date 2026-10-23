"""Grouped expert probability weighting and empty-rank adapter gradients."""

import copy
import importlib

import pytest
import torch

from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_bf16_experts_weight_activations_before_fc2(monkeypatch, use_lora, num_tokens):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)
    torch.manual_seed(13)
    base = GroupedMLP(
        num_local_experts=2, use_te=False,
        config=MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=3, gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            params_dtype=torch.bfloat16,
        ),
    )
    if use_lora:
        base.requires_grad_(False)
    candidate = GroupedMLPLoRA(base, "default", r=2, lora_alpha=4) if use_lora else base
    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.normal_(std=.2)
    reference = copy.deepcopy(candidate)
    x = torch.randn(num_tokens, 4, dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    probs = torch.tensor([.5005, .1237, .9953])[:num_tokens].requires_grad_()
    ref_probs = probs.detach().clone().requires_grad_()
    counts = torch.tensor([2, 1] if num_tokens else [0, 0])
    actual, _ = candidate(x, counts, probs)

    ref_base = reference.base_layer if use_lora else reference
    weight1 = ref_base.weight1.view(2, 4, 6)
    weight2 = ref_base.weight2.view(2, 3, 4)
    contributions = []
    chunks = zip(ref_x.split(counts.tolist()), ref_probs.split(counts.tolist()))
    for expert, (tokens, weights) in enumerate(chunks):
        hidden = tokens @ weight1[expert]
        if use_lora:
            delta = (
                tokens.float() @ reference.lora_A["default"].weight[expert]
                @ reference.lora_B["default"].weight[expert]
            )
            hidden = hidden + delta.to(hidden.dtype) * reference.scaling["default"]
        gate, up = hidden.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * up
        activation = (activation * weights[:, None]).to(hidden.dtype)
        output = activation @ weight2[expert]
        if use_lora:
            delta = (
                activation.float() @ reference.lora_A2["default"].weight[expert]
                @ reference.lora_B2["default"].weight[expert]
            )
            output = output + delta.to(output.dtype) * reference.scaling["default"]
        contributions.append(output)
    expected = torch.cat(contributions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    for parameter, ref_parameter in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("num_local_experts", [1, 2])
def test_zero_token_forward_keeps_adapters_in_graph(num_local_experts):
    torch.manual_seed(0)
    base = GroupedMLP(
        num_local_experts=num_local_experts,
        use_te=False,
        config=MoEConfig(hidden_size=4, moe_ffn_hidden_size=3, gated_linear_unit=True),
    )
    lora = GroupedMLPLoRA(base, "default", r=2, lora_alpha=4)
    hidden_states = torch.empty(0, 4, requires_grad=True)

    output, _ = lora(
        hidden_states,
        torch.zeros(num_local_experts, dtype=torch.long),
        torch.empty(0),
    )
    output.sum().backward()

    assert output.shape == (0, 4)
    adapter_parameters = {
        name: parameter
        for name, parameter in lora.named_parameters()
        if name.startswith("lora_")
    }
    assert len(adapter_parameters) == 4
    for name, parameter in adapter_parameters.items():
        # Every rank must reduce-scatter the same FSDP gradients, so ranks
        # without tokens still produce zero adapter gradients.
        assert parameter.grad is not None, name
        assert not parameter.grad.any(), name
