"""Bailing MoE routing, clamping, shared experts, and checkpoint gradients."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchturbo.engine.config import EngineConfig
from torchturbo.models.bailing_hybrid.parallel_plan import _replace_moe_only
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.models.modules.moe.utils import RoutingResult
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    def cpu_moe_config(*, moe_permute_fusion, **kwargs):
        assert moe_permute_fusion is True
        return MoEConfig(moe_permute_fusion=False, **kwargs)

    monkeypatch.setattr(
        "torchturbo.models.bailing_hybrid.parallel_plan.MoEConfig",
        cpu_moe_config,
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    runtime = CpuRuntime()
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def _experts(num_experts, config):
    dispatcher = MoEAlltoAllTokenDispatcher(
        num_experts, list(range(num_experts)), config, object(),
    )
    return GroupedMLP(num_experts, False, config, layer_number=0), dispatcher


def _routed_forward(experts, dispatcher, tokens, routing_map, probs):
    indices = routing_map.nonzero(as_tuple=True)[1].reshape(
        probs.shape[0], dispatcher.config.moe_router_topk,
    )
    routing = RoutingResult(probs, routing_map, indices)
    packed, packed_probs = dispatcher.dispatch_preprocess(tokens, routing)
    received, received_probs = dispatcher.token_dispatch(packed, packed_probs)
    local, counts, local_probs = dispatcher.dispatch_postprocess(received, received_probs)
    output, _ = experts(local, counts, local_probs)
    returned = dispatcher.token_combine(dispatcher.combine_preprocess(output))
    return dispatcher.combine_postprocess(returned), None


def _reference(tokens, probs, routing_map, weight1, weight2, limit):
    rows = []
    for token, scores, assignments in zip(tokens, probs, routing_map):
        contributions = []
        for expert in assignments.nonzero().flatten().tolist():
            gate, up = (token @ weight1[expert]).chunk(2, dim=-1)
            gate = F.silu(gate)
            if limit:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            activation = (gate * up * scores[expert]).to(token.dtype)
            contributions.append(activation @ weight2[expert])
        rows.append(torch.stack(contributions).sum(0))
    return torch.stack(rows)


@pytest.mark.parametrize("limit", [0.0, 0.3])
@pytest.mark.parametrize("freeze_weight", [False, True])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_expert_outputs_and_gradients(limit, freeze_weight, checkpoint_mode):
    torch.manual_seed(42)
    config = MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=4,
        gated_linear_unit=True, activation_func=F.silu,
        moe_swiglu_limit=[limit],
        moe_swiglu_clamp_gate_after_activation=True,
        moe_permute_fusion=False,
    )
    experts, dispatcher = _experts(4, config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.normal_(std=0.5)
            parameter.requires_grad_(not freeze_weight)
    routing_map = torch.tensor([
        [True, True, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [True, True, False, False],
    ])
    tokens = torch.randn(4, 4, requires_grad=True)
    probs = torch.rand(4, 4, requires_grad=True)
    ref_tokens = tokens.detach().clone().requires_grad_()
    ref_probs = probs.detach().clone().requires_grad_()
    ref_w1 = experts.weight1.detach().clone().requires_grad_(not freeze_weight)
    ref_w2 = experts.weight2.detach().clone().requires_grad_(not freeze_weight)

    if checkpoint_mode is None:
        output, _ = _routed_forward(experts, dispatcher, tokens, routing_map, probs)
    else:
        output, _ = checkpoint(
            lambda x, r, p: _routed_forward(experts, dispatcher, x, r, p),
            tokens, routing_map, probs, use_reentrant=checkpoint_mode,
        )
    expected = _reference(
        ref_tokens, ref_probs, routing_map,
        ref_w1.view(4, 4, 6), ref_w2.view(4, 3, 4), limit,
    )
    torch.testing.assert_close(output, expected)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(tokens.grad, ref_tokens.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    if not freeze_weight:
        torch.testing.assert_close(experts.weight1.grad, ref_w1.grad)
        torch.testing.assert_close(experts.weight2.grad, ref_w2.grad)
        assert not experts.weight1.grad.view(4, 4, 6)[3].any()


def test_empty_tokens_still_connect_expert_parameters_to_backward():
    experts, dispatcher = _experts(
        2,
        MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=2,
            gated_linear_unit=True, activation_func=F.silu,
            moe_permute_fusion=False,
        ),
    )
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(0.1)
    x = torch.empty(0, 4, requires_grad=True)
    probs = torch.empty(0, 2, requires_grad=True)
    output, _ = _routed_forward(experts, dispatcher, x, torch.empty(0, 2, dtype=torch.bool), probs)
    output.sum().backward()
    assert output.shape == (0, 4)
    assert x.grad is not None
    assert probs.grad is not None
    for parameter in experts.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.any()


def _replace():
    hf_config = SimpleNamespace(
        hidden_size=4, moe_intermediate_size=3, num_experts=4,
        num_experts_per_tok=2, n_group=1, topk_group=1,
        routed_scaling_factor=1.5, scoring_func="sigmoid", router_dtype="fp32",
        torch_dtype=torch.float32, expert_swiglu_limit_list=[0.3],
    )
    original = nn.Module()
    original.experts = nn.Identity()
    original.gate = nn.Identity()
    original.shared_experts = nn.Linear(4, 4, bias=False)
    original.layer_idx = 0
    return _replace_moe_only(
        original, hf_config, {"sp_group": None},
        {"ep_group": object(), "ep_size": 1},
        MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        EngineConfig(),
    )


def test_bailing_replacement_preserves_shared_experts_and_parameter_layout():
    torch.manual_seed(5)
    layer = _replace()
    assert type(layer.experts) is GroupedMLP
    assert isinstance(layer.token_dispatcher, MoEAlltoAllTokenDispatcher)
    assert layer.experts.weight1.shape == (16, 6)
    assert layer.experts.weight2.shape == (12, 4)
    assert "shared_experts.weight" in layer.state_dict()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.1)
    x = torch.randn(2, 3, 4, requires_grad=True)
    routing, expected_logits = layer.router(x)
    expected = _reference(
        x.reshape(-1, 4), routing.probs, routing.routing_map,
        layer.experts.weight1.view(4, 4, 6), layer.experts.weight2.view(4, 3, 4), 0.3,
    ).reshape_as(x) + layer.shared_experts(x)
    output, logits = layer(x)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(logits, expected_logits.reshape(-1, 4))
    output.square().sum().backward()
    assert layer.shared_experts.weight.grad.norm() > 0


@pytest.mark.parametrize("num_forwards", [1, 2])
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_expert_only_checkpoint_preserves_routing_and_gradients(num_forwards, use_reentrant):
    from contextlib import nullcontext

    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.parallel import FSDPPlanSpec

    torch.manual_seed(427)
    reference, candidate = _replace(), _replace()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.5)
    candidate.load_state_dict(reference.state_dict())
    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
        prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
    )
    plan = SimpleNamespace(modules={"experts": FSDPPlanSpec(activation_checkpoint=True)})
    FSDP2Backend._apply_activation_checkpointing(backend, candidate, plan)
    inputs = [torch.randn(1, 3 + i, 4, requires_grad=True) for i in range(num_forwards)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    outputs = [candidate(x)[0] for x in inputs]
    expected = [reference(x)[0] for x in ref_inputs]
    for actual, ref in zip(outputs, expected):
        torch.testing.assert_close(actual, ref)
    sum(y.square().sum() for y in outputs).backward()
    sum(y.square().sum() for y in expected).backward()
    for actual, ref in zip(inputs, ref_inputs):
        torch.testing.assert_close(actual.grad, ref.grad)
    assert reference.router.weight.grad.norm() > 0
    for (name, parameter), (ref_name, ref_parameter) in zip(
        candidate.named_parameters(), reference.named_parameters(),
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)
