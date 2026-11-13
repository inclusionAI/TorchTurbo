"""Model adapter expert selection and routing probabilities."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torchturbo.engine.config import EngineConfig
from torchturbo.models.bailing_hybrid.configuration_bailing_moe_v3 import BailingMoeV3Config
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import BailingMoeV3Gate
from torchturbo.models.bailing_hybrid.parallel_plan import _replace_moe_only as replace_bailing
from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Gate
from torchturbo.models.deepseek_v4.parallel_plan import _replace_moe_only as replace_deepseek
from torchturbo.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from torchturbo.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE
from torchturbo.models.glm_moe_dsa.parallel_plan import _replace_moe_only as replace_glm
from torchturbo.models.kimi_k25.configuration_deepseek import DeepseekV3Config
from torchturbo.models.kimi_k25.modeling_deepseek import DeepseekV3TopkRouter
from torchturbo.models.kimi_k25.parallel_plan import _replace_kimi_moe_only
from torchturbo.models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from torchturbo.models.llada2_moe.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
from torchturbo.models.llada2_moe.parallel_plan import _replace_llada2_moe_only
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def replace_layer(replace, original, config):
    return replace(
        original, config, sp_info={"sp_group": None},
        ep_info={"ep_group": object(), "ep_size": 1},
        moe_submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        engine_config=EngineConfig(),
    )


@pytest.mark.parametrize("score", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("topk", [1, 2])
def test_deepseek_router_keeps_native_selection_and_megatron_normalization(score, hash_routing, topk):
    torch.manual_seed(71)
    config = DeepseekV4Config(
        hidden_size=4, moe_intermediate_size=3, n_routed_experts=4,
        num_experts_per_tok=topk, vocab_size=8, num_hash_layers=int(hash_routing),
        scoring_func=score, route_scale=1.7, routed_scaling_factor=1.7,
        torch_dtype=torch.float32,
    )
    gate = DeepseekV4Gate(config, layer_idx=0)
    original = torch.nn.Module()
    original.gate, original.experts, original.layer_idx = gate, torch.nn.Identity(), 0
    router = replace_layer(replace_deepseek, original, config).router
    with torch.no_grad():
        gate.weight.copy_(torch.arange(16).view(4, 4) / 17)
        router.weight.copy_(gate.weight)
        if hash_routing:
            router.tid2eid.copy_(gate.tid2eid)
        else:
            gate.bias.copy_(torch.tensor([1.0, 0.0, 0.0, -1.0]))
            router.expert_bias.copy_(gate.bias)
    inputs = torch.tensor([[[1., 0., -.5, 2.], [-1., .5, 1., 0.]]], requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    ids = torch.tensor([[0, 1]])
    routing_probs, _, _, _ = router(inputs, ids)
    weights, indices = gate(ref_inputs.reshape(-1, config.hidden_size), ids.flatten())
    if score != "softmax" and topk == 1:
        logits = F.linear(ref_inputs.reshape(-1, config.hidden_size), gate.weight)
        scores = logits.sigmoid() if score == "sigmoid" else F.softplus(logits).sqrt()
        weights = scores.gather(1, indices) * config.route_scale
    expected = torch.zeros_like(routing_probs).scatter(1, indices.view(2, topk), weights.view(2, topk))
    torch.testing.assert_close(routing_probs, expected, rtol=0, atol=0)
    gradient = torch.arange(8, dtype=torch.float32).view(2, 4)
    actual_grads = torch.autograd.grad(routing_probs, (inputs, router.weight), gradient)
    expected_grads = torch.autograd.grad(expected, (ref_inputs, gate.weight), gradient)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk,num_groups,group_topk", [(2, 8, 1), (4, 4, 2), (8, 8, 4)])
def test_llada_native_routing_matches_when_group_score_formulas_agree(topk, num_groups, group_topk):
    config = LLaDA2MoeConfig(
        hidden_size=2, moe_intermediate_size=2, num_experts=16,
        num_experts_per_tok=topk, n_group=num_groups, topk_group=group_topk,
        torch_dtype=torch.float32,
    )
    original = LLaDA2MoeSparseMoeBlock(config)
    router = replace_layer(_replace_llada2_moe_only, original, config).router
    with torch.no_grad():
        original.gate.weight.zero_()
        original.gate.weight[:, 0] = torch.arange(-11, 5)
        router.weight.copy_(original.gate.weight)
        router.expert_bias.copy_(original.gate.expert_bias)
    inputs = torch.tensor([[[1., 0.], [-1., .5]]], requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    actual_probs, _, _, _ = router(inputs)
    indices, weights, _ = original.gate(ref_inputs)
    expected = torch.zeros_like(actual_probs).scatter(1, indices, weights)
    torch.testing.assert_close(actual_probs, expected, rtol=0, atol=0)
    gradient = torch.arange(32, dtype=torch.float32).view(2, 16)
    actual_grads = torch.autograd.grad(actual_probs, (inputs, router.weight), gradient)
    expected_grads = torch.autograd.grad(expected, (ref_inputs, original.gate.weight), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_group_limited_routing_accepts_an_empty_source_rank():
    logits = torch.empty(0, 16, requires_grad=True)
    probs, routing_map, expert_indices = topk_softmax_with_capacity(
        logits, 4, num_groups=4, group_topk=2, score_function="sigmoid",
    )
    assert probs.shape == routing_map.shape == (0, 16)
    assert expert_indices.shape == (0, 4)
    probs.sum().backward()
    torch.testing.assert_close(logits.grad, torch.empty_like(logits))


@pytest.mark.parametrize("model", ["bailing", "kimi", "glm"])
def test_group_adapters_preserve_default_native_routing(model):
    torch.manual_seed(93)
    common = dict(hidden_size=4, moe_intermediate_size=3, torch_dtype=torch.float32)
    original = torch.nn.Module()
    original.experts = torch.nn.Identity()
    if model == "bailing":
        config = BailingMoeV3Config(**common)
        original.gate = BailingMoeV3Gate(config)
        router = replace_layer(replace_bailing, original, config).router
        bias = original.gate.expert_bias
    elif model == "kimi":
        config = DeepseekV3Config(**common)
        original.gate = DeepseekV3TopkRouter(config)
        router = replace_layer(_replace_kimi_moe_only, original, SimpleNamespace(text_config=config)).router
        bias = original.gate.e_score_correction_bias
    else:
        config = GlmMoeDsaConfig(**common)
        original = GlmMoeDsaMoE(config)
        router = replace_layer(replace_glm, original, config).router
        bias = original.gate.e_score_correction_bias
    with torch.no_grad():
        original.gate.weight.normal_(std=0.2)
        bias.copy_(torch.linspace(-0.02, 0.02, bias.numel()))
        router.weight.copy_(original.gate.weight)
        router.expert_bias.copy_(bias)
    inputs = torch.randn(1, 5, 4, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    actual_probs, _, actual_indices, _ = router(inputs)
    if model == "bailing":
        indices, weights, _ = original.gate(ref_inputs)
    elif model == "kimi":
        indices, weights = original.gate(ref_inputs)
    else:
        indices, weights = original.route_tokens_to_experts(original.gate(ref_inputs))
    expected = torch.zeros_like(actual_probs).scatter(1, indices, weights)
    torch.testing.assert_close(
        actual_indices.sort(-1).values, indices.sort(-1).values,
    )
    torch.testing.assert_close(actual_probs, expected)
    gradient = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual_probs, (inputs, router.weight), gradient)
    expected_grads = torch.autograd.grad(expected, (ref_inputs, original.gate.weight), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "replace", [replace_bailing, _replace_kimi_moe_only, replace_glm, _replace_llada2_moe_only],
    ids=["bailing", "kimi", "glm", "llada2"],
)
def test_group_adapters_derive_group_score_count_from_megatron_topk(replace):
    config = SimpleNamespace(
        hidden_size=4, moe_intermediate_size=3, num_experts=8, n_routed_experts=8,
        num_experts_per_tok=4, n_group=2, topk_group=1,
        routed_scaling_factor=1.0, scoring_func="sigmoid", router_dtype="fp32",
        torch_dtype=torch.float32,
    )
    original = torch.nn.Module()
    original.gate, original.experts = torch.nn.Identity(), torch.nn.Identity()
    if replace is _replace_llada2_moe_only:
        config = LLaDA2MoeConfig(**vars(config))
        original = LLaDA2MoeSparseMoeBlock(config)
    hf_config = SimpleNamespace(text_config=config) if replace is _replace_kimi_moe_only else config
    router = replace_layer(replace, original, hf_config).router
    # Top-4 sums favor group 1 (2.4 > 1.72); fixed top-2 favors group 0 (1.7 > 1.2).
    scores = torch.tensor([[0.9, 0.8, 0.01, 0.01, 0.6, 0.6, 0.6, 0.6]])
    actual_probs, _, actual_indices = router.routing(scores.logit().unsqueeze(0))
    torch.testing.assert_close(actual_indices.sort(-1).values, torch.tensor([[4, 5, 6, 7]]))
    torch.testing.assert_close(actual_probs, torch.tensor([[0., 0., 0., 0., .25, .25, .25, .25]]))


@pytest.mark.parametrize("model", ["kimi", "glm"])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_sigmoid_adapters_follow_megatron_normalization(model, topk, normalize):
    config = GlmMoeDsaConfig(
        hidden_size=4, moe_intermediate_size=3, n_routed_experts=16,
        num_experts_per_tok=topk, n_group=1, topk_group=1,
        norm_topk_prob=normalize, routed_scaling_factor=1.7,
    )
    if model == "kimi":
        original = torch.nn.Module()
        original.gate = DeepseekV3TopkRouter(config)
        original.experts = torch.nn.Identity()
        router = replace_layer(_replace_kimi_moe_only, original, SimpleNamespace(text_config=config)).router
    else:
        original = GlmMoeDsaMoE(config)
        router = replace_layer(replace_glm, original, config).router
    with torch.no_grad():
        original.gate.weight.copy_(torch.arange(64).view(16, 4) / 67)
        router.weight.copy_(original.gate.weight)
        router.expert_bias.copy_(original.gate.e_score_correction_bias)
    inputs = torch.tensor([[[1., 0., -.5, 2.], [-1., .5, 1., 0.]]], requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    actual_probs, _, _, _ = router(inputs)
    if model == "kimi":
        indices, _ = original.gate(ref_inputs)
    else:
        indices, _ = original.route_tokens_to_experts(original.gate(ref_inputs))
    logits = F.linear(ref_inputs.reshape(-1, config.hidden_size), original.gate.weight)
    weights = logits.sigmoid().gather(1, indices)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * config.routed_scaling_factor
    expected = torch.zeros_like(actual_probs).scatter(1, indices, weights)
    torch.testing.assert_close(actual_probs, expected, rtol=0, atol=0)
    gradient = torch.arange(32, dtype=torch.float32).view(2, 16)
    actual_grads = torch.autograd.grad(actual_probs, (inputs, router.weight), gradient)
    expected_grads = torch.autograd.grad(expected, (ref_inputs, original.gate.weight), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
