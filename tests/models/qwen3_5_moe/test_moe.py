"""Qwen text and multimodal MoE replacement preserves shared expert gates."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3MoeConfig

from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.qwen3_5_moe.parallel_plan import _replace_qwen3_5_moe_only
from torchturbo.models.qwen3_moe.parallel_plan import QWen3MoeLayerWrapper
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    def cpu_moe_config(*, moe_permute_fusion, **kwargs):
        assert moe_permute_fusion is True
        return MoEConfig(moe_permute_fusion=False, **kwargs)

    monkeypatch.setattr(
        "torchturbo.models.qwen3_5_moe.parallel_plan.MoEConfig",
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


@pytest.mark.parametrize("normalize", [False, True])
def test_qwen3_router_respects_model_normalization(normalize):
    config = Qwen3MoeConfig(
        hidden_size=4, moe_intermediate_size=3, num_experts=3, num_experts_per_tok=2,
        norm_topk_prob=normalize, torch_dtype=torch.float32,
    )
    layer = QWen3MoeLayerWrapper(
        nn.Module(), config, {"sp_group": None}, {"ep_group": object(), "ep_size": 1},
        MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        EngineConfig(),
    )
    logits = torch.tensor([[[2.0, 1.0, 0.0]]])
    expected = logits.softmax(-1)[..., :2]
    if normalize:
        expected = expected / expected.sum(-1, keepdim=True)

    probs = layer.router.routing(logits).probs

    torch.testing.assert_close(probs[:, :2], expected.reshape(1, 2))


@pytest.mark.parametrize("topk", [1, 2])
def test_qwen3_5_router_preserves_normalized_probabilities(topk):
    config = SimpleNamespace(
        hidden_size=4, moe_intermediate_size=3, num_experts=3, num_experts_per_tok=topk,
    )
    original = nn.Module()
    original.experts = nn.Identity()
    original.gate = nn.Identity()
    layer = _replace_qwen3_5_moe_only(
        original, config, {"sp_group": None}, {"ep_group": object(), "ep_size": 1},
        MoESubmodules(experts=ModuleSpec(module=GroupedMLP)), EngineConfig(),
    )
    logits = torch.tensor([[[2.0, 1.0, 0.0]]])
    expected = logits.softmax(-1)[..., :topk]
    expected = expected / expected.sum(-1, keepdim=True)

    probs = layer.router.routing(logits).probs

    torch.testing.assert_close(probs[:, :topk], expected.reshape(1, topk))


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_qwen_preserves_shared_gate_outputs_and_gradients(multimodal, checkpoint_mode):
    torch.manual_seed(29)
    text_config = SimpleNamespace(
        hidden_size=4, moe_intermediate_size=3, num_experts=4, num_experts_per_tok=2,
    )
    hf_config = SimpleNamespace(text_config=text_config) if multimodal else text_config

    def replace():
        original = nn.Module()
        original.experts = nn.Identity()
        original.gate = nn.Identity()
        original.shared_expert = nn.Sequential(nn.Linear(4, 3), nn.SiLU(), nn.Linear(3, 4))
        original.shared_expert_gate = nn.Linear(4, 1, bias=False)
        layer = _replace_qwen3_5_moe_only(
            original, hf_config, {"sp_group": None},
            {"ep_group": object(), "ep_size": 1},
            MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            EngineConfig(),
        )
        assert layer.shared_expert is original.shared_expert
        assert layer.shared_expert_gate is original.shared_expert_gate
        return layer

    regular, candidate = replace(), replace()
    assert type(regular.experts) is GroupedMLP
    assert type(candidate.experts) is GroupedMLP
    assert candidate.config.moe_token_dispatcher_type == "alltoall"
    with torch.no_grad():
        for parameter in regular.parameters():
            parameter.normal_(std=0.2)
    candidate.load_state_dict(regular.state_dict(), strict=True)
    x = torch.randn(2, 3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    if checkpoint_mode is None:
        actual, logits = candidate(x)
    else:
        actual, logits = checkpoint(candidate, x, use_reentrant=checkpoint_mode)
    expected, ref_logits = regular(ref_x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, ref_logits)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        candidate.named_parameters(), regular.named_parameters(),
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)
