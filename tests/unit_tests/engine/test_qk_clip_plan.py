from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from torchturbo.engine.config import EngineConfig, MuonOptimizerConfig, OptimizerConfig, QKClipConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2 import optimizers as muon_backend
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.engine.optimizers import (
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonPlan,
    MuonRecipe,
    OptimizersContainer,
    QKClipSpec,
)
from torchturbo.models.modules.attention.qk_clip import get_qk_clip_state
from torchturbo.runtime.cpu import CpuRuntime


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False)
        self.kv_b_proj = nn.Linear(4, 8, bias=False)


class _SingleProjectionSpec(QKClipSpec):
    num_heads: int = 2

    @property
    def projection_rows(self) -> Mapping[str, int]:
        return {"q_proj": 6}


def _model_and_recipe():
    model = nn.Module()
    model.attention = _Attention()
    spec = MLAQKClipSpec("q_proj", "kv_b_proj", 2, 2, 1, 2)
    return model, MuonRecipe({}, qk_clip_specs={"attention": spec})


def _plan(model, recipe, enabled=True):
    return MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(
            lr=0.1, weight_decay=0.0, qk_clip=QKClipConfig() if enabled else None
        ),
        model_fqn_map=build_model_fqn_map(model),
    )


def test_qk_clip_config_parses_through_existing_optimizer_config():
    config = EngineConfig.from_dict({
        "optimizer": {"name": "muon", "muon": {
            "qk_clip": {"threshold": 80.0, "alpha": 0.25}
        }}
    })
    assert config.optimizer.muon.qk_clip == QKClipConfig(80.0, 0.25)
    assert MuonOptimizerConfig().qk_clip is None


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0}, {"threshold": float("inf")},
    {"alpha": -0.1}, {"alpha": 1.1}, {"alpha": float("nan")},
])
def test_qk_clip_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        QKClipConfig(**kwargs)


def test_qk_clip_plan_accepts_public_spec_contract():
    model, _ = _model_and_recipe()
    spec = _SingleProjectionSpec()
    recipe = MuonRecipe({}, qk_clip_specs={"attention": spec})

    layer, = _plan(model, recipe).qk_clip_plan.layers

    assert recipe.qk_clip_specs["attention"] is spec
    assert layer.spec is spec
    assert layer.module is model.attention
    assert tuple(layer.weights) == ("q_proj",)
    assert layer.weights["q_proj"] is model.attention.q_proj.weight
    assert get_qk_clip_state(model.attention) is None


def test_mla_qk_clip_requires_distinct_projection_names():
    with pytest.raises(ValueError, match="distinct projection names"):
        MLAQKClipSpec("q_proj", "q_proj", 2, 2, 1, 2)


def test_recipe_binds_checkpoint_wrapped_active_parameters():
    model, recipe = _model_and_recipe()
    attention = model.attention
    attention.q_proj = checkpoint_wrapper(attention.q_proj)
    model.attention = checkpoint_wrapper(attention)

    plan = _plan(model, recipe)

    layer, = plan.qk_clip_plan.layers
    assert layer.module is attention
    assert layer.weights["q_proj"] is attention.q_proj._checkpoint_wrapped_module.weight
    assert layer.weights["kv_b_proj"] is attention.kv_b_proj.weight
    assert get_qk_clip_state(attention) is None
    assert "qk_clip_state" not in vars(attention)


def test_disabled_clipping_does_not_bind_or_require_attention_recipe():
    model, _ = _model_and_recipe()
    assert _plan(model, MuonRecipe({}), enabled=False).qk_clip_plan is None
    assert get_qk_clip_state(model.attention) is None
    assert "qk_clip_state" not in vars(model.attention)


def test_frozen_pair_is_skipped_but_partial_freeze_is_rejected():
    model, recipe = _model_and_recipe()
    model.extra = nn.Linear(4, 4, bias=False)
    model.attention.requires_grad_(False)
    assert not _plan(model, recipe).qk_clip_plan.layers
    assert get_qk_clip_state(model.attention) is None
    assert "qk_clip_state" not in vars(model.attention)
    model.attention.q_proj.requires_grad_(True)
    with pytest.raises(ValueError, match="all final projections"):
        _plan(model, recipe)


@pytest.mark.parametrize("invalid", ["missing", "shape", "overlap", "bias", "adapter"])
def test_recipe_rejects_invalid_attention_binding(invalid):
    model, recipe = _model_and_recipe()
    spec = recipe.qk_clip_specs["attention"]
    if invalid == "missing":
        recipe = MuonRecipe({})
        message = "explicit QK clip recipe"
    elif invalid == "shape":
        recipe = MuonRecipe({}, qk_clip_specs={"attention": replace(spec, num_heads=3)})
        message = "dimensions"
    elif invalid == "overlap":
        recipe = MuonRecipe({}, qk_clip_specs={"attention": spec, "*": spec})
        message = "overlap"
    elif invalid == "bias":
        model.attention.q_proj = nn.Linear(4, 6, bias=True)
        message = "bias-free"
    else:
        projection = model.attention.q_proj
        model.attention.q_proj = nn.Sequential(projection)
        message = "without adapters"
    with pytest.raises(ValueError, match=message):
        _plan(model, recipe)


@pytest.mark.parametrize("q_lora_rank", [None, 4])
def test_bailing_recipe_selects_only_mla_and_configured_query(q_lora_rank):
    from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
        BailingMoeV3KimiDeltaAttention,
        BailingMoeV3MultiLatentAttention,
    )
    from torchturbo.models.bailing_hybrid.optimizer_recipe import get_muon_recipe

    config = SimpleNamespace(
        q_lora_rank=q_lora_rank, num_attention_heads=2,
        qk_nope_head_dim=2, qk_rope_head_dim=1, v_head_dim=2,
    )
    # The recipe reads module types without running attention kernels.
    layers = [
        SimpleNamespace(attention=object.__new__(attention_type))
        for attention_type in (
            BailingMoeV3MultiLatentAttention,
            BailingMoeV3KimiDeltaAttention,
            BailingMoeV3MultiLatentAttention,
        )
    ]
    model = SimpleNamespace(config=config, get_decoder=lambda: SimpleNamespace(layers=layers))
    recipe = get_muon_recipe(model)
    query = "q_proj" if q_lora_rank is None else "q_b_proj"
    assert tuple(recipe.qk_clip_specs) == (
        "model.layers.0.attention", "model.layers.2.attention"
    )
    assert all(spec.query_projection == query for spec in recipe.qk_clip_specs.values())
    assert f"model.layers.0.attention.{query}.weight" in recipe.parameter_overrides
    assert "model.layers.1.attention.q_proj.weight" in recipe.parameter_overrides


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "frozen"])
def test_backend_installs_clipping_on_final_optimizer(monkeypatch, fallback, mode):
    model, recipe = _model_and_recipe()
    model.extra = nn.Linear(4, 4, bias=fallback)
    if mode == "frozen":
        model.attention.requires_grad_(False)
    engine = _engine(monkeypatch, model, recipe, enabled=mode != "disabled")

    optimizer = engine.prepare_optimizer()

    assert engine.optimizer is optimizer
    assert isinstance(optimizer, OptimizersContainer) == fallback
    if not fallback:
        assert type(optimizer) is torch.optim.SGD
    assert (get_qk_clip_state(model.attention) is not None) == (mode == "enabled")
    assert ("qk_clip_state" in vars(model.attention)) == (mode == "enabled")
    assert len(optimizer._optimizer_step_pre_hooks) == int(mode == "enabled")
    assert len(optimizer._optimizer_step_post_hooks) == int(mode == "enabled")
    if fallback:
        assert all(not child._optimizer_step_pre_hooks for child in optimizer)
        assert all(not child._optimizer_step_post_hooks for child in optimizer)


def _engine(monkeypatch, model, recipe, *, enabled=True):
    # Exercise the real backend assembly and hooks with CPU update kernels.
    def prepare_children(plan, **kwargs):
        return {
            name: torch.optim.SGD(
                [group.get_optimizer_param_group() for group in groups], lr=0.1
            )
            for name, groups in plan.get_param_groups_by_optimizer_name().items()
        }

    monkeypatch.setattr(muon_backend, "prepare_fsdp_muon_optimizers", prepare_children)
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = CpuRuntime()
    backend.fsdp_config = SimpleNamespace(cpu_offload=SimpleNamespace(enabled=False))
    backend._model_fqn_map = build_model_fqn_map(model)
    backend._sp_group = None
    backend._sp_size = 1
    backend._sp_rank = 0
    model.get_optimizer_recipe = lambda name: recipe

    engine = object.__new__(TorchTurboEngine)
    engine.model = model
    engine.backend = backend
    engine.optimizer = None
    engine.config = EngineConfig(optimizer=OptimizerConfig(
        name="muon", muon=MuonOptimizerConfig(
            lr=0.1, weight_decay=0.0, momentum=0.0, nesterov=False,
            qk_clip=QKClipConfig() if enabled else None,
        ),
    ))
    return engine


def test_backend_clips_after_both_query_and_key_optimizers(monkeypatch):
    model, recipe = _model_and_recipe()
    recipe = replace(recipe, parameter_overrides={
        "attention.kv_b_proj.weight": ExcludeFromMuon(),
    })
    engine = _engine(monkeypatch, model, recipe)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
            parameter.grad = torch.ones_like(parameter)
    optimizer = engine.prepare_optimizer()
    assert optimizer.optimizer_names == ["Muon", "AdamW"]

    with pytest.raises(RuntimeError, match="statistics from every configured layer"):
        optimizer.step()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))

    state = model.attention.qk_clip_state
    state.record(torch.tensor([400.0, 100.0]))
    optimizer.step()

    expected_query = torch.full_like(model.attention.q_proj.weight, 0.9)
    expected_query[:2].mul_(0.5)
    expected_query[2].mul_(0.25)
    expected_key_value = torch.full_like(model.attention.kv_b_proj.weight, 0.9)
    expected_key_value[:2].mul_(0.5)
    torch.testing.assert_close(model.attention.q_proj.weight, expected_query)
    torch.testing.assert_close(model.attention.kv_b_proj.weight, expected_key_value)
    assert torch.isneginf(state.max_logits).all()
    assert not state.observed


def test_backend_validates_optimizer_before_binding_attention_stats(monkeypatch):
    model, recipe = _model_and_recipe()
    engine = _engine(monkeypatch, model, recipe)
    prepare_children = muon_backend.prepare_fsdp_muon_optimizers
    children = {}

    def prepare_invalid_children(plan, **kwargs):
        children.update(prepare_children(plan, **kwargs))
        children["Muon"].param_groups[0]["lr"] = 1.0
        return children

    monkeypatch.setattr(muon_backend, "prepare_fsdp_muon_optimizers", prepare_invalid_children)
    with pytest.raises(ValueError, match="changed parameter-group option 'lr'"):
        engine.prepare_optimizer()

    assert engine.optimizer is None
    assert get_qk_clip_state(model.attention) is None
    assert "qk_clip_state" not in vars(model.attention)
    assert not children["Muon"]._optimizer_step_pre_hooks
    assert not children["Muon"]._optimizer_step_post_hooks


def test_repeated_preparation_preserves_installed_optimizer_and_statistics(monkeypatch):
    model, recipe = _model_and_recipe()
    engine = _engine(monkeypatch, model, recipe)
    optimizer = engine.prepare_optimizer()
    state = model.attention.qk_clip_state
    maxima = torch.tensor([200.0, 400.0])
    state.record(maxima)

    with pytest.raises(ValueError, match="already installed"):
        engine.prepare_optimizer()

    assert engine.optimizer is optimizer
    assert model.attention.qk_clip_state is state
    torch.testing.assert_close(state.max_logits, maxima)
    assert state.observed
    assert len(optimizer._optimizer_step_pre_hooks) == 1
    assert len(optimizer._optimizer_step_post_hooks) == 1
