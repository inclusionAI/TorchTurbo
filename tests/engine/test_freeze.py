"""Tests for engine-owned freeze policy application."""

import os

import pytest
import torch.nn as nn

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


class _GroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)

    def get_parameter_groups(self):
        from torchturbo.engine.config import ParameterGroup

        return {
            ParameterGroup.VISION_TOWER: self.vision.named_parameters(
                prefix="vision"
            ),
            ParameterGroup.LANGUAGE_MODEL: self.language.named_parameters(
                prefix="language"
            ),
        }


def test_engine_applies_freeze_policy_to_model_parameter_group():
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.engine.config import FreezePolicy, ParameterGroup

    engine = object.__new__(TorchTurboEngine)
    model = _GroupedModel()

    engine._apply_freeze_policy(
        model, FreezePolicy((ParameterGroup.VISION_TOWER,))
    )

    assert all(not p.requires_grad for p in model.vision.parameters())
    assert all(p.requires_grad for p in model.language.parameters())


def test_engine_rejects_unsupported_freeze_group():
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.engine.config import FreezePolicy, ParameterGroup

    engine = object.__new__(TorchTurboEngine)
    model = _GroupedModel()

    with pytest.raises(ValueError, match="unsupported parameter group"):
        engine._apply_freeze_policy(
            model, FreezePolicy((ParameterGroup.MULTI_MODAL_PROJECTOR,))
        )


# ---------------------------------------------------------------------------
# Pattern-based freeze (wildcard FQN matching, same semantics as parallel plans)
# ---------------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)


class _LayeredModel(nn.Module):
    """Exposes names like ``model.layers.<i>.{q_proj,mlp}.weight``."""

    def __init__(self):
        super().__init__()
        model = nn.Module()
        model.add_module("layers", nn.ModuleList([_Layer(), _Layer()]))
        model.add_module("lm_head", nn.Linear(2, 2, bias=False))
        self.add_module("model", model)


class _GroupedLayeredModel(_LayeredModel):
    def __init__(self):
        super().__init__()
        self.add_module("vision", nn.Linear(2, 2, bias=False))

    def get_parameter_groups(self):
        from torchturbo.engine.config import ParameterGroup

        return {
            ParameterGroup.VISION_TOWER: self.vision.named_parameters(
                prefix="vision"
            ),
            ParameterGroup.LANGUAGE_MODEL: self.model.named_parameters(
                prefix="model"
            ),
        }


class _PeftLikeModel(nn.Module):
    """Mimics PEFT naming: ``base_model.model.`` prefix, ``base_layer`` and LoRA adapters."""

    def __init__(self):
        super().__init__()
        base_model = nn.Module()
        model = nn.Module()
        layer = nn.Module()
        q_proj = nn.Module()
        q_proj.add_module("base_layer", nn.Linear(2, 2, bias=False))
        for adapter in ("lora_A", "lora_B"):
            module = nn.Module()
            module.add_module("default", nn.Linear(2, 2, bias=False))
            q_proj.add_module(adapter, module)
        layer.add_module("q_proj", q_proj)
        model.add_module("layers", nn.ModuleList([layer]))
        base_model.add_module("model", model)
        self.add_module("base_model", base_model)


def _make_engine():
    from torchturbo.engine.core import TorchTurboEngine

    return object.__new__(TorchTurboEngine)


def test_pattern_freezes_only_matching_parameters():
    from torchturbo.engine.config import FreezePolicy

    model = _LayeredModel()

    _make_engine()._apply_freeze_policy(
        model, FreezePolicy(patterns=("model.layers.*.q_proj.weight",))
    )

    params = dict(model.named_parameters())
    assert not params["model.layers.0.q_proj.weight"].requires_grad
    assert not params["model.layers.1.q_proj.weight"].requires_grad
    assert params["model.layers.0.mlp.weight"].requires_grad
    assert params["model.lm_head.weight"].requires_grad


def test_pattern_star_stays_within_one_segment():
    from torchturbo.engine.config import FreezePolicy

    model = _LayeredModel()

    # '*' cannot cross the q_proj/mlp segment, so this pattern matches nothing.
    with pytest.raises(ValueError, match="matched no parameters"):
        _make_engine()._apply_freeze_policy(
            model, FreezePolicy(patterns=("model.layers.*.weight",))
        )

    assert all(p.requires_grad for p in model.parameters())


def test_pattern_double_star_crosses_segments():
    from torchturbo.engine.config import FreezePolicy

    model = _LayeredModel()

    _make_engine()._apply_freeze_policy(
        model, FreezePolicy(patterns=("model.layers.**",))
    )

    params = dict(model.named_parameters())
    assert not params["model.layers.0.q_proj.weight"].requires_grad
    assert not params["model.layers.1.mlp.weight"].requires_grad
    assert params["model.lm_head.weight"].requires_grad


def test_pattern_matching_nothing_raises():
    from torchturbo.engine.config import FreezePolicy

    model = _LayeredModel()

    with pytest.raises(ValueError, match="matched no parameters"):
        _make_engine()._apply_freeze_policy(
            model, FreezePolicy(patterns=("no.such.**",))
        )


def test_patterns_do_not_require_get_parameter_groups():
    from torchturbo.engine.config import FreezePolicy

    # _LayeredModel defines no get_parameter_groups(); patterns must still work.
    model = _LayeredModel()

    _make_engine()._apply_freeze_policy(
        model, FreezePolicy(patterns=("model.lm_head.weight",))
    )

    params = dict(model.named_parameters())
    assert not params["model.lm_head.weight"].requires_grad
    assert params["model.layers.0.q_proj.weight"].requires_grad


def test_patterns_match_peft_stripped_names():
    from torchturbo.engine.config import FreezePolicy

    model = _PeftLikeModel()
    raw_name = "base_model.model.layers.0.q_proj.base_layer.weight"
    assert raw_name in dict(model.named_parameters())

    # Pattern is written against the base-model FQN.
    _make_engine()._apply_freeze_policy(
        model, FreezePolicy(patterns=("layers.*.q_proj.weight",))
    )

    params = dict(model.named_parameters())
    assert not params[raw_name].requires_grad
    # LoRA adapter weights stay trainable unless a pattern selects them.
    assert params["base_model.model.layers.0.q_proj.lora_A.default.weight"].requires_grad
    assert params["base_model.model.layers.0.q_proj.lora_B.default.weight"].requires_grad


def test_patterns_can_freeze_base_weights_and_selected_lora_weights():
    from torchturbo.engine.config import FreezePolicy

    model = _PeftLikeModel()

    # Freeze the base weight plus only lora_B; lora_A stays trainable.
    _make_engine()._apply_freeze_policy(
        model,
        FreezePolicy(
            patterns=(
                "layers.*.q_proj.weight",
                "layers.*.q_proj.lora_B.**",
            )
        ),
    )

    params = dict(model.named_parameters())
    assert not params[
        "base_model.model.layers.0.q_proj.base_layer.weight"
    ].requires_grad
    assert not params[
        "base_model.model.layers.0.q_proj.lora_B.default.weight"
    ].requires_grad
    assert params[
        "base_model.model.layers.0.q_proj.lora_A.default.weight"
    ].requires_grad


def test_groups_and_patterns_combine():
    from torchturbo.engine.config import FreezePolicy, ParameterGroup

    model = _GroupedLayeredModel()

    _make_engine()._apply_freeze_policy(
        model,
        FreezePolicy(
            parameter_groups=(ParameterGroup.VISION_TOWER,),
            patterns=("model.layers.*.q_proj.weight",),
        ),
    )

    params = dict(model.named_parameters())
    assert not params["vision.weight"].requires_grad
    assert not params["model.layers.0.q_proj.weight"].requires_grad
    assert params["model.layers.0.mlp.weight"].requires_grad
    assert params["model.lm_head.weight"].requires_grad
