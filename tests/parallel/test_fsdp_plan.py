"""Unit tests for FSDP parallel plan spec semantics."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest

from torchturbo.engine.utils import match_fqn
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
    VisionBalanceMode,
    VisionBalanceSpec,
    resolve_plan_spec,
)


def test_standalone_world_shard_spec_is_valid():
    spec = FSDPPlanSpec(standalone_world_shard=True)

    assert spec.standalone_world_shard is True
    assert spec.no_shard is False
    assert spec.placement is None


@pytest.mark.parametrize("mode", list(VisionBalanceMode))
def test_vision_balance_plan_accepts_spec_with_enum(mode):
    vision_spec = VisionBalanceSpec(mode=mode)
    assert FSDPPlanSpec(vision_balance=vision_spec).vision_balance is vision_spec
    assert vision_spec.mode is mode


@pytest.mark.parametrize("mode", ["patch", "patch_sq", "pixel", True, False])
def test_vision_balance_spec_requires_enum_mode(mode):
    with pytest.raises(TypeError, match="VisionBalanceMode"):
        VisionBalanceSpec(mode=mode)


@pytest.mark.parametrize("value", ["patch", VisionBalanceMode.PATCH, True, {}])
def test_vision_balance_plan_requires_spec(value):
    with pytest.raises(TypeError, match="VisionBalanceSpec"):
        FSDPPlanSpec(vision_balance=value)


def test_vision_balance_spec_defaults_to_disabled():
    spec = VisionBalanceSpec()

    assert spec.mode is None
    assert spec.io is None


def test_no_shard_conflicts_with_standalone_world_shard():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, standalone_world_shard=True)


def test_no_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, placement=FSDPPlacementSpec(object(), "world"))


def test_standalone_world_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="redundant"):
        FSDPPlanSpec(
            standalone_world_shard=True,
            placement=FSDPPlacementSpec(object(), "edp"),
        )


def test_parallel_plan_still_normalizes_tuple_placement_specs():
    plan = FSDPParallelPlan(plan={"layers.*.mlp.experts": (object(), "edp")})

    spec = plan.plan["layers.*.mlp.experts"]
    assert isinstance(spec, FSDPPlanSpec)
    assert isinstance(spec.placement, FSDPPlacementSpec)
    assert spec.placement.mesh_dim == "edp"


def test_parallel_plan_round_trip_preserves_partial_overrides():
    plan = FSDPParallelPlan.from_dict({
        "sp_size": 2,
        "ep_size": 4,
        "tp_size": 2,
        "plan": {"model.visual": {"vision_balance": {"mode": "patch_sq"}}},
    })
    restored = FSDPParallelPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert (restored.sp_size, restored.ep_size, restored.tp_size) == (2, 4, 2)
    spec = restored.plan["model.visual"]
    assert spec.vision_balance.mode is VisionBalanceMode.PATCH_SQ
    assert spec.standalone_world_shard is None
    assert spec.activation_checkpoint is None
    assert spec.module_kwargs is None


def _plan_with_user_patterns(defaults, user):
    plan = FSDPParallelPlan.from_dict({"plan": defaults})
    user_plan = FSDPParallelPlan.from_dict({"plan": user})
    setattr(plan, "_tt_user_plan_patterns", dict(user_plan.plan))
    return plan


def test_user_pattern_overrides_default_first_match_per_fqn():
    plan = _plan_with_user_patterns(
        {
            "model.layers.*": {"activation_checkpoint": True},
        },
        {"model.layers.1": {"activation_checkpoint": False}},
    )

    spec = resolve_plan_spec(plan, "model.layers.1")
    assert spec.activation_checkpoint is False
    assert spec.replace_module is None

    spec = plan.match("model.layers.1")
    assert spec.activation_checkpoint is True


def test_user_pattern_keeps_unspecified_default_fields():
    plan = _plan_with_user_patterns(
        {"model.layers.*.mlp": {"activation_checkpoint": True, "no_shard": True}},
        {"model.layers.*.mlp": {"activation_checkpoint": False}},
    )

    spec = resolve_plan_spec(plan, "model.layers.0.mlp")
    assert spec.activation_checkpoint is False
    assert spec.no_shard is True


def test_user_patterns_use_declared_order_first_match():
    plan = _plan_with_user_patterns(
        {"model.layers.*": {"activation_checkpoint": True}},
        {
            "model.layers.*": {"activation_checkpoint": False},
            "model.layers.1": {"activation_checkpoint": True},
        },
    )

    assert resolve_plan_spec(plan, "model.layers.1").activation_checkpoint is False
    assert resolve_plan_spec(plan, "model.layers.2").activation_checkpoint is False


def test_user_pattern_without_default_creates_entry():
    plan = _plan_with_user_patterns(
        {}, {"model.layers.*.attention": {"activation_checkpoint": True}}
    )

    spec = resolve_plan_spec(plan, "model.layers.3.attention")
    assert spec.activation_checkpoint is True
    assert resolve_plan_spec(plan, "model.layers.3.mlp") is None


def test_match_fqn_star_stays_within_one_segment():
    assert match_fqn(
        "model.layers.*.q_*.weight",
        "model.layers.0.q_proj.weight",
    )
    assert not match_fqn(
        "model.layers.*.weight",
        "model.layers.0.projection.weight",
    )


def test_match_fqn_double_star_crosses_segment_boundaries():
    assert match_fqn(
        "model.layers.**.weight",
        "model.layers.0.attention.q_proj.weight",
    )


def test_parallel_plan_matches_modeling_fqn_only():
    spec = FSDPPlanSpec(no_shard=True)
    plan = FSDPParallelPlan(
        plan={"model.layers.*.projection.weight": spec}
    )

    assert plan.match("model.layers.0.projection.weight") is spec
    assert plan.match(
        "base_model.model.model.layers.0.projection.base_layer.weight"
    ) is None
