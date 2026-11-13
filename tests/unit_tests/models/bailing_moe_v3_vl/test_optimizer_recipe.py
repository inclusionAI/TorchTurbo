"""Muon parameter and QK-clip plans for the real Bailing V3 VL module tree."""

from unittest.mock import patch

import pytest
from torch.distributed.tensor import Shard

from torchturbo.engine.config import MuonOptimizerConfig, QKClipConfig
from torchturbo.engine.optimizers import (
    AttentionPerHeadComputeView,
    ExpertDownComputeView,
    ExpertUpComputeView,
    MuonPlan,
    Owned,
)
from torchturbo.models.bailing_moe_v3_vl.model import (
    TurboBailingMoeV3VLForConditionalGeneration,
)
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig

from .test_model import make_config


def _plan(model, *, qk_clip=False):
    with patch("torchturbo.engine.optimizers.muon.logger") as logger:
        plan = MuonPlan(
            model,
            recipe=model.get_optimizer_recipe("muon"),
            config=MuonOptimizerConfig(
                lr=0.01,
                weight_decay=0.1,
                qk_clip=QKClipConfig() if qk_clip else None,
            ),
        )
    logger.warning_rank0.assert_not_called()
    return plan


def _assert_per_head(plan, model, fqn, matrix_shape):
    spec = plan.parameter_specs[fqn]
    assert spec.compute_placement == Shard(0)
    assert spec.compute_view == AttentionPerHeadComputeView(
        model.config.text_config.num_attention_heads
    )
    resolved = spec.compute_view.resolve(model.get_parameter(fqn).shape)
    assert resolved.matrix_shape == matrix_shape


@pytest.mark.parametrize("q_lora_rank", [None, 4])
@pytest.mark.parametrize("gate_type", [None, "head_wise", "element_wise"])
@pytest.mark.parametrize("no_kda_lora", [False, True])
def test_hybrid_muon_plan_resolves_configured_attention_and_mtp(
    q_lora_rank, gate_type, no_kda_lora
):
    config = make_config(hybrid=True, mtp=1)
    text = config.text_config
    text.num_attention_heads = 4
    text.num_key_value_heads = 4
    text.v_head_dim = 6
    text.q_lora_rank = q_lora_rank
    text.gated_attention_proj_granularity_type = gate_type
    text.no_kda_lora = no_kda_lora
    model = TurboBailingMoeV3VLForConditionalGeneration(config)

    plan = _plan(model, qk_clip=True)
    adamw_fqns = set(plan.adamw_decay_fqns + plan.adamw_no_decay_fqns)
    query = "q_proj" if q_lora_rank is None else "q_b_proj"
    query_columns = text.hidden_size if q_lora_rank is None else q_lora_rank
    for index in (1, 2):
        attention = f"model.layers.{index}.attention"
        _assert_per_head(
            plan, model, f"{attention}.{query}.weight", (12, query_columns)
        )
        _assert_per_head(plan, model, f"{attention}.kv_b_proj.weight", (10, 8))
        gate_fqn = f"{attention}.g_proj.weight"
        if gate_type == "head_wise":
            assert gate_fqn in adamw_fqns
        elif gate_type == "element_wise":
            _assert_per_head(plan, model, gate_fqn, (6, 16))
        else:
            assert gate_fqn not in dict(model.named_parameters())
        for projection in ("kv_a_proj_with_mqa", "dense"):
            spec = plan.parameter_specs[f"{attention}.{projection}.weight"]
            assert spec.compute_placement == Owned()
        if q_lora_rank is not None:
            spec = plan.parameter_specs[f"{attention}.q_a_proj.weight"]
            assert spec.compute_placement == Owned()

    attention = "model.layers.0.attention"
    for projection in ("q_proj", "k_proj", "v_proj"):
        _assert_per_head(plan, model, f"{attention}.{projection}.weight", (8, 16))
    for gate in ("f", "g"):
        projection = f"{gate}_proj" if no_kda_lora else f"{gate}_b_proj"
        columns = 16 if no_kda_lora else 8
        _assert_per_head(
            plan, model, f"{attention}.{projection}.weight", (8, columns)
        )
        if not no_kda_lora:
            spec = plan.parameter_specs[f"{attention}.{gate}_a_proj.weight"]
            assert spec.compute_placement == Owned()
    assert f"{attention}.b_proj.weight" in adamw_fqns
    assert plan.parameter_specs[
        f"{attention}.o_proj.weight"
    ].compute_placement == Owned()

    layers = plan.qk_clip_plan.layers
    assert len(layers) == 2
    for layer, index in zip(layers, (1, 2), strict=True):
        assert layer.module is model.model.layers[index].attention
        assert tuple(layer.weights) == (query, "kv_b_proj")
        assert layer.spec.num_heads == 4
        assert layer.spec.qk_nope_head_dim == 4
        assert layer.spec.qk_rope_head_dim == 8
        assert layer.spec.v_head_dim == 6


def test_muon_keeps_visual_projector_and_embeddings_on_adamw():
    model = TurboBailingMoeV3VLForConditionalGeneration(make_config())

    plan = _plan(model)

    adamw_fqns = set(plan.adamw_decay_fqns + plan.adamw_no_decay_fqns)
    visual_fqns = {
        name for name, _ in model.named_parameters()
        if name.startswith(("model.visual.", "linear_proj."))
    }
    assert visual_fqns
    assert visual_fqns <= adamw_fqns
    assert {"model.word_embeddings.weight", "lm_head.weight"} <= adamw_fqns
    assert plan.qk_clip_plan is None
    for fqn in (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
    ):
        assert plan.parameter_specs[fqn].compute_placement == Owned()


def test_muon_preserves_fused_expert_module_layouts():
    model = TurboBailingMoeV3VLForConditionalGeneration(make_config(hybrid=True))
    model.model.layers[1].mlp.experts = GroupedMLP(
        num_local_experts=4,
        use_te=False,
        config=MoEConfig(
            hidden_size=16,
            moe_ffn_hidden_size=8,
            gated_linear_unit=True,
        ),
    )

    plan = _plan(model)

    prefix = "model.layers.1.mlp.experts"
    up = plan.parameter_specs[f"{prefix}.weight1"]
    down = plan.parameter_specs[f"{prefix}.weight2"]
    assert up.compute_placement == down.compute_placement == Shard(0)
    assert up.compute_view == ExpertUpComputeView(4, fused_gate_up=True)
    assert down.compute_view == ExpertDownComputeView(4)
    resolved = up.compute_view.resolve(
        model.get_parameter(f"{prefix}.weight1").shape
    )
    assert resolved.matrix_shape == (16, 8)
    assert resolved.column_groups == 2


def test_other_optimizer_recipes_still_raise():
    model = TurboBailingMoeV3VLForConditionalGeneration(make_config())

    with pytest.raises(NotImplementedError, match="'adamw'"):
        model.get_optimizer_recipe("adamw")
