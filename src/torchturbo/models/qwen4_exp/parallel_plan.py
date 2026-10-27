from __future__ import annotations

from typing import Any

from torch.distributed._tensor import Shard

from torchturbo.models.modules.moe.config_converter import get_base_moe_config
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules
from torchturbo.parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec

from .modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock


def _replace_qwen4_moe(
    original_module: Any,
    hf_config: Any,
    sp_info: dict[str, Any],
    ep_info: dict[str, Any],
    moe_submodules: MoESubmodules,
    engine_config: Any,
    **kwargs: Any,
) -> Any:
    """Replace a Qwen4 sparse MoE block with TorchTurbo's EP implementation."""

    del kwargs
    if not isinstance(original_module, Qwen4ExpTextSparseMoeBlock):
        return original_module

    text_config = hf_config.text_config
    moe_config_kwargs = get_base_moe_config(
        hf_config=text_config,
        ep_size=ep_info["ep_size"],
        num_moe_experts=text_config.num_experts,
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
    )
    moe_layer = MoELayer(
        config=MoEConfig(**moe_config_kwargs),
        ep_group=ep_info["ep_group"],
        submodules=moe_submodules,
        sp_group=sp_info["sp_group"],
    )
    moe_layer.shared_expert = original_module.shared_expert
    moe_layer.shared_expert_gate = original_module.shared_expert_gate
    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    """Return the initial FSDP and expert-parallel plan for Qwen4."""

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    return FSDPParallelPlan(
        plan={
            "model.language_model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.language_model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp")
            ),
            "model.language_model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen4_moe,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.visual": FSDPPlanSpec(standalone_world_shard=True),
            "model.language_model.embed_tokens": FSDPPlanSpec(standalone_world_shard=True),
            "lm_head": FSDPPlanSpec(standalone_world_shard=True),
        },
        sp_size=1,
        ep_size=1,
    )
