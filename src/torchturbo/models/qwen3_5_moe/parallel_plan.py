from typing import Any
import torch.nn.functional as F

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


def _is_qwen3_5_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_qwen3_5_moe_only(
    original_module,
    hf_config,
    sp_info,
    ep_info,
    moe_submodules,
    engine_config,
    **kwargs,
):
    if not _is_qwen3_5_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    moe_config_kwargs = get_base_moe_config(
        hf_config=text_config,
        ep_size=ep_size,
        num_moe_experts=text_config.num_experts,
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
    )
    moe_config = MoEConfig(**moe_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        sp_group=sp_info["sp_group"],
    )

    if hasattr(original_module, "shared_expert"):
        moe_layer.shared_expert = original_module.shared_expert
    if hasattr(original_module, "shared_expert_gate"):
        moe_layer.shared_expert_gate = original_module.shared_expert_gate

    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.language_model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.visual.blocks.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.language_model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.language_model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.visual": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "model.language_model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "mtp.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            # mtp.layers is an nn.ModuleList, so the default layer-wrap policy
            # wraps each Qwen3_5MoeMTPLayer as a standalone world-mesh unit
            # automatically; this entry only needs to request activation
            # checkpointing (the experts inside are carved onto edp above).
            "mtp.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "mtp.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
        },
        sp_size=1,
        ep_size=1,
    )
