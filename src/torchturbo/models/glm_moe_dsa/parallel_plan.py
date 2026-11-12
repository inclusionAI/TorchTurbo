from typing import Any

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


def _is_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules,
    engine_config, **kwargs
):
    if not _is_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    glm5_config_kwargs = get_base_moe_config(
        hf_config=hf_config,
        ep_size=ep_size,
        num_moe_experts=hf_config.n_routed_experts,
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**glm5_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        sp_group=sp_info["sp_group"],
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
