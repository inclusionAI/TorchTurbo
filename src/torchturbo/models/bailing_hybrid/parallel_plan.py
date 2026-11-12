from typing import Any

import torch

from ...parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules


def _is_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _dtype_from_config(hf_config):
    dtype = getattr(hf_config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def _replace_moe_only(
    original_module,
    hf_config,
    sp_info,
    ep_info,
    moe_submodules,
    engine_config,
    **kwargs,
):
    if not _is_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    moe_config_kwargs = get_base_moe_config(
        hf_config=hf_config,
        ep_size=ep_size,
        num_moe_experts=hf_config.num_experts,
        params_dtype=_dtype_from_config(hf_config),
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_dtype=hf_config.router_dtype,
        moe_router_enable_expert_bias=True,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
        moe_swiglu_limit=getattr(hf_config, "expert_swiglu_limit_list", None),
        # Match BailingMoeV3MLP, which computes min(silu(gate), limit).
        moe_swiglu_clamp_gate_after_activation=True,
    )
    moe_config = MoEConfig(**moe_config_kwargs)
    layer_number = getattr(original_module, "layer_idx", None)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        layer_number=layer_number,
        sp_group=sp_info["sp_group"],
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ..modules.moe.experts import GroupedMLP
    from ..modules.moe.module import ModuleSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
