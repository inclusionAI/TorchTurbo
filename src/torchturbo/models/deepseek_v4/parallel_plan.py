from typing import Any

import torch

from ...parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules

_NO_SHARD_PARAM_NAMES = [
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
    "attn_sink",
]


def _is_moe_module(module: Any) -> bool:
    return getattr(module, "experts", None) is not None and getattr(module, "gate", None) is not None


def _dtype_from_config(hf_config):
    dtype = getattr(hf_config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype = dtype.removeprefix("torch.")
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float32)
    return torch.float32


def _replace_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules,
    engine_config, **kwargs
):
    if not _is_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0)
    num_mtp_layers = (
        hf_config.dspark_num_layers
        if getattr(hf_config, "dspark_block_size", 0) > 0
        else getattr(hf_config, "num_nextn_predict_layers", 0)
    )
    num_layers = getattr(hf_config, "num_hidden_layers", 0) + num_mtp_layers
    moe_swiglu_limit = [swiglu_limit] * num_layers if swiglu_limit else None
    dsv4_config_kwargs = get_base_moe_config(
        hf_config=hf_config,
        ep_size=ep_size,
        num_moe_experts=hf_config.n_routed_experts,
        params_dtype=_dtype_from_config(hf_config),
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_swiglu_limit=moe_swiglu_limit,
        moe_router_enable_expert_bias=True,
        num_hash_layers=getattr(hf_config, "num_hash_layers", 0),
        vocab_size=getattr(hf_config, "vocab_size", 0),
    )
    moe_config = MoEConfig(**dsv4_config_kwargs)
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
            "layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "layers.*.ffn.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "layers.*.ffn": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "mtp.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "mtp.*.ffn.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "mtp.*.ffn": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            **{name: FSDPPlanSpec(no_shard=True) for name in _NO_SHARD_PARAM_NAMES},
            **{
                f"**.{name}": FSDPPlanSpec(no_shard=True)
                for name in _NO_SHARD_PARAM_NAMES
            },
        },
        sp_size=1,
        ep_size=1,
    )
