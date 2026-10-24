"""FSDP2 parallel plan for Kimi-K3.

Activation-checkpoint targets on the decoder layers, ``Shard(0)`` on the
``edp`` mesh dim for experts, standalone world shards for embed/lm_head,
and ``replace_module`` for the MoE block and Ulysses attention.
"""
from functools import partial
from typing import Any

from ...parallel import FSDPPlanSpec, FSDPParallelPlan, FSDPPlacementSpec
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.attention import ulysses
from torch.distributed._tensor import Shard

# K3 LatentMoE expert (fused grouped_matmul in latent space + down/up proj,
# reusing torchturbo EP/token_dispatcher — no framework change).
from .model import K3LatentGroupedMLP


def _is_kimi_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_kimi_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
):
    if not _is_kimi_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    language_config = getattr(hf_config, "text_config", hf_config)

    num_experts = getattr(language_config, "num_experts", 896)
    scoring_func = getattr(language_config, "moe_router_activation_func", "sigmoid")
    routed_scaling_factor = getattr(language_config, "routed_scaling_factor", 1.0)

    # K3 Stable LatentMoE: expert runs in routed_expert_hidden_size space (3584),
    # not hidden_size (7168). Build fused expert (K3LatentGroupedMLP) with the
    # latent dim so w1/w2 match weights [3072, 3584]; down/up proj convert
    # hidden<->latent around the expert. Falls back to plain GroupedMLP
    # (hidden_size) if not LatentMoE (matches K2.6/bailing behavior).
    raw_hidden_size = language_config.hidden_size
    moe_hidden_size = getattr(language_config, "routed_expert_hidden_size", None)
    use_latent_moe = moe_hidden_size is not None
    if not use_latent_moe:
        moe_hidden_size = raw_hidden_size
    latent_moe_use_norm = getattr(language_config, "latent_moe_use_norm", False)
    rms_norm_eps = getattr(language_config, "rms_norm_eps", 1e-6)

    if use_latent_moe:
        # Pass latent params via ModuleSpec.params; build_module merges them
        # with MoELayer's (num_local_experts, use_te, config, layer_number).
        expert_spec = ModuleSpec(
            module=K3LatentGroupedMLP,
            params={
                "moe_hidden_size": moe_hidden_size,
                "raw_hidden_size": raw_hidden_size,
                "latent_moe_use_norm": latent_moe_use_norm,
                "rms_norm_eps": rms_norm_eps,
            },
        )
        moe_submodules = MoESubmodules(experts=expert_spec)

    kimi_k3_config_kwargs = get_base_moe_config(
        hf_config=language_config,
        ep_size=ep_size,
        num_moe_experts=num_experts,
        moe_router_num_groups=getattr(language_config, "num_expert_group", 1),
        moe_router_group_topk=getattr(language_config, "topk_group", 1),
        moe_router_topk_scaling_factor=routed_scaling_factor,
        moe_router_score_function=scoring_func,
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**kimi_k3_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    # K3 LatentMoE: add routed_expert_down/up_proj/norm on the MoE layer (not
    # inside experts — weight_loader treats experts.* as expert weights and
    # skips plain Linear, leaving them meta). These live at block_sparse_moe
    # level, matching the official weight keys. Forward wrapping via hooks so
    # MoELayer.forward stays unchanged (EP/token_dispatcher untouched).
    if use_latent_moe:
        import torch.nn as nn
        moe_layer.routed_expert_down_proj = nn.Linear(
            raw_hidden_size, moe_hidden_size, bias=False
        )
        moe_layer.routed_expert_up_proj = nn.Linear(
            moe_hidden_size, raw_hidden_size, bias=False
        )
        if latent_moe_use_norm:
            from .modeling_kimi_linear import KimiRMSNorm
            moe_layer.routed_expert_norm = KimiRMSNorm(moe_hidden_size, eps=rms_norm_eps)

        # Hook experts: pre -> down_proj (hidden->latent); post -> norm+up_proj
        def _latent_pre_hook(_module, args):
            (dispatched_input, *rest) = args
            return (moe_layer.routed_expert_down_proj(dispatched_input), *rest)

        def _latent_post_hook(_module, _input, output):
            expert_output, mlp_bias = output
            if latent_moe_use_norm:
                expert_output = moe_layer.routed_expert_norm(expert_output)
            expert_output = moe_layer.routed_expert_up_proj(expert_output)
            return (expert_output, mlp_bias)

        moe_layer.experts.register_forward_pre_hook(_latent_pre_hook)
        moe_layer.experts.register_forward_hook(_latent_post_hook)

    return moe_layer


def _replace_self_attn(original_module, hf_config, sp_info, ep_info, **kwargs):
    if hasattr(original_module, "_flash_attention_forward"):
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        patched_forward = partial(
            ulysses.ulysses_flash_attention_forward,
            sp_group=sp_group,
            sp_size=sp_size,
        )
        original_module._flash_attention_forward = patched_forward
    return original_module


def get_parallel_plan(**kwargs) -> FSDPParallelPlan:
    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            # 1. All layers: AC target (applied when --gradient_checkpointing on)
            "language_model.model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            # 2. Expert weights: edp sharding. K3's MoE block is
            # `block_sparse_moe` (not `mlp` like K2.5/bailing); targeting
            # `mlp` would skip the MoE replace and leave the official
            # KimiSparseMoeBlock running, whose gate asserts
            # `not self.training`.
            "language_model.model.layers.*.block_sparse_moe.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            # 3. MoE module: replace with framework MoELayer
            "language_model.model.layers.*.block_sparse_moe": FSDPPlanSpec(
                replace_module=_replace_kimi_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            # 4. Attention: replace with Ulysses SP
            "language_model.model.layers.*.self_attn": FSDPPlanSpec(
                replace_module=_replace_self_attn,
            ),
            # 5. embed_tokens: standalone world shard
            "language_model.model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),

            # 6. lm_head: standalone world shard
            "language_model.lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
