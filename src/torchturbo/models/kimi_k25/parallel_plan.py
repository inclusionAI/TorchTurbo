from sys import modules
from typing import Any
from functools import partial
from torch import Tensor, nn
import torch.distributed as dist

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.attention import ulysses
from ..modules.attention.qk_clip import get_qk_clip_state


def _is_kimi_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_kimi_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules,
    engine_config, **kwargs
):
    if not _is_kimi_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    language_config = hf_config.text_config
    kimi_k25_config_kwargs = get_base_moe_config(
        hf_config=language_config,
        ep_size=ep_size,
        num_moe_experts=language_config.n_routed_experts,
        moe_aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        moe_z_loss_coeff=engine_config.moe.z_loss_coeff,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
        moe_router_num_groups=language_config.n_group,
        moe_router_group_topk=language_config.topk_group,
        moe_router_topk_scaling_factor=language_config.routed_scaling_factor,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**kimi_k25_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        sp_group=sp_info["sp_group"],
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def _replace_self_attn(
    original_module: nn.Module,
    hf_config: Any,
    sp_info: dict[str, Any],
    ep_info: Any,
    **kwargs: Any,
) -> nn.Module:
    # Standard SP attention injection for old-style modeling: HF's legacy
    # attention classes pick their implementation at __init__ time
    # (ATTENTION_CLASSES[config._attn_implementation]) and never consult
    # ALL_ATTENTION_FUNCTIONS at forward time, so the registered
    # "turbo_ulysses" interface used by the qwen3 families cannot reach
    # them. Instead, bind the Ulysses forward onto the instance here in the
    # parallel plan: the class forward calls self._flash_attention_forward,
    # and an instance attribute shadows that method. The injection stays
    # per-module (vision towers are untouched by construction) with no
    # process-global state.
    from .modeling_deepseek import DeepseekV3FlashAttention2

    if isinstance(original_module, DeepseekV3FlashAttention2):
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        patched_forward = partial(
            ulysses.ulysses_flash_attention_forward, sp_group=sp_group, sp_size=sp_size
        )

        def attention_forward(*args: Any, **kwargs: Any) -> Tensor:
            return patched_forward(
                *args,
                qk_clip_state=(
                    get_qk_clip_state(original_module)
                    if original_module.training else None
                ),
                **kwargs,
            )

        original_module._flash_attention_forward = attention_forward
    return original_module


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "language_model.model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "language_model.model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "language_model.model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_kimi_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "language_model.model.layers.*.self_attn": FSDPPlanSpec(
                replace_module=_replace_self_attn,
            ),
            # "vision_tower": FSDPPlanSpec(
            #     no_shard=True,
            # ),
            "language_model.model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "language_model.lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
