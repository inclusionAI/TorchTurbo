"""FSDP2/EP placement for the language and vision towers."""

import torch
from ...parallel import FSDPPlanSpec, FSDPParallelPlan, FSDPPlacementSpec
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.module import ModuleSpec
from ..modules.moe.router import TopKRouter
from .modeling_glm5_3_flash import Glm5NextTextMoE


class Glm5NextGroupedRouter(TopKRouter):
    def routing(self, logits, input_ids=None):
        # GLM uses top-2 for every group and normalizes even when top_k=1.
        # These differ from the generic router's group/top-k conventions.
        scores = logits.reshape(-1, self.num_experts).float().sigmoid()
        choice = scores + self.expert_bias
        groups = self.config.moe_router_num_groups
        group_k = self.config.moe_router_group_topk
        group_scores = (
            choice.view(-1, groups, self.num_experts // groups)
            .topk(2, dim=-1)
            .values.sum(-1)
        )
        selected_groups = group_scores.topk(group_k, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter(
            1, selected_groups, True
        )
        mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, groups, self.num_experts // groups)
            .reshape_as(choice)
        )
        selected = (
            choice.masked_fill(~mask, float("-inf"))
            .topk(self.topk, dim=-1, sorted=False)
            .indices
        )
        weights = scores.gather(1, selected)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.config.moe_router_topk_scaling_factor
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(
            1, selected, True
        )
        probabilities = torch.zeros_like(scores).scatter(1, selected, weights)
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                counts = routing_map.sum(0).float()
                if not hasattr(self, "local_tokens_per_expert"):
                    self.register_buffer(
                        "local_tokens_per_expert",
                        torch.zeros_like(counts),
                        persistent=False,
                    )
                self.local_tokens_per_expert.add_(counts)
        return probabilities, routing_map


def _replace_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
):
    if not isinstance(original_module, Glm5NextTextMoE):
        return original_module
    config = hf_config.text_config
    moe_config = MoEConfig(
        **get_base_moe_config(
            config,
            ep_info["ep_size"],
            num_moe_experts=config.n_routed_experts,
            moe_router_num_groups=config.n_group,
            moe_router_group_topk=config.topk_group,
            moe_router_topk_scaling_factor=config.routed_scaling_factor,
            moe_router_score_function=getattr(config, "scoring_func", "sigmoid"),
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_swiglu_limit=[config.swiglu_limit] * config.num_hidden_layers,
        )
    )
    result = MoELayer(
        moe_config,
        ep_info["ep_group"],
        submodules=moe_submodules,
        layer_number=original_module.layer_idx,
    )
    result.router = Glm5NextGroupedRouter(moe_config, original_module.layer_idx)
    result.shared_experts = original_module.shared_experts
    return result


def get_parallel_plan(sp_size=1, ep_size=1):
    from torch.distributed._tensor import Shard

    experts = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    return FSDPParallelPlan(
        plan={
            "model.language_model.layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.visual.blocks.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.language_model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp")
            ),
            "model.language_model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": experts},
            ),
            "model.language_model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True
            ),
            "model.visual": FSDPPlanSpec(standalone_world_shard=True),
            "lm_head": FSDPPlanSpec(standalone_world_shard=True),
            **{
                f"**.hc_{kind}_{part}": FSDPPlanSpec(no_shard=True)
                for kind in ("attn", "ffn")
                for part in ("fn", "base", "scale")
            },
        },
        sp_size=sp_size,
        ep_size=ep_size,
    )
