"""Install the fused MoonEP and MegaMoE backend inside expert FSDP hooks."""
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...models.modules.moe.experts import GroupedMLP
from ...models.modules.moe.moe_layer import MoELayer


def configure_token_bound(config: Any, args: Any) -> None:
    """Derive a fixed per-rank token capacity from Swift's training arguments."""
    length = args.packing_length if args.packing else args.max_length
    if length is None:
        length = args.max_length
    if length is None or length <= 0:
        raise ValueError("mega_moe requires a positive packing_length or max_length.")
    tokens = length * args.per_device_train_batch_size
    sp_size = config.parallel_plan.sp_size
    local_tokens = (tokens + sp_size - 1) // sp_size
    required = ((local_tokens + 127) // 128) * 128
    capacity = config.extra.get("mega_moe_local_tokens", required)
    if not isinstance(capacity, int) or capacity < required or capacity % 128:
        raise ValueError(
            "mega_moe_local_tokens must be 128-aligned and at least "
            f"{required} for the configured sequence length, batch size and SP size."
        )
    config.extra["mega_moe_local_tokens"] = capacity


class RoutedExpertsBackend:
    """Adapt the existing router mask without changing expert parameter ownership."""

    def __init__(self, ep_group: dist.ProcessGroup, token_bound: int) -> None:
        self.ep_group = ep_group
        self.token_bound = token_bound

    def route(
        self,
        experts: GroupedMLP,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple:
        """Gather selected probabilities, including selected zero-weight experts."""
        num_experts = routing_map.shape[-1]
        keys = torch.arange(num_experts, device=routing_map.device, dtype=torch.float32)
        keys = torch.where(routing_map, keys, float(num_experts))
        indices = keys.topk(
            experts.config.moe_router_topk, dim=-1, largest=False, sorted=True
        ).values.long()
        # Match the dispatcher probability cast at the expert-output boundary.
        selected_probs = probs.gather(1, indices).to(hidden_states.dtype)
        return experts(hidden_states, indices, selected_probs)


def install(model: nn.Module, config: Any, ep_group: dist.ProcessGroup) -> None:
    """Attach one shared MoonEP and MegaMoE backend before FSDP wrapping."""
    token_bound = config.extra.get("mega_moe_local_tokens")
    if not isinstance(token_bound, int) or token_bound <= 0 or token_bound % 128:
        raise ValueError(
            "mega_moe needs a positive 128-aligned per-rank token capacity. "
            "Swift derives it automatically; engine callers set "
            "config.extra['mega_moe_local_tokens'] before building the model."
        )
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError("mega_moe requires a model using TorchTurbo MoELayer.")
    for layer in layers:
        experts = layer.experts
        if not isinstance(experts, GroupedMLP) or experts.use_te:
            raise ValueError("mega_moe requires TorchTurbo GroupedMLP experts without TE.")
        if layer.config.activation_func is not F.silu or not layer.config.gated_linear_unit:
            raise ValueError("mega_moe requires SwiGLU experts.")
        if layer.config.moe_apply_probs_on_input:
            raise ValueError("mega_moe requires routing weights on expert outputs.")
    geometries = {
        (
            layer.config.hidden_size,
            layer.config.moe_ffn_hidden_size,
            layer.config.num_moe_experts,
            layer.config.moe_router_topk,
        )
        for layer in layers
    }
    if len(geometries) != 1:
        raise ValueError("mega_moe shared workspaces require matching MoE layer geometry.")
    num_experts = layers[0].config.num_moe_experts
    ranks = dist.get_world_size(ep_group)
    if num_experts % ranks:
        raise ValueError("mega_moe requires experts evenly divisible by EP size.")
    modes = set()
    for layer in layers:
        limit = float(layer.experts.swiglu_limit or 0.0)
        mode = (
            "post_activation"
            if layer.config.moe_swiglu_clamp_gate_after_activation
            else "pre_activation"
        ) if limit else "none"
        modes.add((mode, limit))
    from .moonep.backend import MoonEPMegaMoeBackend

    backend = MoonEPMegaMoeBackend(ep_group, token_bound, tuple(sorted(modes)))
    for layer in layers:
        layer._routed_experts_backend = backend
        layer.experts._execution_backend = backend
