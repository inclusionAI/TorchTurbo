"""Install the optional backend without changing expert parameter ownership."""
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...models.modules.moe.experts import GroupedMLP
from ...models.modules.moe.moe_layer import MoELayer
from .execution import execute_mega_moe


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
    config.extra["mega_moe_local_tokens"] = ((local_tokens + 127) // 128) * 128


class MegaMoeBackend:
    """Route tokens and execute inside the existing experts' FSDP boundary."""

    def __init__(self, ep_group: dist.ProcessGroup, token_bound: int) -> None:
        self.ep_group = ep_group
        self.token_bound = token_bound

    def route(self, experts: GroupedMLP, hidden_states: torch.Tensor,
              probs: torch.Tensor, routing_map: torch.Tensor) -> tuple:
        """Adapt the unchanged main router's dense assignment mask."""
        num_experts = routing_map.shape[-1]
        keys = torch.arange(num_experts, device=routing_map.device, dtype=torch.float32)
        keys = torch.where(routing_map, keys, float(num_experts))
        indices = keys.topk(experts.config.moe_router_topk, dim=-1,
                           largest=False, sorted=True).values.long()
        return experts(hidden_states, indices, probs.gather(1, indices))

    def execute(self, experts: GroupedMLP, hidden_states: torch.Tensor,
                indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Use the current unsharded parameters; never cache parameter views."""
        return execute_mega_moe(
            hidden_states, indices, probs, experts.weight1, experts.weight2,
            experts.config, self.ep_group, experts.swiglu_limit, self.token_bound,
        )


def install(model: nn.Module, config: Any, ep_group: dist.ProcessGroup) -> None:
    """Attach the backend before FSDP wrapping and checkpoint loading."""
    token_bound = config.extra.get("mega_moe_local_tokens")
    if not isinstance(token_bound, int) or token_bound <= 0 or token_bound % 128:
        raise ValueError("mega_moe needs a positive 128-aligned per-rank token capacity. "
                         "Swift derives it automatically; engine callers set "
                         "config.extra['mega_moe_local_tokens'] before building the model.")
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
        backend = MegaMoeBackend(ep_group, token_bound)
        layer._routed_experts_backend = backend
        experts._execution_backend = backend
