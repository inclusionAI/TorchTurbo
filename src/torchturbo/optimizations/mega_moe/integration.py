"""Install the fused MoonEP and MegaMoE backend inside expert FSDP hooks."""
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...models.modules.moe.experts import GroupedMLP
from ...models.modules.moe.moe_layer import MoELayer


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

