# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Modified by the TorchTurbo contributors.

"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement."""

from __future__ import annotations

import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import VirtualExpertPlan

from .moe_config import MoEConfig


def plan_virtual_expert_routes(
    top_indices: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> VirtualExpertPlan:
    """Plan deterministic virtual-expert placement for one EP group and map this rank's routes."""
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Virtual-expert planner takes int32/int64 [num_tokens, topk] expert ids."
        )
    return ops.plan_virtual_expert_routes(
        top_indices.contiguous(), num_experts, ep_group,
    )


class VirtualExpertLoadBalancer:
    """Mixin that plans virtual-expert routes and coordinates replica execution."""

    def initialize_virtual_expert_load_balancer(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: MoEConfig,
    ) -> None:
        """Initialize virtual-expert state without initializing the transport parent."""
        self.ep_size = dist.get_world_size(group) if group is not None else 1
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.num_owned_experts = num_local_experts
        self._plan: VirtualExpertPlan | None = None

    @property
    def num_runtime_experts(self) -> int:
        """Natives plus virtual-expert slots per rank."""
        return 2 * self.num_owned_experts

    def plan_dispatch(self, top_indices: torch.Tensor) -> None:
        """Plan global virtual-expert placement and remap local routes."""
        self._plan = plan_virtual_expert_routes(
            top_indices, self.ep_size * self.num_owned_experts, self.group,
        )

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states and the planned [num_tokens, topk] runtime expert IDs."""
        virtual_experts, self._plan.virtual_experts = (
            self._plan.virtual_experts,
            None,
        )
        return hidden_states, virtual_experts

    def prepare_virtual_expert_combine(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Hook before transport combine."""
        return hidden_states
