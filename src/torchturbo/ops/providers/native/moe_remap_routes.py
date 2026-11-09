"""Exact occurrence-quota remapping and stable top-k sorting."""

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_remap_routes",
    impl_name="torch_moe_remap_routes",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoERemapRoutes(OpImpl):
    def __call__(
        self, indices: torch.Tensor, route_positions: torch.Tensor,
        remap_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts = remap_table.shape[0]
        num_ranks = (remap_table.shape[1] - 1) // 2
        num_local_experts = num_experts // num_ranks
        experts = indices.flatten().long()
        ordinals = route_positions.flatten().long() + remap_table[experts, 0]

        # Flatten disjoint expert ranges to avoid a routes-by-ranks temporary.
        totals = remap_table[:, num_ranks]
        starts = totals.cumsum(0) - totals
        boundaries = (
            remap_table[:, 1:1 + num_ranks] + starts[:, None]
        ).flatten()
        destinations = torch.searchsorted(
            boundaries, ordinals + starts[experts], right=True,
        ) % num_ranks
        slots = remap_table[experts, 1 + num_ranks + destinations]
        mapped = destinations * (2 * num_local_experts) + slots
        mapped = mapped.to(indices.dtype).reshape_as(indices)
        return mapped.sort(dim=1, stable=True)
