"""Stable route positions and logical expert counts."""

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_route_metadata",
    impl_name="torch_moe_route_metadata",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoERouteMetadata(OpImpl):
    def __call__(
        self, indices: torch.Tensor, num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = indices.flatten()
        order = flat.argsort(stable=True)
        positions = torch.empty_like(order, dtype=torch.int32).scatter_(
            0, order, torch.arange(flat.numel(), device=indices.device, dtype=torch.int32),
        )
        counts = torch.bincount(flat, minlength=num_experts)
        return positions.reshape_as(indices), counts
