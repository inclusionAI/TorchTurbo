"""Fused stable route metadata without moving differentiable activations."""

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_route_metadata",
    impl_name="npu_moe_route_metadata",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoERouteMetadata(OpImpl):
    def available(self) -> bool:
        try:
            from torch_npu import npu_moe_init_routing_v2  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self, indices: torch.Tensor, num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch_npu import npu_moe_init_routing_v2

        if indices.numel() == 0:
            return (
                torch.empty_like(indices, dtype=torch.int32),
                indices.new_zeros(num_experts, dtype=torch.int64),
            )
        # The routing API has no native backward. Only discrete metadata and a
        # one-column dummy payload enter it; activations keep their autograd path.
        dummy = torch.zeros((indices.shape[0], 1), device=indices.device, dtype=torch.float32)
        _, positions, counts, _ = npu_moe_init_routing_v2(
            dummy, indices.to(torch.int32).contiguous(),
            active_num=indices.numel(), expert_num=num_experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            quant_mode=-1, active_expert_range=[0, num_experts], row_idx_type=0,
        )
        return positions.reshape_as(indices), counts
