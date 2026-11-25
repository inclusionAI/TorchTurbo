"""Return one FC's replica gradients to their original expert owners."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...moe import ep_all_to_all
from ...registry import register_op


@register_op(
    op_name="virtual_expert_grad_reduce",
    impl_name="torch_virtual_expert_grad_reduce",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchVirtualExpertGradReduce(OpImpl):
    def __call__(
        self,
        native_grad: torch.Tensor,
        replica_grad: torch.Tensor,
        context: object,
        *,
        grad_buffer: torch.Tensor,
        arena_offset: int = 0,
        buffer_handle: object | None = None,
        allow_inplace: bool = False,
    ) -> torch.Tensor:
        """Add returned replica gradients, reusing an owned accumulator when possible."""
        if not context.has_global_replicas:
            return native_grad if allow_inplace else native_grad.clone()
        packet = grad_buffer[arena_offset:arena_offset + replica_grad.numel()].view(
            replica_grad.shape,
        )
        packet.copy_(replica_grad)
        returned = ep_all_to_all(
            context.ep_group, packet, context.send_splits, context.receive_splits,
        )
        accum_dtype = torch.promote_types(native_grad.dtype, torch.float32)
        result = native_grad.to(dtype=accum_dtype, copy=not allow_inplace)
        indices = context.send_indices.to(device=result.device)
        result.index_add_(0, indices, returned.to(accum_dtype))
        return result.to(native_grad.dtype)
