"""Single-FC SHMEM replica gradient reduction for native autograd."""

from __future__ import annotations

import torch
import torch.distributed as dist

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="virtual_expert_grad_reduce",
    impl_name="npu_shmem_virtual_expert_grad_reduce",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuShmemVirtualExpertGradReduce(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
            import triton  # noqa: F401
            from torchturbo.runtime.npu import NpuRuntime

            NpuRuntime.shmem_library()
        except (ImportError, OSError):
            return False
        return True

    def __call__(
        self,
        native_grad: torch.Tensor,
        replica_grad: torch.Tensor,
        context: object,
        *,
        grad_buffer: torch.Tensor,
        arena_offset: int = 0,
        buffer_handle: object | None = None,
    ) -> torch.Tensor:
        """Publish this FC's slots, sum peer gradients, and add native gradients."""
        from ._kernels.virtual_expert_weights import reduce_fc_gradients

        if not context.has_global_replicas:
            return native_grad.clone()
        if buffer_handle is None:
            raise ValueError("Direct SHMEM gradient reduction requires a buffer_handle.")
        copies = context.experts_to_copy.to(device=native_grad.device)
        rank = dist.get_rank(context.ep_group) if copies.shape[0] > 1 else 0
        copies.record_stream(torch.npu.current_stream())
        buffer_handle.peer_ptrs.record_stream(torch.npu.current_stream())
        grad_buffer[arena_offset:arena_offset + replica_grad.numel()].copy_(
            replica_grad.reshape(-1),
        )
        buffer_handle.barrier()
        result = torch.empty_like(native_grad, memory_format=torch.contiguous_format)
        reduce_fc_gradients(result, copies, buffer_handle.peer_ptrs, rank, arena_offset)
        buffer_handle.barrier()
        return result.add_(native_grad)
