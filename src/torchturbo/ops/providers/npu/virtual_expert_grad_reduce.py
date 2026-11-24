"""Single-FC SHMEM replica gradient reduction for native autograd."""

from __future__ import annotations

import torch
import torch.distributed as dist

from torchturbo.runtime import TurboDevice, get_device_runtime

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
    requires_peer_memory = True

    def available(self) -> bool:
        runtime = get_device_runtime()
        if not runtime.shmem.available():
            return False
        try:
            import triton  # noqa: F401
        except (ImportError, OSError, AttributeError):
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
        allow_inplace: bool = False,
    ) -> torch.Tensor:
        """Publish this FC's slots, sum peer gradients, and add native gradients."""
        from ._kernels.virtual_expert_weights import reduce_fc_gradients

        if not context.has_global_replicas:
            return native_grad if allow_inplace else native_grad.clone()
        if buffer_handle is None:
            raise ValueError("Direct SHMEM gradient reduction requires a buffer_handle.")
        copies = context.experts_to_copy.to(device=native_grad.device)
        rank = dist.get_rank(context.ep_group) if copies.shape[0] > 1 else 0
        copies.record_stream(torch.npu.current_stream())
        grad_buffer[arena_offset:arena_offset + replica_grad.numel()].copy_(
            replica_grad.reshape(-1),
        )
        result = (
            native_grad.contiguous() if allow_inplace
            else native_grad.clone(memory_format=torch.contiguous_format)
        )
        buffer_handle.barrier()
        reduce_fc_gradients(
            result, grad_buffer, copies, buffer_handle.buffer_ptrs_dev, rank, arena_offset,
            include_native=True, num_local_copies=context.send_indices.numel(),
        )
        buffer_handle.barrier()
        return result
