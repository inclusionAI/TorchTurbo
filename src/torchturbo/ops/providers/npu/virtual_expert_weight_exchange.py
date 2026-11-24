"""Direct SHMEM replica push and owner gradient reduction on Ascend."""

from __future__ import annotations

import torch
import torch.distributed as dist

from torchturbo.runtime import TurboDevice, get_device_runtime

from ...base import OpImpl, OpImplKind
from ...moe import VirtualExpertPlan
from ...registry import register_op


class _DirectExpertWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight1, weight2, hidden, probs, context, prefetched, out, grad_buffer, handles):
        from ._kernels.virtual_expert_weights import push_weights

        count = weight1.shape[0]
        rank = dist.get_rank(context.ep_group) if context.experts_to_copy.shape[0] > 1 else 0
        copies = context.experts_to_copy.to(device=weight1.device)
        copies.record_stream(torch.npu.current_stream())
        ctx.count, ctx.rank, ctx.copies = count, rank, copies
        ctx.shapes = (weight1.shape, weight2.shape)
        ctx.device = weight1.device
        ctx.grad_buffer, ctx.handles = grad_buffer, handles
        replicas = sum(context.receive_splits)
        if prefetched is None:
            handle = handles[0]
            # Every peer must finish reading the previous layer before writes.
            handle.barrier()
            push_weights(weight1.contiguous(), weight2.contiguous(), copies, handle.buffer_ptrs_dev, rank, count)
            handle.barrier()
            prefetched = tuple(weight[:replicas] for weight in out)
        # Different graphs restore these slots before reading their saved
        # weights; they need independent autograd version counters.
        replica1, replica2 = (
            tensor.new_empty(0).set_(
                tensor.untyped_storage(), tensor.storage_offset(), tensor.size(), tensor.stride(),
            )
            for tensor in prefetched
        )
        return replica1, replica2, hidden, probs

    @staticmethod
    def backward(ctx, replica_grad1, replica_grad2, hidden_grad, probs_grad):
        from ._kernels.virtual_expert_weights import reduce_fc_gradients

        grad1 = grad2 = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            elements1 = ctx.shapes[0][1:].numel()
            elements2 = ctx.shapes[1][1:].numel()
            slots1, slots2 = ctx.grad_buffer.split((ctx.count * elements1, ctx.count * elements2))
            slots1[:replica_grad1.numel()].copy_(replica_grad1.reshape(-1))
            slots2[:replica_grad2.numel()].copy_(replica_grad2.reshape(-1))
            handle = ctx.handles[1]
            ctx.copies.record_stream(torch.npu.current_stream())
            # Owners read only after every peer has published its gradients.
            handle.barrier()
            grad1 = torch.empty(ctx.shapes[0], dtype=ctx.grad_buffer.dtype, device=ctx.device)
            grad2 = torch.empty(ctx.shapes[1], dtype=ctx.grad_buffer.dtype, device=ctx.device)
            for gradient, offset in ((grad1, 0), (grad2, grad1.numel())):
                reduce_fc_gradients(
                    gradient, ctx.grad_buffer, ctx.copies, handle.buffer_ptrs_dev, ctx.rank, offset,
                    include_native=False,
                )
            # Reuse the gradient arena only after every owner has read it.
            handle.barrier()
        return (
            grad1 if ctx.needs_input_grad[0] else None,
            grad2 if ctx.needs_input_grad[1] else None,
            hidden_grad, probs_grad, None, None, None, None, None,
        )


@register_op(
    op_name="virtual_expert_weight_exchange",
    impl_name="npu_shmem_virtual_expert_weight_exchange",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuShmemVirtualExpertWeightExchange(OpImpl):
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
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        hidden_states: torch.Tensor | None,
        probs: torch.Tensor | None,
        context: object,
        *,
        prefetched: tuple[torch.Tensor, torch.Tensor] | None = None,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
        grad_buffer: torch.Tensor | None = None,
        buffer_handles: tuple[object, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if weight1.dtype != weight2.dtype:
            raise ValueError("Direct SHMEM FC1 and FC2 weights must have the same dtype.")
        weight_handle, grad_handle = buffer_handles or (None, None)
        if prefetched is None and (out is None or weight_handle is None):
            raise ValueError("Direct SHMEM weight push requires out and its weight handle.")
        if isinstance(context, VirtualExpertPlan):
            if not torch.is_grad_enabled() and prefetched is None:
                from ._kernels.virtual_expert_weights import push_weights

                copies = context.experts_to_copy.to(device=weight1.device)
                rank = dist.get_rank(context.ep_group) if copies.shape[0] > 1 else 0
                copies.record_stream(torch.npu.current_stream())
                weight_handle.barrier()
                push_weights(
                    weight1.contiguous(), weight2.contiguous(), copies,
                    weight_handle.buffer_ptrs_dev, rank, weight1.shape[0],
                )
                weight_handle.barrier()
                # The number of active slots is host metadata. Consumers slice
                # these fixed slots after token dispatch has finished the plan.
                return *out, hidden_states, probs
            context = context.wait().weight_context
        if not context.has_global_replicas:
            replicas = (weight1[:0], weight2[:0])
            if not torch.is_grad_enabled():
                replicas = tuple(weight.detach() for weight in replicas)
            return *replicas, hidden_states, probs
        if torch.is_grad_enabled() and (weight1.requires_grad or weight2.requires_grad):
            if grad_buffer is None or grad_handle is None:
                raise ValueError("Direct SHMEM weight backward requires grad_buffer and its gradient handle.")
            return _DirectExpertWeights.apply(
                weight1, weight2, hidden_states, probs, context, prefetched, out,
                grad_buffer, buffer_handles,
            )
        with torch.no_grad():
            replica1, replica2, _, _ = _DirectExpertWeights.apply(
                weight1, weight2, hidden_states, probs, context, prefetched, out,
                grad_buffer, buffer_handles,
            )
        return replica1, replica2, hidden_states, probs
