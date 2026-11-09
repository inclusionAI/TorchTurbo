"""Direct SHMEM replica push and owner gradient reduction on Ascend."""

from __future__ import annotations

from functools import cache

import torch
import torch.distributed as dist

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import VirtualExpertPlan
from ...registry import register_op


class _StandaloneStorage:
    """Reusable transport scratch for callers without model-managed storage."""

    def __init__(self, shape1, shape2, dtype, device, group):
        from torchturbo.runtime.npu import NpuRuntime

        runtime = NpuRuntime()
        numels = (shape1.numel(), shape2.numel())
        with torch.inference_mode(False):
            arena, weight_handle = runtime.allocate_communication_buffer(
                sum(numels), dtype=dtype, device=device, group=group,
            )
            self.grad_buffer, grad_handle = runtime.allocate_communication_buffer(
                sum(numels), dtype=dtype, device=device, group=group,
            )
        self.out = tuple(section.view(shape) for section, shape in zip(arena.split(numels), (shape1, shape2)))
        self.handles = (weight_handle, grad_handle)
        self.last_use = None

    def wait(self):
        if self.last_use is not None:
            torch.npu.current_stream().wait_event(self.last_use)

    def record_use(self):
        self.last_use = torch.npu.Event()
        self.last_use.record(torch.npu.current_stream())


@cache
def _standalone_storage(shape1, shape2, dtype, device, group):
    # The installed SHMEM allocator retains rendezvous allocations for the
    # process lifetime. Keep one pair of arenas per layout for that lifetime.
    return _StandaloneStorage(shape1, shape2, dtype, device, group)


class _DirectExpertWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight1, weight2, hidden, probs, context, prefetched, out, grad_buffer, handles, standalone):
        from ._kernels.virtual_expert_weights import push_weights

        count = weight1.shape[0]
        rank = dist.get_rank(context.ep_group) if context.experts_to_copy.shape[0] > 1 else 0
        copies = context.experts_to_copy.to(device=weight1.device)
        copies.record_stream(torch.npu.current_stream())
        ctx.count, ctx.rank, ctx.copies = count, rank, copies
        ctx.shapes = (weight1.shape, weight2.shape)
        ctx.dtype, ctx.device = weight1.dtype, weight1.device
        ctx.grad_buffer, ctx.handles = grad_buffer, handles
        ctx.standalone = standalone
        replicas = sum(context.receive_splits)
        if prefetched is None:
            if standalone is not None:
                standalone.wait()
            handle = handles[0]
            handle.peer_ptrs.record_stream(torch.npu.current_stream())
            # Every peer must finish reading the previous layer before writes.
            handle.barrier()
            push_weights(weight1.contiguous(), weight2.contiguous(), copies, handle.peer_ptrs, rank, count)
            handle.barrier()
            prefetched = tuple(weight[:replicas] for weight in out)
            if standalone is not None:
                # Standalone outputs may outlive later calls; only model hooks
                # restore shared slots before backward reads saved weights.
                prefetched = tuple(weight.clone() for weight in prefetched)
                standalone.record_use()
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
        from ._kernels.virtual_expert_weights import reduce_gradients

        grad1 = grad2 = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.standalone is not None:
                ctx.standalone.wait()
            elements1 = ctx.shapes[0][1:].numel()
            elements2 = ctx.shapes[1][1:].numel()
            slots1, slots2 = ctx.grad_buffer.split((ctx.count * elements1, ctx.count * elements2))
            slots1[:replica_grad1.numel()].copy_(replica_grad1.reshape(-1))
            slots2[:replica_grad2.numel()].copy_(replica_grad2.reshape(-1))
            handle = ctx.handles[1]
            handle.peer_ptrs.record_stream(torch.npu.current_stream())
            ctx.copies.record_stream(torch.npu.current_stream())
            # Owners read only after every peer has published its gradients.
            handle.barrier()
            grad1 = torch.empty(ctx.shapes[0], dtype=ctx.grad_buffer.dtype, device=ctx.device)
            grad2 = torch.empty(ctx.shapes[1], dtype=ctx.grad_buffer.dtype, device=ctx.device)
            reduce_gradients(grad1, grad2, ctx.copies, handle.peer_ptrs, ctx.rank, ctx.count)
            # Reuse the gradient arena only after every owner has read it.
            handle.barrier()
            if ctx.standalone is not None:
                ctx.standalone.record_use()
        return (
            grad1 if ctx.needs_input_grad[0] else None,
            grad2 if ctx.needs_input_grad[1] else None,
            hidden_grad, probs_grad, None, None, None, None, None, None,
        )


@register_op(
    op_name="virtual_expert_weight_exchange",
    impl_name="npu_shmem_virtual_expert_weight_exchange",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuShmemVirtualExpertWeightExchange(OpImpl):
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
        if isinstance(context, VirtualExpertPlan):
            if not torch.is_grad_enabled() and buffer_handles is not None and prefetched is None:
                from ._kernels.virtual_expert_weights import push_weights

                copies = context.experts_to_copy.to(device=weight1.device)
                rank = dist.get_rank(context.ep_group) if copies.shape[0] > 1 else 0
                copies.record_stream(torch.npu.current_stream())
                handle = buffer_handles[0]
                handle.peer_ptrs.record_stream(torch.npu.current_stream())
                handle.barrier()
                push_weights(
                    weight1.contiguous(), weight2.contiguous(), copies,
                    handle.peer_ptrs, rank, weight1.shape[0],
                )
                handle.barrier()
                # The number of active slots is host metadata. Consumers slice
                # these fixed slots after token dispatch has finished the plan.
                return *out, hidden_states, probs
            context = context.wait().weight_context
        standalone = None
        if buffer_handles is None:
            if out is not None or grad_buffer is not None:
                raise ValueError("Direct SHMEM requires buffer_handles for caller-managed buffers.")
            # Initialize collectively even for an idle EP group, then reuse.
            standalone = _standalone_storage(
                weight1.shape, weight2.shape, weight1.dtype, weight1.device, context.ep_group,
            )
            out, grad_buffer, buffer_handles = standalone.out, standalone.grad_buffer, standalone.handles
        if not context.has_global_replicas:
            replicas = (weight1[:0], weight2[:0])
            if not torch.is_grad_enabled():
                replicas = tuple(weight.detach() for weight in replicas)
            return *replicas, hidden_states, probs
        if torch.is_grad_enabled() and (weight1.requires_grad or weight2.requires_grad):
            return _DirectExpertWeights.apply(
                weight1, weight2, hidden_states, probs, context, prefetched, out,
                grad_buffer, buffer_handles, standalone,
            )
        with torch.no_grad():
            replica1, replica2, _, _ = _DirectExpertWeights.apply(
                weight1, weight2, hidden_states, probs, context, prefetched, out,
                grad_buffer, buffer_handles, standalone,
            )
        return replica1, replica2, hidden_states, probs
