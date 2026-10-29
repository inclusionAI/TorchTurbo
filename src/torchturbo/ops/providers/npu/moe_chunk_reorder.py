from __future__ import annotations

import torch
import torch_npu

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


class _MoeChunkReorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, chunk_counts, probs):
        counts = chunk_counts.to(device=tokens.device, non_blocking=True).contiguous()
        if probs is None:
            ranks, experts = counts.shape
            destinations = torch.arange(
                counts.numel(), dtype=torch.int32, device=tokens.device,
            ).view(experts, ranks).T.flatten()
            indices = torch.repeat_interleave(
                destinations, counts.flatten(), output_size=tokens.shape[0],
            ).unsqueeze(1)
            output, inverse_indices = torch_npu.npu_moe_token_permute(tokens.contiguous(), indices)
            scales = None
        else:
            output, scales, row_indices, _ = torch_npu.npu_moe_re_routing(
                tokens.contiguous(), counts, per_token_scales=probs.float().contiguous(),
            )
            if any(ctx.needs_input_grad):
                positions = torch.arange(
                    row_indices.numel(), dtype=row_indices.dtype, device=row_indices.device,
                )
                inverse_indices = torch_npu.npu_scatter_nd_update_(
                    torch.empty_like(row_indices), row_indices.unsqueeze(1), positions,
                )
        if any(ctx.needs_input_grad):
            # Each invocation owns its row map, including interleaved microbatches.
            ctx.save_for_backward(inverse_indices)
        ctx.probs_dtype = None if probs is None else probs.dtype
        ctx.set_materialize_grads(False)
        return output, None if probs is None else scales.to(probs.dtype)

    @staticmethod
    def backward(ctx, grad_output, grad_probs):
        if grad_output is None and grad_probs is None:
            return None, None, None
        (inverse_indices,) = ctx.saved_tensors
        grad_tokens = grad_scales = None
        if grad_output is not None and ctx.needs_input_grad[0]:
            grad_tokens = grad_output.index_select(0, inverse_indices)
        if grad_probs is not None and ctx.needs_input_grad[2]:
            grad_scales = grad_probs.index_select(0, inverse_indices).to(ctx.probs_dtype)
        return grad_tokens, None, grad_scales


@register_op(
    op_name="moe_chunk_reorder",
    impl_name="npu_moe_re_routing",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoeChunkReorder(OpImpl):
    def __call__(
        self,
        tokens: torch.Tensor,
        chunk_counts: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transpose rank/expert chunks using the device routing copy operator."""
        if tokens.shape[0] == 0:
            return tokens, probs
        return _MoeChunkReorder.apply(tokens, chunk_counts, probs)
