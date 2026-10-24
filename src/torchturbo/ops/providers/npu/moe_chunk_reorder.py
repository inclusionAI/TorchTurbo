from __future__ import annotations

import torch
import torch_npu

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


class _MoeChunkReorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, chunk_counts, probs):
        # Keep device counts per invocation so later micro-batches cannot replace them.
        counts = chunk_counts.to(device=tokens.device, non_blocking=True).contiguous()
        ctx.save_for_backward(counts)
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype
        ctx.probs_dtype = None if probs is None else probs.dtype
        ctx.set_materialize_grads(False)
        output, scales, _, _ = torch_npu.npu_moe_re_routing(
            tokens.contiguous(), counts,
            per_token_scales=None if probs is None else probs.float().contiguous(),
        )
        return output, None if probs is None else scales.to(probs.dtype)

    @staticmethod
    def backward(ctx, grad_output, grad_probs):
        (counts,) = ctx.saved_tensors
        if grad_output is None and grad_probs is None:
            return None, None, None
        # Transposing the count matrix describes the inverse block permutation.
        inverse_counts = counts.T.contiguous()
        if grad_output is None:
            grad_output = torch.zeros(
                ctx.tokens_shape, dtype=ctx.tokens_dtype, device=counts.device
            )
        grad_tokens, grad_scales, _, _ = torch_npu.npu_moe_re_routing(
            grad_output.contiguous(), inverse_counts,
            per_token_scales=None if grad_probs is None else grad_probs.float().contiguous(),
        )
        return (
            grad_tokens,
            None,
            None if grad_probs is None else grad_scales.to(ctx.probs_dtype),
        )


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
