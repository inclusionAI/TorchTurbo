from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchturbo.ops import ops


CHUNK_ROWS = 16384


def _chunks(counts: tuple[int, ...], chunk_rows: int):
    ends = tuple(accumulate(counts))
    starts = (0, *ends[:-1])
    for start in range(0, ends[-1], chunk_rows):
        end = min(start + chunk_rows, ends[-1])
        first = bisect_right(ends, start)
        last = bisect_left(ends, end) + 1
        sizes = torch.tensor(
            [min(end, ends[i]) - max(start, starts[i]) for i in range(first, last)],
            dtype=torch.int64,
            device="cpu",
        )
        yield start, end, first, last, sizes


def _compute(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    counts: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    hidden = ops.grouped_matmul(x, weight1, counts, trans_b=False)
    hidden = ops.gated_activation(hidden, activation_func=F.silu, limit=limit)
    return ops.grouped_matmul(hidden, weight2, counts, trans_b=False)


class _ChunkedExpertMLP(torch.autograd.Function):
    """Recompute expert MLP chunks with first-order gradients only."""

    @staticmethod
    def forward(ctx, x, weight1, weight2, counts, limit, chunk_rows):
        ctx.save_for_backward(x, weight1, weight2)
        ctx.counts = tuple(counts.tolist())
        ctx.limit = limit
        ctx.chunk_rows = chunk_rows
        output = x.new_empty((x.shape[0], weight2.shape[-1]))
        for start, end, first, last, sizes in _chunks(ctx.counts, chunk_rows):
            output[start:end] = _compute(
                x[start:end], weight1[first:last], weight2[first:last], sizes, limit
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight1, weight2 = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        grad_x = torch.empty_like(x) if needs[0] else None
        # Accumulate across token chunks before casting to parameter dtype.
        grad_w1 = torch.zeros_like(weight1, dtype=torch.float32) if needs[1] else None
        grad_w2 = torch.zeros_like(weight2, dtype=torch.float32) if needs[2] else None
        for start, end, first, last, sizes in _chunks(ctx.counts, ctx.chunk_rows):
            with torch.enable_grad():
                inputs = (
                    x[start:end].detach().requires_grad_(needs[0]),
                    weight1[first:last].detach().requires_grad_(needs[1]),
                    weight2[first:last].detach().requires_grad_(needs[2]),
                )
                output = _compute(*inputs, sizes, ctx.limit)
                gradients = iter(torch.autograd.grad(
                    output,
                    tuple(value for value, need in zip(inputs, needs) if need),
                    grad_output[start:end],
                ))
            if needs[0]:
                grad_x[start:end] = next(gradients)
            if needs[1]:
                grad_w1[first:last].add_(next(gradients))
            if needs[2]:
                grad_w2[first:last].add_(next(gradients))
            del output, inputs, gradients
        return (
            grad_x,
            grad_w1.to(weight1.dtype) if needs[1] else None,
            grad_w2.to(weight2.dtype) if needs[2] else None,
            None,
            None,
            None,
        )


def chunked_expert_mlp(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    counts: torch.Tensor,
    *,
    limit: float,
    chunk_rows: int = CHUNK_ROWS,
) -> torch.Tensor:
    """Compute clipped SwiGLU experts without retaining full hidden activations.

    Args:
        x: Tokens sorted by expert, shaped [T, H].
        weight1: First projection weights, shaped [E, H, 2I].
        weight2: Second projection weights, shaped [E, I, H].
        counts: Number of tokens per expert, shaped [E].
        limit: Positive clipping limit for the SwiGLU gate and up projection.
        chunk_rows: Maximum number of token rows per recomputed chunk.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    return _ChunkedExpertMLP.apply(x, weight1, weight2, counts, limit, chunk_rows)
