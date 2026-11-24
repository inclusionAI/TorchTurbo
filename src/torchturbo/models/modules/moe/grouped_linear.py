"""Grouped linear with optional external accumulation of weight gradients."""

from collections.abc import Callable

import torch

from torchturbo.ops import ops


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, counts, wgrad_counts, wgrad_store, *weights):
        ctx.save_for_backward(x, *weights)
        ctx.counts, ctx.wgrad_counts, ctx.wgrad_store = counts, wgrad_counts, wgrad_store
        weight = weights[0]
        if x.shape[0]:
            return ops.grouped_matmul(x, weight if len(weights) == 1 else weights, counts)
        dtype = x.dtype
        if torch.is_autocast_enabled(x.device.type) and torch.float64 not in (x.dtype, weight.dtype):
            dtype = torch.get_autocast_dtype(x.device.type)
        return x.new_empty((0, weight.shape[-1]), dtype=dtype)

    @staticmethod
    def backward(ctx, dy):
        x, *weights = ctx.saved_tensors
        dx = None
        # Recreate forward AMP operands here so only the original weights stay live.
        with torch.autocast(device_type=dy.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                if x.shape[0]:
                    banks = tuple(w.to(dy.dtype) for w in weights)
                    dx = ops.grouped_matmul(
                        dy, banks[0] if len(banks) == 1 else banks, ctx.counts, trans_b=True,
                    )
                else:
                    dx = torch.empty_like(x)
            if ctx.wgrad_store is not None:
                weight = weights[0]
                dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
                dw = weight.new_zeros((ctx.wgrad_counts.numel(), *weight.shape[1:]), dtype=dtype)
                ops.grouped_matmul_add(dw, x.to(dy.dtype), dy, ctx.wgrad_counts)
                ctx.wgrad_store(dw)
        return (dx, None, None, None) + (None,) * len(weights)


def grouped_linear(
    x: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    weight: torch.Tensor | tuple[torch.Tensor, ...],
    wgrad_store: Callable[[torch.Tensor], None] | None = None,
    wgrad_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply packed weight banks to contiguous groups of input rows.

    By default the selected matmul provider supplies autograd. An external
    wgrad store receives one packed gradient before dX is returned. Its counts
    may include empty groups omitted from the forward banks. A frozen weight
    with an external layout needs only dX and leaves the store unset.
    """
    banks = (weight,) if isinstance(weight, torch.Tensor) else weight
    if wgrad_store is None and (wgrad_counts is None or any(w.requires_grad for w in banks)):
        if x.shape[0] or isinstance(weight, tuple):
            return ops.grouped_matmul(x, weight, tokens_per_expert)
        # Keep matmul's autocast dtype and zero gradients for both operands.
        return x[:, :0] @ weight[:0].reshape(0, weight.shape[-1])
    if wgrad_counts is None:
        wgrad_counts = tokens_per_expert
    return _GroupedLinear.apply(x, tokens_per_expert, wgrad_counts, wgrad_store, *banks)
