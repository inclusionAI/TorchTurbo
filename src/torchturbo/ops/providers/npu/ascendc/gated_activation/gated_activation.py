"""Autograd interface for the precompiled AscendC gated activation kernel."""
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from .._library import load_library


class FusedGatedActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gelu: bool, limit: float, after: bool) -> torch.Tensor:
        load_library("gated_activation")
        ctx.save_for_backward(x)
        ctx.gelu, ctx.limit, ctx.after = gelu, limit, after
        return torch.ops.torchturbo_ascendc.gated_forward(x, gelu, limit, after)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        dx = torch.ops.torchturbo_ascendc.gated_backward(x, dy.contiguous(), ctx.gelu, ctx.limit, ctx.after)
        return dx, None, None, None
