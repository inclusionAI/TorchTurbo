from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchturbo.runtime import TurboDevice
from ...base import OpImpl, OpImplKind
from ...registry import register_op
from ..native.gated_activation import _torch_gated_activation


_CHUNK_ROWS = 4096


class _ChunkedClippedSwiGLU(torch.autograd.Function):
    """Save the input only and recompute bounded chunks during backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.limit = limit
        output = x.new_empty((x.shape[0], x.shape[1] // 2))
        for start in range(0, x.shape[0], _CHUNK_ROWS):
            end = start + _CHUNK_ROWS
            output[start:end] = _torch_gated_activation(
                x[start:end], activation_func=F.silu, limit=limit
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        for start in range(0, x.shape[0], _CHUNK_ROWS):
            end = start + _CHUNK_ROWS
            with torch.enable_grad():
                chunk = x[start:end].detach().requires_grad_(True)
                output = _torch_gated_activation(
                    chunk, activation_func=F.silu, limit=ctx.limit
                )
                (grad,) = torch.autograd.grad(output, chunk, grad_output[start:end])
            grad_input[start:end] = grad
            del output, chunk, grad
        return grad_input, None


@register_op(
    op_name="gated_activation",
    impl_name="npu_chunked_gated_activation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuChunkedGatedActivation(OpImpl):
    def __call__(
        self,
        x: torch.Tensor,
        *,
        activation_func: Callable[[torch.Tensor], torch.Tensor],
        limit: float = 0.0,
        clamp_gate_after_activation: bool = False,
    ) -> torch.Tensor:
        if clamp_gate_after_activation:
            return _torch_gated_activation(
                x, activation_func=activation_func, limit=limit,
                clamp_gate_after_activation=True,
            )
        if activation_func is F.silu and limit > 0 and x.ndim == 2:
            return _ChunkedClippedSwiGLU.apply(x, float(limit))
        return _torch_gated_activation(x, activation_func=activation_func, limit=limit)
