from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


_SUPPORTED_ACTIVATIONS = (F.silu, F.gelu)
_CHUNK_BYTES = 256 * 2**20


def _eager_gated_activation(
    x: torch.Tensor,
    activation_func: Callable[[torch.Tensor], torch.Tensor],
    limit: float,
    clamp_gate_after_activation: bool,
) -> torch.Tensor:
    gate, up = torch.chunk(x, 2, dim=-1)
    if limit == 0:
        return (activation_func(gate) * up).to(x.dtype)

    up = torch.clamp(up.float(), min=-limit, max=limit)
    if clamp_gate_after_activation:
        activated_gate = torch.clamp(activation_func(gate.float()), max=limit)
    else:
        activated_gate = activation_func(torch.clamp(gate.float(), max=limit))
    return (activated_gate * up).to(x.dtype)


class _ChunkedGatedActivation(torch.autograd.Function):
    """Save the input and recompute bounded FP32 intermediates in backward."""

    @staticmethod
    def forward(ctx, x, activation_func, limit, clamp_gate_after_activation):
        ctx.save_for_backward(x)
        ctx.activation_args = (activation_func, limit, clamp_gate_after_activation)
        ctx.chunk_rows = max(1, _CHUNK_BYTES // (x.size(-1) * 4))
        x2 = x.view(-1, x.size(-1))
        output = x.new_empty((*x.shape[:-1], x.size(-1) // 2))
        output2 = output.view(-1, output.size(-1))
        for start in range(0, x2.size(0), ctx.chunk_rows):
            end = start + ctx.chunk_rows
            output2[start:end] = _eager_gated_activation(
                x2[start:end], *ctx.activation_args
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        grad_input = torch.empty_like(x)
        grad_input2 = grad_input.view(-1, x.size(-1))
        grad_output2 = grad_output.reshape(-1, grad_output.size(-1))
        # Reuse eager autograd so activation derivatives and clamp boundaries
        # match the reference, including when higher-order gradients are needed.
        with torch.enable_grad():
            x2 = x.view(-1, x.size(-1))
            for start in range(0, x2.size(0), ctx.chunk_rows):
                end = start + ctx.chunk_rows
                chunk = x2[start:end]
                output = _eager_gated_activation(chunk, *ctx.activation_args)
                (grad_chunk,) = torch.autograd.grad(
                    output,
                    chunk,
                    grad_output2[start:end],
                    create_graph=create_graph,
                )
                grad_input2[start:end] = grad_chunk
        return grad_input, None, None, None


def _torch_gated_activation(
    x: torch.Tensor,
    *,
    activation_func: Callable[[torch.Tensor], torch.Tensor],
    limit: float = 0.0,
    clamp_gate_after_activation: bool = False,
) -> torch.Tensor:
    """Apply ``act(gate) * up`` to ``x = [gate, up]``.

    A positive ``limit`` clamps ``up`` to ``[-limit, limit]`` and caps the
    gate either before the activation (``act(min(gate, limit))``) or, with
    ``clamp_gate_after_activation``, after it (``min(act(gate), limit)``).
    """
    if activation_func not in _SUPPORTED_ACTIVATIONS:
        raise ValueError("gated_activation only supports F.silu and F.gelu.")

    limit = float(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative.")

    if (
        limit > 0
        and not clamp_gate_after_activation
        and x.is_contiguous()
        and x.numel() > 0
        and x.size(-1) % 2 == 0
    ):
        return _ChunkedGatedActivation.apply(
            x, activation_func, limit, clamp_gate_after_activation
        )
    return _eager_gated_activation(
        x, activation_func, limit, clamp_gate_after_activation
    )


@register_op(
    op_name="gated_activation",
    impl_name="torch_gated_activation",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchGatedActivation(OpImpl):
    def __call__(
        self,
        x: torch.Tensor,
        *,
        activation_func: Callable[[torch.Tensor], torch.Tensor],
        limit: float = 0.0,
        clamp_gate_after_activation: bool = False,
    ) -> torch.Tensor:
        return _torch_gated_activation(
            x,
            activation_func=activation_func,
            limit=limit,
            clamp_gate_after_activation=clamp_gate_after_activation,
        )
