from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


_SUPPORTED_ACTIVATIONS = (F.silu, F.gelu)


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

    gate, up = torch.chunk(x, 2, dim=-1)
    if limit == 0:
        return (activation_func(gate) * up).to(x.dtype)

    up = torch.clamp(up.float(), min=-limit, max=limit)
    if clamp_gate_after_activation:
        activated_gate = torch.clamp(activation_func(gate.float()), max=limit)
    else:
        activated_gate = activation_func(torch.clamp(gate.float(), max=limit))
    return (activated_gate * up).to(x.dtype)


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
