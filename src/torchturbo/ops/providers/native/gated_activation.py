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
) -> torch.Tensor:
    if activation_func not in _SUPPORTED_ACTIVATIONS:
        raise ValueError("gated_activation only supports F.silu and F.gelu.")

    limit = float(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative.")

    gate, up = torch.chunk(x, 2, dim=-1)
    if limit > 0:
        up = torch.clamp(up.float(), min=-limit, max=limit)
        gate = torch.clamp(gate.float(), max=limit)
    return (activation_func(gate) * up).to(x.dtype)


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
    ) -> torch.Tensor:
        return _torch_gated_activation(
            x,
            activation_func=activation_func,
            limit=limit,
        )
