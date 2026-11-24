"""Triton and ahead-of-time AscendC providers for fused gated activations."""
from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="gated_activation",
    impl_name="triton_gated_activation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TritonGatedActivation(OpImpl):
    def available(self) -> bool:
        return find_spec("triton") is not None and find_spec("torch_npu") is not None

    def __call__(
        self,
        x: torch.Tensor,
        *,
        activation_func: Callable[[torch.Tensor], torch.Tensor],
        limit: float = 0.0,
        clamp_gate_after_activation: bool = False,
    ) -> torch.Tensor:
        if activation_func not in (F.silu, F.gelu):
            raise ValueError("gated_activation only supports F.silu and F.gelu.")
        limit = float(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        # Odd widths follow torch.chunk broadcasting; other dtypes retain the
        # native contract instead of narrowing their precision in the kernel.
        if x.numel() == 0 or x.shape[-1] % 2 or x.dtype not in (
            torch.float16, torch.bfloat16, torch.float32,
        ):
            from ..native.gated_activation import _torch_gated_activation

            return _torch_gated_activation(
                x, activation_func=activation_func, limit=limit,
                clamp_gate_after_activation=clamp_gate_after_activation,
            )
        from .triton._gated_activation import FusedGatedActivation

        return FusedGatedActivation.apply(
            x.contiguous(), activation_func is F.gelu, limit,
            clamp_gate_after_activation,
        )


@register_op(
    op_name="gated_activation",
    impl_name="ascendc_gated_activation",
    kind=OpImplKind.CUSTOM,
    priority=90,
    devices=(TurboDevice.NPU,),
)
class AscendCGatedActivation(OpImpl):
    def available(self) -> bool:
        from .ascendc._library import library_path

        return find_spec("torch_npu") is not None and library_path("gated_activation").is_file()

    def __call__(
        self,
        x: torch.Tensor,
        *,
        activation_func: Callable[[torch.Tensor], torch.Tensor],
        limit: float = 0.0,
        clamp_gate_after_activation: bool = False,
    ) -> torch.Tensor:
        if activation_func not in (F.silu, F.gelu):
            raise ValueError("gated_activation only supports F.silu and F.gelu.")
        limit = float(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        if x.numel() == 0 or x.shape[-1] % 2 or x.dtype not in (
            torch.float32, torch.float16, torch.bfloat16,
        ):
            from ..native.gated_activation import _torch_gated_activation

            return _torch_gated_activation(
                x, activation_func=activation_func, limit=limit,
                clamp_gate_after_activation=clamp_gate_after_activation,
            )
        from .ascendc.gated_activation import FusedGatedActivation

        return FusedGatedActivation.apply(
            x.contiguous(), activation_func is F.gelu, limit,
            clamp_gate_after_activation,
        )
