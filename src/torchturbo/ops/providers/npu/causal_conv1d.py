"""NPU (AscendC) provider for the causal short-conv1d op.

Wraps the vendored AscendC conv1d in ``ops/providers/npu/_fla_npu_gdn`` which is a
drop-in for fla's ``causal_conv1d`` (same ``[B,T,D]`` x / ``[D,W]`` weight
layout, returns ``(y, final_state)``). priority=100 so `auto` on NPU picks it
over the fla native (priority=0).
"""

from __future__ import annotations

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="causal_conv1d",
    impl_name="ascendc_causal_conv1d",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCCausalConv1d(OpImpl):
    def available(self) -> bool:
        try:
            import fla_npu  # noqa: F401
            from torchturbo.ops.providers.npu._fla_npu_gdn import (  # noqa: F401
                ascendc_causal_conv1d,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "ascendc_causal_conv1d unavailable (will fall back to fla "
                "under auto): %r\n%s",
                exc,
                traceback.format_exc(),
            )
            return False
        return True

    def __call__(
        self,
        x,
        weight,
        bias=None,
        *,
        activation=None,
        initial_state=None,
        output_final_state: bool = False,
        cu_seqlens=None,
        packed_params=None,
        **kwargs,
    ):
        from torchturbo.ops.providers.npu._fla_npu_gdn import ascendc_causal_conv1d

        return ascendc_causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            packed_params=packed_params,
        )
