from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="rotary_position_embedding",
    impl_name="npu_rotary_position_embedding",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuRotaryPositionEmbedding(OpImpl):
    def available(self) -> bool:
        try:
            from mindspeed.ops.npu_rotary_position_embedding import (  # noqa: F401
                npu_rotary_position_embedding,
            )
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int = 0,
    ):
        from mindspeed.ops.npu_rotary_position_embedding import (
            npu_rotary_position_embedding,
        )

        return npu_rotary_position_embedding(x, cos, sin, mode)
