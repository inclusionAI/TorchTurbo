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
            import torch_npu

            torch_npu.npu_rotary_mul  # noqa: B018
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int = 0,
    ):
        import torch_npu

        if mode == 0:
            rotary_mode = "half"
        elif mode == 1:
            rotary_mode = "interleave"
        else:
            raise ValueError(f"RoPE mode must be 0 or 1, but got {mode}.")

        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode=rotary_mode)
