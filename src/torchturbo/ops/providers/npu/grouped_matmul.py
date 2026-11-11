from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="grouped_matmul",
    impl_name="mindspeed_grouped_matmul",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class MindspeedGroupedMatmul(OpImpl):
    def available(self) -> bool:
        try:
            from mindspeed.core.fusions.grouped_matmul import Ops  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        from mindspeed.core.fusions.grouped_matmul import Ops

        return Ops.gmm(x, weight, group_list, trans_b=trans_b)
