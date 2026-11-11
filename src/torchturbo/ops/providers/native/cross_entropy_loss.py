from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="cross_entropy_loss",
    impl_name="torch_cross_entropy_loss",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchCrossEntropyLoss(OpImpl):
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        reduction: str = "sum",
        ignore_index: int = -100,
    ):
        return F.cross_entropy(
            logits,
            labels,
            reduction=reduction,
            ignore_index=ignore_index,
        )
