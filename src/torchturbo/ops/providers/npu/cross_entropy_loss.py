from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="cross_entropy_loss",
    impl_name="npu_cross_entropy_loss",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuCrossEntropyLoss(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        reduction: str = "sum",
        ignore_index: int = -100,
    ):
        import torch_npu

        loss = torch_npu.npu_cross_entropy_loss(
            logits,
            labels,
            reduction=reduction,
            ignore_index=ignore_index,
        )
        if isinstance(loss, tuple):
            loss = loss[0]
        if reduction != "none" and torch.is_tensor(loss) and loss.dim() > 0:
            loss = loss.squeeze()
        return loss
