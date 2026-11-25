from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op
from ..native.grouped_matmul import empty_bank_matmul


@register_op(
    op_name="grouped_matmul",
    impl_name="grouped_gemm",
    kind=OpImplKind.CUSTOM,
    priority=0,
    devices=(TurboDevice.CUDA,),
)
class GroupedGemmMatmul(OpImpl):
    def available(self) -> bool:
        try:
            import grouped_gemm  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | tuple[torch.Tensor, ...],
        group_list: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        import grouped_gemm

        if isinstance(weight, torch.Tensor):
            return grouped_gemm.ops.gmm(x, weight, group_list, trans_b=trans_b)

        bank_counts = group_list.split([bank.shape[0] for bank in weight])
        chunks = x.split([int(counts.sum()) for counts in bank_counts])
        outputs = [
            grouped_gemm.ops.gmm(chunk, bank, counts, trans_b=trans_b)
            if chunk.shape[0] else empty_bank_matmul(chunk, bank, trans_b)
            for chunk, bank, counts in zip(chunks, weight, bank_counts)
        ]
        return torch.cat(outputs, dim=0)
