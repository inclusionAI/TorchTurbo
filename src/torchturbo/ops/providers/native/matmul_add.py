from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="matmul_add",
    impl_name="torch_matmul_add",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMatmulAdd(OpImpl):
    def __call__(
        self,
        accumulator: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        transpose_lhs: bool = False,
    ) -> None:
        # Native addmm generally requires the accumulator and matmul operands
        # to share a dtype. Per-chunk casts avoid a full [V, H] temporary.
        if transpose_lhs:
            lhs = lhs.t()
        accumulator.addmm_(
            lhs.to(accumulator.dtype),
            rhs.to(accumulator.dtype),
        )


@register_op(
    op_name="grouped_matmul_add",
    impl_name="torch_grouped_matmul_add",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchGroupedMatmulAdd(OpImpl):
    _matmul_add = TorchMatmulAdd()

    def __call__(
        self,
        accumulator: torch.Tensor,
        x: torch.Tensor,
        dy: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> None:
        start = 0
        for expert, count in enumerate(tokens_per_expert.tolist()):
            end = start + count
            if count:
                self._matmul_add(accumulator[expert], x[start:end], dy[start:end], transpose_lhs=True)
            start = end
