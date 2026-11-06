from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="matmul_add",
    impl_name="torch_npu_matmul_add",
    kind=OpImplKind.NATIVE,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TorchNpuMatmulAdd(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        accumulator: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        transpose_lhs: bool = False,
    ) -> None:
        # torch-npu supports BF16/FP16 inputs with an FP32 accumulator.
        if transpose_lhs:
            lhs = lhs.t().contiguous()
        accumulator.addmm_(lhs, rhs)
