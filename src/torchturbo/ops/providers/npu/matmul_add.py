from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op
from ..native.matmul_add import TorchGroupedMatmulAdd


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


@register_op(
    op_name="grouped_matmul_add",
    impl_name="torch_npu_grouped_addmm",
    kind=OpImplKind.NATIVE,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TorchNpuGroupedAddmm(TorchGroupedMatmulAdd):
    _matmul_add = TorchNpuMatmulAdd()

    def available(self) -> bool:
        return self._matmul_add.available()


@register_op(
    op_name="grouped_matmul_add",
    impl_name="torch_npu_grouped_matmul_add",
    kind=OpImplKind.CUSTOM,
    priority=200,
    devices=(TurboDevice.NPU,),
)
class TorchNpuGroupedMatmulAdd(TorchNpuGroupedAddmm):
    def available(self) -> bool:
        try:
            from torch_npu import npu_grouped_matmul_add_  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        accumulator: torch.Tensor,
        x: torch.Tensor,
        dy: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> None:
        if x.shape[0] == 0:
            return
        if x.dtype not in (torch.bfloat16, torch.float16) or accumulator.dtype != torch.float32:
            return super().__call__(accumulator, x, dy, tokens_per_expert)

        from torch_npu import npu_grouped_matmul_add_

        # The in-place wgrad kernel takes cumulative ends, including repeated ends for empty slots.
        ends = tokens_per_expert.cumsum(0, dtype=torch.int64).to(x.device, non_blocking=True)
        npu_grouped_matmul_add_(
            accumulator, x.contiguous(), dy.contiguous(), ends,
            transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0,
        )
