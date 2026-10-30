from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


class _TorchNpuGroupedMatmul(torch.autograd.Function):
    """Differentiable grouped matmul backed by torch_npu."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        trans_b: bool,
    ) -> torch.Tensor:
        import torch_npu

        kernel_weight = weight.transpose(1, 2) if trans_b else weight
        counts = group_list.detach().to(
            device=x.device, dtype=torch.int64, non_blocking=True
        )
        ctx.save_for_backward(x, kernel_weight)
        ctx.counts = counts
        ctx.trans_b = trans_b
        output = torch_npu.npu_grouped_matmul(
            [x],
            [kernel_weight],
            group_list=counts,
            split_item=3,
            group_type=0,
            group_list_type=1,
        )
        return output[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch_npu

        x, kernel_weight = ctx.saved_tensors
        counts = ctx.counts
        grad_x = None
        grad_kernel_weight = None

        if ctx.needs_input_grad[0]:
            grad_x = torch_npu.npu_grouped_matmul(
                [grad_output],
                [kernel_weight.transpose(1, 2)],
                group_list=counts,
                split_item=3,
                group_type=0,
                group_list_type=1,
            )[0]

        if ctx.needs_input_grad[1]:
            grad_kernel_weight = torch_npu.npu_grouped_matmul(
                [x.transpose(0, 1)],
                [grad_output],
                group_list=counts,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]
            if ctx.trans_b:
                grad_kernel_weight = grad_kernel_weight.transpose(1, 2)

        return grad_x, grad_kernel_weight, None, None


@register_op(
    op_name="grouped_matmul",
    impl_name="torch_npu_grouped_matmul",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TorchNpuGroupedMatmul(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu

            torch_npu.npu_grouped_matmul  # noqa: B018
        except ImportError:
            return False
        except AttributeError:
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
        return _TorchNpuGroupedMatmul.apply(x, weight, group_list, trans_b)
