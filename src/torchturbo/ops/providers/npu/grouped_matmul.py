from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _grouped_matmul(x, weights, counts, trans_b):
    import torch_npu

    if not x.shape[0]:
        return x.new_empty((0, weights[0].shape[-2 if trans_b else -1]))
    if trans_b:
        weights = tuple(w.transpose(1, 2) for w in weights)
    # CANN accepts either one packed weight or a list of 2-D weights in distinct allocations.
    if len(weights) == 1:
        kernel_weights = [weights[0]]
    else:
        kernel_weights = [expert for bank in weights for expert in bank.unbind()]
    return torch_npu.npu_grouped_matmul(
        [x], kernel_weights, group_list=counts, split_item=3, group_type=0, group_list_type=1,
    )[0]


class _TorchNpuGroupedMatmul(torch.autograd.Function):
    """Differentiable grouped matmul backed by torch_npu."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group_list: torch.Tensor,
        trans_b: bool,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        counts = group_list.detach().to(
            device=x.device, dtype=torch.int64, non_blocking=True
        )
        ctx.save_for_backward(x, counts, *weights)
        ctx.trans_b = trans_b
        return _grouped_matmul(x, weights, counts, trans_b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch_npu

        x, counts, *weights = ctx.saved_tensors
        grad_x = None
        grad_weights = (None,) * len(weights)
        if ctx.needs_input_grad[0]:
            grad_x = _grouped_matmul(grad_output, weights, counts, not ctx.trans_b)

        if any(ctx.needs_input_grad[3:]) and x.shape[0]:
            grad_weight = torch_npu.npu_grouped_matmul(
                [x.transpose(0, 1)],
                [grad_output],
                group_list=counts,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]
            if ctx.trans_b:
                grad_weight = grad_weight.transpose(1, 2)
            grad_weights = grad_weight.split([w.shape[0] for w in weights])
        elif any(ctx.needs_input_grad[3:]):
            grad_weights = tuple(torch.zeros_like(w) for w in weights)
        return grad_x, None, None, *grad_weights


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
        weight: torch.Tensor | tuple[torch.Tensor, ...],
        group_list: torch.Tensor,
        *,
        trans_b: bool = False,
    ) -> torch.Tensor:
        weights = (weight,) if isinstance(weight, torch.Tensor) else weight
        if (
            torch.is_autocast_enabled(x.device.type)
            and x.dtype != torch.float64 and weights[0].dtype != torch.float64
        ):
            # Cast outside the custom Function so autograd restores input dtypes.
            dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(dtype)
            weights = tuple(w.to(dtype) for w in weights)
        # aclnnGroupedMatmulV5 accepts at most 128 weight-list entries. Packed
        # 3-D banks occupy one entry each even for models with more experts.
        if len(weights) > 1 and sum(w.shape[0] for w in weights) > 128:
            bank_counts = group_list.split([w.shape[0] for w in weights])
            chunks = x.split([int(counts.sum()) for counts in bank_counts])
            return torch.cat([
                _TorchNpuGroupedMatmul.apply(chunk, counts, trans_b, bank)
                for chunk, counts, bank in zip(chunks, bank_counts, weights)
            ])
        return _TorchNpuGroupedMatmul.apply(x, group_list, trans_b, *weights)
