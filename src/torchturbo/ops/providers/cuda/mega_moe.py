"""CUDA provider for the MegaMoE dispatch entry point."""

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import MegaMoeContext
from ...registry import register_op


@register_op(
    op_name="mega_moe",
    impl_name="cuda_mega_moe",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class CudaMegaMoe(OpImpl):
    def __call__(
        self,
        inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        experts: torch.nn.Module,
        *,
        context: MegaMoeContext,
    ) -> torch.Tensor:
        raise NotImplementedError("MegaMoE is not yet implemented on CUDA.")
