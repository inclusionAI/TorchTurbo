from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_token_permutation",
    impl_name="npu_moe_token_permute",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoeTokenPermutation(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu

            torch_npu.npu_moe_token_permute  # noqa: B018
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

    def __call__(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
    ):
        import torch_npu

        return torch_npu.npu_moe_token_permute(tokens, expert_indices)
