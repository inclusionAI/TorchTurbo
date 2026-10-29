from __future__ import annotations

from typing import Optional

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_token_unpermutation",
    impl_name="npu_moe_token_unpermute",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoeTokenUnpermutation(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu

            torch_npu.npu_moe_token_unpermute  # noqa: B018
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

    def __call__(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        import torch_npu

        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens, sorted_indices, probs=probs
        )
