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
            from mindspeed.ops.npu_moe_token_unpermute import (  # noqa: F401
                npu_moe_token_unpermute,
            )
        except ImportError:
            return False
        return True

    def __call__(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute

        return npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
