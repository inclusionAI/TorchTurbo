from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="grouped_matmul",
    impl_name="torch_grouped_matmul",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchGroupedMatmul(OpImpl):
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        if weight.ndim != 3:
            raise ValueError(
                "grouped_matmul expects weight with shape (experts, in_features, out_features)."
            )
        if not weight.is_contiguous():
            raise ValueError("grouped_matmul expects contiguous weight.")

        if isinstance(tokens_per_expert, torch.Tensor):
            counts = [int(item) for item in tokens_per_expert.tolist()]
        else:
            counts = [int(item) for item in tokens_per_expert]

        if len(counts) != weight.shape[0]:
            raise ValueError(
                "tokens_per_expert must match the number of expert weight slices."
            )

        outputs: list[torch.Tensor] = []
        start = 0
        for expert_idx, count in enumerate(counts):
            end = start + count
            chunk = x[start:end]
            if chunk.numel() != 0:
                expert_weight = weight[expert_idx]
                if trans_b:
                    expert_weight = expert_weight.transpose(-1, -2)
                outputs.append(chunk.matmul(expert_weight))
            start = end

        if outputs:
            return torch.cat(outputs, dim=0)

        out_features = weight.shape[-2] if trans_b else weight.shape[-1]
        return x.new_empty((0, out_features))
