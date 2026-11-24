from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def empty_bank_matmul(x: torch.Tensor, bank: torch.Tensor, trans_b: bool) -> torch.Tensor:
    """Return zero rows with matmul's autocast dtype and zero gradients for the empty operands."""
    out_features = bank.shape[-2 if trans_b else -1]
    # A nonempty batch already receives dX from its active banks. Attaching that
    # full input here would allocate another batch-sized zero gradient in backward.
    lhs = x[:, :0] if x.shape[0] == 0 else x.new_empty((0, 0))
    return lhs.matmul(bank[:0].reshape(0, out_features))


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
        weight: torch.Tensor | tuple[torch.Tensor, ...],
        tokens_per_expert: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        banks = (weight,) if isinstance(weight, torch.Tensor) else weight
        if any(bank.ndim != 3 for bank in banks):
            raise ValueError(
                "grouped_matmul expects weight with shape (experts, in_features, out_features)."
            )

        if isinstance(tokens_per_expert, torch.Tensor):
            counts = [int(item) for item in tokens_per_expert.tolist()]
        else:
            counts = [int(item) for item in tokens_per_expert]

        if len(counts) != sum(bank.shape[0] for bank in banks):
            raise ValueError(
                "tokens_per_expert must match the number of expert weight slices."
            )

        outputs: list[torch.Tensor] = []
        start = 0
        remaining_counts = iter(counts)
        for bank in banks:
            bank_start = start
            for expert_weight, count in zip(bank.unbind(), remaining_counts):
                end = start + count
                chunk = x[start:end]
                if chunk.numel() != 0:
                    if trans_b:
                        expert_weight = expert_weight.transpose(-1, -2)
                    outputs.append(chunk.matmul(expert_weight))
                start = end
            if bank_start == start and isinstance(weight, tuple):
                outputs.append(empty_bank_matmul(x, bank, trans_b))

        if outputs:
            return torch.cat(outputs, dim=0)

        out_features = banks[0].shape[-2] if trans_b else banks[0].shape[-1]
        return x.new_empty((0, out_features))
