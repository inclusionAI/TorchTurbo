# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# Modified by the TorchTurbo contributors.

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_chunk_reorder",
    impl_name="torch_moe_chunk_reorder",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoeChunkReorder(OpImpl):
    def __call__(
        self,
        tokens: torch.Tensor,
        chunk_counts: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transpose rank/expert chunks with PyTorch split and concatenate."""
        order = torch.arange(chunk_counts.numel(), device="cpu")
        order = order.reshape(chunk_counts.shape).T.reshape(-1).tolist()
        split_sizes = chunk_counts.reshape(-1).tolist()
        token_chunks = torch.split(tokens, split_sizes, dim=0)
        output = torch.cat([token_chunks[index] for index in order], dim=0)
        if probs is None:
            return output, None
        prob_chunks = torch.split(probs, split_sizes, dim=0)
        permuted_probs = torch.cat([prob_chunks[index] for index in order], dim=0)
        return output, permuted_probs
