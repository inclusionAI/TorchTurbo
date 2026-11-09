# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.

"""PyTorch compact permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...moe import MoEPermutation
from ...registry import register_op


@register_op(
    op_name="moe_permutation",
    impl_name="torch_moe_mask_permutation",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoEMaskPermutation(OpImpl):
    def __call__(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
    ) -> MoEPermutation:
        """Group compact assignments stably by expert, then source token."""
        topk = expert_indices.shape[1]
        order = expert_indices.flatten().argsort(stable=True)
        permuted_input = tokens.index_select(0, order // topk)
        permuted_probs = None if probs is None else probs.flatten().index_select(0, order)
        packed_rows = torch.empty_like(order, dtype=torch.int32).scatter_(
            0, order, torch.arange(order.numel(), device=tokens.device, dtype=torch.int32),
        )
        return MoEPermutation(
            permuted_input,
            permuted_probs,
            packed_rows.view_as(expert_indices),
            expert_indices,
        )


@register_op(
    op_name="moe_unpermutation",
    impl_name="torch_moe_mask_unpermutation",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoEMaskUnpermutation(OpImpl):
    def __call__(
        self,
        permutation: MoEPermutation,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weight expert outputs and sum them into their source rows."""
        permuted_tokens = permutation.tokens
        input_dtype = permuted_tokens.dtype
        num_tokens, topk = permutation.row_indices.shape
        packed_rows = permutation.row_indices.flatten().long()
        source_rows = (
            torch.arange(num_tokens, device=permuted_tokens.device)
            .unsqueeze(1)
            .expand(num_tokens, topk)
            .reshape(-1)
        )
        # Invert only the row metadata; keep activations in their packed layout.
        token_indices = torch.empty_like(source_rows).scatter_(
            0, packed_rows, source_rows
        )
        values = permuted_tokens
        if probs is not None:
            selected_probs = probs.gather(1, permutation.expert_indices).flatten()
            packed_probs = torch.empty_like(selected_probs).scatter(
                0, packed_rows, selected_probs
            )
            # Retain the router probability dtype through weighting and accumulation.
            values = values * packed_probs.unsqueeze(-1)
        output = values.new_zeros(restore_shape)
        output.scatter_add_(0, token_indices.unsqueeze(1).expand_as(values), values)
        return output.to(input_dtype)


@register_op(
    op_name="moe_sort_chunks",
    impl_name="torch_moe_sort_chunks",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoESortChunks(OpImpl):
    def __call__(
        self,
        input: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Move chunk metadata to the host for PyTorch split and concatenation."""
        sizes = split_sizes.cpu().tolist()
        order = sorted_idxs.cpu().tolist()
        input = torch.split(input, sizes, dim=0)
        output = torch.cat([input[i] for i in order], dim=0)
        permuted_probs = None
        if probs is not None:
            probs = torch.split(probs, sizes, dim=0)
            permuted_probs = torch.cat([probs[i] for i in order], dim=0)
        return output, permuted_probs
