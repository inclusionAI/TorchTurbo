# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.

"""PyTorch mask permutation and expert chunk sorting."""

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
        routing_map: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> MoEPermutation:
        """Group tokens by expert and retain the packed row of every assignment."""
        num_tokens = tokens.shape[0]
        num_experts = routing_map.shape[1]
        routing_map = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device)
            .unsqueeze(0)
            .expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)
        permuted_input = tokens.index_select(0, sorted_indices)
        permuted_probs = None
        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
        packed_rows = torch.empty_like(routing_map, dtype=torch.int32)
        packed_rows.masked_scatter_(
            routing_map,
            torch.arange(sorted_indices.numel(), device=tokens.device, dtype=torch.int32),
        )
        return MoEPermutation(
            permuted_input,
            permuted_probs,
            packed_rows.T.gather(1, expert_indices),
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
