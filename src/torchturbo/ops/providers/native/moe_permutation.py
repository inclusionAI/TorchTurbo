# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.

"""PyTorch mask permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_mask_permutation",
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Group tokens by expert and return source row indices."""
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
        return permuted_input, permuted_probs, sorted_indices


@register_op(
    op_name="moe_mask_unpermutation",
    impl_name="torch_moe_mask_unpermutation",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoEMaskUnpermutation(OpImpl):
    def __call__(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
        routing_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weight expert outputs and sum them into their source rows."""
        _, hidden = restore_shape
        input_dtype = permuted_tokens.dtype
        if probs is not None:
            assert routing_map is not None, "Mask must be provided to permute the probs."
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
            # Retain the router probability dtype through weighting and accumulation.
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
        output_tokens = torch.zeros(
            restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device,
        )
        if torch.are_deterministic_algorithms_enabled():
            output_tokens.index_add_(0, sorted_indices, permuted_tokens)
        else:
            output_tokens.scatter_add_(
                0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens,
            )
        return output_tokens.to(dtype=input_dtype)


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
        """Reorder expert chunks using host split sizes and indices."""
        input = torch.split(input, split_sizes.tolist(), dim=0)
        output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
        permuted_probs = None
        if probs is not None:
            probs = torch.split(probs, split_sizes.tolist(), dim=0)
            permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
        return output, permuted_probs
