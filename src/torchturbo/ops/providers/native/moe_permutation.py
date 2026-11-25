# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.

"""PyTorch compact permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...moe import MoEPermutation, MoESortedSlots
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
        routing_map: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> MoEPermutation:
        """Group compact assignments stably by expert, then source token."""
        if expert_indices is None:
            expert_indices = routing_map
        elif routing_map is not None and probs is not None:
            probs = probs.gather(1, expert_indices)
        sorted_slots = expert_indices.flatten().argsort(stable=True)
        valid_slots = sorted_slots[:num_out_tokens]
        source_rows = valid_slots // expert_indices.shape[1]
        packed = tokens.index_select(0, source_rows)
        packed_probs = None if probs is None else probs.flatten().index_select(0, valid_slots)
        mapping = MoESortedSlots(expert_indices, sorted_slots, source_rows)
        return MoEPermutation(packed, packed_probs, mapping=mapping)


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
        values = permutation.tokens
        input_dtype = values.dtype
        slots, source_rows = permutation.mapping.packed_indices(values.shape[0])
        if probs is not None:
            selected_probs = probs.gather(1, permutation.expert_indices).flatten()
            values = values * selected_probs.index_select(0, slots).unsqueeze(-1)
        output = values.new_zeros(restore_shape)
        if torch.are_deterministic_algorithms_enabled():
            output.index_add_(0, source_rows, values)
        else:
            output.scatter_add_(0, source_rows.unsqueeze(1).expand_as(values), values)
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
