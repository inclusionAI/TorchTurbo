"""Shared tensor contract for MoE permutation and unpermutation operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MoEPermutation:
    """Data interface shared by ``moe_permutation`` and ``moe_unpermutation``.

    Each source token is assigned to ``topk`` distinct experts without token
    dropping. Packing creates one feature row per token/expert assignment,
    ordered by expert ID, then by source token ID. Unpermutation sums expert
    outputs in that row order back into their source tokens.

    All tensors are on the token device. The two index tensors describe the
    same assignments: slot order may differ between providers, but each
    ``row_indices[i, j]`` stays paired with ``expert_indices[i, j]``.
    """

    # Feature tensor of shape [num_tokens * topk, hidden_size]. After permutation,
    # rows copy source token features with their original dtype and values;
    # routing weights have not been applied. Before unpermutation, supply expert
    # outputs in the same packed row order.
    tokens: torch.Tensor

    # Optional routing weights of shape [num_tokens * topk], aligned with tokens
    # row by row and retaining the input probability dtype. None means no weights
    # were supplied for packing. Expert computation can use these weights;
    # unpermutation does not apply this field implicitly.
    probs: torch.Tensor | None

    # int32 tensor of shape [num_tokens, topk]. Entry [i, j] is the packed row for
    # source token i assigned to expert_indices[i, j]. Every packed row appears
    # exactly once. For example, [0, 4] in row i means packed rows 0 and 4
    # contribute to source token i during unpermutation.
    row_indices: torch.Tensor

    # int64 tensor of shape [num_tokens, topk]. Entry [i, j] is the global expert
    # ID in [0, num_experts) for the assignment at row_indices[i, j]. Unpermutation
    # uses these IDs to gather explicitly supplied dense merging weights. Output
    # slot order need not match the order of the expert IDs passed into permutation.
    expert_indices: torch.Tensor


def get_expert_indices(routing_map: torch.Tensor) -> torch.Tensor:
    """Derive fixed-top-k expert IDs when the caller has no cached selection."""
    num_tokens = routing_map.shape[0]
    if num_tokens == 0:
        return routing_map.new_empty((0, 0), dtype=torch.long)
    rows, experts = routing_map.nonzero(as_tuple=True)
    topk = experts.numel() // num_tokens
    expected_rows = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(1)
        .expand(num_tokens, topk)
        .reshape(-1)
    )
    if not torch.equal(rows, expected_rows):
        raise ValueError(
            "MoE permutation requires the same top-k count for every token."
        )
    return experts.view(num_tokens, topk)
