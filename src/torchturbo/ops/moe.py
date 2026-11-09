"""Shared MoE routing, transport and permutation contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


class _EPAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int] | None,
        input_split_sizes: list[int] | None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input_tensor

        input_tensor = input_tensor.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input_tensor)
        else:
            output = input_tensor.new_empty(
                size=[sum(output_split_sizes)] + list(input_tensor.size()[1:]),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, None]:
        return (
            None,
            _EPAllToAll.apply(
                ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes,
            ),
            None,
            None,
        )


def ep_all_to_all(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Exchange variable-length chunks across an EP group with autograd support."""
    return _EPAllToAll.apply(
        group, input_tensor, output_split_sizes, input_split_sizes,
    )


@dataclass(frozen=True, slots=True)
class HybridEPMetadata:
    """CPU dispatch statistics and device route ordering from one plan.

    The sorted IDs and source column order describe the plan's original
    compact routes. They may be reused with equal routes and new probabilities.
    """

    received_counts: torch.Tensor
    tokens_per_expert: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]
    sorted_expert_indices: torch.Tensor
    route_order: torch.Tensor


@dataclass(slots=True)
class VirtualExpertPlan:
    """Explicit state owned by one forward and retained through its backward.

    Routes retain the input top-k column order. Contexts are provider-owned:
    models pass them to dispatch and weight exchange without inspecting them.
    All ranks agree on has_global_replicas, including ranks with no local work.
    Consuming or copying route tensors does not consume the contexts.
    """

    virtual_experts: torch.Tensor | None
    experts_to_copy: torch.Tensor
    dispatch_context: object | None
    weight_context: object
    has_global_replicas: bool


@dataclass(frozen=True, slots=True)
class HybridEPHandle:
    """Opaque dispatch state passed from ``hybrid_ep_dispatch`` to ``hybrid_ep_combine``."""

    row_indices: torch.Tensor
    expert_indices: torch.Tensor
    restore_shape: torch.Size
    ep_group: dist.ProcessGroup
    num_local_experts: int
    input_splits: list[int] | None
    output_splits: list[int] | None
    num_global_tokens_per_local_expert: torch.Tensor | None
    restore_output_by_local_experts: torch.Tensor | None


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
    # slot order need not match the sorted expert IDs passed into permutation.
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
