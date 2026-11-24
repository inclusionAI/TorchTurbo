"""Shared MoE routing, transport and permutation contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import invert_permutation


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
    """CPU dispatch statistics for the compact routes returned by one plan."""

    received_counts: torch.Tensor
    tokens_per_expert: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


class VirtualExpertPlannerWorkspace:
    """Reusable local histogram for native planning in one EP group."""

    def __init__(
        self, *, num_experts: int, device: torch.device, group: dist.ProcessGroup | None,
    ) -> None:
        self.ep_group = group
        self.num_experts = num_experts
        self.rank = dist.get_rank(group)
        ep_size = dist.get_world_size(group)
        # Evaluation may initialize a layer before its first training forward.
        with torch.inference_mode(False):
            self.gathered_counts = torch.empty((ep_size, num_experts), dtype=torch.int64, device=device)

    @property
    def device(self) -> torch.device:
        """Device holding the histogram for this expert group."""
        return self.gathered_counts.device

    def destroy(self) -> None:
        """Release the histogram after all uses of the workspace have finished."""
        self.gathered_counts = None


@dataclass(slots=True)
class VirtualExpertPlan:
    """Explicit state owned by one forward and retained through its backward.

    Routes retain the input top-k column order. Contexts are provider-owned:
    models pass them to dispatch and weight exchange without inspecting them.
    num_local_replicas counts the active replica slots on this rank.
    has_global_replicas is shared by every rank, including ranks with zero
    local replicas. Both values are available after wait() and may be None
    while host metadata is pending.
    Consuming or copying route tensors does not consume the contexts.
    With async_finish=True, routes and copy decisions remain device tensors
    that may be consumed in stream order. Native transport contexts become
    valid after wait() returns.
    """

    virtual_experts: torch.Tensor | None
    experts_to_copy: torch.Tensor
    dispatch_context: HybridEPMetadata | None
    weight_context: object
    has_global_replicas: bool | None
    _materialize: Callable[[], VirtualExpertPlan] | None = None
    ep_group: dist.ProcessGroup | None = None
    num_local_replicas: int | None = None

    def wait(self) -> VirtualExpertPlan:
        """Finish deferred host metadata before native token or weight transport."""
        if self._materialize is not None:
            result = self._materialize()
            self.experts_to_copy = result.experts_to_copy
            self.dispatch_context = result.dispatch_context
            self.weight_context = result.weight_context
            self.has_global_replicas = result.has_global_replicas
            self.num_local_replicas = result.num_local_replicas
            self._materialize = None
        return self


@dataclass(frozen=True, slots=True)
class MoEPackedRows:
    """Packed row for each [source token, top-k slot], as returned by fused kernels."""

    row_indices: torch.Tensor
    expert_indices: torch.Tensor

    def packed_indices(self, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        slots = invert_permutation(self.row_indices.flatten().long())[:num_rows]
        return slots, slots // self.expert_indices.shape[1]


@dataclass(frozen=True, slots=True)
class MoESortedSlots:
    """Source slots in packed order and the token rows selected by native permute."""

    expert_indices: torch.Tensor
    slots: torch.Tensor
    source_rows: torch.Tensor

    @property
    def row_indices(self) -> torch.Tensor:
        """Packed row per source slot, in the int32 format used by fused kernels."""
        rows = torch.empty_like(self.slots, dtype=torch.int32)
        positions = torch.arange(self.slots.numel(), device=rows.device, dtype=rows.dtype)
        return rows.scatter_(0, self.slots, positions).view_as(self.expert_indices)

    def packed_indices(self, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.slots[:num_rows], self.source_rows[:num_rows]


MoEPermutationMap = MoEPackedRows | MoESortedSlots


@dataclass(frozen=True, slots=True)
class HybridEPHandle:
    """Opaque dispatch state passed from ``hybrid_ep_dispatch`` to ``hybrid_ep_combine``."""

    mapping: MoEPermutationMap
    restore_shape: torch.Size
    ep_group: dist.ProcessGroup
    input_splits: list[int] | None
    output_splits: list[int] | None
    num_global_tokens_per_local_expert: torch.Tensor | None
    restore_output_by_local_experts: torch.Tensor | None


@dataclass(frozen=True, slots=True, init=False)
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

    mapping: MoEPermutationMap

    def __init__(
        self, tokens: torch.Tensor, probs: torch.Tensor | None,
        row_indices: torch.Tensor | None = None, expert_indices: torch.Tensor | None = None,
        *, mapping: MoEPermutationMap | None = None,
    ) -> None:
        if mapping is None:
            if row_indices is None or expert_indices is None:
                raise TypeError("MoEPermutation requires row_indices and expert_indices or a mapping.")
            mapping = MoEPackedRows(row_indices, expert_indices)
        # dataclasses.replace supplies the existing mapping along with any
        # explicitly replaced public index tensor.
        elif row_indices is not None:
            if expert_indices is None:
                expert_indices = mapping.expert_indices
            mapping = MoEPackedRows(row_indices, expert_indices)
        elif expert_indices is not None:
            mapping = replace(mapping, expert_indices=expert_indices)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "probs", probs)
        object.__setattr__(self, "mapping", mapping)

    @property
    def row_indices(self) -> torch.Tensor:
        """Packed row for each [source token, top-k slot], as an int32 tensor."""
        return self.mapping.row_indices

    @property
    def expert_indices(self) -> torch.Tensor:
        """Expert IDs paired with row_indices, in the producing kernel's slot order."""
        return self.mapping.expert_indices


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
