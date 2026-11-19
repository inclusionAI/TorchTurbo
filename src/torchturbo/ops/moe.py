"""Shared MoE routing, transport and permutation contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torchturbo.runtime import DeviceRuntime


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


def _scratch_layout(num_experts: int, ep_size: int) -> tuple[dict[str, tuple[int, tuple[int, ...]]], int]:
    """Offsets and shapes in the reusable planner scratch arena."""
    fields = (
        ("program_histogram", (32, num_experts)),
        ("tokens_per_expert", (num_experts,)),
        ("global_tokens_per_expert", (num_experts,)),
        ("balance", (ep_size,)),
        ("source_prefixes", (ep_size, num_experts)),
        ("allocation", (num_experts, ep_size)),
        ("destination_boundaries", (num_experts, ep_size)),
        ("virtual_expert_slots", (num_experts, ep_size)),
        ("statistics", (ep_size * (1 + 2 * (num_experts // ep_size)),)),
    )
    layout, offset = {}, 0
    for name, shape in fields:
        layout[name] = (offset, shape)
        offset += prod(shape)
    return layout, offset


class VirtualExpertPlannerWorkspace:
    """Reusable planner scratch and histogram window for one EP group.

    Communication storage is allocated and bound to the EP group at creation.
    Plans retain their own routes and copy tables; scratch is overwritten by
    each call. Providers must order all readers before reusing the workspace.
    """

    def __init__(
        self, *, num_experts: int, device: torch.device, group: dist.ProcessGroup | None,
        runtime: DeviceRuntime | None = None,
    ) -> None:
        from torchturbo.runtime import get_device_runtime

        self._runtime = get_device_runtime() if runtime is None else runtime
        self.ep_group = group
        self.rank = dist.get_rank(group)
        ep_size = dist.get_world_size(group)
        self._layout, numel = _scratch_layout(num_experts, ep_size)
        self.last_use: object | None = None
        # Evaluation may initialize a layer before its first training forward.
        with torch.inference_mode(False):
            self.scratch = torch.empty(numel, dtype=torch.int64, device=device)
            window, self.histogram_handle = self._runtime.allocate_communication_buffer(
                ep_size * num_experts, dtype=torch.int64, device=self.scratch.device, group=group,
            )
            self.gathered_counts = window.view(ep_size, num_experts)

    def field(self, name: str) -> torch.Tensor:
        """View one field of the scratch arena."""
        offset, shape = self._layout[name]
        return self.scratch[offset:offset + prod(shape)].view(shape)

    def destroy(self) -> None:
        """Drop the workspace's buffers while its process group is still alive."""
        if self.histogram_handle is not None or self.last_use is not None:
            self._runtime.synchronize()
        self.histogram_handle = self.gathered_counts = self.scratch = self.last_use = None


@dataclass(slots=True)
class VirtualExpertPlan:
    """Explicit state owned by one forward and retained through its backward.

    Routes retain the input top-k column order. Contexts are provider-owned:
    models pass them to dispatch and weight exchange without inspecting them.
    All ranks agree on has_global_replicas once host metadata is complete,
    including ranks with no local work. None means host metadata is pending.
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

    def wait(self) -> VirtualExpertPlan:
        """Finish deferred host metadata before native token or weight transport."""
        if self._materialize is not None:
            result = self._materialize()
            self.experts_to_copy = result.experts_to_copy
            self.dispatch_context = result.dispatch_context
            self.weight_context = result.weight_context
            self.has_global_replicas = result.has_global_replicas
            self._materialize = None
        return self


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
