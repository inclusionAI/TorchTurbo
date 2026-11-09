"""Virtual-expert route planning and placement fallback."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...base import OpImpl, OpImplKind
from ...moe import HybridEPMetadata, VirtualExpertPlan
from ...ops import ops
from ...registry import register_op
from .virtual_expert_weight_exchange import build_weight_exchange_context


def plan_placement(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Place overloaded logical experts into replica slots greedily.

    Args:
        counts: CPU int64 tensor of shape [num_ranks, num_experts] with the
            number of routes sent from each source rank to each logical expert.

    Returns:
        A tuple ``(allocations, copies)`` where ``allocations`` is an int64
        tensor of shape [num_experts, num_ranks] giving the token quota of each
        logical expert on each destination rank, and ``copies`` is an int64
        tensor of shape [num_ranks, num_local_experts] giving the logical expert
        ID materialized in each replica slot (-1 for unused slots).
    """
    counts = counts.to(dtype=torch.int64)
    num_ranks, num_experts = counts.shape
    num_local_experts = num_experts // num_ranks
    totals = counts.sum(dim=0).tolist()
    total_routes = sum(totals)
    balances = [
        sum(totals[rank * num_local_experts : (rank + 1) * num_local_experts])
        - total_routes // num_ranks - (rank < total_routes % num_ranks)
        for rank in range(num_ranks)
    ]

    quotas = [[0] * num_ranks for _ in range(num_ranks)]
    for _ in range(num_ranks):
        if max(balances, default=0) <= 0:
            break
        overloaded = max(range(num_ranks), key=lambda r: (balances[r], -r))
        receiver = max(range(num_ranks), key=lambda r: (-balances[r], -r))
        moved = -balances[receiver]
        quotas[overloaded][receiver] = moved
        balances[overloaded] -= moved
        balances[receiver] = 0

    allocation_rows = [[0] * num_ranks for _ in range(num_experts)]
    for placing_rank in range(num_ranks):
        start = placing_rank * num_local_experts
        remaining = totals[start : start + num_local_experts]
        for local_expert, total in enumerate(remaining):
            allocation_rows[start + local_expert][placing_rank] = total
        rank_quotas = quotas[placing_rank]
        for _ in range(num_ranks + num_local_experts):
            if max(rank_quotas, default=0) <= 0:
                break
            destination = max(range(num_ranks), key=lambda r: (rank_quotas[r], -r))
            local_expert = max(
                range(num_local_experts), key=lambda e: (remaining[e], -e),
            )
            moved = min(rank_quotas[destination], remaining[local_expert])
            allocation_rows[start + local_expert][destination] += moved
            allocation_rows[start + local_expert][placing_rank] -= moved
            remaining[local_expert] -= moved
            rank_quotas[destination] -= moved
    allocations = torch.tensor(allocation_rows, dtype=torch.int64).reshape(
        num_experts, num_ranks,
    )
    # Stable sorting reversed IDs makes equal quotas prefer higher expert IDs.
    expert_ids = torch.arange(num_experts - 1, -1, -1)
    remote_counts = allocations.T.flip(dims=(1,))
    native = expert_ids[None, :] // num_local_experts == torch.arange(num_ranks)[:, None]
    remote_counts.masked_fill_(native, -1)
    selected = remote_counts.argsort(dim=1, descending=True, stable=True)[
        :, :num_local_experts
    ]
    copies = num_experts - 1 - selected
    copies.masked_fill_(remote_counts.gather(1, selected) <= 0, -1)

    return allocations, copies


def count_routes(
    counts: torch.Tensor,
    allocations: torch.Tensor,
    copies: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive one rank's sent splits and per-slot received counts from placement.

    Args:
        counts: CPU int64 tensor of shape [num_ranks, num_experts].
        allocations: CPU int64 tensor of shape [num_experts, num_ranks].
        copies: CPU int64 tensor of shape [num_ranks, num_local_experts].
        rank: Rank in [0, num_ranks) to query.

    Returns:
        A tuple ``(sent, received)`` where ``sent`` has shape [num_ranks] and
        ``received`` has shape [num_ranks, 2 * num_local_experts].
    """
    counts = counts.to(dtype=torch.int64)
    allocations = allocations.to(dtype=torch.int64)
    copies = copies.to(dtype=torch.int64)
    num_ranks, num_local_experts = copies.shape

    prefix_sources = counts.cumsum(dim=0) - counts
    prefix_targets = allocations.cumsum(dim=1) - allocations

    src_start = prefix_sources[rank, :, None]
    src_end = src_start + counts[rank, :, None]
    dst_start = prefix_targets
    dst_end = dst_start + allocations
    sent = (
        torch.minimum(src_end, dst_end) - torch.maximum(src_start, dst_start)
    ).clamp_min_(0).sum(dim=0)

    received = torch.zeros(
        (num_ranks, 2 * num_local_experts), dtype=torch.int64,
    )
    dst_e_start = prefix_targets[:, rank]
    dst_e_end = dst_e_start + allocations[:, rank]
    overlaps = (
        torch.minimum(prefix_sources + counts, dst_e_end[None, :])
        - torch.maximum(prefix_sources, dst_e_start[None, :])
    ).clamp_min_(0)
    local_start = rank * num_local_experts
    received[:, :num_local_experts] = overlaps[
        :, local_start : local_start + num_local_experts
    ]
    replica_experts = copies[rank]
    active = replica_experts >= 0
    received[:, num_local_experts:] = overlaps[:, replica_experts.clamp_min(0)] * active
    return sent, received


def build_remap_table(
    counts: torch.Tensor,
    allocations: torch.Tensor,
    copies: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """Pack one source rank's view of a CPU placement for ops.moe_remap_routes.

    Args:
        counts: CPU int64 tensor of shape [num_ranks, num_experts].
        allocations: CPU int64 tensor of shape [num_experts, num_ranks].
        copies: CPU int64 tensor of shape [num_ranks, num_local_experts].
        source_rank: Rank in [0, num_ranks) whose routes are being remapped.

    Returns:
        CPU int64 tensor of shape [num_experts, 1 + 2 * num_ranks].
    """
    counts = counts.to(dtype=torch.int64)
    allocations = allocations.to(dtype=torch.int64)
    copies = copies.to(dtype=torch.int64)
    num_ranks, num_local_experts = copies.shape
    num_experts = num_ranks * num_local_experts
    table = torch.full(
        (num_experts, 1 + 2 * num_ranks), -1, dtype=torch.int64,
    )
    local_counts = counts[source_rank]
    local_starts = local_counts.cumsum(dim=0) - local_counts
    source_prefix = counts[:source_rank].sum(dim=0)
    table[:, 0] = source_prefix - local_starts
    table[:, 1 : 1 + num_ranks] = allocations.cumsum(dim=1)
    expert_ids = torch.arange(num_experts)
    table[expert_ids, 1 + num_ranks + expert_ids // num_local_experts] = (
        expert_ids % num_local_experts
    )
    replica_slots = torch.arange(num_local_experts)[:, None] + num_local_experts
    replica_slots = replica_slots.expand(num_local_experts, num_ranks)
    replica_slots = replica_slots.masked_fill(copies.T < 0, -1)
    # Maximum slot preserves the final write if a copy table repeats an expert.
    table[:, 1 + num_ranks:].scatter_reduce_(
        0, copies.T.clamp_min(0), replica_slots, reduce="amax", include_self=True,
    )
    return table


@register_op(
    op_name="plan_virtual_expert_routes",
    impl_name="torch_plan_virtual_expert_routes",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchPlanVirtualExpertRoutes(OpImpl):
    def __call__(
        self,
        top_indices: torch.Tensor,
        num_experts: int,
        ep_group: dist.ProcessGroup,
    ) -> VirtualExpertPlan:
        """Plan global virtual-expert placement and remap local routes."""
        ep_size = dist.get_world_size(ep_group)
        num_local_experts = num_experts // ep_size
        if ep_size == 1:
            experts_to_copy = torch.full((1, num_local_experts), -1, dtype=torch.int64)
            context = build_weight_exchange_context(experts_to_copy, ep_group)
            return VirtualExpertPlan(
                top_indices.to(torch.int64), experts_to_copy, None, context, False,
            )

        ep_rank = dist.get_rank(ep_group)
        route_positions, local_counts = ops.moe_route_metadata(top_indices, num_experts)
        global_counts = torch.empty(
            ep_size * num_experts, dtype=local_counts.dtype, device=local_counts.device,
        )
        dist.all_gather_into_tensor(global_counts, local_counts, group=ep_group)
        counts = global_counts.reshape(ep_size, num_experts).to("cpu", dtype=torch.int64)

        allocations, experts_to_copy = plan_placement(counts)
        sent, received = count_routes(counts, allocations, experts_to_copy, ep_rank)
        remap_table = build_remap_table(
            counts, allocations, experts_to_copy, ep_rank,
        ).to(device=top_indices.device)
        mapped, order = ops.moe_remap_routes(top_indices, route_positions, remap_table)
        mapped = mapped.to(torch.int64)
        virtual_experts = torch.empty_like(mapped).scatter_(1, order, mapped)
        dispatch_context = HybridEPMetadata(
            received_counts=received,
            tokens_per_expert=received.sum(0),
            input_splits=sent.tolist(),
            output_splits=received.sum(1).tolist(),
            sorted_expert_indices=mapped,
            route_order=order,
        )
        weight_context = build_weight_exchange_context(experts_to_copy, ep_group)
        return VirtualExpertPlan(
            virtual_experts, experts_to_copy, dispatch_context, weight_context,
            weight_context.has_global_replicas,
        )
