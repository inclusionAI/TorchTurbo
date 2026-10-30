"""Reuse MoonEP allocation counts in the MegaMoE route preparation."""

import math

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class BalancedRoute:
    """Current-batch routes and their exact source-major counts."""

    virtual_ids: torch.Tensor
    counts_by_source: torch.Tensor
    copies: torch.Tensor
    capacity: int
    extra_experts: int
    maximum_capacity: int
    nominal_schedule: bool = False
    batch_experts: bool = False
    routed_tokens: torch.Tensor | None = None
    unpermute_mapping: torch.Tensor | None = None


def virtual_counts(
    global_counts: torch.Tensor,
    cumulative: torch.Tensor,
    virtual_table: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Intersect source and destination intervals for every logical expert."""
    ranks, _experts = global_counts.shape
    source_ends = global_counts.cumsum(0, dtype=torch.int32)
    source_starts = source_ends - global_counts
    destination_ends = cumulative.T.contiguous()
    destination_starts = torch.cat(
        (torch.zeros_like(destination_ends[:1]), destination_ends[:-1]), dim=0
    )
    overlap = (
        torch.minimum(source_ends[:, None, :], destination_ends[None, :, :])
        - torch.maximum(source_starts[:, None, :], destination_starts[None, :, :])
    ).clamp_min_(0)
    columns = (
        torch.arange(ranks, device=global_counts.device)[:, None] * width
        + virtual_table
    ).reshape(1, -1)
    result = torch.zeros(
        (ranks, ranks * width), device=global_counts.device, dtype=torch.int32
    )
    return result.scatter_add_(1, columns.expand(ranks, -1), overlap.reshape(ranks, -1))


@lru_cache(maxsize=8)
def _routing_scratch(tokens: int, device: torch.device) -> torch.Tensor:
    """Supply a narrow input to native routing while retaining only its indices."""
    return torch.zeros((tokens, 32), dtype=torch.bfloat16, device=device)


@lru_cache(maxsize=16)
def _no_expert_copies(ranks: int, local_experts: int, device: torch.device) -> torch.Tensor:
    """Share the immutable empty placement table across unchanged routes."""
    return torch.full((ranks, local_experts), -1, device=device, dtype=torch.int32)


@torch.no_grad()
def build_route(
    ids: torch.Tensor,
    experts: int,
    group: dist.ProcessGroup,
    allocator: Any,
    minimum_imbalance: float = 1.3,
    extra_width: Callable[[int, int], int] | None = None,
    hidden_states: torch.Tensor | None = None,
) -> BalancedRoute:
    """Plan once, then derive virtual routes and counts without a second gather."""
    import torch_npu

    ranks, rank = dist.get_world_size(group), dist.get_rank(group)
    local_experts = experts // ranks
    flat = ids.reshape(-1)
    count = flat.numel()
    _expanded, inverse, histogram, _unused = torch_npu.npu_moe_init_routing_v2(
        _routing_scratch(ids.shape[0], ids.device) if hidden_states is None else hidden_states,
        ids.to(torch.int32),
        active_num=count,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        row_idx_type=0,
    )
    histogram = histogram.to(torch.int32)
    if ranks == 1:
        gathered = histogram.reshape(1, experts)
    else:
        gathered = torch.empty((ranks, experts), device=ids.device, dtype=torch.int32)
        work = dist.all_gather_into_tensor(gathered, histogram, group=group, async_op=True)
        work.wait()
    from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.metadata import route_summary, select_nominal_schedule

    summary = route_summary(gathered)
    # Read only EP destination loads. Native routing needs the same capacity
    # boundary; expert-copy plans and token permutations remain on the device.
    host_loads = summary[:ranks]
    maximum_load = max(host_loads)
    # Batch sub-tile GEMMs or sparse routes; retain pipelining for dense work.
    batch_experts = summary[ranks] <= 4 * 128 or summary[ranks + 2] <= ranks
    reserved_rows = math.ceil(min(minimum_imbalance, ranks) * count / 128) * 128
    if maximum_load <= reserved_rows:
        # Sparse but long expert queues retain the pipelined kernel. The
        # sparse-route batching rule below is for redistributed workloads.
        batch_experts = summary[ranks] <= 4 * 128 and summary[ranks + 2] > ranks
        copies = _no_expert_copies(ranks, local_experts, ids.device)
        return BalancedRoute(
            flat.reshape_as(ids),
            gathered,
            copies,
            max(1, host_loads[rank]),
            0,
            maximum_load,
            nominal_schedule=select_nominal_schedule(summary, ids.shape[0], ids.shape[1], experts, ranks),
            batch_experts=batch_experts,
            routed_tokens=_expanded if hidden_states is not None else None,
            unpermute_mapping=inverse if hidden_states is not None else None,
        )
    cumulative, _, _, copies = allocator.allocate(gathered, count)
    needed = int((copies >= 0).sum(1).max().item())
    extra = needed if extra_width is None else extra_width(needed, local_experts)
    from .render import render_route

    virtual, counts = render_route(
        flat.reshape_as(ids),
        inverse,
        histogram,
        gathered,
        cumulative,
        copies,
        rank,
        extra,
    )
    return BalancedRoute(virtual, counts, copies, count, extra, count, batch_experts=batch_experts)


def prepare_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    spec: Any,
    workspace: Any,
    planned: BalancedRoute,
) -> Any:
    """Reuse the planner histogram without another count all-gather."""
    from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe import route as native

    assert spec.num_experts == planned.counts_by_source.shape[1]
    assert spec.ep_size == planned.counts_by_source.shape[0]
    assert topk_ids.shape == planned.virtual_ids.shape
    counts_by_source = planned.counts_by_source
    counts = counts_by_source[spec.rank_id]
    start = spec.rank_id * spec.local_experts
    received = counts_by_source[:, start : start + spec.local_experts]
    if spec.dispatch_mode == "pull":
        assert workspace.in_use and workspace.source_buffer is not None
        routed, mapping = native._permute_topk_input_out(
            hidden_states, topk_ids, workspace.source_buffer
        )
    elif planned.routed_tokens is not None:
        workspace.wait_for_reuse()
        routed = planned.routed_tokens
        mapping = planned.unpermute_mapping.reshape(-1).to(torch.int32).contiguous()
    else:
        workspace.wait_for_reuse()
        import torch_npu

        routed, mapping, _histogram, _unused = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids.to(torch.int32),
            active_num=topk_ids.numel(),
            expert_num=spec.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, spec.num_experts],
            row_idx_type=0,
        )
        mapping = mapping.reshape(-1).to(torch.int32).contiguous()
    from ..runtime.modules.mega_moe.metadata import compute_metadata

    metadata = compute_metadata(
        counts, counts_by_source, received, spec, planned.capacity
    )
    return native.PreparedTopKRoute(
        routed,
        mapping,
        counts,
        received,
        metadata,
        maximum_received_slots=planned.maximum_capacity,
    )
