"""Communication metadata using AscendC and existing NPU reductions."""

import torch

from .route import RouteMetadata
from .spec import MegaMoeSpec


def _native_metadata(
    counts: torch.Tensor,
    matrix: torch.Tensor,
    received: torch.Tensor,
    spec: MegaMoeSpec,
    capacity: int,
) -> RouteMetadata:
    ranks, experts = matrix.shape
    local, rank, hidden = experts // ranks, spec.rank_id, spec.hidden_size
    rows = matrix.cumsum(1, dtype=torch.int64) - matrix
    sources = matrix.cumsum(0, dtype=torch.int64)
    totals = sources[-1]
    sources = sources - matrix
    groups = totals.reshape(ranks, local).cumsum(1)
    prefix = (groups - totals.reshape(ranks, local)).flatten()
    start, end = rank * local, (rank + 1) * local
    combine_src = ((sources[:, start:end] + prefix[start:end]) * hidden).flatten()
    combine_target = (rows[:, start:end] * hidden).flatten()
    combine_size = (received * hidden).flatten()
    return RouteMetadata(
        dispatch_src_off=combine_target if spec.dispatch_mode == "pull" else rows[rank] * hidden,
        dispatch_target_off=combine_src if spec.dispatch_mode == "pull" else (prefix + sources[rank]) * hidden,
        dispatch_size=combine_size if spec.dispatch_mode == "pull" else counts * hidden,
        combine_src_off=combine_src, combine_target_off=combine_target,
        combine_size=combine_size, group_list=groups[rank].contiguous(), expert_capacity=capacity,
    )


def compute_metadata(
    counts: torch.Tensor, counts_by_source: torch.Tensor,
    received_counts: torch.Tensor, spec: MegaMoeSpec, expert_capacity: int,
) -> RouteMetadata:
    """Build the standalone operator offsets using PyTorch tensor operations."""
    return _native_metadata(counts, counts_by_source, received_counts, spec, expert_capacity)


def route_summary(counts: torch.Tensor) -> list[int]:
    """Reduce nonnegative [source, expert] counts with INT64 accumulation."""
    if counts.ndim != 2:
        raise ValueError("route_summary expects a [source, expert] count matrix")
    p, e = counts.shape
    if p == 0 or e == 0 or e % p:
        raise ValueError("route_summary requires positive dimensions and experts divisible by ranks")
    if counts.dtype not in (torch.int32, torch.int64):
        raise TypeError("route_summary requires int32 or int64 counts")
    if counts.device.type != "npu":
        raise ValueError("route_summary requires NPU counts")
    totals = counts.sum(0, dtype=torch.int64)
    return torch.cat((
        totals.reshape(p, e // p).sum(1),
        totals.max().reshape(1),
        counts.max().to(torch.int64).reshape(1),
        (totals > 0).sum().reshape(1),
    )).tolist()


def select_nominal_schedule(
    values: list[int], tokens: int, topk: int, experts: int, ranks: int,
    dispatch_split: int = 128, activation_split: int = 128,
) -> bool:
    """Use fewer partitions when each lane covers at most two nominal tiles."""
    slots = tokens * topk
    communication_lanes = max(1, (slots + experts * dispatch_split - 1) // (experts * dispatch_split))
    local = experts // ranks
    activation_lanes = max(1, (slots + local * activation_split - 1) // (local * activation_split))
    return (values[ranks + 1] <= 2 * dispatch_split * communication_lanes
            and values[ranks] <= 2 * activation_split * activation_lanes)
