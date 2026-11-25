# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Adapted from Megatron-LM's
# megatron/core/transformer/moe/ops/virtual_expert_triton.py.

"""Staged Ascend kernels, imported only when the NPU planner executes."""

from __future__ import annotations

from math import prod

import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import sort as cann_sort


@triton.jit
def _argmax_lowest(values, ids, valid, SENTINEL: tl.constexpr):
    maximum = tl.max(tl.where(valid, values, -9223372036854775807), axis=0)
    selected = tl.min(tl.where(valid & (values == maximum), ids, SENTINEL), axis=0)
    return maximum, selected


@triton.jit
def _gather_int32(values, indices):
    # Ascend gather supports float32 but not int32; move the bits unchanged.
    bits = tl.gather(values.to(tl.float32, bitcast=True), indices, axis=0)
    return bits.to(tl.int32, bitcast=True)


@triton.jit
def _gather_int64(values, indices):
    low = _gather_int32(values.to(tl.int32), indices)
    high = _gather_int32((values >> 32).to(tl.int32), indices)
    # Ascend sign-extends even through uint32; clear the widened sign bits.
    return (low.to(tl.int64) & 4294967295) | (high.to(tl.int64) << 32)


@triton.jit(do_not_specialize=["NUM_ROUTES", "ROUTES_PER_PROGRAM"])
def _histogram_kernel(
    routes, histograms,
    NUM_ROUTES, NUM_EXPERTS: tl.constexpr,
    ROUTES_PER_PROGRAM, BLOCK_EXPERTS: tl.constexpr,
    ROUTE_TILE: tl.constexpr,
):
    program = tl.program_id(0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    offsets = tl.arange(0, ROUTE_TILE)
    start = program * ROUTES_PER_PROGRAM
    end = tl.minimum(start + ROUTES_PER_PROGRAM, NUM_ROUTES)
    counts = tl.zeros((BLOCK_EXPERTS,), tl.int64)
    for tile in range(start, end, ROUTE_TILE):
        positions = tile + offsets
        valid = positions < end
        ids = tl.load(routes + positions, mask=valid, other=0).to(tl.int32)
        # Triton 3.2 has no histogram mask; remove padding counted in bin zero.
        padding = tl.sum((~valid).to(tl.int32), axis=0)
        counts += tl.histogram(ids, BLOCK_EXPERTS).to(tl.int64) - tl.where(experts == 0, padding, 0)
    tl.store(histograms + program * NUM_EXPERTS + experts, counts, experts < NUM_EXPERTS)


@triton.jit
def _totals_kernel(
    counts, totals, loads, prefixes,
    EP_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr, BLOCK_EP: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr,
):
    owner = tl.program_id(0)
    sources = tl.arange(0, BLOCK_EP)
    local = tl.arange(0, BLOCK_LOCAL)
    experts = owner * NUM_LOCAL_EXPERTS + local
    valid = (sources[:, None] < EP_SIZE) & (local[None, :] < NUM_LOCAL_EXPERTS)
    values = tl.load(
        counts + sources[:, None] * NUM_EXPERTS + experts[None, :],
        mask=valid, other=0,
    ).to(tl.int64)
    expert_totals = tl.sum(values, axis=0)
    tl.store(totals + experts, expert_totals, local < NUM_LOCAL_EXPERTS)
    tl.store(loads + owner, tl.sum(expert_totals, axis=0))
    tl.store(
        prefixes + sources[:, None] * NUM_EXPERTS + experts[None, :],
        tl.cumsum(values, axis=0) - values, valid,
    )


@triton.jit
def _allocation_kernel(
    totals, loads, allocations, boundaries,
    EP_SIZE: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_EP: tl.constexpr, BLOCK_LOCAL: tl.constexpr,
):
    owner = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_EP)
    local = tl.arange(0, BLOCK_LOCAL)
    valid_ranks = ranks < EP_SIZE
    valid_local = local < NUM_LOCAL_EXPERTS
    native_totals = tl.load(
        totals + owner * NUM_LOCAL_EXPERTS + local, valid_local, other=0,
    ).to(tl.int64)
    rank_loads = tl.load(loads + ranks, valid_ranks, other=0).to(tl.int64)
    total = tl.sum(rank_loads, axis=0)
    targets = total // EP_SIZE + (ranks < total % EP_SIZE).to(tl.int64)
    balances = tl.where(valid_ranks, rank_loads - targets, 0)
    quotas = tl.zeros((BLOCK_EP,), tl.int64)
    step = 0
    # A receiver takes its entire deficit from one sender so its replicas fit.
    while (step < EP_SIZE) & (tl.max(balances, axis=0) > 0):
        _, sender = _argmax_lowest(balances, ranks, valid_ranks, BLOCK_EP)
        moved, receiver = _argmax_lowest(-balances, ranks, valid_ranks, BLOCK_EP)
        quotas = tl.where((sender == owner) & (ranks == receiver), moved, quotas)
        balances = tl.where(ranks == sender, balances - moved, balances)
        balances = tl.where(ranks == receiver, 0, balances)
        step += 1

    # Ascend does not provide the int64 2D last-axis broadcast used by [L, EP].
    cells = tl.arange(0, BLOCK_LOCAL * BLOCK_EP)
    cell_experts = cells // BLOCK_EP
    cell_ranks = cells % BLOCK_EP
    cell_totals = tl.load(
        totals + owner * NUM_LOCAL_EXPERTS + cell_experts,
        cell_experts < NUM_LOCAL_EXPERTS, other=0,
    )
    allocation = tl.where(cell_ranks == owner, cell_totals, 0)
    cumulative = tl.where(cell_ranks >= owner, cell_totals, 0)
    remaining = native_totals
    placement_step = 0
    while (placement_step < EP_SIZE + NUM_LOCAL_EXPERTS) & (tl.max(quotas, axis=0) > 0):
        placement_step += 1
        quota, destination = _argmax_lowest(quotas, ranks, valid_ranks, BLOCK_EP)
        count, expert = _argmax_lowest(remaining, local, valid_local, BLOCK_LOCAL)
        moved = tl.minimum(quota, count)
        transfer = tl.where(
            cell_ranks == destination, moved, tl.where(cell_ranks == owner, -moved, 0),
        )
        # Compare before broadcasting: Ascend can miscompile the comparison
        # against flattened, divided expert indices inside this loop.
        expert_cells = tl.broadcast_to(
            local[:, None] == expert, (BLOCK_LOCAL, BLOCK_EP),
        ).reshape(BLOCK_LOCAL * BLOCK_EP)
        allocation += tl.where(expert_cells, transfer, 0)
        prefix_transfer = tl.where(cell_ranks >= destination, moved, 0) - tl.where(cell_ranks >= owner, moved, 0)
        cumulative += tl.where(expert_cells, prefix_transfer, 0)
        remaining = tl.where(local == expert, remaining - moved, remaining)
        quotas = tl.where(ranks == destination, quotas - moved, quotas)
    # Affine addresses avoid scalarized stores and their cross-launch locks.
    offsets = (owner * NUM_LOCAL_EXPERTS + local[:, None]) * EP_SIZE + ranks[None, :]
    valid = valid_local[:, None] & valid_ranks[None, :]
    tl.store(allocations + offsets, allocation.reshape(BLOCK_LOCAL, BLOCK_EP), valid)
    tl.store(boundaries + offsets, cumulative.reshape(BLOCK_LOCAL, BLOCK_EP), valid)


@triton.jit
def _copies_kernel(
    allocations, copies, slots,
    EP_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr, BLOCK_EXPERTS: tl.constexpr,
):
    destination = tl.program_id(0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    native = experts // NUM_LOCAL_EXPERTS == destination
    valid_remote = (experts < NUM_EXPERTS) & ~native
    values = tl.load(
        allocations + experts * EP_SIZE + destination, valid_remote, other=-1,
    ).to(tl.int64)
    expert_slots = tl.full((BLOCK_EXPERTS,), -1, tl.int64)
    copy_ids = tl.full((BLOCK_EXPERTS,), -1, tl.int64)
    for slot in range(NUM_LOCAL_EXPERTS):
        maximum = tl.max(values, axis=0)
        expert = tl.max(tl.where(valid_remote & (values == maximum), experts, -1), axis=0)
        selected = tl.where(maximum > 0, expert, -1)
        copy_ids = tl.where(experts == slot, selected, copy_ids)
        expert_slots = tl.where(experts == selected, slot, expert_slots)
        values = tl.where(experts == expert, -1, values)
    # Write IDs together: the Ascend scalar-store loop can corrupt copy IDs.
    tl.store(
        copies + destination * NUM_LOCAL_EXPERTS + experts, copy_ids,
        experts < NUM_LOCAL_EXPERTS,
    )
    tl.store(slots + experts * EP_SIZE + destination, expert_slots, experts < NUM_EXPERTS)


# Keep rank dynamic so EP ranks share the same compiled planner kernels.
@triton.jit(do_not_specialize=["EP_RANK"])
def _statistics_kernel(
    counts, prefixes, boundaries, copies, statistics,
    EP_RANK, EP_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr, BLOCK_EXPERTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    peer = tl.program_id(0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < NUM_EXPERTS
    # Ascend can touch masked load addresses at an unmapped allocation edge.
    # Keep every lane's address valid while retaining the semantic masks.
    load_experts = tl.where(valid, experts, 0)
    previous_peer = tl.maximum(peer - 1, 0)
    source_start = tl.load(prefixes + EP_RANK * NUM_EXPERTS + load_experts, valid, other=0)
    source_end = source_start + tl.load(counts + EP_RANK * NUM_EXPERTS + load_experts, valid, other=0)
    target_start = tl.load(boundaries + load_experts * EP_SIZE + previous_peer, valid & (peer > 0), other=0)
    target_end = tl.load(boundaries + load_experts * EP_SIZE + peer, valid, other=0)
    overlaps = tl.maximum(tl.minimum(source_end, target_end) - tl.maximum(source_start, target_start), 0)
    tl.store(statistics + peer, tl.sum(overlaps, axis=0))

    slots = tl.arange(0, BLOCK_SLOTS)
    replica_slot = (slots >= NUM_LOCAL_EXPERTS) & (slots < 2 * NUM_LOCAL_EXPERTS)
    load_slots = tl.where(replica_slot, slots - NUM_LOCAL_EXPERTS, 0)
    replica = tl.load(
        copies + EP_RANK * NUM_LOCAL_EXPERTS + load_slots, replica_slot, other=-1,
    )
    experts = tl.where(slots < NUM_LOCAL_EXPERTS, EP_RANK * NUM_LOCAL_EXPERTS + slots, replica)
    active = experts >= 0
    load_experts = tl.where(active, experts, 0)
    previous_rank = tl.maximum(EP_RANK - 1, 0)
    source_start = tl.load(prefixes + peer * NUM_EXPERTS + load_experts, active, other=0)
    source_end = source_start + tl.load(counts + peer * NUM_EXPERTS + load_experts, active, other=0)
    target_start = tl.load(
        boundaries + load_experts * EP_SIZE + previous_rank, active & (EP_RANK > 0), other=0,
    )
    target_end = tl.load(boundaries + load_experts * EP_SIZE + EP_RANK, active, other=0)
    overlaps = tl.maximum(tl.minimum(source_end, target_end) - tl.maximum(source_start, target_start), 0)
    tl.store(
        statistics + EP_SIZE + peer * (2 * NUM_LOCAL_EXPERTS) + slots,
        overlaps, slots < 2 * NUM_LOCAL_EXPERTS,
    )


@triton.jit
def _prefix_maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _sort_route_keys(keys):
    # Encode the packed keys as ordered, normal FP32 values for CANN's vector sort.
    encoded = (keys + 0x00800000).to(tl.float32, bitcast=True)
    return cann_sort(encoded).to(tl.int32, bitcast=True) - 0x00800000


@triton.jit(do_not_specialize=["NUM_ROUTES", "EP_RANK", "ROUTES_PER_PROGRAM"])
def _remap_kernel(
    routes, histograms, prefixes, boundaries, slots, mapped,
    NUM_ROUTES, NUM_EXPERTS: tl.constexpr,
    EP_RANK, EP_SIZE: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr, LOG_EP: tl.constexpr,
    ROUTES_PER_PROGRAM, BLOCK_EXPERTS: tl.constexpr,
    BLOCK_PROGRAMS: tl.constexpr, ROUTE_TILE: tl.constexpr,
):
    program = tl.program_id(0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    programs = tl.arange(0, BLOCK_PROGRAMS)
    earlier = tl.load(
        histograms + programs[:, None] * NUM_EXPERTS + experts[None, :],
        (programs[:, None] < program) & (experts[None, :] < NUM_EXPERTS), other=0,
    )
    running = tl.sum(earlier, axis=0) + tl.load(
        prefixes + EP_RANK * NUM_EXPERTS + experts, experts < NUM_EXPERTS, other=0,
    )
    lanes = tl.arange(0, ROUTE_TILE)
    start = program * ROUTES_PER_PROGRAM
    end = tl.minimum(start + ROUTES_PER_PROGRAM, NUM_ROUTES)
    for tile in range(start, end, ROUTE_TILE):
        positions = tile + lanes
        valid = positions < end
        ids = tl.load(routes + positions, valid, other=0).to(tl.int32)
        # Sort by expert, breaking ties by original lane to preserve route order.
        packed = _sort_route_keys(tl.where(valid, ids, NUM_EXPERTS) * ROUTE_TILE + lanes)
        sorted_ids = packed // ROUTE_TILE
        previous_ids = _gather_int32(sorted_ids, tl.maximum(lanes - 1, 0))
        first = (lanes == 0) | (previous_ids != sorted_ids)
        starts = tl.associative_scan(tl.where(first, lanes, 0), 0, _prefix_maximum)
        # Restore original lanes in the tile so the output store stays contiguous.
        original = packed % ROUTE_TILE
        earlier_in_tile = _sort_route_keys(original * ROUTE_TILE + (lanes - starts)) % ROUTE_TILE
        ordinal = _gather_int64(running, ids) + earlier_in_tile.to(tl.int64)
        destination = tl.zeros((ROUTE_TILE,), tl.int32)
        for bit in tl.static_range(LOG_EP - 1, -1, -1):
            probe = destination + (1 << bit)
            boundary = tl.load(
                boundaries + ids * EP_SIZE + probe - 1,
                valid & (probe < EP_SIZE), other=9223372036854775807,
            )
            destination = tl.where(boundary <= ordinal, probe, destination)
        remote = valid & (destination != ids // NUM_LOCAL_EXPERTS)
        slot = tl.load(slots + ids * EP_SIZE + destination, remote, other=0)
        runtime = destination * (2 * NUM_LOCAL_EXPERTS) + tl.where(
            remote, NUM_LOCAL_EXPERTS + slot, ids % NUM_LOCAL_EXPERTS,
        )
        tl.store(mapped + positions, runtime, valid)
        padding = tl.sum((~valid).to(tl.int32), axis=0)
        running += tl.histogram(ids, BLOCK_EXPERTS).to(tl.int64) - tl.where(experts == 0, padding, 0)


@triton.jit(do_not_specialize=["rank"])
def _publish_histograms_kernel(
    local_counts, peer_ptrs, rank,
    NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr,
):
    peer = tl.program_id(0)
    experts = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = experts < NUM_EXPERTS
    offsets = tl.where(valid, experts, 0)
    counts = tl.load(local_counts + offsets, valid, other=0)
    bases = peer_ptrs.to(tl.pointer_type(tl.int64))
    window = tl.load(bases + peer).to(tl.pointer_type(tl.int64))
    tl.store(window + rank * NUM_EXPERTS + offsets, counts, valid)


def _allocate_scratch(num_experts: int, ep_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate persistent views of the staged kernels' int64 scratch arena."""
    shapes = {
        "program_histogram": (32, num_experts),
        "tokens_per_expert": (num_experts,),
        "global_tokens_per_expert": (num_experts,),
        "balance": (ep_size,),
        "source_prefixes": (ep_size, num_experts),
        "allocation": (num_experts, ep_size),
        "destination_boundaries": (num_experts, ep_size),
        "virtual_expert_slots": (num_experts, ep_size),
        "statistics": (ep_size * (1 + 2 * (num_experts // ep_size)),),
    }
    sizes = [prod(shape) for shape in shapes.values()]
    # An evaluation warmup may allocate scratch needed by later training forwards.
    with torch.inference_mode(False):
        arena = torch.empty(sum(sizes), dtype=torch.int64, device=device)
        return {
            name: section.view(shape)
            for (name, shape), section in zip(shapes.items(), arena.split(sizes), strict=True)
        }


def _place_routes(
    counts: torch.Tensor, rank: int, scratch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return device allocations, copies, boundaries, slots, prefixes and splits."""
    ep_size, num_experts = counts.shape
    num_local_experts = num_experts // ep_size
    block_ep = triton.next_power_of_2(ep_size)
    block_local = triton.next_power_of_2(num_local_experts)
    block_experts = triton.next_power_of_2(num_experts)
    totals = scratch["global_tokens_per_expert"]
    loads = scratch["balance"]
    prefixes = scratch["source_prefixes"]
    allocations = scratch["allocation"]
    boundaries = scratch["destination_boundaries"]
    slots = scratch["virtual_expert_slots"]
    copies = torch.empty((ep_size, num_local_experts), dtype=torch.int64, device=counts.device)
    statistics = scratch["statistics"]
    _totals_kernel[(ep_size,)](
        counts, totals, loads, prefixes, ep_size, num_experts, num_local_experts,
        block_ep, block_local,
    )
    _allocation_kernel[(ep_size,)](
        totals, loads, allocations, boundaries, ep_size, num_local_experts,
        block_ep, block_local,
    )
    _copies_kernel[(ep_size,)](
        allocations, copies, slots, ep_size, num_experts, num_local_experts, block_experts,
    )
    _statistics_kernel[(ep_size,)](
        counts, prefixes, boundaries, copies, statistics, rank, ep_size, num_experts,
        num_local_experts, block_experts, triton.next_power_of_2(2 * num_local_experts),
    )
    return allocations, copies, boundaries, slots, prefixes, statistics


def launch_virtual_expert_planner(
    top_indices: torch.Tensor, workspace, alignment: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange device histograms and remap stable local route occurrences."""
    if alignment != 1:
        raise ValueError("Compact virtual-expert transport requires alignment=1.")
    ep_size, num_experts = workspace.gathered_counts.shape
    if workspace.scratch is None:
        workspace.scratch = _allocate_scratch(num_experts, ep_size, workspace.device)
    scratch = workspace.scratch
    rank = workspace.rank
    routes = top_indices.contiguous()
    num_routes = routes.numel()
    route_tile = 512
    # Keep reduction shapes fixed so changing token counts reuses the kernels.
    histograms = scratch["program_histogram"]
    num_programs = histograms.shape[0]
    routes_per_program = triton.cdiv(num_routes, num_programs * route_tile) * route_tile
    block_experts = triton.next_power_of_2(num_experts)
    _histogram_kernel[(num_programs,)](
        routes, histograms, num_routes, num_experts, routes_per_program,
        block_experts, route_tile,
    )
    local_counts = scratch["tokens_per_expert"]
    torch.sum(histograms, dim=0, out=local_counts)
    gathered = workspace.gathered_counts
    handle = workspace.histogram_handle
    # Finish all readers of the previous plan before publishing this rank's
    # row into each peer window, then expose complete rows to placement.
    handle.barrier()
    _publish_histograms_kernel[(ep_size, triton.cdiv(num_experts, 256))](
        local_counts, handle.buffer_ptrs_dev, rank, num_experts, 256,
    )
    handle.barrier()
    _, copies, boundaries, slots, prefixes, _ = _place_routes(
        gathered, rank, scratch,
    )
    mapped = torch.empty(top_indices.shape, dtype=torch.int64, device=routes.device)
    if num_routes:
        _remap_kernel[(num_programs,)](
            routes, histograms, prefixes, boundaries, slots, mapped,
            num_routes, num_experts, rank, ep_size, num_experts // ep_size,
            (ep_size - 1).bit_length(), routes_per_program, block_experts,
            triton.next_power_of_2(num_programs), route_tile,
        )
    return mapped, copies
