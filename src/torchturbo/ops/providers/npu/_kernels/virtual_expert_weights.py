# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Adapted from Megatron-LM's
# megatron/core/transformer/moe/ops/virtual_expert_triton.py.

"""Direct Ascend SHMEM transport, imported only when the NPU provider executes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _push_fc_layer(
    weights, copies, peer_ptrs, entries, mine, ordinal, active, program, rank,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    ELEMENTS: tl.constexpr, ARENA_OFFSET: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr, BLOCK: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (ELEMENTS + BLOCK - 1) // BLOCK
    # Divide active replicas into contiguous segments to fill the device even
    # when the plan contains only one copy. Adjacent programs vary the peer first.
    safe_active = tl.maximum(active, 1)
    segments = tl.maximum(NUM_PROGRAMS // safe_active, 1)
    lanes = tl.arange(0, BLOCK)
    for unit in range(program, active * segments, NUM_PROGRAMS):
        replica = unit % safe_active
        segment = unit // safe_active
        entry = tl.sum(tl.where(mine & (ordinal == replica), entries, 0), axis=0)
        peer = entry // NUM_LOCAL_EXPERTS
        slot = (entry % NUM_LOCAL_EXPERTS).to(tl.int64)
        expert = tl.load(copies + entry) - rank * NUM_LOCAL_EXPERTS
        arena = tl.load(peer_ptrs + peer).to(tl.pointer_type(weights.dtype.element_ty))
        source = weights + expert * ELEMENTS
        destination = arena + ARENA_OFFSET + slot * ELEMENTS
        start = segment * NUM_TILES // segments
        end = (segment + 1) * NUM_TILES // segments
        for tile in range(start, tl.minimum(end, ELEMENTS // BLOCK)):
            tile_base = tile.to(tl.int64) * BLOCK
            # Preserve contiguous offsets so Ascend can use DMA transfers.
            values = tl.load(source + tile_base + lanes)
            tl.store(destination + tile_base + lanes, values)
        if ELEMENTS % BLOCK != 0:
            if (start <= ELEMENTS // BLOCK) & (ELEMENTS // BLOCK < end):
                tile_base = (ELEMENTS // BLOCK) * BLOCK
                valid = lanes < ELEMENTS % BLOCK
                offsets = tl.where(valid, lanes, 0)
                values = tl.load(source + tile_base + offsets, valid, other=0)
                tl.store(destination + tile_base + offsets, values, valid)


# Keep rank dynamic so all EP ranks share one compiled transport kernel.
@triton.jit(do_not_specialize=["rank"])
def _push_weights_kernel(
    weight1, weight2, copies, peer_ptrs, rank,
    EP_SIZE: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
    FC1_ELEMENTS: tl.constexpr, FC2_ELEMENTS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr, BLOCK_PLAN: tl.constexpr,
    FC1_BLOCK: tl.constexpr, FC2_BLOCK: tl.constexpr,
):
    peer_ptrs = peer_ptrs.to(tl.pointer_type(tl.int64))
    program = tl.program_id(0)
    entries = tl.arange(0, BLOCK_PLAN)
    planned = entries < EP_SIZE * NUM_LOCAL_EXPERTS
    # Keep masked addresses inside the allocation on Ascend.
    load_entries = tl.where(planned, entries, 0)
    experts = tl.load(copies + load_entries, planned, other=-1)
    mine = planned & (experts >= rank * NUM_LOCAL_EXPERTS)
    mine &= experts < (rank + 1) * NUM_LOCAL_EXPERTS
    ordinal = tl.cumsum(mine.to(tl.int32), axis=0) - 1
    active = tl.sum(mine.to(tl.int32), axis=0)

    # Compact the plan once, then sweep each FC section with its own geometry.
    if FC1_ELEMENTS > 0:
        _push_fc_layer(
            weight1, copies, peer_ptrs, entries, mine, ordinal, active, program,
            rank, NUM_LOCAL_EXPERTS, FC1_ELEMENTS, 0, NUM_PROGRAMS, FC1_BLOCK,
        )
    if FC2_ELEMENTS > 0:
        _push_fc_layer(
            weight2, copies, peer_ptrs, entries, mine, ordinal, active, program,
            rank, NUM_LOCAL_EXPERTS, FC2_ELEMENTS,
            NUM_LOCAL_EXPERTS * FC1_ELEMENTS, NUM_PROGRAMS, FC2_BLOCK,
        )


@triton.jit
def _reduce_fc_tile(
    gradients, peer_buffer, peer_ptrs, entries, sources, ordinal, num_sources, expert, tile_base,
    NUM_LOCAL_EXPERTS: tl.constexpr, ELEMENTS: tl.constexpr,
    ARENA_OFFSET: tl.constexpr, BLOCK: tl.constexpr, IS_TAIL: tl.constexpr,
    INCLUDE_NATIVE: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    destination = gradients + expert.to(tl.int64) * ELEMENTS + tile_base
    if IS_TAIL:
        valid = lanes < ELEMENTS % BLOCK
        offsets = tl.where(valid, lanes, 0)
    if INCLUDE_NATIVE:
        if IS_TAIL:
            accumulated = tl.load(destination + offsets, valid, other=0).to(tl.float32)
        else:
            accumulated = tl.load(destination + lanes).to(tl.float32)
    else:
        accumulated = tl.zeros((BLOCK,), tl.float32)
    # Start from the owner gradient, then add peer slots in destination order.
    for source in range(num_sources):
        entry = tl.sum(tl.where(sources & (ordinal == source), entries, 0), axis=0)
        peer = entry // NUM_LOCAL_EXPERTS
        slot = (entry % NUM_LOCAL_EXPERTS).to(tl.int64)
        arena = tl.load(peer_ptrs + peer).to(tl.pointer_type(peer_buffer.dtype.element_ty))
        address = arena + ARENA_OFFSET + slot * ELEMENTS + tile_base
        if IS_TAIL:
            values = tl.load(address + offsets, valid, other=0)
        else:
            values = tl.load(address + lanes)
        accumulated += values.to(tl.float32)
    if IS_TAIL:
        tl.store(destination + offsets, accumulated.to(gradients.dtype.element_ty), valid)
    else:
        tl.store(destination + lanes, accumulated.to(gradients.dtype.element_ty))


@triton.jit(do_not_specialize=["rank"])
def _reduce_gradients_kernel(
    gradients, peer_buffer, copies, peer_ptrs, rank,
    EP_SIZE: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
    ELEMENTS: tl.constexpr, ARENA_OFFSET: tl.constexpr,
    NUM_TILES: tl.constexpr, PROGRAMS_PER_EXPERT: tl.constexpr,
    BLOCK_PLAN: tl.constexpr, BLOCK: tl.constexpr, INCLUDE_NATIVE: tl.constexpr,
    SPARSE: tl.constexpr,
):
    peer_ptrs = peer_ptrs.to(tl.pointer_type(tl.int64))
    program = tl.program_id(0)
    group = program // PROGRAMS_PER_EXPERT
    segment = program % PROGRAMS_PER_EXPERT
    entries = tl.arange(0, BLOCK_PLAN)
    planned = entries < EP_SIZE * NUM_LOCAL_EXPERTS
    load_entries = tl.where(planned, entries, 0)
    experts = tl.load(copies + load_entries, planned, other=-1)
    if SPARSE:
        # Map each group to an owner via a materialized plan entry. If multiple
        # peers copied that owner, only its first entry reduces all contributors.
        mine = planned & (experts >= rank * NUM_LOCAL_EXPERTS)
        mine &= experts < (rank + 1) * NUM_LOCAL_EXPERTS
        selected = mine & (tl.cumsum(mine.to(tl.int32), axis=0) == group + 1)
        entry = tl.sum(tl.where(selected, entries, 0), axis=0)
        expert = tl.sum(tl.where(selected, experts, 0), axis=0) - rank * NUM_LOCAL_EXPERTS
    else:
        expert = group
    sources = planned & (experts == rank * NUM_LOCAL_EXPERTS + expert)
    ordinal = tl.cumsum(sources.to(tl.int32), axis=0) - 1
    num_sources = tl.sum(sources.to(tl.int32), axis=0)

    start = segment * NUM_TILES // PROGRAMS_PER_EXPERT
    end = (segment + 1) * NUM_TILES // PROGRAMS_PER_EXPERT
    if SPARSE:
        first = tl.min(tl.where(sources, entries, BLOCK_PLAN), axis=0)
        end = tl.where(entry == first, end, start)
    elif INCLUDE_NATIVE:
        # The exchange-only backward starts from an empty output and must write zeros.
        end = tl.where(num_sources > 0, end, start)
    for tile in range(start, tl.minimum(end, ELEMENTS // BLOCK)):
        tile_base = tile.to(tl.int64) * BLOCK
        _reduce_fc_tile(
            gradients, peer_buffer, peer_ptrs, entries, sources, ordinal, num_sources,
            expert, tile_base, NUM_LOCAL_EXPERTS, ELEMENTS, ARENA_OFFSET,
            BLOCK, False, INCLUDE_NATIVE,
        )
    if ELEMENTS % BLOCK != 0:
        if (start <= ELEMENTS // BLOCK) & (ELEMENTS // BLOCK < end):
            _reduce_fc_tile(
                gradients, peer_buffer, peer_ptrs, entries, sources, ordinal, num_sources,
                expert, (ELEMENTS // BLOCK) * BLOCK, NUM_LOCAL_EXPERTS,
                ELEMENTS, ARENA_OFFSET, BLOCK, True, INCLUDE_NATIVE,
            )


def push_weights(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    copies: torch.Tensor,
    peer_ptrs: int,
    rank: int,
    num_local_experts: int,
) -> None:
    """Push contiguous native FC matrices into the plan's peer replica slots.

    The int64 device tables contain copies shaped [EP, local_experts] and peer
    arena addresses shaped [EP]. Each arena holds all FC1 slots followed by all
    FC2 slots, in the weights' common dtype. The caller synchronizes peer access
    and keeps the source weights, tables, and arenas alive until completion.
    """
    ep_size = copies.shape[0]
    block_plan = triton.next_power_of_2(ep_size * num_local_experts)
    elements1 = weight1.numel() // num_local_experts
    elements2 = weight2.numel() // num_local_experts
    if elements1 + elements2 == 0:
        return
    # Leave UB space for plan vectors and compiler-generated buffering.
    block1 = min(4096, triton.next_power_of_2(max(elements1, 1)))
    block2 = min(4096, triton.next_power_of_2(max(elements2, 1)))
    num_tiles = max(triton.cdiv(elements1, block1), triton.cdiv(elements2, block2))
    num_programs = min(32, ep_size * num_local_experts * num_tiles)
    _push_weights_kernel[(num_programs,)](
        weight1, weight2, copies, peer_ptrs, rank, ep_size, num_local_experts,
        elements1, elements2, num_programs, block_plan, block1, block2,
    )


def reduce_fc_gradients(
    gradients: torch.Tensor,
    grad_buffer: torch.Tensor,
    copies: torch.Tensor,
    peer_ptrs: int,
    rank: int,
    arena_offset: int,
    *,
    include_native: bool,
    num_local_copies: int | None = None,
) -> None:
    """Reduce published peer slots in FP32, optionally including the owner gradient."""
    num_local_experts = gradients.shape[0]
    elements = gradients.numel() // num_local_experts
    sparse = include_native and num_local_copies is not None and num_local_copies < num_local_experts
    groups = num_local_copies if sparse else num_local_experts
    if elements == 0 or groups == 0:
        return
    block = min(4096, triton.next_power_of_2(elements))
    num_tiles = triton.cdiv(elements, block)
    programs_per_expert = min(32, num_tiles)
    _reduce_gradients_kernel[(groups * programs_per_expert,)](
        gradients, grad_buffer, copies, peer_ptrs, rank, copies.shape[0], num_local_experts,
        elements, arena_offset, num_tiles, programs_per_expert,
        triton.next_power_of_2(copies.shape[0] * num_local_experts), block, include_native, sparse,
    )
