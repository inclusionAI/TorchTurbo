"""Private Ascend Triton kernel for exact quota lookup and top-k sorting."""

import torch
import triton
import triton.language as tl


@triton.jit
def _remap_routes_kernel(
    indices, positions, table, runtime, order,
    num_tokens, topk: tl.constexpr, padded_topk: tl.constexpr,
    num_ranks: tl.constexpr, num_local_experts: tl.constexpr,
    search_steps: tl.constexpr, block_tokens: tl.constexpr, use_key32: tl.constexpr,
    index_stride_t, index_stride_k, position_stride_t, position_stride_k,
    table_stride_e, table_stride_c, key_shift: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64) * block_tokens + tl.arange(0, block_tokens)
    k = tl.arange(0, padded_topk).to(tl.int64)
    mask = (token[:, None] < num_tokens) & (k[None, :] < topk)
    expert = tl.load(
        indices + token[:, None] * index_stride_t + k[None, :] * index_stride_k,
        mask, other=0,
    )
    packed = tl.load(
        positions + token[:, None] * position_stride_t + k[None, :] * position_stride_k,
        mask, other=0,
    ).to(tl.int64)
    table_base = expert.to(tl.int64) * table_stride_e
    ordinal = packed + tl.load(table + table_base, mask, other=0)
    destination = tl.full((block_tokens, padded_topk), 0, tl.int32)
    # Valid ordinals precede the final boundary. Build the upper-bound index
    # from high to low bits, capping probes at the last possible rank.
    for bit in tl.static_range(search_steps):
        step = 1 << (search_steps - 1 - bit)
        column = tl.minimum(destination + step, num_ranks - 1)
        end = tl.load(
            table + table_base + column.to(tl.int64) * table_stride_c,
            mask, other=0,
        )
        destination = tl.where(ordinal >= end, column, destination)
    slot = tl.load(
        table + table_base + (1 + num_ranks + destination.to(tl.int64)) * table_stride_c,
        mask, other=0,
    )
    mapped = destination * (2 * num_local_experts) + slot
    # Include the source column to preserve probability order on equal IDs.
    if use_key32:
        key = (mapped * padded_topk + k[None, :]).to(tl.int32)
        key = tl.where(mask, key, 0x7FFFFFFF)
    else:
        key = tl.where(mask, mapped * padded_topk + k[None, :], 0x7FFFFFFFFFFFFFFF)
    sorted_key = tl.sort(key, descending=False, dim=1)
    output_offset = token[:, None] * topk + k[None, :]
    # Decode positive power-of-two keys exactly; int32 division rounds near INT32_MAX.
    tl.store(runtime + output_offset, sorted_key >> key_shift, mask)
    tl.store(order + output_offset, sorted_key & (padded_topk - 1), mask)


def remap_routes(
    indices: torch.Tensor, route_positions: torch.Tensor, remap_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = torch.empty(indices.shape, device=indices.device, dtype=indices.dtype)
    order = torch.empty(indices.shape, device=indices.device, dtype=torch.int64)
    num_tokens, topk = indices.shape
    if num_tokens and topk:
        num_ranks = (remap_table.shape[1] - 1) // 2
        num_local_experts = remap_table.shape[0] // num_ranks
        padded_topk = triton.next_power_of_2(topk)
        block_tokens = min(32, max(1, 256 // padded_topk))
        # The installed Ascend compiler miscompiles fused int32 width-4 sorts.
        use_key32 = (
            padded_topk != 4
            and 2 * remap_table.shape[0] * padded_topk <= 2**31
        )
        _remap_routes_kernel[(triton.cdiv(num_tokens, block_tokens),)](
            indices, route_positions, remap_table, runtime, order,
            num_tokens, topk, padded_topk, num_ranks, num_local_experts,
            (num_ranks - 1).bit_length(), block_tokens, use_key32,
            *indices.stride(), *route_positions.stride(), *remap_table.stride(),
            padded_topk.bit_length() - 1,
        )
    return runtime, order
