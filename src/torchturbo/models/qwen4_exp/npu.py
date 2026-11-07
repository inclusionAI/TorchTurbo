"""CANN LightningIndexerV2 integration for Qwen3.8-flash-next QSA indexer.

The torch pre-processing retains block pooling, RMSNorm, and RoPE.  Each packed
sample is processed independently so sequence-parallel queries use their
segment-local causal prefixes.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from ..modules.comm.comm import gather_from_sequence_parallel_region
from .modeling_qwen4_exp import apply_rotary_pos_emb


def _import_cann_ops():
    """Import the CANN operator package, preferring the compiled sub-package."""
    try:
        import cann_ops_transformer_qwen38liv2 as ops
    except ImportError:
        import cann_ops_transformer as ops
    return ops


def has_cann_liv2() -> bool:
    """Return True when the LightningIndexerV2 CANN op is importable."""
    try:
        ops = _import_cann_ops()
        return callable(ops.lightning_indexer)
    except (ImportError, AttributeError, OSError):
        return False


def _compressed_index_keys(
    raw_keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_norm: torch.nn.Module,
    compress_ratio: int,
) -> torch.Tensor:
    """Pool raw keys into per-block representatives and apply norm + RoPE."""
    batch, kv_length, head_dim = raw_keys.shape
    blocks = kv_length // compress_ratio
    grouped = raw_keys[:, : blocks * compress_ratio].reshape(
        batch, blocks, compress_ratio, head_dim
    )
    pooled = grouped.float().mean(dim=2).to(raw_keys.dtype)
    normalised = k_norm(pooled)
    block_cos = cos[:, : blocks * compress_ratio : compress_ratio]
    block_sin = sin[:, : blocks * compress_ratio : compress_ratio]
    return apply_rotary_pos_emb(
        normalised.unsqueeze(2),
        cos=block_cos,
        sin=block_sin,
        unsqueeze_dim=2,
    ).squeeze(2).to(normalised.dtype).contiguous()


def _blocks_to_token_ids(
    block_ids: torch.Tensor,
    segment_start: int,
    query_positions: torch.Tensor,
    ratio: int,
    budget: int,
) -> torch.Tensor:
    """Match the torch dense-prefix and causal-tail output contract."""
    local_positions = query_positions - segment_start
    complete_blocks = (local_positions + 1).div(ratio, rounding_mode="floor")
    valid_blocks = (block_ids >= 0) & (block_ids < complete_blocks[:, None])
    output = torch.full(
        (1, block_ids.shape[0], budget + ratio - 1),
        -1,
        dtype=torch.int32,
        device=block_ids.device,
    )
    block_offsets = torch.arange(ratio, device=block_ids.device)
    token_ids = (
        segment_start + block_ids.unsqueeze(-1) * ratio + block_offsets
    ).to(torch.int32)
    token_ids.masked_fill_(~valid_blocks.unsqueeze(-1), -1)
    token_ids = token_ids.flatten(1)
    output[0, :, : token_ids.shape[1]] = token_ids

    tail_offsets = block_offsets[:-1]
    tail_start = complete_blocks * ratio
    tail_length = local_positions + 1 - tail_start
    # Invalid blocks are filtered by the attention KV gather. Place the tail
    # after all complete blocks so it cannot overwrite a valid selection.
    destinations = complete_blocks.clamp(max=block_ids.shape[1])[:, None] * ratio
    destinations = destinations + tail_offsets[None, :]
    tail_ids = segment_start + tail_start[:, None] + tail_offsets[None, :]
    tail_ids = torch.where(
        tail_offsets[None, :] < tail_length[:, None], tail_ids, -1
    ).to(torch.int32)
    output[0].scatter_(1, destinations, tail_ids)
    return output


def _rank_segments(cu_seq_lens, local_length: int, sp_size: int, sp_rank: int):
    """Build local queries and their causal KV prefixes for each packed sample."""
    rank_start = sp_rank * local_length
    rank_stop = rank_start + local_length
    segments = []
    for segment_start, segment_stop in pairwise(cu_seq_lens.tolist()):
        query_start = max(segment_start, rank_start)
        query_stop = min(segment_stop, rank_stop)
        if query_start >= query_stop:
            continue
        segments.append((segment_start, query_start, query_stop))
    return segments


@torch.no_grad()
def qsa_indexer_forward_cann(
    indexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seq_lens: torch.Tensor,
    *,
    sp_group=None,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Return global packed token IDs for local queries using CANN LIv2."""
    ops = _import_cann_ops()
    batch, local_length, _ = hidden_states.shape
    if batch != 1:
        raise ValueError("Packed QSA expects a single flattened batch row.")

    cos, sin = position_embeddings
    head_dim = indexer.index_head_dim
    ratio = indexer.compress_ratio
    qk = indexer.index_qk_proj(hidden_states.detach())
    query, raw_keys = qk.split(
        (indexer.index_n_heads * head_dim, head_dim), dim=-1
    )
    query = indexer.q_layernorm(query.unflatten(-1, (indexer.index_n_heads, head_dim)))
    query = apply_rotary_pos_emb(query, cos=cos, sin=sin, unsqueeze_dim=2).to(raw_keys.dtype)

    if sp_size > 1:
        raw_key_dtype = raw_keys.dtype
        widths = (head_dim, cos.shape[-1], sin.shape[-1])
        selection_state = torch.cat((raw_keys, cos, sin), dim=-1)
        selection_state = gather_from_sequence_parallel_region(
            selection_state.movedim(1, 0).contiguous(),
            tensor_parallel_output_grad=False,
            group=sp_group,
        ).movedim(0, 1)
        raw_keys, cos, sin = selection_state.split(widths, dim=-1)
        raw_keys = raw_keys.to(raw_key_dtype)

    segments = _rank_segments(cu_seq_lens, local_length, sp_size, sp_rank)
    query_offset = sp_rank * local_length
    device = hidden_states.device
    result = torch.full(
        (1, local_length, indexer.token_budget + ratio - 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for segment_start, query_start, query_stop in segments:
        local_start = query_start - query_offset
        local_stop = query_stop - query_offset
        positions = torch.arange(
            query_start,
            query_stop,
            device=device,
            dtype=torch.int32,
        )
        compressed_length = (query_stop - segment_start) // ratio * ratio
        if compressed_length == 0:
            offsets = torch.arange(ratio - 1, device=device)[None, :]
            tail_ids = segment_start + offsets
            tail_ids = torch.where(
                offsets <= (positions - segment_start)[:, None], tail_ids, -1
            ).to(torch.int32)
            result[0, local_start:local_stop, : ratio - 1] = tail_ids
            continue
        segment_query = query[:, local_start:local_stop]
        segment_keys = _compressed_index_keys(
            raw_keys[:, segment_start : segment_start + compressed_length],
            cos[:, segment_start : segment_start + compressed_length],
            sin[:, segment_start : segment_start + compressed_length],
            indexer.k_layernorm,
            ratio,
        )
        kv_length = query_stop - segment_start
        residual_k = torch.tensor([kv_length % ratio], dtype=torch.int32, device=device)
        weights = torch.full(
            (local_stop - local_start, indexer.index_n_heads),
            head_dim ** -0.5,
            dtype=torch.float32,
            device=device,
        )
        selected_blocks, _ = ops.lightning_indexer(
            segment_query.contiguous(),
            segment_keys.unsqueeze(2).contiguous(),
            weights.unsqueeze(0),
            indexer.block_topk,
            cmp_residual_k=residual_k,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=3,
            cmp_ratio=ratio,
            return_value=0,
        )
        result[0, local_start:local_stop] = _blocks_to_token_ids(
            selected_blocks[0, :, 0], segment_start, positions, ratio, indexer.token_budget
        )
    return result
