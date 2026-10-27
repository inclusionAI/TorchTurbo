"""Packed QSA with local queries and sequence-parallel key/value gathering."""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch.distributed import ProcessGroup
from torch.utils.checkpoint import checkpoint
from transformers.integrations.flash_attention import get_target_dtype

from ..modules.attention.interface import attention_forward
from ..modules.attention.ulysses import all_gather_sp_with_grad_reduce
from ..modules.comm.comm import gather_from_sequence_parallel_region
from .modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)

_QSA_INDEXER_QUERY_CHUNK_SIZE = 256
# Expanded KV rows are transient: each chunk is checkpointed, so this bounds
# peak memory rather than saved activations. Real head sizes need about 4 MiB
# per query, and small chunks make the varlen kernels launch-bound.
_QSA_ATTENTION_MAX_KV_BYTES = 1 << 30


def qsa_indexer_forward(
    indexer: Qwen4ExpTextQSAIndexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seq_lens: torch.Tensor,
    *,
    sp_group: ProcessGroup | None = None,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Select global token indices for each local packed query.

    Queries retain all indexer heads on their owning rank. Only raw keys and
    their rotary positions are gathered. Blocks start at each packed sample's
    boundary; selection includes complete top-k blocks and the causal tail.

    Returns:
        Int32 indices of shape ``[1, local_length, budget + ratio - 1]``.
        Heads share one selection per query. Unused slots and padding queries
        contain -1.
    """
    batch_size, local_length, _ = hidden_states.shape
    if batch_size != 1:
        raise ValueError("Packed QSA expects a single flattened batch row.")

    head_dim = indexer.index_head_dim
    cos, sin = position_embeddings
    # Do not cache a no-grad input cast that Q/K/V would reuse without gradients.
    query, raw_keys = indexer.index_qk_proj(hidden_states.detach()).split(
        (indexer.index_n_heads * head_dim, head_dim), dim=-1
    )
    query = indexer.q_layernorm(query.unflatten(-1, (indexer.index_n_heads, head_dim)))
    query = apply_rotary_pos_emb(query, cos=cos, sin=sin, unsqueeze_dim=2)

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
        # RoPE can stay FP32 under autocast; pooling must round to the key dtype.
        raw_keys = raw_keys.to(raw_key_dtype)

    ratio = indexer.compress_ratio
    selected_token_ids = torch.full(
        (1, local_length, indexer.token_budget + ratio - 1),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    query_offset = sp_rank * local_length
    token_positions = torch.arange(raw_keys.shape[1], device=hidden_states.device)
    block_offsets = torch.arange(ratio, device=hidden_states.device)
    tail_offsets = block_offsets[:-1]

    for segment_start, segment_stop in itertools.pairwise(cu_seq_lens.tolist()):
        query_start = max(segment_start, query_offset)
        query_stop = min(segment_stop, query_offset + local_length)
        if query_start >= query_stop:
            continue

        # Later blocks cannot be visible to any query owned by this rank.
        num_blocks = (query_stop - segment_start) // ratio
        if num_blocks:
            block_stop = segment_start + num_blocks * ratio
            pooled_keys = raw_keys[:, segment_start:block_stop].unflatten(
                1, (num_blocks, ratio)
            )
            pooled_keys = indexer.k_layernorm(
                pooled_keys.float().mean(dim=2).to(raw_keys.dtype)
            )
            block_starts = token_positions[segment_start:block_stop:ratio]
            block_keys = apply_rotary_pos_emb(
                pooled_keys.unsqueeze(2),
                cos=cos.index_select(1, block_starts),
                sin=sin.index_select(1, block_starts),
                unsqueeze_dim=2,
            )[0, :, 0]

        for chunk_start in range(
            query_start, query_stop, _QSA_INDEXER_QUERY_CHUNK_SIZE
        ):
            chunk_stop = min(chunk_start + _QSA_INDEXER_QUERY_CHUNK_SIZE, query_stop)
            local_start = chunk_start - query_offset
            local_stop = chunk_stop - query_offset
            query_positions = token_positions[chunk_start:chunk_stop] - segment_start
            complete_blocks = (query_positions + 1) // ratio
            chunk_ids = selected_token_ids[0, local_start:local_stop]
            num_selected_blocks = torch.zeros_like(complete_blocks)
            num_candidate_blocks = (chunk_stop - segment_start) // ratio

            if num_candidate_blocks:
                scores = torch.matmul(
                    query[0, local_start:local_stop].float(),
                    block_keys[:num_candidate_blocks].float().transpose(0, 1),
                )
                scores = torch.relu(scores).sum(dim=1) / math.sqrt(head_dim)
                visible_blocks = (
                    token_positions[:num_candidate_blocks] < complete_blocks[:, None]
                )
                scores.masked_fill_(~visible_blocks, torch.finfo(scores.dtype).min)
                topk = min(indexer.block_topk, num_candidate_blocks)
                block_ids = scores.topk(topk, dim=-1).indices
                valid_blocks = block_ids < complete_blocks[:, None]
                num_selected_blocks = valid_blocks.sum(dim=1)
                token_ids = (
                    segment_start + block_ids.unsqueeze(-1) * ratio + block_offsets
                ).to(torch.int32)
                token_ids.masked_fill_(~valid_blocks.unsqueeze(-1), -1)
                token_ids = token_ids.flatten(1)
                chunk_ids[:, : token_ids.shape[1]] = token_ids

            tail_start = complete_blocks * ratio
            tail_length = query_positions + 1 - tail_start
            destinations = num_selected_blocks[:, None] * ratio + tail_offsets[None, :]
            tail_ids = segment_start + tail_start[:, None] + tail_offsets[None, :]
            tail_ids = torch.where(
                tail_offsets[None, :] < tail_length[:, None], tail_ids, -1
            ).to(torch.int32)
            chunk_ids.scatter_(1, destinations, tail_ids)

    return selected_token_ids


def _qsa_attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    dropout: float,
    softmax_scale: float | None,
    target_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Pack each query and its selected KV rows into one varlen sequence."""
    batch_size, query_length = query.shape[:2]
    valid_tokens = token_ids >= 0
    lengths = valid_tokens.sum(dim=-1).flatten()
    query_indices = torch.nonzero(lengths, as_tuple=False).flatten()
    # Match HF's FP32 correction even when this chunk has no valid queries.
    output_dtype = query.dtype
    if query.dtype == torch.float32 and target_dtype is not None:
        output_dtype = target_dtype
    output = query.new_zeros(
        batch_size * query_length, query.shape[2], value.shape[-1], dtype=output_dtype
    )
    if query_indices.numel() == 0:
        # Padding-only ranks must still join the K/V backward collective.
        zero = query[:, :0].sum() + key[:, :0].sum() + value[:, :0].sum()
        return output.view(batch_size, query_length, *output.shape[1:]) + zero.to(
            output_dtype
        )

    batch_offsets = (
        torch.arange(batch_size, device=token_ids.device) * key.shape[1]
    ).view(batch_size, 1, 1)
    key_indices = (token_ids + batch_offsets)[valid_tokens].long()
    selected_query = query.flatten(0, 1).index_select(0, query_indices)
    selected_key = key.flatten(0, 1).index_select(0, key_indices)
    selected_value = value.flatten(0, 1).index_select(0, key_indices)
    lengths = lengths.index_select(0, query_indices).to(torch.int32)
    num_queries = query_indices.numel()
    num_kv_heads = key.shape[2]
    heads_per_kv = query.shape[2] // num_kv_heads
    # Queries sharing one KV head are independent softmax rows. Folding those
    # heads into the query sequence preserves every score and KV derivative.
    selected_query = selected_query.unflatten(1, (num_kv_heads, heads_per_kv))
    selected_query = selected_query.transpose(1, 2).flatten(0, 1)
    cu_query = torch.arange(
        0, (num_queries + 1) * heads_per_kv, heads_per_kv,
        dtype=torch.int32, device=query.device,
    )
    cu_key = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))

    selected_output = attention_forward(
        selected_query.unsqueeze(0),
        selected_key.unsqueeze(0),
        selected_value.unsqueeze(0),
        None,
        selected_query.shape[0],
        False,
        dropout=dropout,
        softmax_scale=softmax_scale,
        target_dtype=target_dtype,
        cu_seq_lens_q=cu_query,
        cu_seq_lens_k=cu_key,
        max_length_q=heads_per_kv,
        max_length_k=token_ids.shape[-1],
        attn_implementation="flash_attention_2",
    )
    selected_output = selected_output.reshape(
        num_queries, heads_per_kv, num_kv_heads, value.shape[-1]
    ).transpose(1, 2).flatten(1, 2)
    output = output.index_copy(0, query_indices, selected_output)
    return output.view(batch_size, query_length, *output.shape[1:])


def qsa_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    qsa_token_ids: torch.Tensor,
    *,
    dropout: float = 0.0,
    softmax_scale: float | None = None,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run local queries against selected global KV rows in BSND layout.

    Selection already enforces causality and sample boundaries. Each query
    and its selected KV rows form an independent varlen sequence. Selected-row
    chunks are checkpointed to bound expanded KV activations.
    """
    key_bytes = key_states.shape[2] * key_states.shape[3] * key_states.element_size()
    value_bytes = (
        value_states.shape[2] * value_states.shape[3] * value_states.element_size()
    )
    tokens_per_query = query_states.shape[0] * qsa_token_ids.shape[-1]
    bytes_per_query = tokens_per_query * (key_bytes + value_bytes)
    chunk_size = max(1, _QSA_ATTENTION_MAX_KV_BYTES // bytes_per_query)
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query_states, key_states, value_states)
    )
    outputs = []
    for start in range(0, query_states.shape[1], chunk_size):
        query = query_states[:, start : start + chunk_size]
        token_ids = qsa_token_ids[:, start : start + chunk_size]
        if needs_grad:
            output = checkpoint(
                _qsa_attention_chunk,
                query,
                key_states,
                value_states,
                token_ids,
                use_reentrant=False,
                dropout=dropout,
                softmax_scale=softmax_scale,
                target_dtype=target_dtype,
            )
        else:
            output = _qsa_attention_chunk(
                query,
                key_states,
                value_states,
                token_ids,
                dropout=dropout,
                softmax_scale=softmax_scale,
                target_dtype=target_dtype,
            )
        outputs.append(output)
    return torch.cat(outputs, dim=1)


def qsa_forward(
    self: Qwen4ExpTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    *,
    sp_group: ProcessGroup | None = None,
    sp_size: int = 1,
    sp_rank: int = 0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep Q local and gather K/V for packed sequence-parallel QSA."""
    cu_seq_lens = kwargs.get("cu_seq_lens_q")
    if past_key_values is not None or cu_seq_lens is None:
        return Qwen4ExpTextAttention.forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    token_ids = qsa_indexer_forward(
        self.indexer,
        hidden_states,
        position_embeddings,
        cu_seq_lens,
        sp_group=sp_group,
        sp_size=sp_size,
        sp_rank=sp_rank,
    )
    input_shape = hidden_states.shape[:-1]
    head_shape = (*input_shape, -1, self.head_dim)
    query, gate = (
        self.q_proj(hidden_states)
        .view(*input_shape, -1, self.head_dim * 2)
        .chunk(2, dim=-1)
    )
    query = self.q_norm(query.reshape(head_shape))
    key = self.k_norm(self.k_proj(hidden_states).view(head_shape))
    value = self.v_proj(hidden_states).view(head_shape)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
    # One collective also keeps backward ordering identical on padding-only ranks.
    key_value = all_gather_sp_with_grad_reduce(
        torch.cat((key, value), dim=-1), 1, sp_group, sp_size
    )
    key, value = key_value.chunk(2, dim=-1)

    output = qsa_attention_forward(
        query,
        key,
        value,
        token_ids,
        dropout=self.attention_dropout if self.training else 0.0,
        softmax_scale=self.scaling,
        target_dtype=get_target_dtype(query, self),
    )
    output = output.reshape(*input_shape, -1)
    output = output * torch.sigmoid(gate.reshape(*input_shape, -1))
    return self.o_proj(output), None
