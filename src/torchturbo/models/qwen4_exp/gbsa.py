"""Packed QSA using CANN GBSA forward and QSAG pattern-zero backward."""

from __future__ import annotations

from functools import lru_cache
import itertools
import math
from typing import Any, Callable

import torch
from torch.autograd.function import once_differentiable

from ...logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_gbsa_ops() -> tuple[Callable, Callable, Callable]:
    """Import optional dependencies only when the GBSA backend is selected."""
    from cann_ops_transformer_qwen4_gbsa.ops.attention.generic_block_sparse_attention import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )
    from cann_ops_transformer_qwen4_gbsag.ops.attention.generic_block_sparse_attention_grad import (
        generic_block_sparse_attention_grad,
    )

    logger.info_rank0("QSA backend: native CANN GBSA forward and QSAG pattern-0 backward.")
    return (
        generic_block_sparse_attention_metadata,
        generic_block_sparse_attention,
        generic_block_sparse_attention_grad,
    )


def _selected_blocks(
    token_ids: torch.Tensor,
    segment_start: int,
    block_size: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complete selected token groups to sorted sample-local blocks.

    The indexer packs complete blocks first and a partial causal block last.
    GBSA residual_block_mode=1 reconstructs that partial block itself, so it
    must not appear in the explicit sparse list.
    """
    budget = token_ids.shape[-1] - block_size + 1
    grouped = token_ids[:, :budget].unflatten(-1, (-1, block_size))
    valid = (grouped >= 0).all(dim=-1)
    blocks = (grouped[..., 0] - segment_start) // block_size
    sentinel = torch.iinfo(torch.int32).max
    blocks = blocks.masked_fill(~valid, sentinel).sort(dim=-1).values
    counts = valid.sum(dim=-1, dtype=torch.int32)
    blocks = blocks.masked_fill(blocks == sentinel, 0).to(torch.int32)
    return (
        blocks.unsqueeze(0).expand(kv_heads, -1, -1).contiguous(),
        counts.unsqueeze(0).expand(kv_heads, -1).contiguous(),
    )


class _GBSAAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        metadata_op, forward_op, backward_op = _load_gbsa_ops()
        q_length, kv_length = query.shape[0], key.shape[0]
        cu_query = torch.tensor([0, q_length], dtype=torch.int64, device=query.device)
        cu_key = torch.tensor([0, kv_length], dtype=torch.int64, device=query.device)
        options = dict(
            cu_seqlens_q=cu_query,
            cu_seqlens_kv=cu_key,
            layout_q="TND",
            layout_kv="TND",
            mask_mode=1,
            quant_mode=0,
            softmax_precision=0,
            residual_block_mode=1,
            is_consistent_topk=False,
            layout_sparse_pattern=4,
        )
        metadata = metadata_op(
            indices, counts, query.shape[1], key.shape[1], query.shape[2],
            [1, block_size], max_seqlen_q=q_length, max_seqlen_kv=kv_length,
            **options,
        )
        output, lse = forward_op(
            query, key, value, indices, counts, [1, block_size],
            metadata=metadata, softmax_scale=scale, return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            query, key, value, output, lse, indices, counts,
            cu_query, cu_key,
        )
        ctx.block_size = block_size
        ctx.scale = scale
        ctx.backward_op = backward_op
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (query, key, value, output, lse, indices, counts,
         cu_query, cu_key) = ctx.saved_tensors
        # Forward uses packed NQT blocks (pattern 4). Pattern 0 backward
        # keeps the same Q-to-KV lists with an explicit singleton batch axis.
        dq, dk, dv = ctx.backward_op(
            query, key, value, dout.contiguous(), output, lse.contiguous(),
            indices.unsqueeze(0), counts.unsqueeze(0), [1, ctx.block_size],
            metadata=None,
            cu_seqlens_q=cu_query,
            cu_seqlens_kv=cu_key,
            layout_q="TND",
            layout_kv="TND",
            layout_sparse_pattern=0,
            softmax_scale=ctx.scale,
            mask_mode=1,
            softmax_precision=0,
            residual_block_mode=1,
            is_consistent_topk=False,
        )
        return dq, dk, dv, None, None, None, None


def qsa_gbsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    block_size: int,
    query_offset: int = 0,
    dropout: float = 0.0,
    softmax_scale: float | None = None,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run packed local Q against global KV without per-query KV expansion.

    Packed samples are separate GBSA calls. A sample crossing an SP boundary
    uses the local query slice and the KV prefix ending at its last local
    query. Right-aligned causality then matches the global query positions.
    The existing outer K/V collective remains responsible for SP gradients.
    """
    if dropout != 0:
        raise ValueError("GBSA QSA does not support attention dropout.")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("GBSA expects BSND Q/K/V with matching K/V shapes.")
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("GBSA QSA requires one flattened packed batch row.")
    if query.shape[-1] != key.shape[-1] or not 1 <= query.shape[-1] <= 256:
        raise ValueError("A3 GBSA requires matching head dimensions in [1, 256].")
    if key.shape[2] == 0 or query.shape[2] % key.shape[2]:
        raise ValueError("GBSA requires query heads divisible by KV heads.")
    if not 1 <= block_size <= 128 or block_size & (block_size - 1):
        raise ValueError("A3 GBSA block_size must be a power of two in [1, 128].")
    budget = token_ids.shape[-1] - block_size + 1
    if budget <= 0 or budget % block_size or budget // block_size > 512:
        raise ValueError("GBSA expects at most 512 complete blocks plus causal-tail slots.")
    if token_ids.shape[:2] != query.shape[:2]:
        raise ValueError("GBSA token selections must match the local query rows.")
    if torch.are_deterministic_algorithms_enabled() and torch.is_grad_enabled():
        raise ValueError("The GBSAG kernel does not support deterministic training.")
    if query.dtype == torch.float32 and target_dtype is not None:
        query, key, value = (tensor.to(target_dtype) for tensor in (query, key, value))
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in (key, value)
    ):
        raise ValueError("GBSA requires matching FP16 or BF16 Q/K/V.")
    scale = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("GBSA softmax_scale must be positive and finite.")
    boundaries = cu_seq_lens.tolist()
    if (not boundaries or boundaries[0] != 0 or boundaries[-1] > key.shape[1]
            or any(start > stop for start, stop in itertools.pairwise(boundaries))):
        raise ValueError("GBSA requires ordered packed boundaries within the global KV rows.")
    if query_offset < 0 or query_offset + query.shape[1] > key.shape[1]:
        raise ValueError("Local query slice must lie within the global sequence.")

    outputs = []
    active_rows = 0
    for segment_start, segment_stop in itertools.pairwise(boundaries):
        query_start = max(segment_start, query_offset)
        query_stop = min(segment_stop, query_offset + query.shape[1])
        if query_start >= query_stop:
            continue
        local_start, local_stop = query_start - query_offset, query_stop - query_offset
        indices, counts = _selected_blocks(
            token_ids[0, local_start:local_stop], segment_start, block_size, key.shape[2],
        )
        output = _GBSAAttention.apply(
            query[0, local_start:local_stop].contiguous(),
            key[0, segment_start:query_stop].contiguous(),
            value[0, segment_start:query_stop].contiguous(),
            indices, counts, block_size, scale,
        )
        outputs.append(output)
        active_rows += local_stop - local_start
    if active_rows < query.shape[1] or not outputs:
        # A padding-only rank must still take part in the outer KV backward.
        zero = key[:, :0].sum() + value[:, :0].sum()
        outputs.append(query[0, active_rows:] * 0 + zero)
    return torch.cat(outputs, dim=0).unsqueeze(0)
