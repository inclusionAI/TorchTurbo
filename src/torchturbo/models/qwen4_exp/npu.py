"""CANN LightningIndexerV2 integration for Qwen3.8-flash-next QSA indexer.

Replaces the torch small-op indexer path with a fused CANN kernel when the
operator package is available.  The adapter keeps the surrounding
pre-processing (block pool + RMSNorm + RoPE) in torch so that the learned
``k_layernorm`` weight and the rotary schedule stay unchanged.
"""

from __future__ import annotations

import torch

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
    """Pool raw keys into per-block representatives and apply norm + RoPE.

    Follows the same numeric path as the torch indexer: FP32 mean pooling,
    ``k_layernorm`` (RMSNorm), then RoPE at each block's first position.
    """
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
        cos=block_cos.unsqueeze(2),
        sin=block_sin.unsqueeze(2),
    ).squeeze(2).contiguous()


def _blocks_to_token_ids(
    block_ids: torch.Tensor,
    kv_length: int,
    ratio: int,
    budget: int,
) -> torch.Tensor:
    """Expand per-query block indices into flat token indices.

    Mirrors the output contract of the torch ``qsa_indexer_forward``:
    shape ``[B, S, budget + ratio - 1]`` int32 with -1 padding.
    """
    batch, seq_len, num_blocks = block_ids.shape
    width = budget + ratio - 1
    token_ids = torch.full(
        (batch, seq_len, width), -1, dtype=torch.int32, device=block_ids.device
    )

    valid = block_ids >= 0
    lanes = torch.arange(ratio, device=block_ids.device, dtype=torch.int32)
    expanded = block_ids.unsqueeze(-1) * ratio + lanes.view(1, 1, 1, -1)
    expanded = expanded.reshape(batch, seq_len, -1)
    valid_expanded = (
        valid.unsqueeze(-1).expand(-1, -1, -1, ratio).reshape(batch, seq_len, -1)
    )

    # Causal visibility: token position must not exceed the query's own index.
    positions = torch.arange(seq_len, device=block_ids.device).view(1, -1, 1)
    in_range = valid_expanded & (expanded >= 0) & (expanded <= positions)

    cols = min(expanded.shape[-1], width)
    token_ids[:, :, :cols] = torch.where(
        in_range[:, :, :cols],
        expanded[:, :, :cols],
        torch.full_like(expanded[:, :, :cols], -1),
    )
    return token_ids


@torch.no_grad()
def qsa_indexer_forward_cann(
    indexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """LightningIndexerV2 drop-in replacement for ``qsa_indexer_forward``.

    Only supports ``sp_size == 1`` (the packed single-rank case).  Returns the
    same ``[1, S, budget + ratio - 1]`` int32 tensor as the torch path.
    """
    ops = _import_cann_ops()

    batch, seq_len, _ = hidden_states.shape
    full_cos, full_sin = position_embeddings
    current_cos = full_cos[:, -seq_len:]
    current_sin = full_sin[:, -seq_len:]

    head_dim = indexer.index_head_dim
    ratio = indexer.compress_ratio
    block_topk = indexer.block_topk

    # -- query branch ------------------------------------------------------
    query, raw_keys = indexer.index_qk_proj(hidden_states.detach()).split(
        (indexer.index_n_heads * head_dim, head_dim), dim=-1
    )
    query = indexer.q_layernorm(query.unflatten(-1, (indexer.index_n_heads, head_dim)))
    query = apply_rotary_pos_emb(
        query, cos=current_cos, sin=current_sin, unsqueeze_dim=2
    )

    # -- compressed block keys ----------------------------------------------
    kv_length = raw_keys.shape[1]
    compressed_keys = _compressed_index_keys(
        raw_keys, full_cos, full_sin, indexer.k_layernorm, ratio
    )

    # -- CANN LightningIndexerV2 --------------------------------------------
    weights = torch.full(
        (batch, seq_len, indexer.index_n_heads),
        head_dim ** -0.5,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    cmp_residual_k = torch.full(
        (batch,), kv_length % ratio, dtype=torch.int32, device=hidden_states.device
    )

    selected_blocks, _ = ops.lightning_indexer(
        query.contiguous(),
        compressed_keys.unsqueeze(2).contiguous(),
        weights,
        block_topk,
        cmp_residual_k=cmp_residual_k,
        metadata=None,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
        return_value=0,
    )
    selected_blocks = selected_blocks.squeeze(2)  # [B, S, topk]

    return _blocks_to_token_ids(
        selected_blocks, kv_length, ratio, indexer.token_budget
    )
