"""NPU fused attention and lightning-indexer integration for DeepSeek V4.1.

NPU training uses fused MLA, lightning indexer, KL and mHC operators.
A positive KL coefficient enables indexer training. KL targets are recomputed without gradients
using the reference softmax, while the attention output stays fused.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import NamedTuple

import torch

from torchturbo.logging import get_logger
from torchturbo.ops import ops

from torchturbo.models.modules.attention.ulysses import (
    all_gather_sp_with_grad_reduce,
)

from . import source_packing_utils as pu
from .modeling_deepseek_v41 import _select_kv_rows, get_window_topk_idxs
from .configuration_deepseek_v41 import text_config_of
from .packed_metadata import PackedAttentionMetadata, cumulative_lengths

logger = get_logger(__name__)

_CANN_WINDOW_SIZE = 128


def _validate_sp_context(sp_group, sp_size: int, sp_rank: int) -> None:
    """Reject incomplete or inconsistent sequence-parallel configuration."""
    if sp_size < 1 or not 0 <= sp_rank < sp_size:
        raise ValueError("SP requires sp_size >= 1 and 0 <= sp_rank < sp_size.")
    if sp_size > 1 and sp_group is None:
        raise ValueError("SP with multiple ranks requires an explicit process group.")


def patch_deepseek_v41_npu_sfa(
        model: torch.nn.Module,
        *,
        sp_group: torch.distributed.ProcessGroup | None = None,
        sp_size: int = 1,
        sp_rank: int = 0,
) -> int:
    """Configure SP state and install instance-local NPU attention."""
    from .modeling_deepseek_v41 import (
        DeepseekV41Compressor,
        DeepseekV41InnerAttentionOps,
        DeepseekV41PreAttentionOps,
        DeepseekV41Indexer,
    )

    _validate_sp_context(sp_group, sp_size, sp_rank)
    for module in model.modules():
        if isinstance(module, DeepseekV41PreAttentionOps) and module.window_size != _CANN_WINDOW_SIZE:
            raise ValueError(f"CANN MLA requires sliding_window={_CANN_WINDOW_SIZE}.")
    modules_with_sp = (
        DeepseekV41InnerAttentionOps, DeepseekV41PreAttentionOps,
        DeepseekV41Compressor, DeepseekV41Indexer,
    )
    count = 0
    for module in model.modules():
        if isinstance(module, modules_with_sp):
            module._sp_group = sp_group
            module._sp_size = sp_size
            module._sp_rank = sp_rank
        if isinstance(module, DeepseekV41InnerAttentionOps):
            module.forward = MethodType(_npu_sfa, module)
            count += 1
    return count


def patch_deepseek_v41_npu_engram(
        model: torch.nn.Module,
        *,
        sp_group: torch.distributed.ProcessGroup | None = None,
        sp_size: int = 1,
        sp_rank: int = 0,
) -> None:
    """Install local-output hashing with global SP token context."""
    from .modeling_deepseek_v41 import NgramHashState

    _validate_sp_context(sp_group, sp_size, sp_rank)
    for module in model.modules():
        if isinstance(module, NgramHashState):
            module._sp_group = sp_group
            module._sp_size = sp_size
            module._sp_rank = sp_rank
            module.forward = MethodType(
                _npu_engram_hash_forward,
                module,
            )


def _npu_engram_hash_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hash global token context and return this rank's token rows."""
    from .modeling_deepseek_v41 import NgramHashState

    local_length = input_ids.shape[1]
    global_input_ids = input_ids

    if self._sp_size > 1:
        global_input_ids = all_gather_sp_with_grad_reduce(
            input_ids,
            dim=1,
            sp_group=self._sp_group,
            sp_size=self._sp_size,
        )

    # Call the original class method; only the instance was patched.
    global_hashes = NgramHashState.forward(
        self,
        global_input_ids,
        attention_mask,
    )

    start = self._sp_rank * local_length
    end = start + local_length
    return global_hashes[:, start:end].contiguous()


def _npu_sfa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        window_topk: torch.Tensor,
        kv_compress: torch.Tensor | None,
        compress_topk: torch.Tensor | None,
        attn_sink: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed: PackedAttentionMetadata | None = None,
) -> torch.Tensor:
    """Dispatch a patched attention instance to the unified fused adapter."""
    if packed is None:
        mask = attention_mask
        if mask is None:
            mask = torch.ones(kv.shape[:2], dtype=torch.long, device=q.device)
        packed = PackedAttentionMetadata(
            mask, q.shape[1], self._sp_rank,
            self.compress_ratio if kv_compress is not None else 1,
        )
    cann_mask, reason = prepare_cann_attention(
        q,
        kv,
        window_topk,
        kv_compress,
        compress_topk,
        attn_sink,
        self.compress_ratio,
        packed.mask,
        position_ids,
        sp_rank=self._sp_rank,
        packed=packed,
    )
    if reason is not None:
        raise ValueError(f"CANN MLA cannot execute this attention: {reason}")
    return run_cann_attention(
        q,
        kv,
        kv_compress,
        compress_topk,
        attn_sink,
        self.softmax_scale,
        self.compress_ratio,
        cann_mask,
        sp_rank=self._sp_rank,
        packed=packed,
        validated=True,
    )


class _AttentionPart(NamedTuple):
    query: torch.Tensor
    original_kv: torch.Tensor
    compressed_kv: torch.Tensor | None
    indices: torch.Tensor | None
    residual: int
    output_indices: torch.Tensor


def run_cann_attention(
        q: torch.Tensor,
        kv: torch.Tensor,
        compressed: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
        sinks: torch.Tensor,
        scale: float,
        ratio: int,
        attention_mask: torch.Tensor | None,
        *,
        sp_rank: int = 0,
        packed: PackedAttentionMetadata | None = None,
        validated: bool = False,
) -> torch.Tensor:
    """Execute prepared attention after sequence-parallel communication.

    Args:
        q: Local queries, [B, L, N, D].
        kv: Gathered original KV, [B, T, D].
        compressed: Replicated compressed KV, [B, C, D], or None for SWA.
        compressed_indices: Local indices, [B, L, K], into each batch row's
            compressed KV. Indices exclude the original-KV offset; -1 is padding.
        sinks: Replicated FP32 sinks, [N].
        scale: Attention score scale.
        ratio: Compression ratio, 1, 2 or 4 when compressed KV is present.
        attention_mask: Global sample IDs, [B, T]. Positive IDs identify
            contiguous samples; zero denotes padding. None selects one sample.
        sp_rank: Rank owning an equally sized, contiguous query shard.
        packed: Forward-local geometry for the same mask, shard and ratio.
        validated: The caller already ran the prepare contract checks
            (check_padding + per-segment local_indices) on this exact
            indices tensor and packed geometry; skip re-validating them.

    Returns:
        Local output, [B, L, N, D], with zeros at padding positions.

    Communication belongs to the caller. Autograd returns contributions for
    gathered/replicated KV and sinks; the caller owns their SP reductions.
    Normal packed samples share one TND call. Irregular selection counts and
    samples without visible compressed blocks use groups of compatible calls.
    """
    if attention_mask is None:
        attention_mask = torch.ones(kv.shape[:2], dtype=torch.long, device=q.device)
    batch_size, local_length, num_heads, head_dim = q.shape
    local_start = sp_rank * local_length
    has_cmp = compressed is not None and compressed_indices is not None
    if packed is None:
        packed = PackedAttentionMetadata(
            attention_mask, local_length, sp_rank, ratio if has_cmp else 1
        )
    query_indices = packed.query_indices
    kv_indices = packed.kv_indices
    output = q * 0 + kv.reshape(-1)[:0].sum() + sinks[:0].sum()
    if has_cmp:
        output = output + compressed.reshape(-1)[:0].sum()
        if not validated:
            packed.check_padding(compressed_indices)
    if query_indices.numel() == 0:
        return output
    query_tnd = q.reshape(-1, num_heads, head_dim).index_select(0, query_indices)
    kv_tnd = kv.reshape(-1, head_dim).index_select(0, kv_indices).unsqueeze(1)
    q_lengths = packed.query_lengths
    kv_lengths = packed.kv_lengths
    queries = query_tnd.split(q_lengths)
    original_kvs = kv_tnd.split(kv_lengths)
    output_indices = query_indices.split(q_lengths)

    groups: dict[int, list[_AttentionPart]] = {}
    for segment_index, segment in enumerate(packed.segments):
        batch, start, end, sample_start, block_start, block_count = segment
        query = queries[segment_index]
        original = original_kvs[segment_index]
        scatter_indices = output_indices[segment_index]
        query_offset = local_start + start - sample_start
        length = end - start
        if not has_cmp:
            groups.setdefault(0, []).append(_AttentionPart(
                query[:length], original[:query_offset + length],
                None, None, 0, scatter_indices[:length],
            ))
            continue

        # Only CSA partitions queries by their visible compressed selections.
        values = compressed_indices[batch, start:end]
        local_indices, visible = packed.local_indices(
            compressed_indices, segment, validate=not validated
        )
        valid = local_indices >= 0
        sort_keys = torch.where(valid, local_indices, block_count)
        # FP32 represents these bounded integer keys exactly and sorts on AiCore.
        if block_count <= 2**24:
            selected = sort_keys.float().sort(dim=-1).values.to(local_indices.dtype)
        else:
            selected = sort_keys.sort(dim=-1).values
        counts = valid.sum(-1)
        expected = visible.clamp_max(values.shape[-1])
        if (query_offset + length) // ratio > 0 and torch.equal(counts, expected):
            chunks = [(0, length, values.shape[-1])]
        else:
            counts = counts.tolist()
            chunks = []
            begin = 0
            for stop in range(1, length + 1):
                if stop == length or counts[stop] != counts[begin]:
                    chunks.append((begin, stop, counts[begin]))
                    begin = stop
        for begin, stop, topk in chunks:
            prefix = query_offset + stop
            cmp_kv = indices = None
            if topk:
                count = prefix // ratio
                cmp_kv = compressed[
                         batch, block_start: block_start + count
                         ].unsqueeze(1)
                indices = selected[begin:stop, :topk]
                indices = (
                    torch.where(indices < block_count, indices, -1)
                    .int()
                    .unsqueeze(1)
                )
            groups.setdefault(topk, []).append(
                _AttentionPart(
                    query[begin:stop],
                    original[:prefix],
                    cmp_kv,
                    indices,
                    prefix % ratio if topk else 0,
                    scatter_indices[begin:stop],
                )
            )

    for topk, parts in groups.items():
        queries = [part.query for part in parts]
        originals = [part.original_kv for part in parts]
        compressed_parts = [part.compressed_kv for part in parts] if topk else []
        query = torch.cat(queries)
        original = torch.cat(originals)
        compressed_kv = torch.cat(compressed_parts) if topk else None
        indices = torch.cat([part.indices for part in parts]) if topk else None
        q_cu = cumulative_lengths([t.shape[0] for t in queries], queries[0].device)
        kv_cu = cumulative_lengths([t.shape[0] for t in originals], originals[0].device)
        cmp_cu = (
            cumulative_lengths([t.shape[0] for t in compressed_parts], compressed_parts[0].device)
            if topk
            else None
        )
        residual = (
            torch.tensor([part.residual for part in parts], dtype=torch.int32, device=q.device)
            if topk else None
        )
        max_seqlens = (
            max(part.query.shape[0] for part in parts),
            max(part.original_kv.shape[0] for part in parts),
            max(part.compressed_kv.shape[0] for part in parts) if topk else 0,
        )
        result = ops.sparse_flash_mla(
            query,
            original,
            compressed_kv,
            indices,
            residual,
            sinks.contiguous(),
            scale,
            ratio if topk else 1,
            _CANN_WINDOW_SIZE - 1,
            0,
            cu_seq_lens_q=q_cu,
            cu_seq_lens_ori_kv=kv_cu,
            cu_seq_lens_cmp_kv=cmp_cu,
            custom_package=True,
            max_seqlens=max_seqlens,
        )
        scatter = torch.cat([part.output_indices for part in parts])
        output = (
            output.reshape(-1, num_heads, head_dim)
            .index_copy(0, scatter, result)
            .view_as(q)
        )
    return output


def packed_window_matches(indices: torch.Tensor, mask: torch.Tensor, *, sp_rank: int = 0) -> bool:
    """Ensure implicit CANN windows reproduce the model's explicit indices."""
    batch, global_length = mask.shape

    local_length = indices.shape[1]

    width = min(global_length, _CANN_WINDOW_SIZE)

    query_start = sp_rank * local_length
    query_end = query_start + local_length

    if (
            global_length == 0
            or query_end > global_length
            or indices.shape != (batch, local_length, width)
    ):
        return False

    expected = get_window_topk_idxs(
        _CANN_WINDOW_SIZE, batch, global_length, indices.device, mask,
        query_start=query_start, query_end=query_end,
    )
    return torch.equal(indices, expected)


def prepare_cann_attention(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        compressed: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
        sinks: torch.Tensor,
        ratio: int,
        mask: torch.Tensor | None,
        positions: torch.Tensor | None,
        *,
        sp_rank: int = 0,
        packed: PackedAttentionMetadata | None = None,
) -> tuple[torch.Tensor | None, str | None]:
    """Validate the model contract and return its global sample mask."""
    if q.device.type != "npu":
        return None, "query must be on NPU"
    if q.dtype not in (torch.float16, torch.bfloat16) or kv.dtype != q.dtype:
        return None, "query and KV must have matching fp16/bf16 dtype"
    if q.shape[-1] != 512 or kv.shape[-1] != 512:
        return None, "head_dim must be 512"
    if q.shape[2] not in (1, 2, 4, 8, 16, 32, 64, 128):
        return None, "unsupported query head count"

    local_length = q.shape[1]

    if sinks.shape != (q.shape[2],) or sinks.dtype != torch.float32:
        return None, "sinks must be FP32 with one entry per local query head"
    has_cmp = compressed is not None and compressed_indices is not None
    if has_cmp and ratio not in (1, 2, 4):
        return None, "A3 CSA requires compression ratio 1, 2, or 4"

    if mask is None:
        mask = torch.ones(
            kv.shape[:2],
            dtype=torch.long,
            device=q.device,
        )

    if not packed_window_matches(indices, mask, sp_rank=sp_rank):
        return None, "window indices must match the packed causal 128-token window"

    try:
        if packed is None:
            packed = PackedAttentionMetadata(
                mask, local_length, sp_rank, ratio if has_cmp else 1
            )
        packed.block_infos
    except ValueError as exc:
        return None, str(exc)
    if not has_cmp:
        return mask, None
    if compressed.dtype != q.dtype or compressed.shape[-1] != 512:
        return None, "compressed KV must match query dtype and head_dim"
    if compressed_indices.shape[-1] > 8192:
        return None, "CSA top-k must not exceed 8192"
    capacity = packed.compressed_capacity
    if compressed.shape[:2] != (q.shape[0], capacity):
        return None, "compressed KV does not match complete packed block ordering"

    try:
        packed.check_padding(compressed_indices)
        for segment in packed.segments:
            packed.local_indices(compressed_indices, segment, positions)
    except ValueError as exc:
        return None, str(exc)

    return mask, None


def _kl_loss_coeff(config) -> float:
    """KL coefficient from the config field. The CLI flag
    --tt_indexer_kl_loss_coeff (folded into the field at model build time;
    see TurboDeepseekV41ForCausalLM._pre_init_patches) overrides the
    checkpoint value. A missing field raises (loud by contract) rather than
    defaulting silently."""
    return float(config.indexer_kl_loss_coeff)


# ---------------------------------------------------------------------------
# Indexer KL loss AutoScaler (DSAIndexerLossAutoScaler pattern).
# ---------------------------------------------------------------------------


class V41IndexerLossAutoScaler(torch.autograd.Function):
    """Attach KL gradients using the scale captured for this forward."""

    main_loss_backward_scale: float = 1.0

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
        ctx.save_for_backward(loss)
        ctx.loss_scale = (
            V41IndexerLossAutoScaler.main_loss_backward_scale
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(loss) * ctx.loss_scale

    @staticmethod
    def set_loss_scale(scale: float):
        V41IndexerLossAutoScaler.main_loss_backward_scale = float(scale)

# ---------------------------------------------------------------------------
# Fused lightning-indexer forward (replaces DeepseekV41Indexer.score).
# ---------------------------------------------------------------------------


def _scatter_tnd_indices(
        indices: torch.Tensor,
        query_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        tnd_to_block: torch.Tensor,
        bsz: int,
        seqlen: int,
) -> torch.Tensor:
    """Map segment-local TND indices back to global packed-block rows.

    ``indices``: [T1, 1, K] int32 segment-local (per LIV2/SLI convention).
    Returns [b, s, K] int64 block-list rows, -1 where invalid.
    """
    topk = indices.size(-1)
    local = indices[:, 0, :]
    segment_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    key_offsets = torch.repeat_interleave(cu_seqlens_k[:-1], segment_lengths)
    rows = local.long() + key_offsets.unsqueeze(1)
    valid = (local >= 0) & (rows < tnd_to_block.size(0))
    safe_rows = rows.clamp(min=0, max=tnd_to_block.size(0) - 1)
    mapped = tnd_to_block[safe_rows]
    valid = valid & (mapped >= 0)
    block_indices = torch.where(valid, mapped, -1)
    topk_idxs = block_indices.new_full((bsz, seqlen, topk), -1)
    topk_idxs.reshape(-1, topk)[query_indices] = block_indices
    return topk_idxs


def _npu_indexer_select(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        candidates: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
):
    """Fused index selection from projected local queries and weights.

    Same contract: (topk block rows [b, s, K] int64 with -1, new candidates
    or None, None). The third return is always None (fused path produces no
    dense index_score; the KL loss consumes the topk slot instead). The
    candidates slot becomes the op-native ``candidate_topk_indices`` tensor
    (TND int32) instead of the small operator's bool mask.
    """

    bsz, local_length = q.shape[:2]
    seqlen = local_length

    sp_size = self._sp_size
    sp_rank = self._sp_rank
    sp_group = self._sp_group

    query_start = sp_rank * local_length
    query_end = query_start + local_length

    # Query RoPE is already local; gather positions for global block mapping.
    global_position_ids = position_ids
    if sp_size > 1:
        global_position_ids = all_gather_sp_with_grad_reduce(
            position_ids,
            dim=1,
            sp_group=sp_group,
            sp_size=sp_size,
        )

    block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
        attention_mask,
        self.compress_ratio,
        global_position_ids,
    )

    num_blocks = block_ids.size(1)

    if index_k is None or index_k.size(1) != num_blocks:
        return None, None, None
    if num_blocks == 0:
        return (
            q.new_full((bsz, seqlen, 0), -1, dtype=torch.long),
            None,
            None,
        )

    # Fused index selection is forward-only.
    # Indexer training gradients are handled by the separate KL path.
    if sp_size > 1:
        q = all_gather_sp_with_grad_reduce(
            q.detach(),
            dim=1,
            sp_group=sp_group,
            sp_size=sp_size,
        )
        weights = all_gather_sp_with_grad_reduce(
            weights.detach(),
            dim=1,
            sp_group=sp_group,
            sp_size=sp_size,
        )

    # The existing TND packing and fused indexer now see global queries.
    seqlen = q.shape[1]

    # Global-view packed geometry: the same PackedAttentionMetadata
    # derivation the fused attention and KL paths use (the full mask as a
    # single shard), replacing the v4 TND helpers.
    global_packed = PackedAttentionMetadata(
        attention_mask, attention_mask.size(1), 0, self.compress_ratio,
    )
    query_indices, cu_seqlens_q = global_packed.query_indices, global_packed.cu_query
    k_tnd, cu_seqlens_k, cmp_residual_k = global_packed.compressed_keys(index_k)
    tnd_to_block = global_packed.block_indices
    # Forward-only ops: detach everything (indexer grads come from the KL
    # path); bf16 q/k and fp32 weights per the op contract.
    query_tnd = (
        q.reshape(-1, self.n_heads, self.head_dim)
        .index_select(0, query_indices)
        .detach()
        .to(torch.bfloat16)
        .contiguous()
    )
    weights_tnd = (
        weights.reshape(-1, self.n_heads)
        .index_select(0, query_indices)
        .detach()
        .float()
        .contiguous()
    )
    k_tnd = k_tnd.detach().to(torch.bfloat16).unsqueeze(1).contiguous()

    # All LI variants consume the same packed query/key geometry. Candidate
    # tensors stay in the provider's TND format across layers.
    op_kwargs = dict(
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k, layout_q="TND", layout_k="TND",
        mask_mode=3, cmp_ratio=self.compress_ratio,
    )
    new_candidates = None
    if candidates is not None and not self.is_candidate_source:
        source_ratio = self.candidate_source_ratio
        if source_ratio is not None and source_ratio != self.compress_ratio:
            raise RuntimeError(
                f"sparse_lightning_indexer requires cmp_ratio == candidate "
                f"source ratio ({source_ratio}), layer has {self.compress_ratio}."
            )
        indices = ops.sparse_lightning_indexer(
            query_tnd, k_tnd, weights_tnd, self.index_topk, candidates,
            candidate_block_size=self.candidate_block_size, **op_kwargs,
        )
    else:
        if self.is_candidate_source:
            op_kwargs.update(
                candidate_mode=True, candidate_topk_blocks=self.candidate_topk_blocks,
                candidate_block_size=self.candidate_block_size,
            )
        indices, candidate_pool = ops.lightning_indexer_v2(
            query_tnd, k_tnd, weights_tnd, self.index_topk, **op_kwargs,
        )
        if self.is_candidate_source:
            new_candidates = candidate_pool.contiguous()

    topk_idxs = _scatter_tnd_indices(
        indices, query_indices, cu_seqlens_q, cu_seqlens_k, tnd_to_block, bsz, seqlen
    )

    # Small-operator parity: re-validate the mapped rows with the packed
    # compress mask (the op's mask_mode=3 and the small operator's boundary
    # rule differ at block edges; the extra check is cheap on [b, s, K]).
    flat_idxs = topk_idxs.reshape(bsz, -1).clamp_min(0)
    batch_index = torch.arange(bsz, device=q.device).unsqueeze(1)
    topk_ids = block_ids[batch_index, flat_idxs].view_as(topk_idxs)
    topk_pos = block_positions[batch_index, flat_idxs].view_as(topk_idxs)
    invalid = pu.packed_compress_mask(
        attention_mask,
        topk_ids,
        topk_pos,
        self.compress_ratio,
        global_position_ids,
    )
    topk_idxs = torch.where(invalid, -1, topk_idxs)

    # Each rank's Attention only consumes its local query rows.
    # A contiguous slice can retain the full global backing storage.
    topk_idxs = topk_idxs[:, query_start:query_end].clone()

    # Keep the candidate pool in the original global TND layout.
    return topk_idxs, new_candidates, None

# ---------------------------------------------------------------------------
# Indexer KL loss (inside the checkpointed attention forward).
# ---------------------------------------------------------------------------


def _indexer_kl_loss(
        attn,
        x: torch.Tensor,
        index_k: torch.Tensor,
        topk_global: torch.Tensor,
        target_probs: torch.Tensor,
        inv_freq: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        packed: PackedAttentionMetadata | None = None,
) -> torch.Tensor:
    """Compute the valid-token mean KL for this rank's local queries."""
    indexer = attn.indexer
    bsz, seqlen, _ = x.shape
    topk = topk_global.size(-1)

    sp_rank = indexer._sp_rank

    if packed is None:
        packed = PackedAttentionMetadata(
            attention_mask, seqlen, sp_rank, indexer.compress_ratio
        )

    # Recompute local projections with gradients enabled.
    qr = attn.q_norm(attn.wq_a(x))
    q, weights = indexer.project_queries(x, qr, inv_freq, position_ids)

    query_indices, cu_seqlens_q = packed.query_indices, packed.cu_query

    # Preserve all gradient paths even on an all-padding rank.
    if query_indices.numel() == 0:
        return (
                q.float().sum() * 0.0
                + weights.float().sum() * 0.0
                + index_k.float().sum() * 0.0
        )

    k_tnd, cu_seqlens_k, cmp_residual_k = packed.compressed_keys(index_k)

    # Convert row-global block IDs to sample-local block IDs.
    # Do not change the Top-K column order.
    sparse_seglocal = torch.full_like(topk_global, -1)

    for segment in packed.segments:
        relative, _ = packed.local_indices(topk_global, segment)
        sparse_seglocal[segment.batch, segment.start:segment.end] = relative

    q_tnd = (
        q.reshape(-1, indexer.n_heads, indexer.head_dim)
        .index_select(0, query_indices)
        .to(torch.bfloat16)
        .contiguous()
    )
    w_tnd = (
        weights.reshape(-1, indexer.n_heads)
        .index_select(0, query_indices)
        .float()
        .contiguous()
    )
    k_tnd = k_tnd.to(torch.bfloat16).unsqueeze(1).contiguous()

    # Targets and sparse indices retain exactly the same slot ordering.
    p_tnd = (
        target_probs.reshape(-1, topk)
        .index_select(0, query_indices)
        .unsqueeze(1)
        .detach()
        .float()
        .contiguous()
    )
    sparse_tnd = (
        sparse_seglocal.reshape(-1, topk)
        .index_select(0, query_indices)
        .unsqueeze(1)
        .int()
        .contiguous()
    )

    metadata = ops.sli_kl_loss_grad_metadata(
        indexer.n_heads,
        1,
        indexer.head_dim,
        topk=topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=cu_seqlens_q.numel() - 1,
        cmp_ratio=indexer.compress_ratio,
    )

    return ops.sli_kl_loss(
        q_tnd,
        k_tnd,
        w_tnd,
        sparse_tnd,
        p_tnd,
        metadata=metadata,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        cmp_ratio=indexer.compress_ratio,
    )


class _IndexerLossContext(NamedTuple):
    """Forward-local KL metadata; never retained by the model."""

    coeff: float
    packed: PackedAttentionMetadata

    @staticmethod
    @torch.no_grad()
    def target_probs(
            inner: torch.nn.Module,
            q: torch.Tensor,
            kv: torch.Tensor,
            window_topk: torch.Tensor,
            kv_compress: torch.Tensor,
            compress_topk: torch.Tensor,
            attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute detached reference probabilities for the indexer KL target."""
        # Shift the compressed block indices past the window KV.
        compress_topk = torch.where(
            compress_topk >= 0, compress_topk + kv.size(1), compress_topk
        )
        window = window_topk.size(-1)
        topk_idxs = torch.cat([window_topk, compress_topk], dim=-1)
        kv_states = torch.cat([kv, kv_compress], dim=1)

        bsz, seqlen = q.shape[:2]
        safe_indices = topk_idxs.clamp_min(0).long()
        selected_kv = _select_kv_rows(kv_states, safe_indices)

        scores = (
                torch.einsum("bshd,bskd->bshk", q, selected_kv).float() * inner.softmax_scale
        )
        scores = scores.masked_fill(topk_idxs.unsqueeze(2) < 0, float("-inf"))

        sink = attn_sink.float().view(1, 1, -1, 1).expand(bsz, seqlen, -1, -1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        return probs[:, :, :, window:].mean(dim=2)

    def attach(
        self, attn: torch.nn.Module, attn_output: torch.Tensor,
        hidden_states: torch.Tensor, index_k: torch.Tensor, used_topk: torch.Tensor,
        target_probs: torch.Tensor, inv_freq: torch.Tensor,
        attention_mask: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor:
        kl_loss = _indexer_kl_loss(
            attn,
            hidden_states,
            index_k,
            used_topk,
            target_probs,
            inv_freq,
            attention_mask,
            position_ids,
            packed=self.packed,
        )
        # The KL operator returns a local valid-token mean.
        # Compensate for parameter-gradient averaging across SP ranks.
        sp_size = attn.indexer._sp_size
        sp_rank = attn.indexer._sp_rank
        local_length = hidden_states.shape[1]
        query_start = sp_rank * local_length
        query_end = query_start + local_length
        local_mask = attention_mask[:, query_start:query_end]
        local_tokens = (local_mask > 0).sum().float()
        global_tokens = (attention_mask > 0).sum().float()
        sp_weight = sp_size * local_tokens / global_tokens.clamp_min(1)
        attn_output = V41IndexerLossAutoScaler.apply(
            attn_output, kl_loss * self.coeff * sp_weight
        )
        return attn_output


def _prepare_indexer_loss(
    self, hidden_states: torch.Tensor, kv_compress: torch.Tensor | None,
    index_k: torch.Tensor | None, used_topk: torch.Tensor | None,
    attention_mask: torch.Tensor,
) -> _IndexerLossContext | None:
    coeff = _kl_loss_coeff(self.config)
    want_kl = (
            coeff > 0
            and self.training
            and torch.is_grad_enabled()
            and self.is_index_source
            and kv_compress is not None
            and index_k is not None
            and used_topk is not None
            and used_topk.size(-1) > 0
    )

    if not want_kl:
        return None
    return _IndexerLossContext(
        coeff, PackedAttentionMetadata(
            attention_mask, hidden_states.shape[1], self.indexer._sp_rank, self.compress_ratio
        )
    )


# ---------------------------------------------------------------------------
# Fused mHC pre-mixing; the decoder forward remains shared.
# ---------------------------------------------------------------------------


def _npu_hc_pre(self, hidden_states, fn, scale, base, pre_mix):
    """Fused mHC for every call: training forward, checkpoint recompute,
    and no-grad forward alike.

    Routing the no-grad path (reentrant AC's initial forward, eval) through
    the fused kernel keeps forward/backward on the same numerics under
    reentrant checkpointing and matches the non-reentrant default, where
    the initial forward already runs fused.
    """
    sinkhorn = self.hc.hc_split_sinkhorn
    hin, post, residual, pre = ops.mhc_pre_sinkhorn_premix(
        hidden_states, fn, scale, base, pre_mix,
        self.hc_mult,
        sinkhorn.hc_sinkhorn_iters,
        sinkhorn.hc_eps,
        self.hc.norm_eps,
    )
    comb = residual.view(
        hidden_states.size(0), hidden_states.size(1), self.hc_mult, self.hc_mult
    )
    return hin, post, comb, pre


# ---------------------------------------------------------------------------
# Patching.
# ---------------------------------------------------------------------------


def patch_deepseek_v41_npu(
        *, enable_li: bool, enable_indexer_kl: bool, model: torch.nn.Module
) -> None:
    """Install the requested indexer paths on this model's instances.

    Inner attention keeps its tensor-returning contract and any installed
    fused MLA implementation. Other model instances remain unchanged.
    """
    from . import modeling_deepseek_v41 as dsv41

    for module in model.modules():
        if enable_li and isinstance(module, dsv41.DeepseekV41Indexer):
            module._select_blocks = MethodType(_npu_indexer_select, module)
        if enable_indexer_kl and isinstance(module, dsv41.DeepseekV41Attention):
            module._prepare_indexer_loss = MethodType(_prepare_indexer_loss, module)
    if enable_li:
        logger.info("deepseek_v41: fused NPU lightning indexer enabled")
    if enable_indexer_kl:
        logger.info("deepseek_v41: indexer KL loss enabled")


def patch_deepseek_v41_npu_sfa_li(model: torch.nn.Module) -> None:
    """Install the required LI/KL bundle, or degrade the block selection.

    KL activates only with a positive loss coefficient. Required operators
    must be available before any indexer instances are patched. The fused
    block selection additionally needs the sparse_lightning_indexer kernel;
    images whose kernel split leaves it out keep the model's reference
    ``_select_blocks`` (the fused source op alone cannot express the
    candidate-consumer contract).
    """
    from torchturbo.ops.providers.npu.lightning_indexer_v2 import has_lightning_indexer_v2
    from torchturbo.ops.providers.npu.sparse_lightning_indexer import has_sparse_lightning_indexer
    from torchturbo.ops.providers.npu.sparse_lightning_indexer_kl_loss_grad import has_sli_kl_loss_grad

    if not has_lightning_indexer_v2():
        raise RuntimeError("NPU training requires lightning_indexer_v2.")
    if not has_sli_kl_loss_grad():
        raise RuntimeError("NPU training requires sparse_lightning_indexer_kl_loss_grad and its metadata operator.")
    enable_li = has_sparse_lightning_indexer()
    if not enable_li:
        logger.warning(
            "deepseek_v41: sparse_lightning_indexer kernel unavailable; "
            "indexer block selection falls back to the reference path "
            "(unverified under sequence parallelism)"
        )
    patch_deepseek_v41_npu(enable_li=enable_li, enable_indexer_kl=True, model=model)


def patch_deepseek_v41_npu_mhc(model: torch.nn.Module) -> None:
    """Install required mHC before activation checkpointing.

    Missing kernels or unsupported configurations fail before patching.
    """
    from torchturbo.ops.providers.npu.mhc_pre_sinkhorn_premix import (
        has_mhc_pre_sinkhorn_premix,
    )

    if not has_mhc_pre_sinkhorn_premix():
        raise RuntimeError("NPU training requires mhc_pre_sinkhorn_premix and its backward operator.")
    config = text_config_of(model.config)
    # Hard constraints of the fused backward kernel.
    if config.hc_mult != 4:
        raise ValueError(f"Fused mHC requires hc_mult=4, got {config.hc_mult}.")
    if config.hidden_size % 128 != 0:
        raise ValueError(f"Fused mHC requires hidden_size divisible by 128, got {config.hidden_size}.")
    if config.hc_sinkhorn_iters != 20:
        raise ValueError(f"Fused mHC requires hc_sinkhorn_iters=20, got {config.hc_sinkhorn_iters}.")
    from . import modeling_deepseek_v41 as dsv41

    for module in model.modules():
        if isinstance(module, dsv41.DeepseekV41DecoderLayer):
            module._hc_pre = MethodType(_npu_hc_pre, module)
    logger.info("deepseek_v41: fused NPU mhc_pre_sinkhorn_premix enabled")


def _env_flag_enabled(name: str) -> bool:
    """True when an env var is set to an explicit true value."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _npu_rms_norm_forward(self, hidden_states):
    """Fused npu_rms_norm replacement for DeepseekV41RMSNorm.forward."""
    return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)


def patch_deepseek_v41_npu_rms_norm(model: torch.nn.Module) -> None:
    """Route DeepseekV41RMSNorm instances through the fused npu_rms_norm op.

    One kernel replaces the fp32 small-op chain; the op keeps gamma in fp32,
    matching the v41 math. Instance-level like the mHC patches, so modules
    constructed outside this model keep the eager chain.
    TORCH_TURBO_DISABLE_V41_FUSED_RMS_NORM restores the eager chain.
    """
    if _env_flag_enabled("TORCH_TURBO_DISABLE_V41_FUSED_RMS_NORM"):
        return
    from . import modeling_deepseek_v41 as dsv41

    patched = 0
    for module in model.modules():
        if isinstance(module, dsv41.DeepseekV41RMSNorm):
            module.forward = MethodType(_npu_rms_norm_forward, module)
            patched += 1
    logger.info("deepseek_v41: fused NPU rms_norm enabled, patched_norms=%d", patched)


def _npu_hc_post_forward(self, x, residual, post, comb):
    """Fused mhc_post replacement for HcPost.forward (bound per instance)."""
    return ops.mhc_post(residual, comb, x, post)


def patch_deepseek_v41_npu_hc_post(model: torch.nn.Module) -> None:
    """Install the fused mhc_post HcPost patch.

    Runs after patch_deepseek_v41_npu_mhc, so hc_mult=4 and hidden_size % 128
    are already validated. Missing kernels fail before patching; the bmm
    HcPost fallback stays installed when disabled via the env flag.
    """
    if _env_flag_enabled("TORCH_TURBO_DISABLE_V41_FUSED_HC_POST"):
        return
    from torchturbo.ops.providers.npu.mhc_post import has_mhc_post

    if not has_mhc_post():
        raise RuntimeError("NPU training requires mhc_post and its backward operator.")
    from . import modeling_deepseek_v41 as dsv41

    for module in model.modules():
        if isinstance(module, dsv41.HcPost):
            module.forward = MethodType(_npu_hc_post_forward, module)
    logger.info("deepseek_v41: fused NPU mhc_post enabled")
