# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch DeepSeek V4.1 model for transformers library.

Packed-sequence native training implementation. The packed inputs follow the
engine convention: ``attention_mask`` carries per-token sequence ids (0 for
padding, 1..N inside each packed segment) and ``position_ids`` carries
per-segment positions. Extra packed metadata keys (``cu_seqlens``,
``max_length_q`` ...) are absorbed via ``**kwargs``.

v4.1-specific mechanisms implemented here:

- CSA2 cross-layer sharing: explicit slot tensors (compressed KV, index K,
  top-k indices, candidate blocks) are published by kv/index source layers
  and passed to Reindex/Reuse layers. Tensors keep their autograd graph, so consumer layers backprop into
  the source layer's compressor.
- Because of that cross-layer graph, the kv-source publish step
  (``DecoderLayer.publish_shared``) runs before the layer call, outside the
  engine's per-layer activation checkpointing; a checkpointed publish would
  orphan the shared tensors' graph.
- Engram conditional memory: the n-gram hash is computed from ``input_ids``
  inside the forward (deterministic buffers, no learned parameters). The
  model loop applies Engram before publishing shared KV and before the
  decoder checkpoint, so every attention branch reads the same stream.
  The full-table ``DeepseekV41Engram`` is the HF/reference variant;
  ``HostOffloadEngram`` (built by the parallel plan) keeps the table on the
  host, EP-sharded, with only the requested rows fetched per lookup
  (FSDPTurbo PR100 design).

Deliberate differences vs deepseek_v4 modeling: delayed hyper-connections
(coefficient sets apply one sublayer later; no learned hc_head parameters),
no extra per-head q norm, no hadamard rotation in the indexer, no ape/overlap
in the compressor, no hash routing, and RMSNorm casts back to the input
dtype. The optional NPU indexer auxiliary loss is attached by the NPU adapter.

Cross-layer tensors (CSA2 shared slots, delayed hc pre-mix) flow between
per-layer checkpoint regions; this requires NON-REENTRANT activation
checkpointing (the training backend enables exactly that). kv-source layers
additionally publish their shared slots outside the checkpointed region.
"""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .source_hyper_connections import (
    DeepseekV4HCSplitSinkhorn,
    HcPost,
)
from . import source_packing_utils as pu
from .configuration_deepseek_v41 import DeepseekV41Config

from torchturbo.models.modules.attention.ulysses import (
    all_gather_sp_with_grad_reduce,
)
from .source_functools import cache

if TYPE_CHECKING:
    from .packed_metadata import PackedAttentionMetadata


# ---------------------------------------------------------------------------
# Rotary helpers (computed from positions on the fly; no 1M-row tables).
# ---------------------------------------------------------------------------


def _yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Inverted frequencies with YaRN correction (v4 formula)."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len <= 0:
        return freqs

    def correction_dim(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp

    return freqs / factor * (1 - smooth) + freqs * smooth


def apply_rope_at_positions(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.LongTensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the last dim of ``x`` (the rope part) at the given positions.

    ``x``: [b, s, d] or [b, s, h, d]; ``positions``: [b, s].
    """
    freqs = positions.unsqueeze(-1).float() * inv_freq.float().to(x.device)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    if inverse:
        freqs_cis = freqs_cis.conj()
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(x_complex.size(0), x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(x_complex.size(0), x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(x.dtype)


def _rope_tail(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.LongTensor,
    rope_head_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate only the trailing rope dims of the last dimension, in place of
    the full-tensor in-place style used by the reference implementation."""
    x_nope, x_rope = torch.split(x, [x.size(-1) - rope_head_dim, rope_head_dim], dim=-1)
    x_rope = apply_rope_at_positions(x_rope, inv_freq, positions, inverse=inverse)
    return torch.cat([x_nope, x_rope], dim=-1)


class DeepseekV41RMSNorm(nn.Module):
    """RMSNorm with fp32 compute and cast back to the input dtype."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(dtype)


# ---------------------------------------------------------------------------
# Delayed hyper-connections.
# ---------------------------------------------------------------------------


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Initial one-hot mix: collapse onto copy 0."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def collapse_hc(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the hc copies into one, weighted by pre_mix.

    x: [b, s, hc, dim]; pre_mix: [b, s, hc].
    """
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


class DeepseekV41HyperConnections(nn.Module):
    """v4.1 hyper-connections with delayed application.

    Coefficient sets are computed from the stream at each sublayer entry but
    applied one sublayer later: the attention input is collapsed with the
    previous layer's ffn pre-mix, the ffn input with this layer's attention
    pre-mix, and the final collapse with the last layer's ffn pre-mix. This
    is why v4.1 has no learned hc_head parameters (unlike v4).
    """

    def __init__(self, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_split_sinkhorn = DeepseekV4HCSplitSinkhorn(hc_mult, sinkhorn_iters, hc_eps)

    def mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b, s, hc, dim] -> (pre, post, comb) coefficient sets."""
        x_flat = x.flatten(2).float()
        # Normalized over the whole flattened hc*d stream, one statistic per
        # token.
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        return self.hc_split_sinkhorn(mixes, hc_scale, hc_base)


class DeepseekV41Head(nn.Module):
    """Output head: fp32 projection over the normed collapsed stream."""

    def __init__(self, text_config):
        super().__init__()
        self.vocab_size = text_config.vocab_size
        self.dim = text_config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        return F.linear(norm(x).float(), self.weight)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# Packed sliding-window indices.
# ---------------------------------------------------------------------------


@cache
def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    device: torch.device,
    attention_mask: torch.Tensor,
    *,
    query_start: int = 0,
    query_end: Optional[int] = None,
) -> torch.Tensor:
    """Sliding-window candidate positions per query, packed-aware.

    Query bounds select a contiguous shard in global sequence coordinates.
    Rows pack multiple segments; window slots belonging to another segment
    (or to padding) are set to -1 via the sequence-id check. Invalid slots
    stay -1 because the selection ``where(mask, window_topk, -1)`` uses the
    original (already -1) values, never the clamped gather indices.
    """
    if query_end is None:
        query_end = seqlen
    base = torch.arange(query_start, query_end, device=device).unsqueeze(1)
    window_topk = (base - window_size + 1).clamp(0) + torch.arange(
        min(seqlen, window_size), device=device
    )
    window_topk = torch.where(window_topk > base, -1, window_topk)  # [s, w]
    window_topk = window_topk.unsqueeze(0).expand(bsz, -1, -1)

    window_token_indices = window_topk.clamp_min(0).reshape(bsz, -1)
    batch_index = torch.arange(bsz, device=device).unsqueeze(1)
    window_ids = attention_mask[batch_index, window_token_indices].view_as(window_topk)
    query_ids = attention_mask[:, query_start:query_end].unsqueeze(-1)
    mask = (window_ids == query_ids) & (query_ids > 0)
    return torch.where(mask, window_topk, torch.full_like(window_topk, -1))


# ---------------------------------------------------------------------------
# Two-level candidate pre-filtering (hierarchical sparse indexer level 1).
# ---------------------------------------------------------------------------


def select_candidate_blocks_packed(
    index_score: torch.Tensor,
    block_ids: torch.Tensor,
    block_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    ratio: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep the ``topk_blocks`` best candidate blocks per query.

    Candidate groups restart at each packed sample, just as they do for
    separate reference-model forwards. Invisible entries have score -inf.
    The group containing the query's newest complete KV block is pinned in.
    Returns a bool mask shaped like ``index_score`` (True = inside the pool).
    """
    keep = torch.zeros_like(index_score, dtype=torch.bool)
    if index_score.size(-1) == 0:
        return keep
    for batch in range(index_score.size(0)):
        starts, ends, sample_ids = pu._segment_spans(block_ids[batch])
        for start, end, sample_id in zip(starts.tolist(), ends.tolist(), sample_ids.tolist()):
            scores = index_score[batch, :, start:end]
            scores = F.pad(scores, (0, -(end - start) % block_size), value=float("-inf"))
            scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
            visible = (
                (block_positions[batch, start:end].unsqueeze(0)
                 < (((position_ids[batch] + 1) // ratio) * ratio).unsqueeze(-1))
                & (attention_mask[batch] == sample_id).unsqueeze(-1)
            ).sum(-1, keepdim=True)
            last_group = (visible - 1) // block_size
            groups = torch.arange(scores.size(-1), device=scores.device)
            scores = scores.masked_fill(groups == last_group, float("inf"))
            top = scores.topk(min(topk_blocks, scores.size(-1)), dim=-1)
            selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                -1, top.indices, top.values > float("-inf")
            )
            keep[batch, :, start:end] = selected.repeat_interleave(block_size, dim=-1)[:, :end-start]
    return keep


# ---------------------------------------------------------------------------
# CSA2 shared slots.
#
# The four slot tensors (compressed KV, index K, top-k indices, candidate
# blocks) flow between layers EXPLICITLY: the model loop resolves them at
# forward time and passes them as arguments to every layer, so a
# checkpointed layer recompute replays against the exact same tensors.
# A mutable shared object would break under reverse-order recomputation
# (later index sources overwrite slots that earlier layers' recomputes
# would then misread). kv-source production runs in publish_shared,
# OUTSIDE the checkpointed region, keeping its autograd graph alive.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compressor and indexer.
# ---------------------------------------------------------------------------


class DeepseekV41Compressor(nn.Module):
    """Packed KV compressor.

    ratio == 1 is a plain bf16 projection (no gate); ratio > 1 pools ratio
    tokens with a learned softmax gate in fp32 (wkv/wgate stored fp32). Both
    paths emit the PRE-rope latent for the packed complete blocks, ordered by
    the block list built by ``packing_utils``.
    """

    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self._sp_group: Optional[torch.distributed.ProcessGroup] = None
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if compress_ratio > 1:
            self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
            self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
        else:
            self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        """Return the pre-rope latent [b, T, head_dim] or None (no block)."""
        dtype = x.dtype
        # Ratio 1 gathers projected KV; gated pooling gathers hidden states
        # so that complete sample blocks reach the compressor provider.
        values = x if self.compress_ratio > 1 else self.wkv(x)
        if self._sp_size > 1:
            values = all_gather_sp_with_grad_reduce(
                values, dim=1, sp_group=self._sp_group, sp_size=self._sp_size,
            )
            position_ids = all_gather_sp_with_grad_reduce(
                position_ids, dim=1, sp_group=self._sp_group, sp_size=self._sp_size,
            )
        if self.compress_ratio > 1:
            # Softmax pooling runs in fp32; the checkpoint stores fp32
            # weights to match. The pooled latent returns to bf16 before the
            # norm so downstream concat with window KV stays single-dtype.
            # NPU auto dispatch prefers the paired PyPTO kernels for CSA2.
            # Other ratios retain the reference provider's block semantics.
            from torchturbo.ops import ops

            return ops.compressor(
                values,
                self.wkv.weight,
                self.wgate.weight,
                self.norm.weight,
                attention_mask,
                position_ids,
                self.compress_ratio,
                self.head_dim,
                self.norm.variance_epsilon,
                impl=None if self.compress_ratio == 2 else "torch_compressor",
            )
        kv = values
        score = torch.zeros_like(kv)
        # v4.1 has no learned position bias; a zero ape keeps the shared
        # packed pooling helper on the plain softmax path.
        ape = torch.zeros(
            self.compress_ratio, self.head_dim, dtype=score.dtype, device=x.device
        )
        latent, _ = pu.compress_packed_kv(
            kv,
            score,
            ape,
            attention_mask,
            position_ids,
            self.compress_ratio,
            self.head_dim,
            overlap=False,
        )
        if latent is None:
            return None
        return self.norm(latent.to(dtype))

    def init_weights(self, _init_std: float):
        for linear in [self.wkv] + ([self.wgate] if self.compress_ratio > 1 else []):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)


class DeepseekV41Indexer(nn.Module):
    """Lightning indexer over the shared compressed KV (no hadamard, no fp4).

    ``owns_k`` layers (kv sources) project their own compressor latent into
    index keys and publish them; the rest score against ``shared.index_k``.
    The candidate-source layer publishes the level-1 block pool; later index
    sources mask their scores into that pool.
    """

    def __init__(self, config, compress_ratio: int, owns_k: bool, is_candidate_source: bool):
        super().__init__()
        self._sp_group: Optional[torch.distributed.ProcessGroup] = None
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.is_candidate_source = is_candidate_source
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim ** -0.5
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        source_id = config.candidate_source_layer_id
        self.candidate_source_ratio = (
            config.compress_ratios[source_id]
            if 0 <= source_id < len(config.compress_ratios) else None
        )

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def compute_index_k(
        self,
        latent: torch.Tensor,
        inv_freq: torch.Tensor,
        block_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """Project this layer's pre-rope latent into shared index keys
        (kv sources only; the tail rotates at block-start positions)."""
        k = self.k_norm(self.wk(latent))
        return _rope_tail(k, inv_freq, block_positions, self.rope_head_dim)

    def score(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor,
        inv_freq: torch.Tensor,
        candidates: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Score the shared index keys and pick the top-k blocks.

        ``index_k``: the group's shared index keys. ``candidates`` masks
        scores into the candidate-source block pool when present. Returns
        (topk row-block indices [b, s, k] with -1 padding, new candidate
        pool or None, index_score [b, s, T] for the future indexer aux
        loss).
        """
        q, weights = self.project_queries(x, qr, inv_freq, position_ids)
        return self._select_blocks(q, weights, index_k, candidates, attention_mask, position_ids)

    def project_queries(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        inv_freq: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project local indexer queries and weights, including query RoPE."""
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = _rope_tail(q, inv_freq, position_ids, self.rope_head_dim)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        return q, weights

    def _select_blocks(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        candidates: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        bsz, seqlen = q.shape[:2]
        global_position_ids = position_ids
        if self._sp_size > 1:
            global_position_ids = all_gather_sp_with_grad_reduce(
                position_ids, dim=1, sp_group=self._sp_group, sp_size=self._sp_size
            )
        block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
            attention_mask, self.compress_ratio, global_position_ids
        )
        if self._sp_size > 1:
            start = self._sp_rank * seqlen
            attention_mask = attention_mask[:, start : start + seqlen]
        num_blocks = block_ids.size(1)
        if index_k is None or index_k.size(1) != num_blocks:
            return None, None, None

        # Index scores in FP32 so the top-k decisions are numerically stable
        # (FSDPTurbo reference: bf16 ties would flip block selection).
        index_score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)

        invisible = pu.packed_compress_mask(
            attention_mask,
            block_ids.unsqueeze(1),
            block_positions.unsqueeze(1),
            self.compress_ratio,
            position_ids,
        )
        invisible = invisible | (attention_mask <= 0).unsqueeze(-1)
        index_score = index_score.masked_fill(invisible, float("-inf"))

        new_candidates = None
        if self.is_candidate_source:
            new_candidates = select_candidate_blocks_packed(
                index_score,
                block_ids,
                block_positions,
                attention_mask,
                position_ids,
                self.compress_ratio,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif candidates is not None:
            index_score = index_score.masked_fill(~candidates, float("-inf"))

        topk = min(self.index_topk, num_blocks)
        if topk == 0:
            return (
                index_score.new_full((bsz, seqlen, 0), -1, dtype=torch.long),
                new_candidates,
                index_score,
            )
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values

        # Flatten before gathering. idxs is [b, s, k]; a [b, 1] batch index
        # only broadcasts against that when b == 1. b > 1 collides s with b.
        flat_idxs = idxs.reshape(bsz, -1)
        batch_index = torch.arange(bsz, device=q.device).unsqueeze(1)
        topk_ids = block_ids[batch_index, flat_idxs].view_as(idxs)
        topk_pos = block_positions[batch_index, flat_idxs].view_as(idxs)
        invalid = pu.packed_compress_mask(
            attention_mask, topk_ids, topk_pos, self.compress_ratio, position_ids
        )
        invalid = invalid | (attention_mask <= 0).unsqueeze(-1)
        return torch.where(invalid, -1, idxs), new_candidates, index_score

    def init_weights(self, _init_std: float):
        for linear in [self.wq_b, self.weights_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        if self.owns_k:
            nn.init.trunc_normal_(self.wk.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.k_norm.weight, mean=1, std=0.02)


# ---------------------------------------------------------------------------
# Attention: pre / inner / post structure (kept for SP patching, see v4).
# ---------------------------------------------------------------------------


class DeepseekV41PreAttentionOps(nn.Module):
    """Projections, window KV, and the shared compressed KV production."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self._sp_group: Optional[torch.distributed.ProcessGroup] = None
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.window_size = config.sliding_window

    def forward(
        self,
        x: torch.Tensor,
        freqs_inv_plain: torch.Tensor,
        freqs_inv_yarn: torch.Tensor,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        indexer,
        compress_kv: Optional[torch.Tensor],
        index_k: Optional[torch.Tensor],
        topk_idxs: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Returns (q, window kv, window topk, kv_compress, new_topk).

        ``new_topk`` is set only by Reindex layers (index sources without
        their own compressor): they run the indexer inside this call. All
        other compressed-KV layers consume the ``topk_idxs`` argument.
        """
        bsz, seqlen, _ = x.size()
        # Layers with compressed KV rope everything with the compress table
        # (YaRN + compress_rope_theta); pure sliding-window layers use the
        # plain table (matches the reference implementation).
        inv_freq = freqs_inv_yarn if self.compress_ratio > 0 else freqs_inv_plain
        rd = self.rope_head_dim

        qr = q_norm(wq_a(x))
        q = wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = _rope_tail(q, inv_freq, position_ids, rd)

        kv = kv_norm(wkv(x))
        kv = _rope_tail(kv, inv_freq, position_ids, rd)

        if self._sp_size > 1:
            kv = all_gather_sp_with_grad_reduce(
                kv, dim=1, sp_group=self._sp_group, sp_size=self._sp_size
            )
        query_start = self._sp_rank * seqlen
        window_topk = get_window_topk_idxs(
            self.window_size, bsz, kv.shape[1], x.device, attention_mask,
            query_start=query_start, query_end=query_start + seqlen,
        )

        new_topk = None
        if self.compress_ratio > 0:
            if indexer is not None:
                # Reindex layer: score the shared index keys here so the
                # recomputation is self-contained.
                new_topk, _, _ = indexer.score(
                    x, qr, index_k, inv_freq, candidates, attention_mask, position_ids
                )
            else:
                new_topk = topk_idxs

        return q, kv, window_topk, compress_kv, new_topk


def _select_kv_rows(kv_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Pick KV rows by index without materializing ``[b, seq, kv_len, dim]``.

    ``kv_states`` is ``[b, kv_len, dim]`` and ``indices`` is ``[b, seq, topk]``
    (already clamped to ``>= 0``). ``torch.gather`` on an expanded KV makes
    NPU allocate the full broadcast, which is larger than one die at 4k.
    ``index_select`` only writes the selected rows, ``[b, seq, topk, dim]``.
    """
    bsz, seqlen, topk = indices.shape
    kv_len, head_dim = kv_states.shape[1], kv_states.shape[-1]
    offsets = torch.arange(bsz, device=indices.device, dtype=indices.dtype)
    flat_index = (indices + offsets.view(bsz, 1, 1) * kv_len).reshape(-1)
    rows = kv_states.contiguous().view(bsz * kv_len, head_dim)
    return rows.index_select(0, flat_index).view(bsz, seqlen, topk, head_dim)


class DeepseekV41InnerAttentionOps(nn.Module):
    """Indexed sparse attention over the selected KV positions.

    Process selected KV rows in 64-index blocks, matching the reference
    online softmax. Round the unnormalized weights to the KV dtype before
    multiplying by V and normalize the FP32 accumulator after adding the
    attention sink. Negative indices are masked out.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self._sp_group: Optional[torch.distributed.ProcessGroup] = None
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        window_topk: torch.Tensor,
        kv_compress: Optional[torch.Tensor],
        compress_topk: Optional[torch.Tensor],
        attn_sink: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        packed: Optional["PackedAttentionMetadata"] = None,
    ) -> torch.Tensor:
        if kv_compress is not None and compress_topk is not None:
            # Shift the compressed block indices past the window KV.
            compress_topk = torch.where(
                compress_topk >= 0, compress_topk + kv.size(1), compress_topk
            )
            topk_idxs = torch.cat([window_topk, compress_topk], dim=-1)
            kv_states = torch.cat([kv, kv_compress], dim=1)
        else:
            topk_idxs = window_topk
            kv_states = kv

        query = q.float()
        row_max = torch.full_like(query[..., 0], -1e30)
        denominator = torch.zeros_like(row_max)
        accumulator = torch.zeros_like(query)
        block_size = 64
        for start in range(0, topk_idxs.size(-1), block_size):
            indices = topk_idxs[..., start : start + block_size]
            values = _select_kv_rows(kv_states, indices.clamp_min(0).long()).float()
            scores = (query @ values.transpose(-1, -2)) * self.softmax_scale
            scores = scores.masked_fill(indices.unsqueeze(2) < 0, float("-inf"))
            next_max = torch.maximum(row_max, scores.amax(dim=-1))
            rescale = torch.exp(row_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(-1))
            denominator = denominator * rescale + weights.sum(dim=-1)
            weighted_values = weights.to(kv_states.dtype).float() @ values
            accumulator = accumulator * rescale.unsqueeze(-1) + weighted_values
            row_max = next_max

        # All-masked rows have a zero numerator. Avoid exp(sink - (-1e30))
        # on these rows so that the sink gradient also stays finite.
        sink_delta = torch.where(denominator > 0, attn_sink.float() - row_max, 0.0)
        denominator = denominator + torch.exp(sink_delta)
        return (accumulator / denominator.unsqueeze(-1)).to(q.dtype)



class DeepseekV41PostAttentionOps(nn.Module):
    """Inverse rope on the output tail plus the grouped low-rank output
    projection."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.rope_head_dim = config.qk_rope_head_dim
        self.head_dim = config.head_dim

    def forward(
        self,
        o: torch.Tensor,
        inv_freq: torch.Tensor,
        position_ids: torch.LongTensor,
        bsz: int,
        seqlen: int,
        n_local_groups: int,
        wo_a,
        wo_b,
    ):
        o = _rope_tail(o, inv_freq, position_ids, self.rope_head_dim, inverse=True)
        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a_weight = wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
        return wo_b(o.reshape(bsz, seqlen, -1))


class DeepseekV41Attention(nn.Module):
    """Latent attention with sliding window plus CSA2 compressed KV.

    Parameters are defined here directly (checkpoint-compatible names);
    the compressor exists only on kv-source layers and the indexer only on
    index-source layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids

        self.q_lora_rank = config.q_lora_rank
        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV41RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, self.eps)

        self.o_lora_rank = config.o_lora_rank
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))

        if self.is_kv_source:
            self.compressor = DeepseekV41Compressor(config, self.compress_ratio)
        if self.is_index_source:
            self.indexer = DeepseekV41Indexer(
                config,
                self.compress_ratio,
                owns_k=self.is_kv_source,
                is_candidate_source=layer_idx == config.candidate_source_layer_id,
            )

        self.pre_attention_ops = DeepseekV41PreAttentionOps(config, layer_idx)
        self.inner_attention_ops = DeepseekV41InnerAttentionOps(config, layer_idx)
        self.post_attention_ops = DeepseekV41PostAttentionOps(config)

        # Lazy rope frequencies: plain (theta, no YaRN) and compress
        # (YaRN + compress theta). Built on first use; None buffers are
        # skipped by FSDP/to_empty.
        self.register_buffer("inv_freq_plain", None, persistent=False)
        self.register_buffer("inv_freq_yarn", None, persistent=False)

    def _ensure_inv_freq(self, device: torch.device):
        if self.inv_freq_plain is None:
            dim = self.rope_head_dim
            freqs = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.inv_freq_plain = freqs.to(device)
        if self.inv_freq_yarn is None:
            self.inv_freq_yarn = _yarn_inv_freq(
                self.rope_head_dim,
                self.config.compress_rope_theta,
                self.config.rope_factor,
                self.config.original_max_position_embeddings,
                self.config.beta_fast,
                self.config.beta_slow,
            ).to(device)

    def _prepare_indexer_loss(self, hidden_states, kv_compress, index_k, used_topk, attention_mask):
        """Optional auxiliary-loss adapter installed before training."""
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        compress_kv: Optional[torch.Tensor],
        index_k: Optional[torch.Tensor],
        topk_idxs: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor],
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        **kwargs,
    ):
        bsz, seqlen, _ = hidden_states.size()
        self._ensure_inv_freq(hidden_states.device)

        # Only Reindex layers (index source without their own compressor)
        # run the indexer inside the layer; kv sources got their top-k from
        # publish_shared.
        indexer = (
            self.indexer
            if self.is_index_source and not self.is_kv_source
            else None
        )

        q, kv, window_topk, kv_compress, new_topk = self.pre_attention_ops(
            hidden_states,
            self.inv_freq_plain,
            self.inv_freq_yarn,
            self.wq_a,
            self.q_norm,
            self.wq_b,
            self.wkv,
            self.kv_norm,
            indexer,
            compress_kv,
            index_k,
            topk_idxs,
            candidates,
            attention_mask,
            position_ids,
        )

        used_topk = new_topk if new_topk is not None else topk_idxs
        auxiliary = self._prepare_indexer_loss(
            hidden_states, kv_compress, index_k, used_topk, attention_mask
        )
        o = self.inner_attention_ops(
            q,
            kv,
            window_topk,
            kv_compress,
            used_topk,
            self.attn_sink,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed=auxiliary.packed if auxiliary is not None else None,
        )
        target_probs = None
        if auxiliary is not None:
            target_probs = auxiliary.target_probs(
                self.inner_attention_ops, q, kv, window_topk,
                kv_compress, used_topk, self.attn_sink,
            )

        inv_freq = self.inv_freq_yarn if self.compress_ratio > 0 else self.inv_freq_plain
        # Groups shrink when SP splits heads; SP=1 keeps all of them.
        n_local_groups = self.n_groups // (self.n_heads // q.shape[2])
        attn_output = self.post_attention_ops(
            o,
            inv_freq,
            position_ids,
            bsz,
            seqlen,
            n_local_groups,
            self.wo_a,
            self.wo_b,
        )
        if auxiliary is not None and target_probs is not None:
            attn_output = auxiliary.attach(
                self, attn_output, hidden_states, index_k, used_topk,
                target_probs, inv_freq, attention_mask, position_ids,
            )
        return attn_output, new_topk

    def init_weights(self, init_std: float):
        for linear in [self.wq_a, self.wq_b, self.wkv, self.wo_a, self.wo_b]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.q_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.kv_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.attn_sink, mean=0.0, std=init_std)
        if self.is_kv_source:
            self.compressor.init_weights(init_std)
        if self.is_index_source:
            self.indexer.init_weights(init_std)


# ---------------------------------------------------------------------------
# Engram conditional memory.
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def _find_next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


def _build_engram_primes(
    num_layers: int,
    max_ngram_size: int,
    n_heads: int,
    engram_vocab_size: int,
) -> torch.Tensor:
    """Bucket modulus per (layer, n-gram size, head); primes are drawn in
    order from a global pool so bucket ranges never overlap."""
    seen: set = set()
    primes = []
    for _ in range(num_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = engram_vocab_size - 1
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    # [num_layers, max_ngram_size - 1, n_heads]
    return torch.tensor(primes, dtype=torch.int64)


def _compute_engram_multipliers(
    layer_ids,
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback); per-layer RNG so layers hash
    differently. Matches the reference (numpy PCG64, seed 10007 * layer)."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)  # [num_layers, max_ngram_size]


def build_compressed_token_map(tokenizer) -> tuple:
    """Map token ids onto the compressed id space where surface variants
    collapse together ("The"/"the"/"THE" hash identically).

    Ported from the reference implementation; the returned compressed vocab
    size is asserted against the config because a mismatch would silently
    rehash the whole engram table.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


class NgramHashState(nn.Module):
    """Deterministic n-gram hash for the engram lookup (no parameters).

    Buffers are built lazily because tokenizer access is only possible once
    the model path is known (engine path passes it via ``build``; the HF
    reference path falls back to ``config._name_or_path``).
    """

    def __init__(self, top_config, text_config):
        super().__init__()
        self._sp_group: Optional[torch.distributed.ProcessGroup] = None
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self.layer_ids = tuple(text_config.engram_layer_ids)
        self.max_ngram_size = text_config.engram_max_ngram_size
        self.n_heads = text_config.engram_n_heads
        self.pad_token_id = text_config.engram_pad_token_id
        self.compressed_vocab_size = text_config.engram_compressed_vocab_size
        self.engram_vocab_size = text_config.engram_vocab_size
        self.vocab_size = text_config.vocab_size
        self._num_embeddings = list(text_config.engram_num_embeddings)
        self._top_config = top_config
        self.hash_fingerprint: Optional[str] = None

        self.register_buffer("token_map", None, persistent=False)
        self.register_buffer("multipliers", None, persistent=False)
        self.register_buffer("primes", None, persistent=False)
        self.register_buffer("offsets", None, persistent=False)

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def build(self, device: torch.device, model_path: Optional[str] = None):
        if self.token_map is not None:
            return
        import hashlib

        from transformers import AutoTokenizer

        path = model_path or self._top_config._name_or_path
        if not path:
            raise RuntimeError(
                "NgramHashState needs a tokenizer path: pass model_path to "
                "build() or set config._name_or_path."
            )
        tokenizer = AutoTokenizer.from_pretrained(path)
        token_map, compressed_vocab_size = build_compressed_token_map(tokenizer)
        if compressed_vocab_size != self.compressed_vocab_size:
            raise RuntimeError(
                f"engram compressed vocab size mismatch: tokenizer gives "
                f"{compressed_vocab_size}, config expects "
                f"{self.compressed_vocab_size} - the table would be silently "
                f"rehashed. Check the tokenizer against the checkpoint."
            )

        # Layout validation (FSDPTurbo reference): layer ids must be unique
        # and the tables must cover every prime bucket range.
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("engram layer ids must be unique.")

        primes = _build_engram_primes(
            len(self.layer_ids), self.max_ngram_size, self.n_heads, self.engram_vocab_size
        )
        for rows, layer_primes in zip(self._num_embeddings, primes.tolist()):
            if rows < sum(sum(heads) for heads in layer_primes):
                raise ValueError(
                    "engram table rows do not cover the generated hash bucket ranges."
                )
        multipliers = _compute_engram_multipliers(
            self.layer_ids, self.max_ngram_size, compressed_vocab_size
        )
        # Offsets: cumulative start index of each (n-gram, head) bucket range
        # inside the layer's table.
        offsets = []
        for layer_primes in primes.tolist():
            flat_sizes = [p for per_ngram in layer_primes for p in per_ngram]
            starts = [0]
            for size in flat_sizes[:-1]:
                starts.append(starts[-1] + size)
            offsets.append(starts)

        # Integrity fingerprint over the full hash ABI (token map + bucket
        # layout + multipliers): a tokenizer revision change rehashes every
        # row id while keeping shapes valid, so shape checks cannot catch it.
        self.hash_fingerprint = hashlib.sha256(
            torch.tensor(token_map, dtype=torch.int64).numpy().tobytes()
            + primes.numpy().tobytes()
            + multipliers.numpy().tobytes()
        ).hexdigest()

        self.token_map = torch.tensor(token_map, dtype=torch.int64).to(device)
        self.multipliers = multipliers.to(device)
        self.primes = primes.to(device)
        self.offsets = torch.tensor(offsets, dtype=torch.int64).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """hash ids [b, s, n_engram_layers, n_hash_cols]."""
        if self.token_map is None:
            self.build(input_ids.device)

        bsz, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]  # [b, s]
        pad_id = self.token_map[self.pad_token_id]
        seq = attention_mask

        positions = torch.arange(seqlen, device=input_ids.device)
        blocked = torch.zeros(bsz, seqlen, dtype=torch.bool, device=input_ids.device)
        tokens = []
        for shift in range(self.max_ngram_size):
            src_pos = (positions - shift).clamp_min(0)
            source = compressed.gather(1, src_pos.unsqueeze(0).expand(bsz, -1))
            src_seq = seq.gather(1, src_pos.unsqueeze(0).expand(bsz, -1))
            # An n-gram never crosses a segment boundary or touches padding.
            blocked = blocked | (positions - shift < 0) | (src_seq != seq) | (src_seq == 0) | (seq == 0)
            tokens.append(torch.where(blocked, pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, max_ngram_size]

        # XOR the multiplied ids one lookback at a time; each running value
        # is the hash of an (i+1)-gram landing in its own prime bucket.
        products = tokens.unsqueeze(2) * self.multipliers  # [b, s, L, n]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        hash_cols = torch.cat(hashes, dim=-1)  # [b, s, L, (n-1)*H]
        return hash_cols + self.offsets


class DeepseekV41Engram(nn.Module):
    """Gated n-gram lookup written into the residual stream.

    The full-table variant used by the HF/reference path; the training path
    swaps in HostOffloadEngram (same param names, host-offloaded table).
    """

    def __init__(self, text_config, layer_idx: int, layer_hash_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = layer_hash_index
        self.dim = text_config.hidden_size
        self.hc_mult = text_config.hc_mult
        self.clamp_value = 1e-6
        self.eps = text_config.rms_norm_eps

        num_rows = text_config.engram_num_embeddings[layer_hash_index]
        n_hash_cols = (text_config.engram_max_ngram_size - 1) * text_config.engram_n_heads
        self.n_hash_cols = n_hash_cols
        self.embed = nn.Embedding(num_rows, text_config.engram_head_dim)
        self.wkv = nn.Linear(
            n_hash_cols * text_config.engram_head_dim,
            text_config.hidden_size * (text_config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))

    def _embed_lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(hash_ids)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """x: [b, s, hc, dim]; hash_ids: [b, s, n_hash_cols]."""
        memory = self._embed_lookup(hash_ids).flatten(-2).to(x.dtype)
        # Host offload keeps FP32 master parameters. Match the inference
        # GEMM dtype instead of using the NPU's mixed BF16/FP32 linear path.
        kv = F.linear(memory, self.wkv.weight.to(memory.dtype))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # Normalized per (token, hc copy) over dim, NOT jointly over copies.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim ** -0.5
        # Signed sqrt before the sigmoid, matching the training kernel; the
        # where-form avoids copysign (NPU portability, FSDPTurbo reference).
        magnitude = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def init_weights(self, _init_std: float):
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wkv.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)


# Capacity of a full packed micro-batch of engram lookups (128k tokens).
MAX_ENGRAM_PACKED_TOKENS = 131072





class DeepseekV41Gate(nn.Module):
    """noaux_tc gating: the bias steers selection only; routing weights come
    from the unbiased scores. bias and bias_vl are trainable fp32 parameters
    (FSDPTurbo reference); bias_vl is consumed on multimodal inputs only.

    The engine training path swaps this module for MoELayer, whose router
    keeps expert_bias as a persistent buffer (torchturbo convention).
    """

    def __init__(self, text_config, vision_enabled: bool):
        super().__init__()
        self.topk = text_config.num_experts_per_tok
        self.score_func = text_config.scoring_func
        self.gate_temp = text_config.gate_temp
        self.norm_topk_prob = text_config.norm_topk_prob
        self.route_scale = text_config.routed_scaling_factor

        self.weight = nn.Parameter(
            torch.empty(text_config.n_routed_experts, text_config.hidden_size)
        )
        self.bias = nn.Parameter(torch.empty(text_config.n_routed_experts, dtype=torch.float32))
        if vision_enabled:
            self.bias_vl = nn.Parameter(
                torch.empty(text_config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.bias_vl = None

    def forward(self, hidden_states: torch.Tensor):
        """x: [n, dim]."""
        scores = F.linear(hidden_states.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()

        # the bias picks experts but does not scale them: weights come from
        # the raw scores
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices

    def init_weights(self, _init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=_init_std)
        nn.init.zeros_(self.bias)
        if self.bias_vl is not None:
            nn.init.zeros_(self.bias_vl)


class DeepseekV41Expert(nn.Module):
    """SwiGLU FFN with the training clamps (up clamped both sides, gate from
    above)."""

    def __init__(self, text_config):
        super().__init__()
        self.swiglu_limit = text_config.swiglu_limit
        self.w1 = nn.Linear(text_config.hidden_size, text_config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(text_config.moe_intermediate_size, text_config.hidden_size, bias=False)
        self.w3 = nn.Linear(text_config.hidden_size, text_config.moe_intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = x * weights
        return self.w2(x.to(dtype))


class DeepseekV41MoE(nn.Module):
    """Reference MoE (per-expert loop). The parallel plan replaces this with
    the fused MoELayer + GroupedMLP and reuses shared_experts."""

    def __init__(self, text_config, layer_idx: int, vision_enabled: bool):
        super().__init__()
        # parallel_plan reads this to pick the per-layer swiglu clamp.
        self.layer_idx = layer_idx
        self.dim = text_config.hidden_size
        self.n_routed_experts = text_config.n_routed_experts
        self.gate = DeepseekV41Gate(text_config, vision_enabled)
        self.experts = nn.ModuleList(
            [DeepseekV41Expert(text_config) for _ in range(self.n_routed_experts)]
        )
        assert text_config.n_shared_experts == 1
        self.shared_experts = DeepseekV41Expert(text_config)

    def forward(self, hidden_states: torch.Tensor):
        shape = hidden_states.size()
        x = hidden_states.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.n_routed_experts):
            mask = indices == i
            if mask.any():
                idx, top = torch.where(mask)
                y[idx] += self.experts[i](x[idx], weights[idx, top].unsqueeze(-1))
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    def init_weights(self, init_std: float):
        self.gate.init_weights(init_std)
        for expert in self.experts:
            for linear in [expert.w1, expert.w2, expert.w3]:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for linear in [self.shared_experts.w1, self.shared_experts.w2, self.shared_experts.w3]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, text_config, layer_idx: int, vision_enabled: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self._tt_publish_shared_fsdp_registered = False
        self._tt_publish_collapse_checkpoint = None
        self.norm_eps = text_config.rms_norm_eps
        self.hc_mult = text_config.hc_mult
        self.is_kv_source = layer_idx in text_config.kv_source_layer_ids

        self.attn = DeepseekV41Attention(text_config, layer_idx)
        self.ffn = DeepseekV41MoE(text_config, layer_idx, vision_enabled)
        self.attn_norm = DeepseekV41RMSNorm(text_config.hidden_size, self.norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(text_config.hidden_size, self.norm_eps)

        self.engram = None
        self.engram_hash_index = -1
        if layer_idx in text_config.engram_layer_ids:
            self.engram_hash_index = text_config.engram_layer_ids.index(layer_idx)
            self.engram = DeepseekV41Engram(text_config, layer_idx, self.engram_hash_index)

        hc_mult = self.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * text_config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        self.hc = DeepseekV41HyperConnections(
            hc_mult, text_config.hc_sinkhorn_iters, text_config.hc_eps, self.norm_eps
        )
        self.hc_post = HcPost()

    def publish_shared(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        compress_kv: Optional[torch.Tensor],
        index_k: Optional[torch.Tensor],
        topk_idxs: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Produce this kv source's shared slots OUTSIDE activation
        checkpointing.

        Returns the updated (compress_kv, index_k, topk_idxs, candidates)
        tuple; the caller threads it into the layer call as explicit
        arguments, so checkpointed recomputes replay against exactly these
        tensors and the cross-layer autograd graph stays alive. The caller
        must apply this layer's Engram before passing hidden_states here.
        """
        if not self.is_kv_source:
            return compress_kv, index_k, topk_idxs, candidates
        attn = self.attn
        attn._ensure_inv_freq(hidden_states.device)
        checkpoint_collapse = self._tt_publish_collapse_checkpoint
        if checkpoint_collapse is not None and self.training and torch.is_grad_enabled():
            x = checkpoint_collapse(collapse_hc, hidden_states, pre_mix)
        else:
            x = collapse_hc(hidden_states, pre_mix)
        x = self.attn_norm(x)
        latent = attn.compressor(x, attention_mask, position_ids)

        global_position_ids = position_ids
        sp_size = attn.compressor._sp_size

        if sp_size > 1:
            global_position_ids = all_gather_sp_with_grad_reduce(
                position_ids,
                dim=1,
                sp_group=attn.compressor._sp_group,
                sp_size=sp_size,
            )

        inv_freq = attn.inv_freq_yarn
        if latent is not None:
            block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
                attention_mask, attn.compress_ratio, global_position_ids
            )
            compress_kv = _rope_tail(
                latent, inv_freq, block_positions, attn.rope_head_dim
            )
            if attn.indexer is not None and attn.indexer.owns_k:
                index_k = attn.indexer.compute_index_k(latent, inv_freq, block_positions)
        else:
            # No complete block in this group: consumers fall back to
            # window-only attention.
            compress_kv = None
        if attn.is_index_source:
            qr = attn.q_norm(attn.wq_a(x))
            topk, new_candidates, _ = attn.indexer.score(
                x, qr, index_k, inv_freq, candidates, attention_mask, position_ids
            )
            if topk is not None:
                topk_idxs = topk
            if new_candidates is not None:
                candidates = new_candidates
        return compress_kv, index_k, topk_idxs, candidates

    def _hc_pre(self, hidden_states, fn, scale, base, pre_mix):
        """Compute the next mixes and collapse using the incoming delayed mix."""
        next_pre, post, comb = self.hc.mixes(hidden_states, fn, scale, base)
        return collapse_hc(hidden_states, pre_mix), post, comb, next_pre

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        compress_kv: Optional[torch.Tensor],
        index_k: Optional[torch.Tensor],
        topk_idxs: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Run the sublayers on the stream after this layer's Engram."""
        # Attention sublayer: mixes from the attention-entry stream, applied
        # to the ffn input; the attention input uses the incoming pre-mix
        # (the previous layer's ffn pre).
        residual = hidden_states
        hidden_states, attn_post, attn_comb, attn_pre = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale,
            self.hc_attn_base, pre_mix,
        )
        hidden_states = self.attn_norm(hidden_states)
        attn_output, new_topk = self.attn(
            hidden_states,
            compress_kv,
            index_k,
            topk_idxs,
            candidates,
            attention_mask,
            position_ids,
        )
        hidden_states = self.hc_post(attn_output, residual, attn_post, attn_comb)

        # FFN sublayer: the ffn pre-mix becomes the next layer's attention
        # input mix (returned to the caller).
        residual = hidden_states
        hidden_states, ffn_post, ffn_comb, ffn_pre = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale,
            self.hc_ffn_base, attn_pre,
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre, new_topk

    def init_weights(self, init_std: float):
        for norm in [self.attn_norm, self.ffn_norm]:
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        self.attn.init_weights(init_std)
        self.ffn.init_weights(init_std)
        if self.engram is not None:
            self.engram.init_weights(init_std)
        for param in [
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
        ]:
            nn.init.trunc_normal_(param, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# Vision tower and aligner (parameter stubs).
#
# Phase 1 is text-only: no multimodal data path exists, so these modules
# carry weights only, for checkpoint exchange and the parameter-group and
# freeze bookkeeping in the model wrapper. Calling them hits the default
# nn.Module "missing forward" error. The full implementation lives in git
# history.
# ---------------------------------------------------------------------------


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.n_heads = vision_config.num_attention_heads
        self.head_dim = vision_config.hidden_size // self.n_heads
        self.wqkv = nn.Linear(vision_config.hidden_size, 3 * vision_config.hidden_size)
        self.wo = nn.Linear(vision_config.hidden_size, vision_config.hidden_size)


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.w1 = nn.Linear(
            vision_config.hidden_size, 2 * vision_config.intermediate_size, bias=False
        )
        self.w2 = nn.Linear(vision_config.intermediate_size, vision_config.hidden_size, bias=False)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.norm1 = DeepseekV41RMSNorm(vision_config.hidden_size)
        self.attn = DeepseekV41VisionAttention(vision_config)
        self.norm2 = DeepseekV41RMSNorm(vision_config.hidden_size)
        self.mlp = DeepseekV41VisionMLP(vision_config)


class _PatchEmbed(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.proj = nn.Linear(
            3 * vision_config.patch_size ** 2, vision_config.hidden_size
        )


class DeepseekV41VisionTower(nn.Module):
    """Parameter stub of the DeepSeek ViT tower (text-only phase 1)."""

    def __init__(self, vision_config):
        super().__init__()
        self.patch_embed = _PatchEmbed(vision_config)
        self.blocks = nn.ModuleList(
            [DeepseekV41VisionBlock(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(vision_config.hidden_size)


class DeepseekV41Aligner(nn.Module):
    """Parameter stub of the vision-text aligner (text-only phase 1)."""

    def __init__(self, vision_config, text_config):
        super().__init__()
        self.downsample_ratio = vision_config.downsample_ratio
        in_dim = vision_config.hidden_size * self.downsample_ratio ** 2
        self.w1 = nn.Linear(in_dim, text_config.hidden_size)
        self.w2 = nn.Linear(text_config.hidden_size, text_config.hidden_size)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = ""  # checkpoint weights have no "model." prefix
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, DeepseekV41DecoderLayer):
            module.init_weights(std)
        elif isinstance(module, DeepseekV41Head):
            module.init_weights(std)
        elif isinstance(module, DeepseekV41Engram):
            module.init_weights(std)
        elif isinstance(module, DeepseekV41RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    """DeepSeek V4.1 for causal LM (flat layout, checkpoint-compatible)."""

    _tied_weights_keys = ["head.weight"]

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        text_config = config.text_config
        if text_config.num_nextn_predict_layers > 0:
            raise NotImplementedError(
                "DSpark MTP layers are not implemented yet (phase 1 trains "
                "the backbone only); enable the engine MTP config once "
                "dspark support lands."
            )
        self.vocab_size = text_config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed = nn.Embedding(
            text_config.vocab_size, text_config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(text_config, i, config.vision_enabled)
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV41RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.head = DeepseekV41Head(text_config)

        if config.vision_enabled:
            self.vision = DeepseekV41VisionTower(config.vision_config)
            self.aligner = DeepseekV41Aligner(config.vision_config, text_config)
            self.image_start = nn.Parameter(torch.empty(text_config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(text_config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(text_config.hidden_size))

        self.engram_hash = (
            NgramHashState(config, text_config)
            if text_config.engram_layer_ids
            else None
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head.weight = (
            new_embeddings.weight if hasattr(new_embeddings, "weight") else new_embeddings
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": kwargs.get("attention_mask"),
        }

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states, self.norm)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """The packed metadata keys added by the engine (``cu_seqlens``,
        ``cu_seq_lens_q``/``k``, ``max_length_q``/``k``, ``max_seqlen``) are
        absorbed by ``**kwargs`` and unused here: the packed semantics live
        in ``attention_mask`` (sequence ids) and ``position_ids``."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)

        if self.engram_hash is not None and input_ids is None:
            raise ValueError("input_ids is required when the model has engram layers")

        engram_hashes = None
        if self.engram_hash is not None:
            engram_hashes = self.engram_hash(input_ids, attention_mask)

        hc_mult = self.config.text_config.hc_mult
        hidden_states = inputs_embeds.unsqueeze(2).repeat(1, 1, hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states, hc_mult)

        all_hidden_states = () if output_hidden_states else None
        # CSA2 slots threaded explicitly: (compress_kv, index_k, topk,
        # candidates). kv sources refresh them in publish_shared (outside
        # any checkpoint); Reindex layers refresh topk via their return
        # value. Checkpointed layers receive the slots as arguments, so
        # recomputes replay against identical tensors.
        slots = (None, None, None, None)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Engram precedes every attention branch, including shared KV
            # production outside the decoder checkpoint. Training replaces
            # this module with a no_shard HostOffloadEngram.
            if decoder_layer.engram is not None:
                hash_ids = engram_hashes[:, :, decoder_layer.engram_hash_index, :]
                hidden_states = decoder_layer.engram(hidden_states, hash_ids)

            if decoder_layer.is_kv_source:
                slots = decoder_layer.publish_shared(
                    hidden_states, pre_mix, *slots, attention_mask, position_ids
                )
            if self.gradient_checkpointing and self.training:
                hidden_states, pre_mix, new_topk = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    pre_mix,
                    *slots,
                    attention_mask,
                    position_ids,
                )
            else:
                hidden_states, pre_mix, new_topk = decoder_layer(
                    hidden_states,
                    pre_mix,
                    *slots,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            if new_topk is not None:
                slots = (slots[0], slots[1], new_topk, slots[3])

        # Final collapse with the last layer's ffn pre-mix (delayed HC), then
        # norm + fp32 projection.
        hidden_states = collapse_hc(hidden_states, pre_mix)
        logits = self._compute_logits(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
