"""Instance-scoped NPU training forwards for GLM-5.3-Flash.

Installed by on_parallelized, after parallel placement and adapter creation.
Unadapted operations retain the reference implementation. No process-wide class
patches, token-level lightning indexer, or indexer KL objective are used here.
"""

from __future__ import annotations

import os
import types
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from torchturbo.logging import get_logger
from torchturbo.ops.providers.npu.mhc_pre_sinkhorn import (
    NpuMhcPreSinkhorn,
    mhc_pre_sinkhorn_unavailable_reason,
    mhc_pre_sinkhorn_unsupported_reason,
)

from .modeling_glm5_3_flash import (
    Glm5NextTextAttention,
    Glm5NextTextDecoderLayer,
    Glm5NextTextLinearAttention,
    Glm5NextTextRMSNorm,
    Glm5NextTextRMSNormGated,
    l2norm,
    sequence_segments,
    recurrent_kimi_delta_attention,
)


logger = get_logger(__name__)
_mhc_pre = NpuMhcPreSinkhorn()


def npu_mhc_pre(self, hidden, kind):
    phi = getattr(self, f"hc_{kind}_fn")
    alpha = getattr(self, f"hc_{kind}_scale")
    bias = getattr(self, f"hc_{kind}_base")
    reason = mhc_pre_sinkhorn_unsupported_reason(
        hidden, phi, alpha, bias, self.hc_mult, self.hc_sinkhorn_iters
    )
    if reason is None:
        reason = mhc_pre_sinkhorn_unavailable_reason()
    if reason is not None:
        logger.warning_rank0_once("GLM-5.3-Flash mHC pre fallback: %s", reason)
        return self._glm5_reference_pre(hidden, kind)
    # Once selected, propagate execution errors instead of silently changing
    # computation paths during training or checkpoint recomputation.
    collapsed, post, comb = _mhc_pre(
        hidden, phi, alpha, bias, self.hc_mult, self.hc_sinkhorn_iters,
        hc_eps=self.hc_eps, norm_eps=self.hc_norm_eps,
    )
    logger.info_rank0_once("GLM-5.3-Flash using fused NPU mhc_pre_sinkhorn")
    b, s, n, _ = hidden.shape
    return collapsed, post.to(hidden.dtype), comb.reshape(b, s, n, n).to(hidden.dtype)


def npu_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type != "npu" or self.weight.dtype != hidden_states.dtype:
        return self._glm5_reference_forward(hidden_states)
    import torch_npu

    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    if hidden_states.device.type != "npu":
        return self._glm5_reference_forward(hidden_states, gate)
    import torch_npu

    # The installed FLA/Triton-Ascend gated norm can return different outputs
    # for identical inputs during FSDP checkpoint recomputation. Keep the
    # reference FP32 normalization/gate arithmetic with the supported NPU norm.
    # Normalize in FP32 so rounding happens after the sigmoid gate, not before.
    normalized = torch_npu.npu_rms_norm(
        hidden_states.float(), self.weight.float(), self.variance_epsilon
    )[0]
    return (normalized * gate.float().sigmoid()).to(hidden_states.dtype)


def _exchange(self, tensor, inverse=False):
    if self.sp_size == 1:
        return tensor
    from ..modules.attention.ulysses.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    fn = gather_heads_scatter_seq if inverse else gather_seq_scatter_heads
    return fn(
        tensor, seq_dim=1, head_dim=2, sp_group=self.sp_group, sp_size=self.sp_size
    )


def _exchange_kda_inputs(self, qkv, gate, beta, merge=True):
    """Exchange all projected inputs together without changing their dtypes."""
    inputs = (*qkv, gate, beta.unsqueeze(-1))
    if (
        self.sp_size > 1
        and merge
        and os.environ.get("TORCHTURBO_GLM5_KDA_EXCHANGE", "merged") == "merged"
        and all(x.dtype == inputs[0].dtype for x in inputs)
    ):
        widths = [x.shape[-1] for x in inputs]
        packed = _exchange(self, torch.cat(inputs, dim=-1))
        q, k, v, gate, beta = packed.split(widths, dim=-1)
        return (q, k, v), gate, beta.squeeze(-1)
    return (
        tuple(_exchange(self, x) for x in qkv),
        _exchange(self, gate),
        _exchange(self, beta),
    )


@lru_cache(maxsize=8)
def _kda_pack_metadata(runs, sequence_length, device):
    """Share immutable batch metadata across KDA layers and checkpoint replay.

    The value-based key includes every segment boundary and the concrete device;
    mutating a caller's segment list cannot produce a stale cache hit. Keep only
    eight layouts so variable packing cannot retain unbounded device tensors.
    """
    locations, lengths, cumulative = [], [], [0]
    for row, row_runs in enumerate(runs):
        for start, end in row_runs:
            locations.extend(
                range(row * sequence_length + start, row * sequence_length + end)
            )
            lengths.append(end - start)
            cumulative.append(cumulative[-1] + end - start)
    indices = torch.tensor(locations, dtype=torch.long, device=device)
    cu_cpu = torch.tensor(cumulative, dtype=torch.int32, device="cpu")
    return indices, tuple(lengths), cu_cpu, cu_cpu.to(device=device)


@lru_cache(maxsize=1)
def _get_npu_causal_conv1d():
    from fla.modules.convolution import causal_conv1d

    return causal_conv1d


def _kda_reference_conv(x, weight, lengths):
    """Original per-segment convolution, also used for explicit A/B validation."""
    outputs, start = [], 0
    for length in lengths:
        value = F.conv1d(
            x[:, start : start + length].transpose(1, 2).to(weight.dtype),
            weight,
            padding=weight.shape[-1] - 1,
            groups=weight.shape[0],
        )[..., :length]
        outputs.append(F.silu(value).transpose(1, 2).to(x.dtype))
        start += length
    return torch.cat(outputs, dim=1)


def _kda_varlen_conv(x, weight, cu_seqlens, cu_seqlens_cpu):
    """Three calls per KDA layer, independent of the packed segment count."""
    # F.conv1d first receives x.to(weight.dtype), then autocast converts both
    # operands. FLA is a custom Function and does not perform that cast for us.
    conv_dtype = weight.dtype
    if torch.is_autocast_enabled(x.device.type):
        conv_dtype = torch.get_autocast_dtype(x.device.type)
    value, _ = _get_npu_causal_conv1d()(
        x=x.to(weight.dtype).to(conv_dtype).contiguous(),
        weight=weight.squeeze(1).to(conv_dtype).contiguous(),
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        # GLM rounds the convolution result before SiLU. Fusing SiLU into FLA
        # would remove that BF16 rounding boundary and change the model.
        activation=None,
        backend="triton",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return F.silu(value).to(x.dtype)


@lru_cache(maxsize=1)
def _get_npu_chunk_kda():
    """Use the same fixed-workspace FLA build as bailing_moe_v3_vl."""
    try:
        import triton
        from fla.ops.kda import chunk_kda, gate
    except ImportError as error:
        raise ImportError(
            "GLM-5.3-Flash NPU KDA requires the fixed-workspace FLA 0.6.0 build "
            "and Triton-Ascend (see docs/glm5_3_flash_npu_kda.md); "
            "kda_backend='torch' selects the reference implementation."
        ) from error

    # Stock BS=32 exceeds Ascend's coreDim limit on long sequences. Mirror
    # bailing's larger tiles; only extend the known kernel, once per process.
    kernel = gate.kda_gate_chunk_cumsum_vector_kernel
    tuner = getattr(kernel, "fn", None)
    if tuner is None or not hasattr(tuner, "configs"):
        tuner = kernel
    existing = {config.kwargs.get("BS") for config in tuner.configs}
    for size, warps in ((64, 2), (128, 4)):
        if size not in existing:
            tuner.configs.append(triton.Config({"BS": size}, num_warps=warps))
    return chunk_kda


def _npu_chunk_kda(
    q, k, v, gate, beta, A_log, dt_bias, lengths, lower_bound, *, cu_seqlens=None
):
    """One varlen call over valid segments; each segment starts with zero state."""
    chunk_kda = _get_npu_chunk_kda()
    # Unlike F.normalize, GLM uses sqrt(sum(x*x) + 1e-6); preserve it in FP32.
    q, k = (l2norm(x.float()).to(x.dtype) for x in (q, k))
    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0] + list(lengths), device=q.device, dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)
    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return out


def npu_kda_forward(
    self, hidden_states, segments=None, attention_mask=None, position_ids=None, **kwargs
):
    if hidden_states.device.type != "npu":
        return self._glm5_reference_forward(
            hidden_states, segments, attention_mask, position_ids, **kwargs
        )
    return _kda_forward(self, hidden_states, segments, attention_mask, position_ids)


def _kda_forward(self, hidden_states, segments, attention_mask=None, position_ids=None):
    """Head-parallel packed KDA implementation, independent of device dispatch."""
    if self.num_heads % self.sp_size:
        raise ValueError("GLM-5.3-Flash KDA heads must be divisible by sp_size")
    b, local_t = hidden_states.shape[:2]
    t, h, d = local_t * self.sp_size, self.num_heads // self.sp_size, self.head_dim
    runs = (
        segments
        if segments is not None
        else sequence_segments(attention_mask, position_ids, b, t)
    )
    first, last = self.sp_rank * h, (self.sp_rank + 1) * h
    locations, lengths, cu_cpu, cu_seqlens = _kda_pack_metadata(
        tuple(tuple(tuple(pair) for pair in row) for row in runs),
        t,
        hidden_states.device,
    )
    qkv = [
        getattr(self, name + "_proj")(hidden_states).view(
            b, local_t, self.num_heads, d
        )
        for name in ("q", "k", "v")
    ]
    raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).view(
        b, local_t, self.num_heads, d
    )
    beta = self.b_proj(hidden_states).sigmoid()
    # For an entirely empty batch keep the original unused-gradient behavior.
    qkv, raw_gate, beta = _exchange_kda_inputs(
        self, qkv, raw_gate, beta, merge=bool(lengths)
    )
    out = qkv[0] * 0
    if lengths:
        use_varlen = (
            self.config.kda_backend in ("auto", "fla")
            and os.environ.get("TORCHTURBO_GLM5_KDA_CONV", "varlen") == "varlen"
        )
        conv_inputs = []
        for name, value in zip(("q", "k", "v"), qkv):
            packed = (
                value.flatten(0, 1).index_select(0, locations).reshape(1, -1, h * d)
            )
            weight = getattr(self, name + "_conv1d").weight[first * d : last * d]
            if use_varlen:
                convolved = _kda_varlen_conv(packed, weight, cu_seqlens, cu_cpu)
            else:
                convolved = _kda_reference_conv(packed, weight, lengths)
            conv_inputs.append(convolved.reshape(1, -1, h, d))
        gates = raw_gate.flatten(0, 1).index_select(0, locations).unsqueeze(0)
        betas = beta.flatten(0, 1).index_select(0, locations).unsqueeze(0)
        lower = self.config.linear_lower_bound
        if self.config.kda_backend in ("auto", "fla"):
            packed_out = _npu_chunk_kda(
                *conv_inputs,
                gates,
                betas,
                self.A_log[first:last],
                self.dt_bias[first * d : last * d],
                lengths,
                lower,
                cu_seqlens=cu_seqlens,
            )
        else:
            outputs = []
            rate = self.A_log[first:last].float().exp().view(1, 1, h, 1)
            bias = self.dt_bias[first * d : last * d].float().view(1, 1, h, d)
            start = 0
            for length in lengths:
                end = start + length
                raw = gates[:, start:end].float() + bias
                decay = (
                    lower * (rate * raw).sigmoid()
                    if lower is not None
                    else -rate * F.softplus(raw)
                )
                value, _ = recurrent_kimi_delta_attention(
                    conv_inputs[0][:, start:end],
                    conv_inputs[1][:, start:end],
                    conv_inputs[2][:, start:end],
                    g=decay,
                    beta=betas[:, start:end],
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                outputs.append(value)
                start = end
            packed_out = torch.cat(outputs, dim=1)
        out = (
            out.flatten(0, 1)
            .index_copy(0, locations, packed_out.squeeze(0))
            .view(b, t, h, d)
        )
    out = _exchange(self, out, inverse=True)
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
        b, local_t, self.num_heads, d
    )
    return self.o_proj(self.o_norm(out, gate).flatten(2))


def _absorb_weight(proj):
    """Effective active LoRA weight, or None when absorption is not equivalent."""
    if not hasattr(proj, "lora_A"):
        return proj.weight if isinstance(proj, nn.Linear) else None
    if getattr(proj, "disable_adapters", False):
        # Disabled but merged adapters need PEFT's unmerge side effect first.
        return None if getattr(proj, "merged", False) else proj.weight
    weight = proj.weight
    merged = set(getattr(proj, "merged_adapters", []))
    for name in proj.active_adapters:
        if name in merged or name not in proj.lora_A:
            continue
        if getattr(proj, "use_dora", {}).get(name, False):
            return None
        dropout = proj.lora_dropout[name]
        if proj.training and getattr(dropout, "p", 0) != 0:
            return None
        weight = weight + proj.get_delta_weight(name).to(weight.dtype)
    return weight


def _compress_kpools(indexer, keys, gate):
    """Compress [..., pool, dim], preserving the reference rounding boundaries."""
    logits = gate.float() + indexer.index_kpool_compress_ape.float()
    return (logits.softmax(-2).to(keys.dtype) * keys).sum(-2)


@torch.no_grad()
def kpool_indices(
    indexer,
    hidden_kv,
    hidden_q,
    q_resid,
    query_start=0,
    query_chunk=256,
    pool_chunk=2048,
    projected_keys=None,
    projected_gate=None,
    projected_pools=None,
):
    """Select k-pools in bounded causal tiles for one contiguous valid segment.

    Queries with at most ``topk // pool`` complete visible pools need no scores:
    every complete pool is selected. Preserve the incomplete tail separately.
    Valid token indices are sorted before the -1 padding for sparse kernels.
    """
    if projected_keys is not None and projected_gate is None:
        raise ValueError("Projected indexer keys require projected gates")
    if projected_pools is not None and projected_keys is not None:
        raise ValueError("Supply pooled keys or token keys, not both")
    b, nq = hidden_q.shape[:2]
    pool, topk = indexer.index_kpool, indexer.index_topk
    tail = indexer.index_kpool_always_select_tail
    width = topk + (pool - 1 if tail else 0)
    result = torch.full((b, nq, width), -1, device=hidden_q.device, dtype=torch.int32)

    # This prefix includes up to pool-1 tail tokens beyond topk. In particular,
    # all-short packed batches can avoid indexer projections and SP collectives.
    prefix = min(nq, max(0, topk + pool - 1 - query_start))
    if prefix:
        visible = torch.arange(
            query_start + 1, query_start + prefix + 1, device=hidden_q.device
        )
        if not tail:
            visible = visible.div(pool, rounding_mode="floor") * pool
        tokens = torch.arange(width, device=hidden_q.device, dtype=torch.int32)
        dense = tokens[None, None].expand(b, prefix, -1)
        result[:, :prefix] = dense.masked_fill(
            tokens[None, None] >= visible[None, :, None], -1
        )
    if prefix == nq:
        return result

    if projected_pools is not None:
        npools = min(projected_pools.shape[1], (query_start + nq) // pool)
        pooled = projected_pools[:, :npools].float()
    else:
        length = (hidden_kv if projected_keys is None else projected_keys).shape[1]
        npools = min(length // pool, (query_start + nq) // pool)
        if projected_keys is None:
            projected_keys = indexer.k_norm(indexer.wk(hidden_kv))
            projected_gate = F.linear(hidden_kv, indexer.index_kpool_compress_gate)
        pooled = _compress_kpools(
            indexer,
            projected_keys[:, :npools * pool].reshape(b, npools, pool, indexer.head_dim),
            projected_gate[:, :npools * pool].reshape(b, npools, pool, indexer.head_dim),
        ).float()
    q = indexer.wq_b(q_resid).view(b, nq, indexer.n_heads, indexer.head_dim)
    weights = indexer.weights_proj(hidden_q).float() * indexer.n_heads**-0.5
    select = min(topk // pool, npools)
    selected_pools = torch.empty(
        (b, nq - prefix, select), device=q.device, dtype=torch.int32
    )
    pool_offsets = torch.arange(pool, device=q.device, dtype=torch.int32)
    tail_offsets = (
        torch.arange(pool - 1, device=q.device, dtype=torch.int32) if tail else None
    )
    scored_positions = torch.arange(
        query_start + prefix, query_start + nq, device=q.device, dtype=torch.int32
    )
    for qs in range(prefix, nq, query_chunk):
        qe = min(qs + query_chunk, nq)
        positions = scored_positions[qs - prefix:qe - prefix]
        # Reuse one FP32 query tile for every pool tile. A single larger GEMM
        # also reuses the shared pooled keys across all queries and heads.
        q_tile = q[:, qs:qe].float().reshape(b, -1, indexer.head_dim)
        best_score = best_pool = None
        visible_pools = min(npools, (query_start + qe) // pool)
        for ps in range(0, visible_pools, pool_chunk):
            pe = min(ps + pool_chunk, visible_pools)
            score = torch.matmul(q_tile, pooled[:, ps:pe].transpose(-1, -2))
            score = score.view(b, qe - qs, indexer.n_heads, pe - ps)
            score.mul_(indexer.softmax_scale).relu_()
            # The original empty FP32 cat promoted TopK inputs under autocast.
            score = (
                torch.matmul(weights[:, qs:qe, None, :], score).squeeze(-2).float()
            )
            # Keep masked slots in the boundary tile: trimming their layout
            # can otherwise change which equal-scoring visible pools TopK picks.
            tile_end = min(ps + pool_chunk, npools)
            if pe < tile_end:
                score = F.pad(
                    score, (0, tile_end - pe), value=torch.finfo(score.dtype).min
                )
            candidates = torch.arange(ps, tile_end, device=q.device)
            score.masked_fill_(
                (candidates + 1) * pool - 1 > positions[:, None],
                torch.finfo(score.dtype).min,
            )
            candidates = candidates.view(1, 1, -1).expand(b, qe - qs, -1)
            if best_score is not None:
                score = torch.cat((best_score, score), -1)
                candidates = torch.cat((best_pool, candidates), -1)
            best_score, slots = score.topk(min(select, score.shape[-1]), -1)
            best_pool = candidates.gather(-1, slots)
        selected_pools[:, qs - prefix:qe - prefix] = best_pool
    # One AI Core sort over K/p pool IDs instead of K token IDs. Sorting a
    # query tile at a time would add many small launches on long sequences.
    selected_pools = selected_pools.float().sort(-1).values.to(torch.int32)
    for qs in range(prefix, nq, query_chunk):
        qe = min(qs + query_chunk, nq)
        positions = scored_positions[qs - prefix:qe - prefix]
        best_pool = selected_pools[:, qs - prefix:qe - prefix]
        indices = best_pool[..., None] * pool + pool_offsets
        valid = (best_pool + 1) * pool - 1 <= positions[None, :, None]
        indices = indices.masked_fill(~valid[..., None], -1).flatten(-2)
        if tail:
            count = (positions + 1).remainder(pool)
            extra = positions[:, None] + 1 - count[:, None] + tail_offsets
            extra = extra.masked_fill(tail_offsets >= count[:, None], -1)
            indices = torch.cat((indices, extra[None].expand(b, -1, -1)), -1)
        # Cast on assignment instead of retaining all INT64 chunks plus their
        # concatenation until the final INT32 conversion.
        result[:, qs:qe, : indices.shape[-1]] = indices
    return result


def _add_sparse_overflow_grads(
    query,
    kv,
    indices,
    grad_out,
    out,
    softmax_max,
    softmax_sum,
    scale,
    dq,
    dkv,
    main_width,
    chunk_size=128,
):
    """Add selected tokens beyond the fused backward width.

    Use the full forward's normalization and output, not a separate softmax:
    dS_i = P_i * (dO @ V_i - dO @ O), including the main/overflow interaction.
    Only query chunks and the extra selected K/V are materialized in FP32.
    """
    # Local Q is the suffix of its KV prefix under sparse_mode=3.
    query_offset = kv.shape[0] - query.shape[0]
    first_overflow_query = max(0, main_width - query_offset)
    for start in range(first_overflow_query, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        selected = indices[start:end, 0, main_width:]
        valid = selected >= 0
        slots = selected.clamp_min(0).long()
        extra = kv[:, 0][slots].float()
        q, do = query[start:end].float(), grad_out[start:end].float()
        scores = torch.bmm(q, extra.transpose(1, 2)) * scale
        scores = scores.masked_fill(~valid[:, None], -float("inf"))
        probs = (scores - softmax_max[0, start:end, :, None]).exp()
        probs = probs / softmax_sum[0, start:end, :, None]
        delta = (do * out[start:end].float()).sum(-1, keepdim=True)
        ds = probs * (torch.bmm(do, extra.transpose(1, 2)) - delta)
        dq[start:end] = (dq[start:end].float() + torch.bmm(ds, extra) * scale).to(
            dq.dtype
        )
        extra_grad = torch.bmm(ds.transpose(1, 2), q) * scale
        extra_grad = extra_grad + torch.bmm(probs.transpose(1, 2), do)
        # Invalid slots were gathered from token zero solely for safe addressing.
        extra_grad = extra_grad.masked_fill(~valid[..., None], 0)
        dkv[:, 0].index_add_(0, slots.flatten(), extra_grad.flatten(0, 1))


class _NpuNopeSparseAttention(torch.autograd.Function):
    """Keep fused forward; never send alignment padding to sparse backward."""

    @staticmethod
    def forward(ctx, query, kv, indices, scale, main_width):
        import torch_npu

        cu_q = torch.tensor([query.shape[0]], device=query.device, dtype=torch.int32)
        cu_kv = torch.tensor([kv.shape[0]], device=query.device, dtype=torch.int32)
        # The forward accepts -1 padding and 64-wide alignment. The deployed
        # backward instead consumes min(K, causal_length), regardless of -1.
        padded = F.pad(indices, (0, (-indices.shape[-1]) % 64), value=-1).contiguous()
        q_rope = query.new_zeros(query.shape[0], query.shape[1], 64)
        k_rope = kv.new_zeros(kv.shape[0], 1, 64)
        out, maximum, total = torch_npu.npu_sparse_flash_attention(
            query,
            kv,
            kv,
            padded,
            scale,
            actual_seq_lengths_query=cu_q,
            actual_seq_lengths_kv=cu_kv,
            query_rope=q_rope,
            key_rope=k_rope,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(query, kv, indices, out, maximum, total, cu_q, cu_kv)
        ctx.scale, ctx.main_width = scale, main_width
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        query, kv, indices, out, maximum, total, cu_q, cu_kv = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        # With always-selected tails, each row has min(topk, causal_length)
        # valid prefix entries, including the KV-derived local-Q offset.
        # K=topk is also a supported backward width.
        # Supply the FULL forward statistics/output so this computes precisely
        # the main entries' contribution to the original attention gradient.
        main = indices[..., : ctx.main_width].contiguous()
        dq, dk, dv, *_ = torch.ops.npu.npu_sparse_flash_attention_grad(
            query,
            kv,
            kv,
            main,
            grad_out,
            out,
            maximum,
            total,
            ctx.scale,
            1,
            query_rope=query.new_zeros(query.shape[0], query.shape[1], 64),
            key_rope=kv.new_zeros(kv.shape[0], 1, 64),
            actual_seq_qlen=cu_q,
            actual_seq_kvlen=cu_kv,
            layout="TND",
            sparse_mode=3,
            attention_mode=2,
        )
        dkv = dk.float() + dv.float()
        if kv.shape[0] > ctx.main_width and indices.shape[-1] > ctx.main_width:
            with torch.autocast(device_type=query.device.type, enabled=False):
                _add_sparse_overflow_grads(
                    query,
                    kv,
                    indices,
                    grad_out,
                    out,
                    maximum,
                    total,
                    ctx.scale,
                    dq,
                    dkv,
                    ctx.main_width,
                )
        return dq, dkv.to(kv.dtype), None, None, None


@torch.no_grad()
def _gather_sp_index_tensor(self, tensor):
    """Gather frozen indexer projections in this layer's SP group."""
    if self.sp_size == 1:
        return tensor
    import torch.distributed as dist

    pieces = [torch.empty_like(tensor) for _ in range(self.sp_size)]
    dist.all_gather(pieces, tensor.contiguous(), group=self.sp_group)
    return torch.cat(pieces, dim=1)


@lru_cache(maxsize=8)
def _kpool_sp_metadata(runs, total_t, sp_size, sp_rank, pool, topk, device):
    """Immutable layout for pool owners and the few raw tokens crossing ranks.

    The rank containing a pool's last token owns that complete pool. Pooling
    starts at each segment's start, including when it is not SP/pool aligned.
    All collective sizes/branches follow global metadata, even on empty ranks.
    """
    local_t = total_t // sp_size
    owned = [[] for _ in range(sp_size)]
    boundary = [[] for _ in range(sp_size)]
    order, ranges = [], {}
    for row, row_runs in enumerate(runs):
        for start, end in row_runs:
            count = (end - start) // pool
            if count <= topk // pool:
                continue
            ranges[row, start, end] = (len(order), count)
            for pi in range(count):
                first = start + pi * pool
                owner = (first + pool - 1) // local_t
                order.append((owner, len(owned[owner]) // pool))
                for position in range(first, first + pool):
                    location = row * total_t + position
                    owned[owner].append(location)
                    source = position // local_t
                    if source != owner:
                        boundary[source].append(location)
    max_pools = max(len(tokens) // pool for tokens in owned)
    max_boundary = max(map(len, boundary))
    boundary_slots = {
        location: rank * max_boundary + slot
        for rank, tokens in enumerate(boundary)
        for slot, location in enumerate(tokens)
    }
    offset = sp_rank * local_t
    local_size = len(runs) * local_t
    sources = []
    for location in owned[sp_rank]:
        row, position = divmod(location, total_t)
        sources.append(
            row * local_t + position - offset
            if offset <= position < offset + local_t
            else local_size + boundary_slots[location]
        )
    boundary_sources = [
        (location // total_t) * local_t + location % total_t - offset
        for location in boundary[sp_rank]
    ]
    # CPU metadata and device index vectors are shared by layers/checkpoint
    # replay, but no hidden-state-dependent values are cached.
    def to_device(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    return (
        to_device(sources),
        to_device(boundary_sources),
        to_device([rank * max_pools + slot for rank, slot in order]),
        ranges,
        max_pools,
        max_boundary,
    )


@torch.no_grad()
def _gather_sp_pooled_keys(self, hidden_states, runs):
    """Compress each complete pool once, before its SP all-gather.

    Only crossing-pool raw K/Gate records use the small boundary collective.
    The main collective sends pooled keys in their original rounded dtype.
    """
    indexer = self.indexer
    b, local_t = hidden_states.shape[:2]
    p, d = indexer.index_kpool, indexer.head_dim
    sources, boundary_sources, order, ranges, max_pools, max_boundary = (
        _kpool_sp_metadata(
            tuple(tuple(tuple(pair) for pair in row) for row in runs),
            local_t * self.sp_size,
            self.sp_size,
            self.sp_rank,
            p,
            indexer.index_topk,
            hidden_states.device,
        )
    )
    keys = indexer.k_norm(indexer.wk(hidden_states))
    gate = F.linear(hidden_states, indexer.index_kpool_compress_gate)
    packed = torch.cat((keys, gate), dim=-1).reshape(b * local_t, 2 * d)
    if max_boundary:
        edge = packed.new_zeros((max_boundary, 2 * d))
        if boundary_sources.numel():
            edge[:boundary_sources.numel()] = packed.index_select(0, boundary_sources)
        gathered_edge = _gather_sp_index_tensor(self, edge.unsqueeze(0)).squeeze(0)
        packed = torch.cat((packed, gathered_edge), dim=0)
    local_pools = keys.new_zeros((max_pools, d))
    if sources.numel():
        grouped = packed.index_select(0, sources).reshape(1, -1, p, 2 * d)
        # cat may promote mixed input dtypes. Restore K's dtype before the
        # probability cast/product/reduction to preserve the reference result.
        value = _compress_kpools(
            indexer, grouped[..., :d].to(keys.dtype), grouped[..., d:]
        ).squeeze(0)
        local_pools[:value.shape[0]] = value
    pooled = _gather_sp_index_tensor(self, local_pools.unsqueeze(0)).squeeze(0)
    return pooled.index_select(0, order), ranges


def npu_dsa_forward(
    self, hidden_states, segments=None, prev_topk_indices=None, **kwargs
):
    # Enable only the shape/dtype combination validated on the target stack.
    weight = _absorb_weight(self.kv_b_proj)
    if (
        hidden_states.device.type != "npu"
        or hidden_states.dtype != torch.bfloat16
        or self.kv_lora_rank != 512
        or self.num_heads != 64
        or not 1024 <= self.config.index_topk <= 8192
        or self.config.index_topk % 1024 != 0
        or self.config.index_topk % self.config.index_kpool != 0
        or not self.config.index_kpool_always_select_tail
        or (self.training and self.config.attention_dropout != 0)
        or weight is None
    ):
        return self._glm5_reference_forward(
            hidden_states, segments, prev_topk_indices, **kwargs
        )
    b, local_t = hidden_states.shape[:2]
    total_t = local_t * self.sp_size
    offset = self.sp_rank * local_t
    runs = (
        segments
        if segments is not None
        else sequence_segments(
            kwargs.get("attention_mask"), kwargs.get("position_ids"), b, total_t
        )
    )
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, local_t, self.num_heads, self.qk_head_dim)
    weight = weight.view(
        self.num_heads, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank
    )
    q = torch.einsum("bthd,hdr->bthr", q, weight[:, : self.qk_head_dim]).contiguous()
    # Q/output projections remain local. Only compact K/V representations
    # cross SP ranks; the main latent gather still reduces its gradients.
    latent = self.gather(
        self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
    )
    index_keys = index_gate = pooled_keys = None
    pool_ranges = None
    # runs is global metadata shared by every SP rank. All ranks must make
    # the same decision before skipping either of the two indexer all-gathers.
    needs_index_scores = self.indexer is not None and any(
        (end - start) // self.config.index_kpool
        > self.config.index_topk // self.config.index_kpool
        for row_runs in runs
        for start, end in row_runs
    )
    if needs_index_scores:
        with torch.no_grad():
            hidden_index = hidden_states.detach()
            if os.environ.get("TORCHTURBO_GLM5_KPOOL_EXCHANGE", "pooled") == "tokens":
                # Explicit A/B and rollback path; all ranks must use one mode.
                index_keys = _gather_sp_index_tensor(
                    self, self.indexer.k_norm(self.indexer.wk(hidden_index))
                )
                index_gate = _gather_sp_index_tensor(
                    self, F.linear(hidden_index, self.indexer.index_kpool_compress_gate)
                )
            else:
                pooled_keys, pool_ranges = _gather_sp_pooled_keys(self, hidden_index, runs)
    width = self.config.index_topk + (
        self.config.index_kpool - 1 if self.config.index_kpool_always_select_tail else 0
    )
    if self.indexer is None:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA layers require preceding full-layer indices")
        if prev_topk_indices.shape[1] == local_t:
            indices = prev_topk_indices
        else:
            indices = prev_topk_indices[:, offset : offset + local_t]
    else:
        indices = torch.full(
            (b, local_t, width), -1, device=q.device, dtype=torch.int32
        )
    output_locations, output_values = [], []
    for row, row_runs in enumerate(runs):
        for start, end in row_runs:
            qs, qe = max(start, offset), min(end, offset + local_t)
            if qs >= qe:
                continue
            ls, le = qs - offset, qe - offset
            if self.indexer is not None:
                projected_pools = None
                if pool_ranges is not None and (row, start, end) in pool_ranges:
                    pool_start, pool_count = pool_ranges[row, start, end]
                    projected_pools = pooled_keys[
                        None, pool_start:pool_start + pool_count
                    ]
                selected = kpool_indices(
                    self.indexer,
                    None,
                    hidden_states[row : row + 1, ls:le],
                    q_resid[row : row + 1, ls:le],
                    qs - start,
                    projected_keys=(
                        index_keys[row : row + 1, start:qe]
                        if index_keys is not None
                        else None
                    ),
                    projected_gate=(
                        index_gate[row : row + 1, start:qe]
                        if index_gate is not None
                        else None
                    ),
                    projected_pools=projected_pools,
                )
                indices[row : row + 1, ls:le] = torch.where(
                    selected >= 0, selected + start, selected
                )
                # kpool_indices already returns sorted, segment-local IDs.
                selected = selected.squeeze(0)
            else:
                selected = indices[row, ls:le]
                selected = torch.where(selected >= 0, selected - start, selected)
                # A shared layer may receive unsorted reference-fallback IDs.
                # Keep this compatibility path; full layers sort only K/p IDs.
                selected = selected.masked_fill(
                    selected < 0, qe - start
                ).float().sort(-1).values.to(selected.dtype)
                selected = selected.masked_fill(selected == qe - start, -1)
            selected = selected.unsqueeze(1).contiguous()
            # Match the right-aligned causal offset to this rank's Q slice.
            kv = latent[row, start:qe, None].contiguous()
            query = q[row, ls:le].contiguous()
            value = _NpuNopeSparseAttention.apply(
                query,
                kv,
                selected,
                self.scaling,
                self.config.index_topk,
            )
            output_locations.append(
                torch.arange(row * local_t + ls, row * local_t + le, device=q.device)
            )
            output_values.append(value)
    # Keep the input in the graph even for an entirely padded batch.
    out = q * 0 + (latent[:, :1].float().sum() * 0).to(q.dtype)
    if output_values:
        out = (
            out.flatten(0, 1)
            .index_copy(0, torch.cat(output_locations), torch.cat(output_values))
            .view_as(q)
        )
    out = torch.einsum("bthr,hvr->bthv", out, weight[:, self.qk_head_dim :]).to(
        hidden_states.dtype
    )
    out = self.o_proj(out.flatten(2))
    return out, indices


def patch_model(model):
    """Bind eligible NPU forwards only on this model's text tower."""
    for module in model.model.language_model.modules():
        # Independent from forward patches, including activation checkpoint's
        # wrapper: the original layer forward dynamically calls self._pre.
        if isinstance(module, Glm5NextTextDecoderLayer) and not hasattr(
            module, "_glm5_reference_pre"
        ):
            module._glm5_reference_pre = module._pre
            module._pre = types.MethodType(npu_mhc_pre, module)
        if hasattr(module, "_glm5_reference_forward"):
            continue
        forward = None
        if isinstance(module, Glm5NextTextRMSNorm):
            forward = npu_rms_norm_forward
        elif isinstance(module, Glm5NextTextRMSNormGated):
            forward = npu_gated_rms_norm_forward
        elif isinstance(module, Glm5NextTextLinearAttention):
            forward = npu_kda_forward
        elif isinstance(module, Glm5NextTextAttention):
            forward = npu_dsa_forward
        if forward is not None:
            module._glm5_reference_forward = module.forward
            module.forward = types.MethodType(forward, module)
