"""Autograd adapter for the full AscendC KDA path (fused fwd + fused bwd).

2026-09-28 head-first rewrite: the Q/K/V path now stays in the AscendC
operators' native head-major spelling end to end, following the approach
validated by MindSpeed-Bridge PR #8.  Layout handling:

* ``head_major_qkv=True`` (production): q/k/v arrive head-first ``[B,H,T,D]``
  straight out of the fused short convolution, the forward runs with
  ``layout="NTD"`` (packed) / ``"BNSD"`` (dense), and the backward hands
  ``dq``/``dk``/``dv`` back head-first with no transpose.  Only ``g``/``beta``
  (projection outputs, inherently sequence-major) and ``do``/``dg``/``db``
  cross the layout boundary.
* ``head_major_qkv=False`` (fallback): fla-org's BSND calling convention is
  kept; the backward converts to head-major for the fused kernel and back.

L2 normalization stays OUTSIDE the kernel and uses the fla-npu triton kernels
(``fla_npu.ops.triton.l2norm_fwd/l2norm_bwd``), because the in-kernel variant
needs the V2 three-stage forward, whose saved-value outlets are A5-only.  The
triton kernels are rank-agnostic (``x.view(-1, D)``), so they work directly on
head-major tensors without any layout conversion.

Platform facts pinned by measurement on Ascend910_93 (A3), wheel a802b290:

* ``NTD``/``BNSD`` produce bit-identical ``o``, intermediates and gradients
  versus ``TND``/``BSND`` (bench/probe_kda_head_major_a3.py);
* the fused backward only supports ``disable_recompute=True`` -- the recompute
  variant routes to the A5-only optimized path (bench/probe_kda_recompute_a3.py);
* ``q_rstd``/``k_rstd`` must NOT be passed: they select the Ascend950-only
  optimized backward, which raises on A3 with no fallback.
"""

from __future__ import annotations

import torch

from fla_npu.ops.ascendc import npu_chunk_kda_bwd, npu_chunk_kda_fwd
from fla_npu.ops.triton import l2norm_bwd, l2norm_fwd

import os

# ---------------------------------------------------------------------------
# Diagnostic switch: force the KDA forward onto the legacy single-fused
# kernel (aclnnChunkKdaFwd) instead of the V2 three-kernel chain.
#
# Motivation: of the source-level risks under investigation, the
# zero-copy `MaybeReshapeToHeadMajor4` reshape bridge and the
# `qgScaledCompute` split across non-adjacent L0 stages (written by the
# Prepare kernel, read by Finalize) exist ONLY in the V2 chain.  Raising
# the V2 work-items gate to infinity yields a clean A/B that differs in
# the forward packing alone: same shapes, same inputs, same backward,
# only the forward swaps single-fused kernel vs three-kernel chain.
#
# Requires `FLA_NPU_STABLE_ABI=ctypes`: the stable launcher has no such
# gate (its comments state it always runs V2); only the ctypes reference
# path reads this constant.  Disable by setting 0 or dropping the env.
# ---------------------------------------------------------------------------
if os.environ.get("FLA_NPU_FORCE_KDA_FWD_V1", "0").strip().lower() not in (
    "0", "", "false", "no",
):
    try:
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ct

        _ct._CHUNK_KDA_FWD_V2_MIN_WORK_ITEMS = 1 << 62
    except Exception:  # noqa: BLE001 - best-effort diagnostic switch
        pass

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _gate_parameter_for_ascendc(tensor):
    """Promote model BF16 gate parameters to the FP32 AscendC contract."""

    if tensor is None:
        return None
    return tensor.float() if tensor.dtype != torch.float32 else tensor


def _host_cu_seqlens(cu_seqlens, cu_seqlens_cpu):
    """Materialize cu_seqlens as a host int tuple for the AscendC binding."""

    if cu_seqlens is None:
        return None
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    return [int(item) for item in src.detach().cpu().tolist()]


def _drop_batch(tensor, is_varlen):
    """``[1,H,T,D]`` -> ``[H,T,D]`` for the packed kernel contract (free view)."""

    if is_varlen and tensor.dim() == 4:
        return tensor.squeeze(0)
    return tensor


def _to_head_major(tensor):
    """[1, T, H, D] -> [H, T, D] (fused-bwd varlen input contract)."""

    if tensor is None:
        return None
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _to_head_major_beta(tensor):
    """[1, T, H] -> [H, T]."""

    if tensor is None:
        return None
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _to_bnsd(tensor):
    """[B, T, H, D] -> [B, H, T, D] (fused-bwd dense input contract)."""

    if tensor is None:
        return None
    return tensor.transpose(1, 2).contiguous()


def _to_bnsd_beta(tensor):
    """[B, T, H] -> [B, H, T]."""

    if tensor is None:
        return None
    return tensor.transpose(1, 2).contiguous()


def _from_head_major(tensor, dtype):
    """[H, T, D] -> contiguous [1, T, H, D] in ``dtype`` (grad outputs)."""

    return tensor.to(dtype).unsqueeze(0).transpose(1, 2).contiguous()


def _from_head_major_beta(tensor, dtype):
    """[H, T] -> contiguous [1, T, H] in ``dtype``."""

    return tensor.to(dtype).unsqueeze(0).transpose(1, 2).contiguous()


def _from_bnsd(tensor, dtype):
    """[B, H, T, D] -> contiguous [B, T, H, D] in ``dtype`` (grad outputs)."""

    return tensor.to(dtype).permute(0, 2, 1, 3).contiguous()


def _from_bnsd_beta(tensor, dtype):
    """[B, H, T] -> contiguous [B, T, H] in ``dtype``."""

    return tensor.to(dtype).transpose(1, 2).contiguous()


def _normalise_intermediate(tensor, expected_rank):
    """Normalise a fwd intermediate to the fused-bwd layout contract."""

    if tensor is None:
        return None
    return tensor if tensor.dim() == expected_rank else tensor.squeeze(0)


def _seq_to_kernel(tensor, is_varlen, beta=False):
    """Sequence-major model tensor -> the kernel's head-major spelling.

    With ``head_major_qkv`` only ``g``/``beta`` take this path (q/k/v already
    arrive head-first); without it every input does.
    """

    if tensor is None:
        return None
    if is_varlen:
        return _to_head_major_beta(tensor) if beta else _to_head_major(tensor)
    return _to_bnsd_beta(tensor) if beta else _to_bnsd(tensor)


class AscendCChunkKDAFunction(torch.autograd.Function):
    """fla-org ``ChunkKDAFunction`` with both passes on AscendC kernels."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context=None,
        head_major_qkv: bool = False,
        export_intermediates: bool = True,
    ):
        # use_beta_sigmoid_in_kernel / allow_neg_eigval / cp_context /
        # return_intermediate_states / state_v_first=True are rejected by the
        # public wrapper; they are kept in the signature so the backward
        # return tuple mirrors fla-org's ChunkKDAFunction one-to-one.
        g_input = g
        cu_host = _host_cu_seqlens(cu_seqlens, cu_seqlens_cpu)
        is_varlen = cu_host is not None
        export_intermediates = bool(export_intermediates)

        # Bring q/k/v into the kernel's spelling.  head_major_qkv keeps them
        # head-first (the packed squeeze is a free view); the fallback path
        # feeds BSND and defers conversion to the backward.
        if head_major_qkv:
            layout = "NTD" if is_varlen else "BNSD"
            q_in = _drop_batch(q, is_varlen)
            k_in = _drop_batch(k, is_varlen)
            v_in = _drop_batch(v, is_varlen)
            g_in = _seq_to_kernel(g_input, is_varlen)
            beta_in = _seq_to_kernel(beta, is_varlen, beta=True)
        else:
            layout = "BSND"
            q_in, k_in, v_in = q, k, v
            g_in, beta_in = g_input, beta

        # L2 normalization runs here on the fla-npu triton kernels, on whatever
        # layout the kernel will consume -- both are row-major over the last
        # (head) dim, so no conversion is needed in either mode.
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q_in, q_rstd = l2norm_fwd(q_in)
            k_in, k_rstd = l2norm_fwd(k_in)

        outputs = list(
            npu_chunk_kda_fwd(
                q_in, k_in, v_in, g_in, beta_in, scale, chunk_size,
                layout=layout,
                initial_state=initial_state,
                output_final_state=bool(output_final_state),
                cu_seqlens=cu_host,
                chunk_indices=None,  # derived internally from host metadata
                safe_gate=bool(safe_gate),
                lower_bound=lower_bound,
                use_gate_in_kernel=bool(use_gate_in_kernel),
                A_log=_gate_parameter_for_ascendc(A_log),
                dt_bias=_gate_parameter_for_ascendc(dt_bias),
                disable_recompute=export_intermediates,
                return_intermediate_states=False,
                state_v_first=False,
                use_qk_l2norm_in_kernel=False,  # external l2norm (see above)
                use_exp2=True,
            )
        )
        if len(outputs) != 12:
            raise RuntimeError(
                f"fla_npu chunk_kda_fwd returned {len(outputs)} values; expected 12."
            )

        (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h,
         initial_state_out) = outputs

        # ``attn_out`` is fixed sequence-major by the operator contract
        # (fla/ops/ascendc/kda/README.md): [T,HV,V] for rank-3 layouts,
        # [B,T,HV,V] otherwise.  Restore the model's rank-4 spelling.
        if o.dim() == 3:
            o = o.unsqueeze(0)

        if export_intermediates:
            # The fused backward consumes head-major intermediates; the packed
            # NTD forward already returns them at the right rank, so the
            # normalisation below is a no-op view there.  ``u`` is dropped: the
            # fused backward does not take it.
            token_rank = 3 if is_varlen else 4
            gk = _normalise_intermediate(gk, token_rank)
            Aqk = _normalise_intermediate(Aqk, token_rank)
            Akk = _normalise_intermediate(Akk, token_rank)
            w = _normalise_intermediate(w, token_rank)
            qg = _normalise_intermediate(qg, token_rank)
            kg = _normalise_intermediate(kg, token_rank)
            v_new = _normalise_intermediate(v_new, token_rank)
            h = _normalise_intermediate(h, token_rank + 1)

        # Saved in the kernel's spelling so the backward needs no conversion
        # for q/k/v/g/beta.  q_in/k_in are the NORMALIZED tensors; q_rstd/
        # k_rstd drive the external L2-norm backward.
        ctx.save_for_backward(
            q_in, q_rstd, k_in, k_rstd, v_in, g_in, beta_in, A_log, dt_bias,
            gk, Aqk, Akk, w, qg, kg, v_new, h,
        )
        ctx.cu_host = cu_host
        ctx.is_varlen = is_varlen
        ctx.head_major_qkv = bool(head_major_qkv)
        ctx.export_intermediates = export_intermediates
        # The rank the caller handed in: the head-major backward restores
        # dq/dk/dv to it and must not assume rank-4 (_drop_batch only
        # squeezes rank-4 inputs).
        ctx.qkv_input_rank = q.dim()
        ctx.qkv_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        ctx.g_dtype = g_input.dtype
        ctx.beta_dtype = beta.dtype
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel

        return o.type_as(q_in), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        (q_hat, q_rstd, k_hat, k_rstd, v_in, g_in, beta_in, A_log, dt_bias,
         gk, Aqk, Akk, w, qg, kg, v_new, h) = ctx.saved_tensors

        # NOTE on dht: the engine materialises a ZERO gradient for the
        # (production-wise discarded) final_state even when it is unused.
        # The fused backward takes no dht; gradients w.r.t. final_state are
        # unsupported and silently ignored.
        if gk is None:
            raise RuntimeError(
                "ascendc_chunk_kda: the fused backward requires the "
                "forward's intermediates, but they were not exported."
            )

        is_varlen = ctx.is_varlen
        head_major_qkv = ctx.head_major_qkv

        # q/k/v/g/beta were saved in the kernel's spelling.  Without
        # head_major_qkv they are sequence-major and still need converting.
        if head_major_qkv:
            q_h, k_h, v_h, g_h, beta_h = q_hat, k_hat, v_in, g_in, beta_in
        else:
            q_h = _seq_to_kernel(q_hat, is_varlen)
            k_h = _seq_to_kernel(k_hat, is_varlen)
            v_h = _seq_to_kernel(v_in, is_varlen)
            g_h = _seq_to_kernel(g_in, is_varlen)
            beta_h = _seq_to_kernel(beta_in, is_varlen, beta=True)
        # ``do`` follows ``o``, which the operator fixes to sequence-major.
        do_h = _seq_to_kernel(do, is_varlen)

        dt_bias_h = _gate_parameter_for_ascendc(dt_bias)
        if dt_bias_h is not None and dt_bias_h.dim() == 1:
            # q_h is head-major: varlen [H,T,D] -> shape[0]=H; dense [B,H,T,D] -> shape[1]=H
            heads = q_h.shape[0] if is_varlen else q_h.shape[1]
            head_dim = q_h.shape[-1]
            dt_bias_h = dt_bias_h.view(heads, head_dim)

        # q_rstd/k_rstd are deliberately NOT passed: they select the
        # Ascend950-only optimized backward (which fuses the L2-norm backward
        # into Finalize) and raise on A3 with no fallback.  The L2-norm
        # backward is applied below on the fla-npu triton kernel instead.
        dq_hat, dk_hat, dv, db, dg, dh0, dA, dbias = npu_chunk_kda_bwd(
            q=q_h,
            k=k_h,
            v=v_h,
            beta=beta_h,
            gk=gk,
            Aqk=Aqk,
            Akk=Akk,
            w=w,
            qg=qg,
            kg=kg,
            v_new=v_new,
            h=h,
            d_o=do_h,
            scale=ctx.scale,
            raw_g=g_h,
            A_log=_gate_parameter_for_ascendc(A_log),
            dt_bias=dt_bias_h,
            cu_seqlens=ctx.cu_host,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=ctx.export_intermediates,
            use_exp2=True,
        )

        # The fused backward returns gradients w.r.t. the NORMALIZED q/k in
        # head-major layout; chain the L2-norm backward in the same layout.
        if ctx.use_qk_l2norm_in_kernel:
            if head_major_qkv:
                q_norm, q_rstd_h = q_hat, q_rstd
                k_norm, k_rstd_h = k_hat, k_rstd
            else:
                q_norm = _seq_to_kernel(q_hat, is_varlen)
                k_norm = _seq_to_kernel(k_hat, is_varlen)
                q_rstd_h = _seq_to_kernel(q_rstd, is_varlen, beta=True)
                k_rstd_h = _seq_to_kernel(k_rstd, is_varlen, beta=True)
            dq_raw = l2norm_bwd(q_norm, q_rstd_h, dq_hat)
            dk_raw = l2norm_bwd(k_norm, k_rstd_h, dk_hat)
        else:
            dq_raw = dq_hat
            dk_raw = dk_hat

        if head_major_qkv:
            # Q/K/V live head-first, so their gradients go back untouched apart
            # from restoring the caller's rank/dtype.  The target rank is the one
            # the caller passed in -- recovering to "rank-4 whenever varlen"
            # would break a caller that already handed in a packed rank-3 tensor.
            def restore(t, dtype):
                t = t.to(dtype)
                if t.dim() < ctx.qkv_input_rank:
                    t = t.unsqueeze(0)
                return t

            dq = restore(dq_raw, ctx.qkv_dtype)
            dk = restore(dk_raw, ctx.k_dtype)
            dv = restore(dv, ctx.v_dtype)
        else:
            if is_varlen:
                dq = _from_head_major(dq_raw, ctx.qkv_dtype)
                dk = _from_head_major(dk_raw, ctx.k_dtype)
                dv = _from_head_major(dv, ctx.v_dtype)
            else:
                dq = _from_bnsd(dq_raw, ctx.qkv_dtype)
                dk = _from_bnsd(dk_raw, ctx.k_dtype)
                dv = _from_bnsd(dv, ctx.v_dtype)

        # g/beta entered sequence-major, so their gradients leave that way.
        if is_varlen:
            dg = _from_head_major(dg, ctx.g_dtype)
            db = _from_head_major_beta(db, ctx.beta_dtype)
        else:
            dg = _from_bnsd(dg, ctx.g_dtype)
            db = _from_bnsd_beta(db, ctx.beta_dtype)

        dA = None if A_log is None else dA.to(A_log)
        if dbias is not None and dt_bias is not None:
            if dbias.dim() == 2:
                # fused bwd returns [H, K]; fla-org's contract is [H*K]
                dbias = dbias.reshape(-1)
            dbias = dbias.to(dt_bias)

        return (dq, dk, dv, dg, db, dA, dbias, None, dh0,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None)
