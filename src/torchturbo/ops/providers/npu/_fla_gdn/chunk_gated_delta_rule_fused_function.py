"""Adaptive autograd adapter around the AscendC GDN large-fusion operators.

The fusion granularity is chosen *independently* for the forward and the
backward, because the two kernels land on different SoCs/releases at
different times (on the pinned v26.9.1 A3 wheel, for example, the fused
forward exists but the composite backward does not):

* ``fwd fused``   -- one ``npu_chunk_gated_delta_rule_fwd`` aclnn call with
  ``disable_recompute=True`` exporting exactly the ``g_cumsum``/``A`` the
  backward needs.
* ``bwd fused``  -- one ``npu_chunk_gated_delta_rule_bwd`` aclnn call
  (consumes the fused forward's exports; requires ``fwd fused``).
* ``bwd composed`` -- the vendored four-operator chain ``_backward_chain``,
  which recomputes w/u/h/v_new itself from the saved q/k/v/g_cumsum/beta/A,
  so it feeds equally well on the fused forward's exports.

When the fused forward is unavailable (outside the K=V=128/chunk=64
envelope, or the probe fails), the wrapper dispatches to the composed-chain
:class:`ChunkGatedDeltaRuleFunction` instead -- both forward and backward,
since the fused backward's ``A`` is only produced by the fused forward.

The calling convention is the exact one of the composed-chain adapter:
head-first ``q/k [B,Hk,T,K]``, ``v [B,Hv,T,V]``, BSN ``g/beta [B,T,Hv]``,
sequence-major output ``o [B,T,Hv,V]``. The q/k l2norm always stays outside
the fused pair (the in-kernel export is an A5-only path), so gradient post-
processing is identical in every combination.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from .chunk_gated_delta_rule_function import (
    ListIndexDict,
    TensorIndexDict,
    _backward_chain,
    _normalize_varlen_metadata,
    _validate_forward_inputs,
)
from .l2norm import l2norm_bwd, l2norm_fwd

__all__ = [
    "ChunkGatedDeltaRuleAdaptiveFunction",
    "fused_bwd_usable",
    "fused_fwd_usable",
    "fused_envelope_supported",
]

# The composite backward kernel is stricter than the forward: K = V = 128 and
# chunk_size = 64 only (the fwd additionally allows V = 256 and chunk 128).
_FUSED_KEY_DIM = 128
_FUSED_VALUE_DIM = 128
_FUSED_CHUNK_SIZE = 64


@lru_cache(maxsize=None)
def _load_fused_ops() -> Optional[tuple]:
    """Import the large-fusion pair or return None.

    Imported lazily + cached so this module never breaks import-time
    availability of the composed chain -- the same rule the torchturbo op
    providers follow for optional dependencies.
    """
    try:
        from fla_npu.ops.ascendc import (
            npu_chunk_gated_delta_rule_bwd,
            npu_chunk_gated_delta_rule_fwd,
        )

        return npu_chunk_gated_delta_rule_fwd, npu_chunk_gated_delta_rule_bwd
    except ImportError:
        return None


def fused_envelope_supported(
    key_dim: int,
    value_dim: int,
    chunk_size: int,
) -> bool:
    """Whether the large-fusion kernels cover this shape configuration."""
    return (
        key_dim == _FUSED_KEY_DIM
        and value_dim == _FUSED_VALUE_DIM
        and chunk_size == _FUSED_CHUNK_SIZE
    )


def _probe_pair(probe_bwd: bool) -> tuple[bool, str]:
    """Run one minimal fused call (fwd, plus bwd when ``probe_bwd``) on a
    tiny dense shape and report ``(usable, failure_reason)``.

    Import availability is not enough: fla-npu 26.9.1 (the version pinned in
    the NPU docker) ships the Python entry of the fused pair but compiles no
    ``chunk_gated_delta_rule_bwd`` kernel for ascend910_93 -- every backward
    call there fails with aclnn error 161002. Probing costs one small launch
    per direction and lets the wrapper adapt instead of crashing mid-training.
    """
    fused_ops = _load_fused_ops()
    if fused_ops is None:
        return False, (
            "fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd/bwd "
            "not importable"
        )
    npu_fwd, npu_bwd = fused_ops
    if probe_bwd and npu_bwd is None:
        return False, "npu_chunk_gated_delta_rule_bwd not importable"
    try:
        import torch  # local import keeps the module importable off-device

        q = torch.randn(1, 1, 64, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(1, 1, 64, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 64, 128, device="npu", dtype=torch.bfloat16)
        g = -torch.rand(1, 64, 1, device="npu", dtype=torch.float32) * 0.1
        beta = torch.rand(1, 64, 1, device="npu", dtype=torch.float32)
        do = torch.randn(1, 64, 1, 128, device="npu", dtype=torch.bfloat16)
        (
            _o,
            _final_state,
            g_cumsum,
            A,
            _beta_eff,
            _h,
            _q_hat,
            _k_hat,
            _q_rstd,
            _k_rstd,
        ) = npu_fwd(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            chunk_size=_FUSED_CHUNK_SIZE,
            scale=1.0,
            disable_recompute=True,
        )
        if probe_bwd:
            npu_bwd(
                q,
                k,
                v,
                g_cumsum,
                beta,
                A,
                do.contiguous(),
                1.0,
                chunk_size=_FUSED_CHUNK_SIZE,
            )
        torch.npu.synchronize()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any launch failure means unusable
        return False, repr(exc)


@lru_cache(maxsize=None)
def fused_fwd_usable() -> bool:
    """Cached forward-direction probe result (one tiny fused fwd launch)."""
    usable, _ = _probe_pair(probe_bwd=False)
    return usable


@lru_cache(maxsize=None)
def fused_bwd_usable() -> bool:
    """Cached backward-direction probe result (tiny fused fwd+bwd launch)."""
    usable, _ = _probe_pair(probe_bwd=True)
    return usable


class ChunkGatedDeltaRuleAdaptiveFunction(torch.autograd.Function):
    """Autograd binding with the fused forward and an adaptive backward.

    ``apply`` extends the composed-chain adapter's signature with one
    trailing ``fused_backward`` flag: the forward always runs the fused
    kernel (the wrapper only dispatches here when the fwd probe passed and
    the envelope is met), while the backward honours the flag -- one
    ``npu_chunk_gated_delta_rule_bwd`` call, or the vendored composed chain
    ``_backward_chain`` over the fused forward's exports.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[TensorIndexDict | torch.Tensor] = None,
        chunk_indices_list: Optional[
            ListIndexDict | list[int] | tuple[int, ...] | torch.Tensor
        ] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = _FUSED_CHUNK_SIZE,
        fused_backward: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        (
            batch,
            _key_heads,
            _value_heads,
            tokens,
            key_dim,
            value_dim,
        ) = _validate_forward_inputs(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            chunk_size,
        )
        if not fused_envelope_supported(key_dim, value_dim, chunk_size):
            raise ValueError(
                "the fused AscendC GDR pair requires K=V=128 and "
                f"chunk_size=64, got K={key_dim}, V={value_dim}, "
                f"chunk_size={chunk_size}"
            )
        if fused_backward and not fused_bwd_usable():
            # The wrapper normally prevents this; the guard keeps an explicit
            # override (see TORCHTURBO_GDN_USE_FUSED) from failing mid-train.
            fused_backward = False
        (
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
        ) = _normalize_varlen_metadata(
            g=g,
            total_tokens=tokens,
            batch=batch,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
        )
        # The fused ops take the host lists; the composed backward chain
        # additionally needs the device tensor for its reverse cumsum.
        ctx.cu_seqlens = cu_seqlens

        if initial_state is not None and ctx.needs_input_grad[6]:
            raise NotImplementedError(
                "initial_state.requires_grad=True is unsupported by this "
                "adapter. Pass a detached initial_state."
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        main_dtype = q.dtype
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd = None
            k_rstd = None

        chunk_list = (
            chunk_indices_list.get(str(chunk_size))
            if chunk_indices_list is not None
            else None
        )
        fused_ops = _load_fused_ops()
        if fused_ops is None:
            raise RuntimeError(
                "the installed fla_npu does not provide the large-fusion "
                "npu_chunk_gated_delta_rule_fwd/bwd pair; the wrapper "
                "should have dispatched to the composed chain"
            )
        npu_fwd = fused_ops[0]
        (
            o,
            final_state,
            g_cumsum,
            A,
            _beta_eff,
            _h,
            _q_hat,
            _k_hat,
            _q_rstd,
            _k_rstd,
        ) = npu_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_list,
            scale=float(scale),
            disable_recompute=True,
            return_intermediate_states=False,
            state_v_first=False,
        )
        # With use_qk_l2norm_in_kernel=False the kernel treats the (already
        # externally normalized) q/k as raw inputs and aliases q_hat/k_hat to
        # them; the fused backward therefore needs no rstd.

        # Saved in the exact spelling both backward paths consume: q/k
        # normalized, g already exported as g_cumsum (still [B,T,Hv] BSN).
        ctx.save_for_backward(
            q, k, v, g_cumsum, beta, A, initial_state, q_rstd, k_rstd
        )
        ctx.has_l2norm = use_qk_l2norm_in_kernel
        ctx.scale = float(scale)
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_list = chunk_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        ctx.fused_backward = fused_backward
        ctx.set_materialize_grads(False)

        # o comes out sequence-major [B, T, Hv, V]; the wrapper transposes.
        if not output_final_state:
            final_state = None
        return o.to(main_dtype), final_state

    @staticmethod
    def backward(
        ctx,
        do: Optional[torch.Tensor],
        _dht: Optional[torch.Tensor],
    ):
        if _dht is not None:
            raise NotImplementedError(
                "final_state gradients are unsupported by this adapter. "
                "Set output_final_state=False, or do not include "
                "final_state in the differentiated loss."
            )
        if do is None:
            # Nothing differentiates through o; no input gradient is owed.
            return (None,) * 15

        (
            q,
            k,
            v,
            g_cumsum,
            beta,
            A,
            initial_state,
            q_rstd,
            k_rstd,
        ) = ctx.saved_tensors
        # do matches o's [B, T, Hv, V] sequence-major layout, which both
        # backward paths expect.
        do = do.contiguous()

        if ctx.fused_backward:
            fused_ops = _load_fused_ops()
            if fused_ops is None:
                raise RuntimeError(
                    "the installed fla_npu lost the large-fusion backward"
                )
            npu_bwd = fused_ops[1]
            (
                dq,
                dk,
                dv,
                d_beta,
                d_g,
                _dh0,
                _d_a_log,
                _d_dt_bias,
            ) = npu_bwd(
                q,
                k,
                v,
                g_cumsum,
                beta,
                A,
                do,
                ctx.scale,
                chunk_size=ctx.chunk_size,
                cu_seqlens=ctx.cu_seqlens_list,
                chunk_indices=ctx.chunk_list,
                initial_state=initial_state,
                dht=None,
                use_qk_l2norm_in_kernel=False,
                use_exp2=False,
            )
            # d_g is already the raw-gate gradient (the composite backward
            # applies the reverse chunk-local cumsum and the GVA head
            # reduction internally); dq/dk are w.r.t. the normalized q/k.
            del _dh0, _d_a_log, _d_dt_bias
        else:
            # Composed backward over the fused forward's exports: the chain
            # recomputes w/u/h/v_new from the saved q/k/v/g_cumsum/beta/A.
            # The fused forward's A export is Hv-headed [B,Hv,T,chunk]; the
            # composed chain's A_head is GVA-reduced when Hv != Hk, matching
            # the k-reduction used by recompute_w_u/dqkwg downstream --
            # _backward_chain reduces the KKT input the same way, so the
            # export feeds straight in.
            dq, dk, dv, d_g, d_beta, _dh0 = _backward_chain(
                q=q,
                k=k,
                v=v,
                cumulative_g=g_cumsum,
                beta=beta,
                A_head=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=None,
                cu_seqlens=ctx.cu_seqlens,
                cu_seqlens_list=ctx.cu_seqlens_list,
                chunk_indices=ctx.chunk_indices,
                chunk_indices_list=ctx.chunk_indices_list,
                chunk_size=ctx.chunk_size,
            )
        if ctx.has_l2norm:
            assert q_rstd is not None and k_rstd is not None
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # The order must match all 15 forward inputs exactly:
        # q, k, v, g, beta, scale, initial_state, output_final_state,
        # cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list,
        # use_qk_l2norm_in_kernel, chunk_size, fused_backward.
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            d_g.to(g_cumsum.dtype),
            d_beta.to(beta.dtype),
            None,  # scale
            None,  # initial_state gradient (rejected in forward)
            None,  # output_final_state
            None,
            None,
            None,
            None,
            None,  # use_qk_l2norm_in_kernel
            None,  # chunk_size
            None,  # fused_backward
        )
