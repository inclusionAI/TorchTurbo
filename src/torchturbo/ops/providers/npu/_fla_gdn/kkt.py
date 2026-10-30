"""Reuse the FLA KKT kernel across packed chunk-count alignment classes."""

import inspect

import torch
import triton

# The KKT kernel lives in different places across fla_npu generations:
# 26.9.1+ wheels vendor it at fla_npu.ops.triton.triton_core (built from the
# repo's fla/ overlay, see fla_npu setup.py), while 26.6.0-era wheels merely
# re-export it from the fork-fla package overlay. Prefer the self-contained
# path (works with a stock fla) and fall back to the fork overlay.
try:
    from fla_npu.ops.triton.triton_core.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd_kernel as _upstream_kernel,
    )
except ImportError:
    from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd_kernel as _upstream_kernel,
    )
from fla_npu.ops.triton import chunk_scaled_dot_kkt_fwd as _upstream_fwd

# Keep the upstream computation while excluding packed work counts from the
# JIT key. A change in NT divisibility must not compile another binary.
_kkt_fwd_kernel = triton.jit(
    _upstream_kernel.fn.fn, do_not_specialize=["T", "NT", "TOTAL_TASKS"]
)


def _kernel_params(kernel) -> frozenset:
    """Parameter names of a (possibly re-wrapped) triton JIT kernel."""
    fn = getattr(kernel, "fn", kernel)
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return frozenset()


# GQA landed in the KKT kernel as H_K/H_V constexprs (replacing the single
# H): newer releases require both, older ones reject them. Probe once which
# generation the installed fla ships so the raw launch below works on either.
_KKT_PARAMS = _kernel_params(_kkt_fwd_kernel)
_KKT_HAS_GQA_PARAMS = "H_K" in _KKT_PARAMS


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute gated KKT for the adapter's head-first keys and packed metadata."""
    if chunk_size != 64:
        return _upstream_fwd(
            k=k,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
        )
    batch, heads, tokens, dim = k.shape
    num_chunks = (
        triton.cdiv(tokens, chunk_size) if cu_seqlens is None else len(chunk_indices)
    )
    beta = beta.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    output = torch.empty(
        batch, tokens, heads, chunk_size, device=k.device, dtype=output_dtype
    )
    if _KKT_HAS_GQA_PARAMS:
        # k is already expanded to value_heads (see _expand_k_for_kkt), so
        # H_K == H_V == heads and HV_PER_HK == 1, matching the old H spelling.
        head_args = {"H_K": heads, "H_V": heads}
    else:
        head_args = {"H": heads}
    _kkt_fwd_kernel[(24,)](
        k=k,
        g=g,
        beta=beta,
        A=output,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=tokens,
        **head_args,
        K=dim,
        BT=chunk_size,
        BK=128,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=True,
        NT=num_chunks,
        B=batch,
        TOTAL_TASKS=batch * num_chunks,
    )
    return output
