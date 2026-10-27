"""Reuse the FLA KKT kernel across packed chunk-count alignment classes."""

import torch
import triton
from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd_kernel as _upstream_kernel,
)
from fla_npu.ops.triton import chunk_scaled_dot_kkt_fwd as _upstream_fwd

# Keep the upstream computation while excluding packed work counts from the
# JIT key. A change in NT divisibility must not compile another binary.
_kkt_fwd_kernel = triton.jit(
    _upstream_kernel.fn.fn, do_not_specialize=["T", "NT", "TOTAL_TASKS"]
)


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
    _kkt_fwd_kernel[(24,)](
        k=k,
        g=g,
        beta=beta,
        A=output,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=tokens,
        H=heads,
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
