"""AscendC implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch

from torchturbo.runtime import TurboDevice
from torchturbo.utils.functools import cache

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _chunk_indices(cu_seqlens):
    indices = []
    for seq, (left, right) in enumerate(pairwise(cu_seqlens)):
        for chunk in range((right - left + 63) // 64):
            indices.extend((seq, chunk))
    return indices


@cache
def _prepare_kda_metadata(cu_seqlens, _version):
    """Cache host boundaries and 64-token chunk indices across layers."""
    cu_host = cu_seqlens.detach().cpu().tolist()
    return cu_host, _chunk_indices(cu_host)


class ChunkKDAFunction(torch.autograd.Function):
    """Connect token-major model tensors to the AscendC KDA operators."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        output_final_state: bool,
        lower_bound: float | None,
        use_gate_in_kernel: bool,
        cu_seqlens: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        use_beta_sigmoid_in_kernel: bool,
        save_backward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla_npu.ops.ascendc import chunk_kda_fwd

        cu_host = indices = None
        if cu_seqlens is not None:
            cu_host, indices = _prepare_kda_metadata(cu_seqlens, cu_seqlens._version)
        packed = cu_host is not None
        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        if packed:
            g, beta = g.squeeze(0), beta.squeeze(0)
        g, beta = g.contiguous(), beta.contiguous()
        q_hat = k_hat = q_rstd = k_rstd = None
        # Normalization caches are head-major regardless of the input layout.
        token_axis = 0 if packed else 1
        head_shape = q.transpose(token_axis, token_axis + 1).shape
        if use_qk_l2norm_in_kernel and save_backward:
            q_hat = torch.empty(head_shape, dtype=q.dtype, device=q.device)
            k_hat = torch.empty(head_shape, dtype=k.dtype, device=k.device)
            q_rstd = torch.empty(head_shape[:-1], dtype=torch.float32, device=q.device)
            k_rstd = torch.empty(head_shape[:-1], dtype=torch.float32, device=k.device)
        beta_eff = None
        if use_beta_sigmoid_in_kernel and save_backward:
            beta_eff = torch.empty(head_shape[:-1], dtype=torch.float32, device=beta.device)
        outputs = chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="TND" if packed else "BSND",
            output_final_state=output_final_state,
            cu_seqlens=cu_host,
            chunk_indices=indices,
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=False,
            use_exp2=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            q_hat_out=q_hat,
            k_hat_out=k_hat,
            q_rstd_out=q_rstd,
            k_rstd_out=k_rstd,
            beta_eff_out=beta_eff,
        )
        output, final_state, _gk, aqk, akk, _w, _u, _qg, _kg, _v_new, _h, _state = outputs
        if q_hat is not None:
            q, k = q_hat, k_hat
        if save_backward:
            ctx.save_for_backward(
                q, k, v, g, beta, A_log, dt_bias, aqk, akk,
                q_rstd, k_rstd, beta_eff,
            )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.cu_seqlens = cu_host
        ctx.chunk_indices = indices
        # An unused final state must arrive as None, not a materialized zero
        # gradient: this kernel does not implement final-state backward.
        ctx.set_materialize_grads(False)
        if packed:
            output = output.unsqueeze(0)
        return output, final_state

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        from fla_npu.ops.ascendc import npu_chunk_kda_bwd

        # A zero grad is the discarded output's placeholder; a non-zero
        # one means the loss depends on final_state, which has no backward.
        if grad_final_state is not None and bool(grad_final_state.any()):
            raise NotImplementedError(
                "AscendC KDA does not support gradients through final_state."
            )
        (
            q, k, v, g, beta, A_log, dt_bias, aqk, akk,
            q_rstd, k_rstd, beta_eff,
        ) = ctx.saved_tensors
        packed = ctx.cu_seqlens is not None
        if packed:
            grad_output = grad_output.squeeze(0)
        # Model tensors and output gradients stay token-major. The upstream
        # backward rebuilds the omitted forward caches.
        dq, dk, dv, dbeta, dg, _dh0, dA, dbias = npu_chunk_kda_bwd(
            q, k, v, beta, None, aqk, akk, None, None, None, None, None,
            grad_output.contiguous(), ctx.scale,
            raw_g=g, A_log=A_log, dt_bias=dt_bias,
            lower_bound=ctx.lower_bound, chunk_size=64,
            safe_gate=True, use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=False, use_exp2=True,
            q_rstd=q_rstd, k_rstd=k_rstd,
            cu_seqlens=ctx.cu_seqlens, chunk_indices=ctx.chunk_indices,
            layout="TND" if packed else "BSND",
            use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
            beta_eff=beta_eff,
        )
        dbias = None if dt_bias is None else dbias.reshape_as(dt_bias)
        # Gate and beta gradients have the original token-major layout.
        if packed:
            dg, dbeta = dg.unsqueeze(0), dbeta.unsqueeze(0)
        return (
            dq,
            dk,
            dv,
            dg.to(ctx.g_dtype),
            dbeta.to(ctx.beta_dtype),
            dA,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def ascendc_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA on [B,T,C] Q/K/V inputs and return time-major output.

    A2/A3 training requires BF16 Q/K/V, safe gate computed in the kernel,
    equal Q/K/V head counts, K=V=128, and no state gradients.
    """
    if not safe_gate:
        raise NotImplementedError(
            "AscendC KDA training requires safe_gate=True. "
            "Select fla_kda for other gate configurations."
        )
    if initial_state is not None:
        raise NotImplementedError(
            "AscendC KDA training does not support initial_state."
        )
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "AscendC KDA training requires use_gate_in_kernel=True."
        )
    if cu_seqlens is not None and q.size(0) != 1:
        raise ValueError("Packed KDA requires B=1.")

    qk_head_dim = g.shape[-1]
    v_head_dim = v.shape[-1] // beta.shape[-1]
    q = q.unflatten(-1, (-1, qk_head_dim))
    k = k.unflatten(-1, (-1, qk_head_dim))
    v = v.unflatten(-1, (-1, v_head_dim))

    # AscendC consumes FP32 gate parameters; these casts retain gradients
    # to the model's mixed-precision parameters and their local head slices.
    if A_log is not None:
        A_log = A_log.float()
    if dt_bias is not None:
        dt_bias = dt_bias.float()
    if scale is None:
        scale = q.size(-1) ** -0.5
    if lower_bound is None:
        lower_bound = -5.0

    inputs = []
    for tensor in (q, k, v):
        if cu_seqlens is not None:
            tensor = tensor.squeeze(0)
        inputs.append(tensor.contiguous())
    # FP32 gate inputs avoid Prepare's BF16 path. Keep the cast in autograd.
    g = g.float()
    output, final_state = ChunkKDAFunction.apply(
        *inputs,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        output_final_state,
        lower_bound,
        use_gate_in_kernel,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        torch.is_grad_enabled(),
    )
    return output, final_state


@register_op(
    op_name="kda",
    impl_name="ascendc_kda",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla_npu.ops.ascendc import (  # noqa: F401
                chunk_kda_fwd,
                npu_chunk_kda_bwd,
            )
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ascendc_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )
