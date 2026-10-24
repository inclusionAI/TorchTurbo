"""Autograd binding for fla-npu v26.9.1 KDA forward and backward."""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.ops.packed_params import PackedParams


class ChunkKDAFunction(torch.autograd.Function):
    """Connect the head-major AscendC KDA training operators."""

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
        packed_params: PackedParams | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla_npu.ops.ascendc import chunk_kda_fwd

        cu_seqlens = None
        chunk_indices = None
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens_list
            chunk_indices = packed_params.chunk_indices_list["64"]
        outputs = chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="NTD" if packed_params is not None else "BNSD",
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=True,
            use_exp2=True,
        )
        output, final_state, gk, aqk, akk, w, _u, qg, kg, v_new, h, _state = outputs
        ctx.save_for_backward(
            q, k, v, g, beta, A_log, dt_bias, gk, aqk, akk, w, qg, kg, v_new, h
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        # An unused final state must arrive as None, not a materialized zero
        # gradient: this kernel does not implement final-state backward.
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        from fla_npu.ops.ascendc import chunk_kda_bwd

        if grad_final_state is not None:
            raise NotImplementedError(
                "AscendC KDA does not support gradients through final_state."
            )
        q, k, v, g, beta, A_log, dt_bias, gk, aqk, akk, w, qg, kg, v_new, h = (
            ctx.saved_tensors
        )
        token_axis = 0 if ctx.cu_seqlens is not None else 1
        grad_output = grad_output.transpose(token_axis, token_axis + 1).contiguous()
        bias = dt_bias
        if bias is not None:
            bias = bias.reshape(q.size(-3), q.size(-1))
        # Forward returns h in chunk-major order. Pass it directly to backward;
        # the other saved intermediates are already head-major.
        dq, dk, dv, dbeta, dg, _dh0, dA, dbias = chunk_kda_bwd(
            q,
            k,
            v,
            beta,
            gk,
            aqk,
            akk,
            w,
            qg,
            kg,
            v_new,
            h,
            grad_output,
            ctx.scale,
            raw_g=g if ctx.use_gate_in_kernel else None,
            A_log=A_log,
            dt_bias=bias,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=ctx.chunk_indices,
            chunk_size=64,
            safe_gate=True,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=True,
            use_exp2=True,
            implementation="legacy",
        )
        if dbias is not None:
            dbias = dbias.reshape_as(dt_bias)
        return (
            dq,
            dk,
            dv,
            dg.to(g.dtype),
            dbeta,
            dA,
            dbias,
            None,
            None,
            None,
            None,
            None,
        )
