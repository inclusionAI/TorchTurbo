"""AscendC implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...packed_params import PackedParams
from ...registry import register_op


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
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    packed_params: PackedParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA on [B,T,H,D] inputs and return time-major output.

    The fla-npu v26.9.1 backward requires safe gate, equal Q/K/V head
    counts, K=V=128, and no initial-state or final-state gradients.
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
    if packed_params is not None and q.size(0) != 1:
        raise ValueError("Packed KDA requires B=1.")

    if use_qk_l2norm_in_kernel:
        from fla_npu.ops.triton import l2norm

        # Keep normalization in the autograd graph so its backward also runs
        # on A2/A3, where the fused KDA backward does not implement L2Norm.
        q = l2norm(q)
        k = l2norm(k)
    if use_gate_in_kernel:
        # AscendC consumes FP32 gate parameters; these casts retain gradients
        # to the model's mixed-precision parameters and their local head slices.
        if A_log is not None:
            A_log = A_log.float()
        if dt_bias is not None:
            dt_bias = dt_bias.float()
    else:
        A_log = dt_bias = None
    if scale is None:
        scale = q.size(-1) ** -0.5

    inputs = []
    for tensor in (q, k, v, g, beta):
        tensor = tensor.transpose(1, 2)
        if packed_params is not None:
            tensor = tensor.squeeze(0)
        inputs.append(tensor.contiguous())

    output, final_state = ChunkKDAFunction.apply(
        *inputs,
        A_log,
        dt_bias,
        scale,
        output_final_state,
        lower_bound,
        use_gate_in_kernel,
        packed_params,
    )
    if packed_params is not None:
        output = output.unsqueeze(0)
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
            from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd  # noqa: F401
            from fla_npu.ops.triton import l2norm  # noqa: F401
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
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        packed_params: PackedParams | None = None,
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
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            packed_params=packed_params,
        )
