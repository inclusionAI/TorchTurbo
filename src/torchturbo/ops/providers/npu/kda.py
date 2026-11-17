"""AscendC implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...packed_params import PackedParams
from ...registry import register_op


_KDA_CHUNK_SIZE = 64
_MAX_KDA_SEQUENCES = 1024


def _chunk_indices(cu_seqlens):
    indices = []
    for seq, (left, right) in enumerate(pairwise(cu_seqlens)):
        for chunk in range((right - left + 63) // 64):
            indices.extend((seq, chunk))
    return indices


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
        use_qk_l2norm_in_kernel: bool,
        export_intermediates: bool,
        packed_metadata: tuple[list[int], list[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cu_host = indices = None
        if packed_metadata is not None:
            cu_host, indices = packed_metadata
        packed = cu_host is not None
        ctx.empty = packed and q.size(1) == 0
        # An unused final state must arrive as None, not a materialized zero
        # gradient: this kernel does not implement final-state backward.
        ctx.set_materialize_grads(False)
        if ctx.empty:
            ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
            final_state = None
            if output_final_state:
                final_state = torch.zeros(
                    (len(cu_host) - 1, q.size(0), q.size(-1), v.size(-1)),
                    dtype=torch.float32, device=q.device,
                )
            return v.transpose(0, 1).unsqueeze(0), final_state

        from fla_npu.ops.ascendc import chunk_kda_fwd

        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)
        if packed:
            g, beta = g.squeeze(0), beta.squeeze(0)
        g, beta = g.contiguous(), beta.contiguous()
        q_hat = k_hat = q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q_hat = torch.empty_like(q)
            k_hat = torch.empty_like(k)
            q_rstd = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
            k_rstd = torch.empty(k.shape[:-1], dtype=torch.float32, device=k.device)
        outputs = chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="NTD" if packed else "BNSD",
            output_final_state=output_final_state,
            cu_seqlens=cu_host,
            chunk_indices=indices,
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=export_intermediates,
            use_exp2=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            q_hat_out=q_hat,
            k_hat_out=k_hat,
            q_rstd_out=q_rstd,
            k_rstd_out=k_rstd,
        )
        output, final_state, gk, aqk, akk, w, _u, qg, kg, v_new, h, _state = outputs
        # Backward consumes normalized Q/K and effective beta, not raw inputs.
        if q_hat is not None:
            q, k = q_hat, k_hat
        if export_intermediates:
            ctx.save_for_backward(
                q, k, v, g, beta, A_log, dt_bias, gk, aqk, akk, w, qg, kg, v_new, h,
                q_rstd, k_rstd,
            )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.cu_seqlens = cu_host
        ctx.chunk_indices = indices
        if packed:
            output = output.unsqueeze(0)
        return output, final_state

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        # A zero grad is the discarded output's placeholder; a non-zero
        # one means the loss depends on final_state, which has no backward.
        if grad_final_state is not None and bool(grad_final_state.any()):
            raise NotImplementedError(
                "AscendC KDA does not support gradients through final_state."
            )
        if ctx.empty:
            return tuple(
                torch.zeros_like(tensor) if tensor is not None else None
                for tensor in ctx.saved_tensors
            ) + (None,) * 8

        from fla_npu.ops.ascendc import (
            npu_chunk_gated_delta_rule_bwd_dhu,
            npu_chunk_kda_bwd_finalize,
            npu_chunk_kda_bwd_prepare,
        )

        (
            q, k, v, g, beta, A_log, dt_bias, gk, aqk, akk, w, qg, kg, v_new, h,
            q_rstd, k_rstd,
        ) = ctx.saved_tensors
        packed = ctx.cu_seqlens is not None
        if packed:
            grad_output = grad_output.squeeze(0)
        token_axis = 0 if packed else 1
        grad_output = grad_output.transpose(token_axis, token_axis + 1).contiguous()

        cu, indices = ctx.cu_seqlens, ctx.chunk_indices
        if packed and any(left == right for left, right in pairwise(cu)):
            # Empty sequences consume no tokens or chunk storage.
            cu = [cu[0]] + [right for left, right in pairwise(cu) if right > left]
            indices = _chunk_indices(cu)
        layout = "NTD" if packed else "BNSD"
        d_aqk, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
            aqk, v_new, grad_output, h, scale=ctx.scale, layout=layout,
            chunk_size=64, cu_seqlens=cu, chunk_indices=indices,
        )
        # Dhu requires rank-4 token tensors even for a packed sequence.
        scan_inputs = (qg, kg, w, grad_output, dv_local, gk)
        if packed:
            scan_inputs = tuple(t.unsqueeze(0) for t in scan_inputs)
        qg_scan, kg_scan, w_scan, do_scan, dv_local_scan, gk_scan = scan_inputs
        dh, _dh0, dv_scan = npu_chunk_gated_delta_rule_bwd_dhu(
            qg_scan, kg_scan, w_scan, do_scan, dv_local_scan, ctx.scale, 64, gK=gk_scan,
            cu_seqlens=cu, chunk_indices=indices, use_exp2=True,
        )
        if packed:
            dh, dv_scan = dh.squeeze(0), dv_scan.squeeze(0)

        if dt_bias is None:
            bias = torch.zeros((q.size(-3), q.size(-1)), dtype=torch.float32, device=q.device)
        else:
            bias = dt_bias.reshape(q.size(-3), q.size(-1))
        dq, dk, dv, dbeta, dg, dA, dbias = npu_chunk_kda_bwd_finalize(
            q, k, v, gk, g, beta.float(), A_log, bias, akk, v_new, h, dh,
            dv_scan, d_aqk, dq_raw,
            scale=ctx.scale, lower_bound=ctx.lower_bound, chunk_size=64,
            safe_gate=True, use_gate_in_kernel=ctx.use_gate_in_kernel,
            use_exp2=True, q_rstd=q_rstd, k_rstd=k_rstd,
            cu_seqlens=cu, chunk_indices=indices, output_layout=layout,
        )
        dbias = None if dt_bias is None else dbias.reshape_as(dt_bias)
        # Finalize already returns token-major gate/beta gradients.
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
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    packed_params: PackedParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA on [B,T,H,D] inputs, batching packed sequences within limits."""
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
    if packed_params is not None and q.size(0) != 1:
        raise ValueError("Packed KDA requires B=1.")

    qk_head_dim = g.shape[-1]
    v_head_dim = v.shape[-1] // beta.shape[-1]
    if q.ndim == 3:
        q = q.unflatten(-1, (-1, qk_head_dim))
        k = k.unflatten(-1, (-1, qk_head_dim))
        v = v.unflatten(-1, (-1, v_head_dim))
    elif q.ndim != 4:
        raise ValueError(f"Q/K/V must be rank-3 or rank-4, got {q.ndim}")

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

    packed = packed_params is not None
    qkv = []
    for tensor in (q, k, v):
        tensor = tensor.transpose(1, 2)
        if packed:
            tensor = tensor.squeeze(0)
        qkv.append(tensor)
    # Keep gate/beta in their input-order coordinates so the staged backward
    # can return token-major gradients directly to autograd.
    g = g.float()

    def run_chunk(query, key, value, gate, beta_input, metadata):
        return ChunkKDAFunction.apply(
            query.contiguous(), key.contiguous(), value.contiguous(),
            gate, beta_input, A_log, dt_bias, scale, output_final_state,
            lower_bound, use_gate_in_kernel,
            use_qk_l2norm_in_kernel,
            True, metadata,
        )

    num_sequences = 0 if packed_params is None else len(packed_params.cu_seqlens_list) - 1
    if packed_params is None or num_sequences <= _MAX_KDA_SEQUENCES:
        metadata = None
        if packed_params is not None:
            metadata = (
                packed_params.cu_seqlens_list,
                packed_params.chunk_indices_list[str(_KDA_CHUNK_SIZE)],
            )
        return run_chunk(qkv[0], qkv[1], qkv[2], g, beta, metadata)

    outputs, final_states = [], []
    cu_host = packed_params.cu_seqlens_list
    for sequence_start in range(0, num_sequences, _MAX_KDA_SEQUENCES):
        sequence_end = min(sequence_start + _MAX_KDA_SEQUENCES, num_sequences)
        token_start, token_end = cu_host[sequence_start], cu_host[sequence_end]
        boundaries = [
            offset - token_start
            for offset in cu_host[sequence_start:sequence_end + 1]
        ]
        chunk_indices = _chunk_indices(boundaries)
        group_qkv = [tensor[:, token_start:token_end] for tensor in qkv]
        output, final_state = run_chunk(
            *group_qkv,
            g[:, token_start:token_end],
            beta[:, token_start:token_end],
            (boundaries, chunk_indices),
        )
        outputs.append(output)
        if output_final_state:
            final_states.append(final_state)
    final_state = torch.cat(final_states, dim=0) if output_final_state else None
    return torch.cat(outputs, dim=1), final_state

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
                npu_chunk_gated_delta_rule_bwd_dhu,
                npu_chunk_kda_bwd_finalize,
                npu_chunk_kda_bwd_prepare,
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
