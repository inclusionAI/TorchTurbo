"""Time-major training interface for the AscendC KDA operators."""

from __future__ import annotations

import torch

from torchturbo.ops.packed_params import PackedParams

from .chunk_kda_function import ChunkKDAFunction


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


__all__ = ["ascendc_chunk_kda"]
