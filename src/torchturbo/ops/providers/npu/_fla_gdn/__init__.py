"""AscendC conv1d / gated-delta-rule backends for the Qwen3.5 linear-attention layer.

This package vendors the two verified autograd adapters from the
``nachuan-flash-linear-attention-npu`` (fla-npu) example suite:

* :class:`AscendCCausalConv1dFunction` — AscendC causal short-conv1d.
* :class:`ChunkGatedDeltaRuleFunction` — AscendC chunk gated-delta-rule chain.

The two ``apply`` functions are the verified entry points, but their signatures
and tensor layouts differ from the ``fla`` (theta-fla triton-ascend) functions
that ``modeling_qwen3_5_moe.py`` / ``model.py`` currently call. To switch the
Qwen3.5 linear-attention path to AscendC *without* rewriting the model code, we
expose two drop-in wrappers that keep the ``fla`` calling convention:

* :func:`ascendc_causal_conv1d`  — mirrors ``fla.modules.convolution.causal_conv1d``:
  ``x`` is ``[B, T, D]``, ``weight`` is ``[D, W]``, returns ``[B, T, D]``.

* :func:`ascendc_chunk_gated_delta_rule` — mirrors
  ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``: ``q/k/v`` are time-first
  ``[B, T, H, D]``, ``g/beta`` are ``[B, T, H]``, returns ``[B, T, H, V]``.

The calling convention (``head_num = 2*nk + nv`` for the conv, head-first
``[B, H, T, D]`` q/k/v and ``scale = dk ** -0.5`` / ``chunk_size = 64`` for the
GDR) follows the verified ``DemoGatedDeltaNet`` reference in
``examples/flash_gated_delta_rule.py`` of the fla-npu repository.

The conv1d is depthwise (one ``[W]`` filter per channel, weight ``[D, W]``), so
``head_num`` is purely the NPU kernel's internal channel tiling: any tiling with
``head_dim % 16 == 0`` yields the identical per-channel result. We pick
``head_dim = 128`` (== the model's ``linear_key_head_dim``) which matches the
verified reference and satisfies the alignment constraint.
"""

from __future__ import annotations

from typing import Optional

import torch

from torchturbo.ops.packed_params import PackedParams

from .ascendc_causal_conv1d_function import AscendCCausalConv1dFunction
from .chunk_gated_delta_rule_function import ChunkGatedDeltaRuleFunction

__all__ = [
    "AscendCCausalConv1dFunction",
    "ChunkGatedDeltaRuleFunction",
    "PackedParams",
    "ascendc_causal_conv1d",
    "ascendc_chunk_gated_delta_rule",
]


# Preferred depthwise-conv head dimension. Equals the model's
# ``linear_key_head_dim`` (128 for Qwen3.5), which makes ``head_num`` collapse
# to ``2 * num_key_heads + num_value_heads`` exactly as in the verified
# ``DemoGatedDeltaNet`` reference. Falling back to smaller multiples of 16 keeps
# the wrapper usable for other head dims while still satisfying the kernel's
# ``head_dim % 16 == 0`` alignment requirement.
_PREFERRED_HEAD_DIMS = (128, 64, 32, 16)


def _conv_head_num(dim: int) -> int:
    """Pick an AscendC conv1d ``head_num`` for feature dim ``dim``.

    The conv is depthwise, so this is only an internal tiling parameter; we just
    need ``dim % head_num == 0`` and ``head_dim = dim // head_num`` to be a
    multiple of 16.
    """
    for head_dim in _PREFERRED_HEAD_DIMS:
        if dim % head_dim == 0:
            return dim // head_dim
    raise ValueError(
        f"AscendC causal-conv1d could not find a valid head_num for dim={dim}; "
        f"dim must be divisible by one of {_PREFERRED_HEAD_DIMS}."
    )


def ascendc_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    packed_params: Optional[PackedParams] = None,
    head_num: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in AscendC replacement for ``fla.modules.convolution.causal_conv1d``.

    Args:
        x: ``[B, T, D]`` (varlen: ``[1, T, D]`` with ``cu_seqlens``).
        weight: depthwise filters ``[D, W]`` (one row per channel).
        bias: optional ``[D]``.
        activation: ``None`` or ``"silu"``/``"swish"``.
        initial_state / output_final_state / cu_seqlens: as in ``fla``.
        head_num: when given, return the operator's native head-first output
            ``[B, head_num, T, D // head_num]`` instead of flattening it back
            to ``[B, T, D]``.  Pass the consumer's head count (e.g. the fused
            KDA operator's) so it can consume the output with no layout
            transpose; the convolution is depthwise, so ``head_num`` only
            fixes the output spelling and never changes the per-channel
            result.

    Returns:
        ``(y, final_state)`` where ``y`` is ``[B, T, D]``, or
        ``[B, head_num, T, D // head_num]`` when ``head_num`` is given, and
        ``final_state`` is ``None`` unless ``output_final_state``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, D], got shape={tuple(x.shape)}")
    batch, seq_len, dim = x.shape
    head_first = head_num is not None
    if head_first:
        head_num = int(head_num)
        if head_num <= 0 or dim % head_num:
            raise ValueError(
                f"head_num={head_num} must be a positive divisor of D={dim}."
            )
    else:
        head_num = _conv_head_num(dim)
    cu_seqlens_list = None
    if packed_params is not None:
        if cu_seqlens is not None:
            raise ValueError("pass either cu_seqlens or packed_params, not both")
        if packed_params.cu_seqlens.device != x.device:
            raise ValueError(
                "packed_params must be built on the same device as x, "
                f"got {packed_params.cu_seqlens.device} and {x.device}"
            )
        cu_seqlens = packed_params.cu_seqlens
        cu_seqlens_list = packed_params.cu_seqlens_list
    if cu_seqlens is not None and cu_seqlens.device != x.device:
        cu_seqlens = cu_seqlens.to(x.device)

    # The AscendC kernel always produces head-first ``[B, H, T, Dh]``. Because
    # the conv is depthwise with contiguous head grouping (``d = h * Dh + dh``),
    # transposing back and flattening recovers the exact per-channel result in
    # ``[B, T, D]`` -- but that copy is pure overhead when the consumer (the
    # fused KDA operator) wants head-first anyway, so ``head_num`` skips it.
    y_head, final_state = AscendCCausalConv1dFunction.apply(
        x,
        weight,
        head_num,
        bias,
        None,            # residual
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
        cu_seqlens_list,
    )
    if head_first:
        return y_head, final_state
    y = y_head.transpose(1, 2).reshape(batch, seq_len, dim).contiguous()
    return y, final_state


def ascendc_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    packed_params: Optional[PackedParams] = None,
    save_backward_intermediates: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in AscendC replacement for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

    Accepts the time-first layout used throughout the Qwen3.5 modeling code
    (``q/k/v`` as ``[B, T, H, D]``, ``g/beta`` as ``[B, T, H]``) and returns
    ``[B, T, H, V]``, matching ``fla``'s ``head_first=False`` convention. The
    AscendC adapter expects head-first ``q/k/v`` (``[B, H, T, D]``), so we
    transpose on the way in. ``scale`` defaults to ``dk ** -0.5`` exactly like
    ``fla`` / the verified ``DemoGatedDeltaNet`` reference.
    """
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be [B, T, H, D], got q.shape={tuple(q.shape)}")
    # time-first [B, T, H, D] -> head-first [B, H, T, D]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    scale = float(k.shape[-1] ** -0.5)

    cu_seqlens_list: Optional[list[int]] = None
    chunk_indices = None
    chunk_indices_list = None
    if packed_params is not None:
        if cu_seqlens is not None:
            raise ValueError("pass either cu_seqlens or packed_params, not both")
        if packed_params.cu_seqlens.device != q.device:
            raise ValueError(
                "packed_params must be built on the same device as q, "
                f"got {packed_params.cu_seqlens.device} and {q.device}"
            )
        cu_seqlens = packed_params.cu_seqlens
        cu_seqlens_list = packed_params.cu_seqlens_list
        chunk_indices = packed_params.chunk_indices
        chunk_indices_list = packed_params.chunk_indices_list
    if cu_seqlens is not None:
        if cu_seqlens.device != q.device:
            cu_seqlens = cu_seqlens.to(q.device)
        if cu_seqlens_list is None:
            cu_seqlens_list = [
                int(item) for item in cu_seqlens.detach().cpu().tolist()
            ]

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
        save_backward_intermediates,
    )
    # o is already [B, T, H, V] (the adapter transposes back internally).
    return o, final_state
