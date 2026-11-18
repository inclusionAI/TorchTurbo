from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="compressor",
    impl_name="torch_compressor",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchCompressor(OpImpl):
    """Reference packed-KV compressor: fp32 projections + softmax pooling
    + RMSNorm.

    Mirrors the softmax-gated path of ``DeepseekV41Compressor.forward``
    (pre-op inline path): projects x with the fp32 wkv/wgate weights, pools
    every complete packed block with a per-channel softmax gate, casts the
    latent back to x's dtype, and applies the module-side RMSNorm. Invalid
    block slots stay zero so the ``[b, max_blocks, head_dim]`` layout is
    identical across implementations.

    The norm tail repeats the ``DeepseekV41RMSNorm.forward`` op sequence on
    the same bf16 latent, so absorbing the norm into the op keeps this path
    bitwise identical to the pre-op module inline computation.
    """

    def __call__(
        self,
        x: torch.Tensor,
        wkv_weight: torch.Tensor,
        wgate_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        ratio: int,
        head_dim: int,
        eps: float,
    ):
        # Imported here so the provider scan does not pull the models package
        # (and its transformers dependency) into every dispatcher creation.
        from torchturbo.models.deepseek_v41.source_packing_utils import compress_packed_kv

        # Softmax pooling runs in fp32; the checkpoint stores fp32 weights to
        # match. The pooled latent returns to x's dtype before the norm so
        # downstream concat with window KV stays single-dtype.
        dtype = x.dtype
        x = x.float()
        kv = F.linear(x, wkv_weight)
        score = F.linear(x, wgate_weight)
        # v4.1 has no learned position bias; a zero ape keeps the shared
        # packed pooling helper on the plain softmax path.
        ape = torch.zeros(ratio, head_dim, dtype=score.dtype, device=x.device)
        latent, _ = compress_packed_kv(
            kv,
            score,
            ape,
            attention_mask,
            position_ids,
            ratio,
            head_dim,
            overlap=False,
        )
        if latent is None:
            return None
        # RMSNorm tail: same op sequence as DeepseekV41RMSNorm.forward on the
        # bf16 latent (fp32 compute, weight in fp32, cast back).
        latent = latent.to(dtype).to(torch.float32)
        variance = latent.pow(2).mean(-1, keepdim=True)
        latent = latent * torch.rsqrt(variance + eps)
        return (norm_weight.float() * latent).to(dtype)
