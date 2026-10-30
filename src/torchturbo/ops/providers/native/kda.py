"""FLA implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="kda",
    impl_name="fla_kda",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
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
        **kwargs,
    ):
        # fla-org's chunk_kda is an autograd function returning (o, final_state).
        from fla.ops.kda import chunk_kda

        # Convert the fused path's head-first Q/K/V and raw beta logits to
        # fla's sequence-major, sigmoided spelling so the model code stays
        # backend-agnostic.
        head_major_qkv = bool(kwargs.pop("head_major_qkv", False))
        if head_major_qkv:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        if kwargs.pop("use_beta_sigmoid_in_kernel", False):
            beta = beta.sigmoid()

        # Fused-only flags the triton path does not understand.
        kwargs.pop("disable_recompute", None)
        kwargs.pop("allow_neg_eigval", None)

        # fla takes the raw cu_seqlens tensor; it would otherwise silently
        # drop varlen boundaries through its **kwargs.
        packed_params = kwargs.pop("packed_params", None)
        if packed_params is not None and kwargs.get("cu_seqlens") is None:
            kwargs["cu_seqlens"] = packed_params.cu_seqlens

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
