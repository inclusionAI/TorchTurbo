"""Native (fla-org) provider for the chunk_kda op.

Direct delegation to ``fla.ops.kda.chunk_kda`` -- the triton-ascend path that
production used before the AscendC integration. ``devices=()`` makes it a
candidate everywhere; on NPU the AscendC hybrid provider (priority=100) wins
under `auto`, and this provider serves as the correctness baseline and the
rollback path.
"""

from __future__ import annotations

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="chunk_kda",
    impl_name="fla_chunk_kda",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaChunkKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        **kwargs,
    ):
        # fla-org's chunk_kda is an autograd function returning (o, final_state).
        from fla.ops.kda import chunk_kda

        head_major_qkv = bool(kwargs.pop("head_major_qkv", False))
        if head_major_qkv:
            # The AscendC provider consumes the fused convolution's head-first
            # Q/K/V directly; fla's triton kernels are sequence-major, so
            # convert here to keep the model code backend-agnostic.
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
