"""KDA (Kimi Delta Attention) chunk op providers.

Two interchangeable implementations of ``chunk_kda`` (the training path of
KDA layers in kimi_k3 and bailing_hybrid):

- ``fla_chunk_kda`` (priority 10): flash-linear-attention's Triton kernel;
  the preferred implementation whenever ``fla.ops.kda`` is importable.
- ``naive_chunk_kda`` (priority 0): the pure-PyTorch implementation in
  ``torchturbo.ops.kda.chunk_kda_naive``; always available, so it is the
  automatic fallback on stacks without a working Triton backend.

With the AUTO policy the dispatcher tries ``fla_chunk_kda`` first and falls
back to ``naive_chunk_kda`` when fla cannot be imported. Stacks where the
kernel imports but deadlocks at runtime (old triton-ascend aicore timeout)
still need an explicit override (per-op override or the model's
``kda_implementation: naive`` config, which the models map to
``impl="naive_chunk_kda"``).
"""

from __future__ import annotations

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="chunk_kda",
    impl_name="fla_chunk_kda",
    kind=OpImplKind.NATIVE,
    priority=10,
    devices=(),
)
class FlaChunkKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(self, *args, **kwargs):
        from fla.ops.kda import chunk_kda

        return chunk_kda(*args, **kwargs)


@register_op(
    op_name="chunk_kda",
    impl_name="naive_chunk_kda",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class NaiveChunkKDA(OpImpl):
    def available(self) -> bool:
        return True

    def __call__(self, *args, **kwargs):
        from ...kda.chunk_kda_naive import chunk_kda_naive

        # Match the layout the kimi_k3 modeling forward has always used for
        # the naive path; fla's kernel manages layouts internally and never
        # receives this flag.
        kwargs.setdefault("transpose_state_layout", True)
        return chunk_kda_naive(*args, **kwargs)
