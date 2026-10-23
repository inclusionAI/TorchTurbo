"""NPU (AscendC) provider for the chunk_kda op.

Full AscendC training mode: both the fused ``ChunkKdaFwd`` forward (with
external L2 normalization via the fla-npu triton kernels) and the fused
``ChunkKdaBwd`` backward from the ``fla_npu`` package (vendored in
``ops/providers/npu/_fla_npu_kda``).

Opt-in by design: ``available()`` stays False even when fla_npu is
importable, so the default `auto` selection on NPU keeps the battle-tested
fla-org triton provider.  ``TORCHTURBO_KDA_ASCENDC=1`` opts in; an explicit
``impl='ascendc_chunk_kda'`` or an ``ops.overrides`` entry selects it
directly regardless of the env.
"""

from __future__ import annotations

import os

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="chunk_kda",
    impl_name="ascendc_chunk_kda",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCChunkKDA(OpImpl):
    def available(self) -> bool:
        if os.environ.get("TORCHTURBO_KDA_ASCENDC", "0").strip() not in (
            "1", "true", "True",
        ):
            return False
        try:
            import fla_npu  # noqa: F401
            # The full-AscendC path needs BOTH fused ops; older wheels
            # (fwd-only) fail here and fall back to the native fla triton
            # provider under `auto`.
            from fla_npu.ops.ascendc import npu_chunk_kda_bwd  # noqa: F401
            from torchturbo.ops.providers.npu._fla_npu_kda import (  # noqa: F401
                ascendc_chunk_kda,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "ascendc_chunk_kda unavailable (explicit selection will "
                "raise; auto falls back to fla triton): %r\n%s",
                exc,
                traceback.format_exc(),
            )
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
        from torchturbo.ops.providers.npu._fla_npu_kda import ascendc_chunk_kda

        return ascendc_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
