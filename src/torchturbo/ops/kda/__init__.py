"""Unified entry for the KDA chunk op.

Routes ``chunk_kda`` through the ops registry (fla Triton preferred,
pure-PyTorch naive as automatic fallback) when the TorchTurbo ops
dispatcher is configured, and resolves the same choice directly when it is
not — so the model files stay runnable outside the engine (stock-HF
parity checks, standalone forwards).

``impl`` mirrors the dispatcher's per-call override:

- ``None`` (default): auto — fla if importable, else naive.
- ``"naive_chunk_kda"``: force the pure-PyTorch implementation.
- ``"fla_chunk_kda"``: force fla (ImportError propagates when missing).
"""

from __future__ import annotations

import logging

from .chunk_kda_naive import chunk_kda_naive

logger = logging.getLogger(__name__)

_standalone_fallback_warned = False


def chunk_kda(*args, impl: str | None = None, **kwargs):
    global _standalone_fallback_warned
    dispatcher = None
    try:
        from torchturbo.ops.ops import get_ops

        dispatcher = get_ops()
    except RuntimeError:
        dispatcher = None
    if dispatcher is not None:
        return dispatcher.chunk_kda(*args, impl=impl, **kwargs)

    if impl != "naive_chunk_kda":
        try:
            from fla.ops.kda import chunk_kda as fla_chunk_kda
        except ImportError:
            if impl == "fla_chunk_kda":
                raise
        else:
            return fla_chunk_kda(*args, **kwargs)
        if not _standalone_fallback_warned:
            logger.warning(
                "fla.ops.kda is not importable (triton backend missing or "
                "fla too old): chunk_kda falls back to the pure-PyTorch "
                "naive implementation. Install flash-linear-attention>=0.6.0 "
                "with a working triton backend for the fused path."
            )
            _standalone_fallback_warned = True

    kwargs.setdefault("transpose_state_layout", True)
    return chunk_kda_naive(*args, **kwargs)


__all__ = ["chunk_kda", "chunk_kda_naive"]
