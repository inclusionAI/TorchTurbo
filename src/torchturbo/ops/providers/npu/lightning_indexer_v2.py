"""NPU providers for the CANN lightning-indexer-v2 ops.

The per-op li wheel hangs its torch interface module on the master
``cann_ops_transformer`` module as ``lightning_indexer_v2`` through entry
points. The fused ``lightning_indexer`` wrapper lives there: the candidate
two-level TopK is an optional parameter pair (routed to
aclnnLightningIndexerV3, four outputs), while omitting them keeps the legacy
two-output contract (aclnnLightningIndexerV2). The master's bare
``lightning_indexer`` attribute is the torch.ops packet with the legacy
schema and does not take the candidate parameters.

arch22 (A2/A3) constraints, enforced here:
- ``metadata`` / ``seqused_*`` / ``output_idx_offset`` are never passed and
  ``max_seqlen_q`` stays -1 (the host rejects them on arch22; the CANN < 9.2
  ``aclnnLightningIndexerV2Metadata`` symbol is not available either).
- ``cmp_residual_k`` is nulled at ``cmp_ratio == 1``: the op rejects a
  residual-counts tensor when nothing is compressed (the counts are
  identically zero there), while the model-side packing helper materializes
  the tensor unconditionally.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op

def _candidate_capable(mod) -> bool:
    """True when the module's lightning_indexer wrapper takes the fused
    candidate parameters (the per-op li wheel generation)."""
    entry = getattr(mod, "lightning_indexer", None)
    if not callable(entry):
        return False
    try:
        params = inspect.signature(entry).parameters
    except (TypeError, ValueError):  # signature-less callable
        return False
    return "candidate_topk_blocks" in params and "candidate_block_size" in params


@cache
def _load_ops():
    """Load and validate the fused wrapper once; failed loads remain retriable."""
    import cann_ops_transformer as master

    mod = getattr(master, "lightning_indexer_v2", None)
    if mod is not None and _candidate_capable(mod):
        return mod
    raise ImportError(
        "lightning_indexer_v2 fused ops not found: install the "
        "cann_ops_transformer_dsv41_li wheel, whose torch interface module "
        "registers as cann_ops_transformer.lightning_indexer_v2 through "
        "entry points"
    )


def has_lightning_indexer_v2() -> bool:
    try:
        _load_ops()
        return True
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


@register_op(
    op_name="lightning_indexer_v2",
    impl_name="npu_lightning_indexer_v2",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuLightningIndexerV2(OpImpl):
    """Lightning indexer v2 forward (TopK index selection).

    Returns ``(sparse_indices, candidate_topk_indices_or_None)``. In
    candidate mode the fused wrapper also emits the level-1 candidate block
    pool for ``sparse_lightning_indexer``.
    """

    def available(self) -> bool:
        return has_lightning_indexer_v2()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        topk: int,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        cmp_residual_k: torch.Tensor | None = None,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        candidate_mode: bool = False,
        candidate_topk_blocks: int = -1,
        candidate_block_size: int = 8,
        **kwargs: Any,
    ) -> Any:
        ops = _load_ops()
        if cmp_ratio == 1:
            # Op contract: cmp_residual_k must be null at cmp_ratio 1.
            cmp_residual_k = None
        if candidate_mode:
            # The fused wrapper returns four outputs in candidate mode (the
            # fourth, candidate_block_length, is a reserved empty tensor).
            indices, _values, candidate_topk_indices, _block_length = (
                ops.lightning_indexer(
                    query,
                    key,
                    weights,
                    topk,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    cmp_residual_k=cmp_residual_k,
                    max_seqlen_q=-1,
                    layout_q=layout_q,
                    layout_k=layout_k,
                    mask_mode=mask_mode,
                    cmp_ratio=cmp_ratio,
                    candidate_topk_blocks=candidate_topk_blocks,
                    candidate_block_size=candidate_block_size,
                    **kwargs,
                )
            )
            return indices, candidate_topk_indices
        indices, _values = ops.lightning_indexer(
            query,
            key,
            weights,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            max_seqlen_q=-1,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=0,
            **kwargs,
        )
        return indices, None
