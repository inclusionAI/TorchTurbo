"""NPU provider for the CANN sparse_lightning_indexer op.

Candidate-consumer variant of the lightning indexer: it eats the
``candidate_topk_indices`` emitted by the fused ``lightning_indexer``
wrapper's candidate mode and runs the two-level TopK selection (in-pool
first, reachable out-of-pool scores demoted to -1e30). The op registers on
the master ``cann_ops_transformer`` module; arch22 only.

The ``candidate_block_size`` / ``cmp_ratio`` contract with the source-side
``lightning_indexer`` candidate-mode call is NOT checked by the op; the
caller (``deepseek_v41/npu.py``) passes both from a single config source.
Like the source-side op, a non-null ``cmp_residual_k`` at ``cmp_ratio == 1``
is rejected (same tiling rule); the provider nulls it there.

Availability goes beyond the callable check: the bundled wrapper dlopens
``aclnnSparseLightningIndexer*`` from the CANN ``libopapi*.so`` libraries,
and per-op vendor packages export the same entry points from their
``libcust_opapi.so``. Split-vendor images that leave this op out of every
package pass the callable check yet abort on the first call, so
``has_sparse_lightning_indexer`` probes those libraries for the aclnn entry
points.
"""

from __future__ import annotations

import ctypes
import glob
from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _load_ops():
    """Master cann_ops_transformer module carrying the op registration."""
    import cann_ops_transformer as master

    return master


_BUNDLED_LIBS = ("libopapi.so", "libopapi_transformer.so")
_VENDOR_LIB_GLOB = "/usr/local/lib/cann_ops/vendors/*/op_api/lib/libcust_opapi.so"
_KERNEL_SYMBOLS = (
    "aclnnSparseLightningIndexer",
    "aclnnSparseLightningIndexerGetWorkspaceSize",
)


def _kernel_available() -> bool:
    """True when a reachable library exports the op's aclnn entry points.

    The bundled wrapper resolves the kernel through the CANN ``libopapi*.so``
    sonames at first call, per-op vendor packages export it from their
    ``libcust_opapi.so``. Missing symbols in all of them mean the first call
    would abort with a runtime error deep into training startup. The wrapper
    object on the master module cannot tell the two apart (a torch.ops
    registration repackages it either way), so both carriers are probed.
    """
    candidates = list(_BUNDLED_LIBS) + sorted(glob.glob(_VENDOR_LIB_GLOB))
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            for symbol in _KERNEL_SYMBOLS:
                getattr(lib, symbol)
        except (OSError, AttributeError):
            continue
        return True
    return False


def has_sparse_lightning_indexer() -> bool:
    try:
        ops = _load_ops()
        fn = ops.sparse_lightning_indexer
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False
    return callable(fn) and _kernel_available()


@register_op(
    op_name="sparse_lightning_indexer",
    impl_name="npu_sparse_lightning_indexer",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSparseLightningIndexer(OpImpl):
    """Candidate-consumer lightning indexer forward.

    Returns the sparse indices tensor (segment-local, int32, -1 padded).
    """

    def available(self) -> bool:
        return has_sparse_lightning_indexer()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        topk: int,
        candidate_topk_indices: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        cmp_residual_k: torch.Tensor | None = None,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        candidate_block_size: int = 8,
        **kwargs: Any,
    ) -> Any:
        if cmp_ratio == 1:
            # Op contract (same tiling rule as lightning_indexer_v2):
            # cmp_residual_k must be null at cmp_ratio 1.
            cmp_residual_k = None
        indices, _values = _load_ops().sparse_lightning_indexer(
            query,
            key,
            weights,
            topk,
            candidate_topk_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            max_seqlen_q=-1,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            candidate_block_size=candidate_block_size,
            **kwargs,
        )
        return indices
