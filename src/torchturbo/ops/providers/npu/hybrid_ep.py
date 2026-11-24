"""HybridEP transport with Ascend token permutation kernels."""

from __future__ import annotations

from torchturbo.runtime import TurboDevice

from ...base import OpImplKind
from ...registry import register_op
from ..native.hybrid_ep import TorchHybridEPCombine, TorchHybridEPDispatch
from .moe_permutation import NpuMoEMaskPermutation, NpuMoEMaskUnpermutation, NpuMoESortChunks


def _permutation_kernels_available() -> bool:
    try:
        from torch_npu import npu_moe_token_permute, npu_moe_token_unpermute  # noqa: F401
    except ImportError:
        return False
    return True


@register_op(
    op_name="hybrid_ep_dispatch",
    impl_name="npu_hybrid_ep_dispatch",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuHybridEPDispatch(TorchHybridEPDispatch):
    _permute = NpuMoEMaskPermutation()
    _sort_chunks = NpuMoESortChunks()
    _chunk_metadata_on_device = True

    def available(self) -> bool:
        return _permutation_kernels_available()


@register_op(
    op_name="hybrid_ep_combine",
    impl_name="npu_hybrid_ep_combine",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuHybridEPCombine(TorchHybridEPCombine):
    _unpermute = NpuMoEMaskUnpermutation()
    _sort_chunks = NpuMoESortChunks()

    def available(self) -> bool:
        return _permutation_kernels_available()
