"""Optional Triton probability packing with the existing NPU token transport."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImplKind
from ...registry import register_op
from .moe_permutation import NpuMoEMaskPermutation, NpuMoESortChunks


class _TritonProbabilityPacking:
    def available(self) -> bool:
        if not super().available():
            return False
        try:
            from triton.backends.ascend.compiler import AscendBackend  # noqa: F401

            from ._kernels.moe_probability import pack_probabilities  # noqa: F401
        except ImportError:
            return False
        return True

    def _pack_probabilities(
        self, probs: torch.Tensor, inverse: torch.Tensor,
    ) -> torch.Tensor:
        if probs.ndim != 1:
            # Chunk probabilities can contain multiple values per packed row.
            return super()._pack_probabilities(probs, inverse)
        from ._kernels.moe_probability import pack_probabilities

        return pack_probabilities(probs, inverse)


@register_op(
    op_name="moe_permutation",
    impl_name="npu_triton_moe_mask_permutation",
    kind=OpImplKind.CUSTOM,
    priority=90,
    devices=(TurboDevice.NPU,),
)
class NpuTritonMoEMaskPermutation(_TritonProbabilityPacking, NpuMoEMaskPermutation):
    """Use Triton for probability packing and torch_npu for token assignments."""


@register_op(
    op_name="moe_sort_chunks",
    impl_name="npu_triton_moe_sort_chunks",
    kind=OpImplKind.CUSTOM,
    priority=90,
    devices=(TurboDevice.NPU,),
)
class NpuTritonMoESortChunks(_TritonProbabilityPacking, NpuMoESortChunks):
    """Use Triton for chunk probabilities and torch_npu for token movement."""
