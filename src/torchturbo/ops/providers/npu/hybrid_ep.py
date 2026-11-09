"""NPU compact source packing with the shared HybridEP transport contract."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImplKind
from ...moe import MoEPermutation
from ...registry import register_op
from ..native.hybrid_ep import TorchHybridEPDispatch
from .moe_permutation import _ProbabilityPacking, _TokenPermutation


@register_op(
    op_name="hybrid_ep_dispatch",
    impl_name="npu_hybrid_ep_dispatch",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuHybridEPDispatch(_ProbabilityPacking, TorchHybridEPDispatch):
    def available(self) -> bool:
        try:
            from torch_npu import npu_moe_token_permute  # noqa: F401
        except ImportError:
            return False
        return True

    def _source_permutation(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor,
        probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int,
    ) -> MoEPermutation:
        """Pack compact routes without allocating token-by-expert tensors."""
        if routing_map.dtype == torch.bool:
            return super()._source_permutation(
                hidden_states, routing_map, probs, expert_indices, num_experts,
            )
        if expert_indices.numel() == 0:
            return MoEPermutation(
                hidden_states[:0], probs.reshape(-1),
                torch.empty_like(expert_indices, dtype=torch.int32), expert_indices,
            )
        output, inverse = _TokenPermutation.apply(hidden_states, expert_indices)
        packed_probs = self._pack_probabilities(probs.flatten(), inverse)
        return MoEPermutation(
            output, packed_probs, inverse.view_as(expert_indices), expert_indices,
        )
