"""NPU token permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_fused_permutation",
    impl_name="npu_moe_mask_permutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoEMaskPermutation(OpImpl):
    def available(self) -> bool:
        try:
            from torch_npu import npu_moe_token_permute  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor],
    ]:
        """Group fixed-top-k tokens and retain expert indices for probability gather."""
        from torch_npu import npu_moe_token_permute

        if num_out_tokens is None:
            raise ValueError("num_out_tokens is required for fused mask permutation.")
        num_tokens = tokens.shape[0]
        if num_tokens == 0:
            inverse = routing_map.new_empty((0,), dtype=torch.int32)
            if expert_indices is None:
                expert_indices = routing_map.new_empty((0, 0), dtype=torch.long)
            permuted_probs = None if probs is None else probs.reshape(-1)
            return tokens, permuted_probs, (inverse, expert_indices)

        if expert_indices is None:
            topk = num_out_tokens // num_tokens
            expert_indices = torch.nonzero(routing_map, as_tuple=True)[1].view(num_tokens, topk)
        output, inverse = npu_moe_token_permute(tokens, expert_indices)
        permuted_probs = None
        if probs is not None:
            selected_probs = probs.gather(1, expert_indices).flatten()
            # Each token/top-k slot maps to its row in the packed tensor.
            permuted_probs = torch.empty_like(selected_probs).scatter(
                0, inverse.long(), selected_probs,
            )
        return output, permuted_probs, (inverse, expert_indices)


@register_op(
    op_name="moe_fused_unpermutation",
    impl_name="npu_moe_mask_unpermutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoEMaskUnpermutation(OpImpl):
    def available(self) -> bool:
        try:
            from torch_npu import npu_moe_token_unpermute  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: tuple[torch.Tensor, torch.Tensor],
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
        routing_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weight and merge expert outputs using cached token/top-k indices."""
        from torch_npu import npu_moe_token_unpermute

        inverse, expert_indices = sorted_indices
        if restore_shape[0] == 0:
            output = permuted_tokens.view(restore_shape)
            if probs is not None:
                output = output * probs.sum(dim=1, keepdim=True).to(output.dtype)
            return output

        weights = None
        if probs is not None:
            weights = probs.gather(1, expert_indices).to(permuted_tokens.dtype)
        elif expert_indices.shape[1] > 1:
            # The index-based kernel infers top-k from the probability shape.
            weights = permuted_tokens.new_ones(expert_indices.shape)
        return npu_moe_token_unpermute(permuted_tokens, inverse, probs=weights)


@register_op(
    op_name="moe_fused_sort_chunks",
    impl_name="npu_moe_sort_chunks",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMoESortChunks(OpImpl):
    def available(self) -> bool:
        try:
            from torch_npu import npu_moe_token_permute  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        input: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reorder chunks with device metadata and the NPU permutation kernel."""
        from torch_npu import npu_moe_token_permute

        if split_sizes.device != input.device or sorted_idxs.device != input.device:
            raise ValueError(
                "Fused chunk sorting requires split_sizes and sorted_idxs on the input device."
            )
        if input.shape[0] == 0:
            return input, probs
        # Assign each source chunk its destination position without copying counts to CPU.
        chunk_positions = torch.arange(
            sorted_idxs.numel(), device=input.device, dtype=torch.int32,
        )
        destinations = torch.empty_like(chunk_positions).scatter_(
            0, sorted_idxs, chunk_positions,
        )
        indices = torch.repeat_interleave(
            destinations, split_sizes, output_size=input.shape[0],
        ).unsqueeze(1)
        output, inverse = npu_moe_token_permute(input, indices)
        permuted_probs = None
        if probs is not None:
            permuted_probs = torch.empty_like(probs).scatter(0, inverse.long(), probs)
        return output, permuted_probs
