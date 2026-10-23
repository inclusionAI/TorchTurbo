"""Transformer Engine mask permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _input_for_probs(tokens: torch.Tensor, probs: torch.Tensor | None) -> torch.Tensor:
    # TE probability backward depends on the activation grad flag.
    # A detached leaf enables router-only training without changing caller tensors.
    if (
        torch.is_grad_enabled()
        and probs is not None
        and probs.requires_grad
        and not tokens.requires_grad
    ):
        return tokens.detach().requires_grad_()
    return tokens


@register_op(
    op_name="moe_fused_permutation",
    impl_name="te_moe_mask_permutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoEMaskPermutation(OpImpl):
    def available(self) -> bool:
        try:
            from transformer_engine.pytorch.permutation import (  # noqa: F401
                moe_permute, moe_permute_with_probs,
            )
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    def __call__(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Group tokens by expert and return TE's opaque inverse mapping."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_permute_with_probs

        if num_out_tokens is None:
            raise ValueError("num_out_tokens is required for fused mask permutation.")
        if probs is None:
            permuted_input, sorted_indices = moe_permute(
                tokens, routing_map, num_out_tokens=num_out_tokens,
            )
            return permuted_input, None, sorted_indices
        permuted_input, permuted_probs, sorted_indices = moe_permute_with_probs(
            _input_for_probs(tokens, probs), probs, routing_map,
            num_out_tokens=num_out_tokens,
        )
        if tokens.numel() == 0:
            # Older TE versions create empty probabilities with the default dtype.
            permuted_probs = probs.reshape(-1)
        return permuted_input, permuted_probs, sorted_indices


@register_op(
    op_name="moe_fused_unpermutation",
    impl_name="te_moe_mask_unpermutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoEMaskUnpermutation(OpImpl):
    def available(self) -> bool:
        try:
            from transformer_engine.pytorch.permutation import moe_unpermute  # noqa: F401
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    def __call__(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
        routing_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore source tokens using a TE permutation mapping."""
        from transformer_engine.pytorch.permutation import moe_unpermute

        return moe_unpermute(
            _input_for_probs(permuted_tokens, probs), sorted_indices,
            merging_probs=probs, restore_shape=restore_shape,
        )


@register_op(
    op_name="moe_fused_sort_chunks",
    impl_name="te_moe_sort_chunks",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoESortChunks(OpImpl):
    def available(self) -> bool:
        try:
            from transformer_engine.pytorch.permutation import (  # noqa: F401
                moe_sort_chunks_by_index, moe_sort_chunks_by_index_with_probs,
            )
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    def __call__(
        self,
        input: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reorder expert chunks using metadata on the input device."""
        from transformer_engine.pytorch.permutation import (
            moe_sort_chunks_by_index, moe_sort_chunks_by_index_with_probs,
        )

        if split_sizes.device != input.device or sorted_idxs.device != input.device:
            raise ValueError(
                "Fused chunk sorting requires split_sizes and sorted_idxs "
                "on the input device."
            )
        if probs is None:
            return moe_sort_chunks_by_index(input, split_sizes, sorted_idxs), None
        return moe_sort_chunks_by_index_with_probs(
            _input_for_probs(input, probs), probs, split_sizes, sorted_idxs,
        )
