"""Transformer Engine mask permutation and expert chunk sorting."""

from __future__ import annotations

from importlib import metadata

import torch
from packaging.version import Version

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import MoEPermutation
from ...registry import register_op


def _supports_packed_mask_map() -> bool:
    # TE 2.6 replaced the expert-major map with packed per-token assignments.
    try:
        return Version(metadata.version("transformer-engine")) >= Version("2.6.0")
    except metadata.PackageNotFoundError:
        return False


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
    op_name="moe_permutation",
    impl_name="te_moe_mask_permutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoEMaskPermutation(OpImpl):
    def available(self) -> bool:
        if not _supports_packed_mask_map():
            return False
        try:
            from transformer_engine.pytorch.permutation import moe_permute  # noqa: F401
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
    ) -> MoEPermutation:
        """Translate TE's mask map into shared per-token assignment metadata."""
        from transformer_engine.pytorch.permutation import moe_permute

        num_experts = routing_map.shape[1]
        topk = expert_indices.shape[1]
        if num_out_tokens == 0:
            return MoEPermutation(
                tokens[:0],
                None if probs is None else probs.reshape(-1)[:0],
                torch.empty_like(expert_indices, dtype=torch.int32),
                expert_indices,
            )
        # The fused probability path treats zero weights as padding. Routing
        # membership depends on the mask, including selected zero-weight tokens.
        permuted_input, row_id_map = moe_permute(
            tokens,
            routing_map,
            num_out_tokens=num_out_tokens,
        )
        # TE packs destination rows first, expert IDs at offset num_experts,
        # and the routed count last. Slot order is opaque but paired.
        row_indices = row_id_map[:, :topk].contiguous()
        expert_indices = (
            row_id_map[:, num_experts : num_experts + topk].long().contiguous()
        )
        permuted_probs = None
        if probs is not None:
            selected_probs = probs.gather(1, expert_indices).flatten()
            permuted_probs = torch.empty_like(selected_probs).scatter(
                0,
                row_indices.flatten().long(),
                selected_probs,
            )
        return MoEPermutation(
            permuted_input,
            permuted_probs,
            row_indices,
            expert_indices,
        )


@register_op(
    op_name="moe_unpermutation",
    impl_name="te_moe_mask_unpermutation",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoEMaskUnpermutation(OpImpl):
    def available(self) -> bool:
        if not _supports_packed_mask_map():
            return False
        try:
            from transformer_engine.pytorch.permutation import moe_unpermute  # noqa: F401
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    def __call__(
        self,
        permutation: MoEPermutation,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a compact TE mask map from provider-independent assignments."""
        from transformer_engine.pytorch.permutation import moe_unpermute

        permuted_tokens = permutation.tokens
        num_tokens, topk = permutation.row_indices.shape
        if permutation.row_indices.numel() == 0:
            # Keep the inverse-map dependency so checkpoint recomputes before EP backward.
            restored = permuted_tokens.index_select(
                0, permutation.row_indices.flatten().long()
            )
            output = restored.view(num_tokens, topk, permuted_tokens.shape[-1]).sum(1)
            if probs is not None:
                output = output + probs.sum() * 0
            return output.view(restore_shape).to(permuted_tokens.dtype)
        # Treat the selected slots as the compact map's experts. Dense routing
        # probabilities are gathered into the same slot order before the call.
        slots = torch.arange(topk, dtype=torch.int32, device=permuted_tokens.device)
        counts = torch.full(
            (num_tokens, 1), topk, dtype=torch.int32, device=permuted_tokens.device
        )
        row_id_map = torch.cat(
            (permutation.row_indices, slots.expand(num_tokens, -1), counts), dim=1
        )
        weights = None if probs is None else probs.gather(1, permutation.expert_indices)
        return moe_unpermute(
            _input_for_probs(permuted_tokens, weights),
            row_id_map,
            merging_probs=weights,
            restore_shape=restore_shape,
        )


@register_op(
    op_name="moe_sort_chunks",
    impl_name="te_moe_sort_chunks",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineMoESortChunks(OpImpl):
    def available(self) -> bool:
        try:
            from transformer_engine.pytorch.permutation import (  # noqa: F401
                moe_sort_chunks_by_index,
                moe_sort_chunks_by_index_with_probs,
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
            moe_sort_chunks_by_index,
            moe_sort_chunks_by_index_with_probs,
        )

        if probs is None:
            return moe_sort_chunks_by_index(input, split_sizes, sorted_idxs), None
        return moe_sort_chunks_by_index_with_probs(
            _input_for_probs(input, probs),
            probs,
            split_sizes,
            sorted_idxs,
        )
