"""Transformer Engine compact permutation and expert chunk sorting."""

from __future__ import annotations

from importlib import metadata

import torch
from packaging.version import Version

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import MoEPermutation
from ...registry import register_op
from ..native.moe_permutation import TorchMoEMaskPermutation


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


def _compact_mask_map(row_indices: torch.Tensor) -> torch.Tensor:
    """Describe selected slots in TE's packed mask-map format."""
    num_tokens, topk = row_indices.shape
    slots = torch.arange(topk, dtype=torch.int32, device=row_indices.device)
    counts = torch.full(
        (num_tokens, 1), topk, dtype=torch.int32, device=row_indices.device,
    )
    return torch.cat((row_indices, slots.expand(num_tokens, -1), counts), dim=1)


class _TokenPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, routing_map, expert_indices):
        if routing_map.dtype == torch.bool and tokens.shape[1] > 0:
            from transformer_engine.pytorch.permutation import moe_permute

            permuted, row_map = moe_permute(
                tokens, routing_map, num_out_tokens=expert_indices.numel(),
            )
            topk = expert_indices.shape[1]
            num_experts = routing_map.shape[1]
            # TE pairs destination rows with expert IDs in its own slot order.
            row_indices = row_map[:, :topk].clone(memory_format=torch.contiguous_format)
            expert_indices = row_map[:, num_experts : num_experts + topk].long().contiguous()
        else:
            # TE's index kernel shares a workspace across streams. PyTorch's
            # compact sort also supports zero-width features and unaligned views.
            packed = TorchMoEMaskPermutation()(tokens, expert_indices)
            permuted, row_indices = packed.tokens, packed.row_indices
        ctx.save_for_backward(row_indices)
        ctx.restore_shape = tokens.shape
        return permuted, row_indices, expert_indices

    @staticmethod
    def backward(ctx, grad_output, _grad_rows, _grad_experts):
        from transformer_engine.pytorch.permutation import moe_unpermute

        (row_indices,) = ctx.saved_tensors
        if grad_output.numel() == 0:
            return grad_output.reshape(ctx.restore_shape), None, None
        # The packed-mask kernel sums BF16/FP16 contributions in FP32.
        grad_input = moe_unpermute(
            grad_output.contiguous(), _compact_mask_map(row_indices),
            merging_probs=None, restore_shape=ctx.restore_shape,
        )
        return grad_input, None, None


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
            from transformer_engine.pytorch.permutation import (  # noqa: F401
                moe_permute, moe_unpermute,
            )
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    def __call__(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> MoEPermutation:
        """Pack dense masks with TE and compact slots with a stream-local sort."""
        if routing_map is None:
            routing_map = expert_indices
        elif expert_indices is None:
            expert_indices = routing_map
        if expert_indices.numel() == 0:
            return MoEPermutation(
                tokens[:0],
                None if probs is None else probs.reshape(-1)[:0],
                torch.empty_like(expert_indices, dtype=torch.int32),
                expert_indices,
            )
        permuted_input, row_indices, expert_indices = _TokenPermutation.apply(
            tokens, routing_map, expert_indices,
        )
        permuted_probs = None
        if probs is not None:
            if routing_map.dtype == torch.bool:
                probs = probs.gather(1, expert_indices)
            selected_probs = probs.flatten()
            permuted_probs = torch.empty_like(selected_probs).scatter(
                0, row_indices.flatten().long(), selected_probs,
            )
        return MoEPermutation(
            permuted_input, permuted_probs, row_indices, expert_indices,
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
        row_indices = permutation.row_indices
        num_tokens, topk = row_indices.shape
        if permuted_tokens.numel() == 0:
            # Keep the inverse-map dependency so checkpoint recomputes before EP backward.
            restored = permuted_tokens.index_select(
                0, row_indices.flatten().long()
            )
            output = restored.view(num_tokens, topk, permuted_tokens.shape[-1]).sum(1)
            if probs is not None:
                output = output + probs.sum() * 0
            return output.view(restore_shape).to(permuted_tokens.dtype)
        # Treat the selected slots as the compact map's experts. Dense routing
        # probabilities are gathered into the same slot order before the call.
        row_id_map = _compact_mask_map(row_indices)
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
