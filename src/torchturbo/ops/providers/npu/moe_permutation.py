"""NPU token permutation and expert chunk sorting."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import MoEPermutation
from ...registry import register_op


class _TokenPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, expert_indices):
        from torch_npu import npu_moe_token_permute

        output, inverse = npu_moe_token_permute(tokens, expert_indices)
        ctx.save_for_backward(inverse)
        ctx.assignment_shape = expert_indices.shape
        return output, inverse

    @staticmethod
    def backward(ctx, grad_output, grad_inverse):
        from torch_npu import npu_moe_token_unpermute

        (inverse,) = ctx.saved_tensors
        weights = None
        if ctx.assignment_shape[1] > 1:
            weights = grad_output.new_ones(ctx.assignment_shape)
        grad_tokens = npu_moe_token_unpermute(grad_output, inverse, probs=weights)
        return grad_tokens, None


class _UnweightedUnpermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, inverse, expert_indices):
        from torch_npu import npu_moe_token_unpermute

        # This linear merge needs only assignment metadata for its backward.
        ctx.save_for_backward(expert_indices)
        weights = None
        if expert_indices.shape[1] > 1:
            weights = tokens.new_ones(expert_indices.shape)
        return npu_moe_token_unpermute(tokens, inverse, probs=weights)

    @staticmethod
    def backward(ctx, grad_output):
        (expert_indices,) = ctx.saved_tensors
        grad_tokens, _ = _TokenPermutation.apply(grad_output, expert_indices)
        return grad_tokens, None, None


@register_op(
    op_name="moe_permutation",
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
    ) -> MoEPermutation:
        """Group fixed-top-k tokens and retain expert indices for probability gather."""
        if num_out_tokens == 0:
            inverse = torch.empty_like(expert_indices, dtype=torch.int32)
            permuted_probs = None if probs is None else probs.reshape(-1)[:0]
            return MoEPermutation(tokens[:0], permuted_probs, inverse, expert_indices)
        output, inverse = _TokenPermutation.apply(tokens, expert_indices)
        permuted_probs = None
        if probs is not None:
            selected_probs = probs.gather(1, expert_indices).flatten()
            # Each token/top-k slot maps to its row in the packed tensor.
            permuted_probs = torch.empty_like(selected_probs).index_put(
                (inverse.long(),), selected_probs,
            )
        return MoEPermutation(
            output,
            permuted_probs,
            inverse.view_as(expert_indices),
            expert_indices,
        )


@register_op(
    op_name="moe_unpermutation",
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
        permutation: MoEPermutation,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weight and merge expert outputs using cached token/top-k indices."""
        from torch_npu import npu_moe_token_unpermute

        permuted_tokens = permutation.tokens
        inverse = permutation.row_indices.flatten()
        expert_indices = permutation.expert_indices
        if inverse.numel() == 0:
            # Keep the inverse-map dependency so checkpoint recomputes before EP backward.
            restored = permuted_tokens.index_select(0, inverse.long())
            output = restored.view(
                *permutation.row_indices.shape, permuted_tokens.shape[-1]
            ).sum(dim=1)
            if probs is not None:
                output = output + probs.sum() * 0
            return output.view(restore_shape).to(permuted_tokens.dtype)

        if probs is None:
            return _UnweightedUnpermutation.apply(
                permuted_tokens, inverse, expert_indices,
            )

        input_dtype = permuted_tokens.dtype
        weights = probs.gather(1, expert_indices)
        # The kernel accepts FP32 weights with BF16/FP16 tokens and accumulates
        # in FP32 without materializing a promoted copy of the packed features.
        return npu_moe_token_unpermute(permuted_tokens, inverse, probs=weights).to(
            input_dtype
        )


@register_op(
    op_name="moe_sort_chunks",
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
        if input.shape[0] == 0:
            return input, probs
        # Assign each source chunk its destination position without copying counts to CPU.
        chunk_positions = torch.arange(
            sorted_idxs.numel(),
            device=input.device,
            dtype=torch.int32,
        )
        destinations = torch.empty_like(chunk_positions).scatter_(
            0,
            sorted_idxs,
            chunk_positions,
        )
        indices = torch.repeat_interleave(
            destinations,
            split_sizes,
            output_size=input.shape[0],
        ).unsqueeze(1)
        output, inverse = _TokenPermutation.apply(input, indices)
        permuted_probs = None
        if probs is not None:
            permuted_probs = torch.empty_like(probs).index_put((inverse.long(),), probs)
        return output, permuted_probs
