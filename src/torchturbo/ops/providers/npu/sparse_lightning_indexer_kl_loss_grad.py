"""NPU providers for the CANN sparse_lightning_indexer_kl_loss_grad ops.

Training-only backward pair (regular ``cann_ops_transformer`` wheel):
``sparse_lightning_indexer_kl_loss_grad_metadata`` (AICPU core-split planner)
and ``sparse_lightning_indexer_kl_loss_grad`` (indexer KL loss backward).

The grad op takes the indexer branch ``q``/``k``/``w``, the forward TopK
``sparse_indices`` and the target distribution ``attn_softmax_l1_norm`` (the
main attention's softmax probabilities at the selected positions, averaged
over heads), and emits ``dq``/``dk``/``dw`` plus the indexer softmax ``y``.
It does NOT emit the loss value; the KL itself,

    L = sum_t sum_k p * (log p - log y),

is computed torch-side in ``SliKlLoss`` from the returned ``softmax_out``
(``xlogy`` keeps the p == 0 entries at zero). ``SliKlLoss`` returns the
token-SUM of the per-token KL; the provider divides by the token count so
the public op is a mean-KL whose backward scale matches a mean loss.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from .sparse_flash_mla import _max_seqlen
from ...registry import register_op

_KL_TOPK_CHOICES = (512, 1024, 2048, 4096, 8192)


def _debug_stats(name: str, t: torch.Tensor) -> str:
    t = t.detach().float()
    finite = torch.isfinite(t)
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"nan={int(t.isnan().sum())} inf={int(torch.isinf(t).sum())} "
        f"min={t[finite].min().item() if finite.any() else 'n/a'} "
        f"max={t[finite].max().item() if finite.any() else 'n/a'}"
    )


def _debug_check_finite(dq, dk, dw, softmax_out, target_probs) -> None:
    """TORCHTURBO_KL_DEBUG=1: raise with summary stats if the KL grad op
    produced non-finite outputs, or if target_probs was already bad."""
    problems = [
        _debug_stats(name, t)
        for name, t in (
            ("dq", dq),
            ("dk", dk),
            ("dw", dw),
            ("softmax_out", softmax_out),
        )
        if not torch.isfinite(t.float()).all()
    ]
    if problems:
        problems.append(_debug_stats("target_probs", target_probs))
        raise RuntimeError(
            "sparse_lightning_indexer_kl_loss_grad produced non-finite "
            "outputs:\n  " + "\n  ".join(problems)
        )
    if not torch.isfinite(target_probs.float()).all():
        raise RuntimeError(
            "target_probs (attn_softmax_l1_norm) already contains non-finite "
            "values before the op:\n  " + _debug_stats("target_probs", target_probs)
        )


def _load_ops():
    """Master cann_ops_transformer module only; no vendor probing.

    The KL ops must already be attributes of the master module (bundled
    wheel, or an entry-point side effect from an earlier import elsewhere).
    The dsv41 vendor subpackage is not imported here, so a wheel where the
    KL ops live only in the vendor package is not discovered:
    ``has_sli_kl_loss_grad`` then reports False and the patch fails loudly."""
    import cann_ops_transformer as master

    return master


def has_sli_kl_loss_grad() -> bool:
    try:
        ops = _load_ops()
        return callable(ops.sparse_lightning_indexer_kl_loss_grad) and callable(
            ops.sparse_lightning_indexer_kl_loss_grad_metadata
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


def _compact_kl_slots(
    sparse_indices: torch.Tensor, target_probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep valid slots first while preserving their order and target pairing."""
    width = sparse_indices.size(-1)
    slots = torch.arange(width, device=sparse_indices.device, dtype=torch.float32)
    # Unique FP32 keys keep the sort on AiCore and preserve within-group order.
    keys = slots + (sparse_indices < 0).to(torch.float32) * width
    permutation = keys.argsort(dim=-1)
    return (
        sparse_indices.gather(-1, permutation).contiguous(),
        target_probs.gather(-1, permutation).contiguous(),
    )


class SliKlLoss(torch.autograd.Function):
    """Autograd bridge for the "forward computes the backward" KL grad op.

    forward calls the grad kernel once, keeps dq/dk/dw, and returns the
    token-summed KL computed torch-side from p and the op's softmax_out.
    backward only scales the stored gradients (LILossTrain pattern).
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        weights,
        sparse_indices,
        target_probs,
        metadata,
        cu_seqlens_q,
        cu_seqlens_k,
        cmp_residual_k,
        cmp_ratio,
    ):
        # The fused kernel consumes a valid prefix; holes or leading -1 slots
        # can yield incorrect gradients or NaNs. Permute targets together with
        # indices only at this KL boundary; attention selection is unchanged.
        sparse_indices, target_probs = _compact_kl_slots(sparse_indices, target_probs)
        dq, dk, dw, softmax_out = _load_ops().sparse_lightning_indexer_kl_loss_grad(
            query,
            key,
            weights,
            sparse_indices,
            target_probs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q="TND",
            layout_k="TND",
            mask_mode=3,
            cmp_ratio=cmp_ratio,
        )
        ctx.save_for_backward(dq, dk, dw)
        if os.environ.get("TORCHTURBO_KL_DEBUG", "0") == "1":
            _debug_check_finite(dq, dk, dw, softmax_out, target_probs)
        log_y = softmax_out.float().clamp_min(1e-30).log()
        p = target_probs.float()
        # KL(p || y) per (token, head-group) entry, summed over the topk axis;
        # xlogy(p, p) keeps p == 0 slots at zero.
        kl = (torch.xlogy(p, p) - p * log_y).sum(-1)
        return kl.sum()

    @staticmethod
    def backward(ctx, grad_output):
        dq, dk, dw = ctx.saved_tensors
        return (
            dq * grad_output,
            dk * grad_output,
            dw * grad_output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@register_op(
    op_name="sli_kl_loss_grad_metadata",
    impl_name="npu_sli_kl_loss_grad_metadata",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSliKlLossGradMetadata(OpImpl):
    """AICPU core-split metadata for sparse_lightning_indexer_kl_loss_grad."""

    def available(self) -> bool:
        return has_sli_kl_loss_grad()

    def __call__(
        self,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        *,
        topk: int,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        cmp_residual_k: torch.Tensor | None = None,
        batch_size: int | None = None,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        **kwargs: Any,
    ) -> Any:
        if topk not in _KL_TOPK_CHOICES:
            raise ValueError(
                f"sparse_lightning_indexer_kl_loss_grad topk must be one of "
                f"{_KL_TOPK_CHOICES} on A2/A3, got {topk}."
            )
        return _load_ops().sparse_lightning_indexer_kl_loss_grad_metadata(
            num_heads_q,
            num_heads_k,
            head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=batch_size,
            max_seqlen_q=_max_seqlen(cu_seqlens_q),
            max_seqlen_k=_max_seqlen(cu_seqlens_k),
            topk=topk,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            **kwargs,
        )


@register_op(
    op_name="sli_kl_loss",
    impl_name="npu_sli_kl_loss",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSliKlLoss(OpImpl):
    """Indexer KL loss (mean over tokens) with fused NPU backward.

    ``target_probs`` is the main attention's softmax probability at each
    selected position (head-averaged, zero at invalid slots), shaped like
    ``sparse_indices``. Returns a scalar mean KL; the indexer q/k/w gradients
    flow through ``SliKlLoss``.
    """

    def available(self) -> bool:
        return has_sli_kl_loss_grad()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        sparse_indices: torch.Tensor,
        target_probs: torch.Tensor,
        *,
        metadata: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        cmp_residual_k: torch.Tensor | None = None,
        cmp_ratio: int = 1,
        **kwargs: Any,
    ) -> Any:
        loss_sum = SliKlLoss.apply(
            query,
            key,
            weights,
            sparse_indices,
            target_probs,
            metadata,
            cu_seqlens_q,
            cu_seqlens_k,
            cmp_residual_k,
            cmp_ratio,
        )
        num_tokens = max(query.size(0), 1)
        return loss_sum / num_tokens
