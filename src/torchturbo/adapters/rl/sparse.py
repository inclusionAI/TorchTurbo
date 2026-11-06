"""Loss-mask row selection for packed, text-only RL output heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .packed import gather_logprobs_entropy


@dataclass
class SparseLogitsPlan:
    """Mapping from shifted packed targets to contiguous SP-local predictors."""

    rows: torch.Tensor
    model_rows: torch.Tensor
    labels: torch.Tensor
    valid_rows: torch.Tensor
    mask: torch.Tensor
    old_logp: torch.Tensor
    local_length: int
    cu_seqlens: torch.Tensor | None
    sp_pad_size: int
    sp_info: dict[str, Any]


def build_sparse_plan(ctx: dict[str, Any], sp_info: dict[str, Any]) -> SparseLogitsPlan:
    """Select predictors for a shifted packed ``ppo_loss_mask``.

    Args:
        ctx: Context returned by ``prepare_packed_mb_inputs``. Its original
            micro-batch must contain ``ppo_loss_mask`` and finite ``old_logp``
            of shape ``[sum(sequence_lengths - 1)]``.
        sp_info: Contiguous Ulysses shard information, or an SP1 layout.

    Returns:
        A plan that leaves transformer inputs unchanged and retains every
        loss-bearing row in the local shard, without a fixed row limit.
    """
    mb = ctx["mb_input"]
    inputs = ctx["model_inputs"]
    for data in (mb, inputs):
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            value = data.get(key)
            if value is not None and not (
                isinstance(value, torch.Tensor) and value.numel() == 0
            ):
                raise ValueError("Sparse logits support text-only micro-batches.")
    if "logits_to_keep" in inputs or inputs.get("labels") is not None:
        raise ValueError(
            "Sparse logits cannot be combined with labels or logits_to_keep."
        )

    ids = mb["input_ids"]
    if ids.ndim != 1 or ids.numel() < 2:
        raise ValueError("Sparse logits require full packed 1-D input_ids.")
    total = ids.numel()
    size = int(sp_info.get("sp_size", 1))
    rank = int(sp_info.get("sp_rank", 0))
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid sparse logits SP rank or size.")
    cu = mb.get("cu_seqlens")
    bounds = [0, total] if cu is None else cu.detach().cpu().tolist()
    if (
        len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != total
        or any(a >= b for a, b in zip(bounds, bounds[1:]))
    ):
        raise ValueError("cu_seqlens must describe the original unpadded input_ids.")
    valid_rows = torch.cat(
        [torch.arange(a, b - 1, device=ids.device) for a, b in zip(bounds, bounds[1:])]
    )
    if valid_rows.numel() == 0:
        raise ValueError("All-singleton packed batches are unsupported.")
    mask = mb["ppo_loss_mask"]
    if mask.ndim != 1 or mask.numel() != valid_rows.numel():
        raise ValueError("ppo_loss_mask must have shifted packed shape [sum(L-1)].")
    mask = mask.to(device=ids.device, dtype=torch.bool)
    old_logp = mb["old_logp"]
    if old_logp.shape != mask.shape or not bool(torch.isfinite(old_logp).all()):
        raise ValueError("old_logp must be finite and match the shifted loss mask.")
    old_logp = old_logp.detach().to(device=ids.device, dtype=torch.float32)

    pad = int(ctx["pad_length"])
    sp_pad = int(ctx["sp_pad_size"])
    padded = total + pad + sp_pad
    if pad < 0 or not 0 <= sp_pad < size or padded % size:
        raise ValueError("Invalid contiguous SP padding sizes.")
    local_length = padded // size
    active = valid_rows[mask]
    start = rank * local_length
    active = active[(active >= start) & (active < start + local_length)]
    rows = active - start
    model_rows = rows if rows.numel() else ids.new_zeros(1, dtype=torch.long)

    if size > 1:
        if sp_info.get("sp_group") is None:
            raise ValueError("SP process group is required for sparse logits.")
        labels = mb["_input_ids_rolled"]
        if labels.ndim != 1 or labels.numel() != local_length:
            raise ValueError(
                "SP rolled labels do not match the contiguous local shard."
            )
        if not torch.equal(labels.index_select(0, rows), ids[active + 1]):
            raise ValueError(
                "SP labels are misaligned with shifted loss-mask predictors."
            )
    else:
        labels = torch.cat((ids[1:], ids.new_zeros(pad + 1)))

    return SparseLogitsPlan(
        rows=rows,
        model_rows=model_rows,
        labels=labels[model_rows],
        valid_rows=valid_rows,
        mask=mask,
        old_logp=old_logp,
        local_length=local_length,
        cu_seqlens=cu,
        sp_pad_size=sp_pad,
        sp_info=dict(sp_info),
    )


def compute_sparse_logprobs_entropy(
    logits: torch.Tensor,
    plan: SparseLogitsPlan,
    *,
    temperature: float = 1.0,
    chunk_size: int = 32,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore shifted packed vectors using the existing SP autograd gathers.

    Masked-out log-probabilities equal ``old_logp`` so PPO ratios there are
    finite and equal to one. Masked-out entropy is zero. Neither is a current
    policy evaluation on those positions; callers must honor the loss mask.
    """
    if logits.ndim != 2 or logits.shape[0] != plan.model_rows.numel():
        raise ValueError("Sparse model output does not match selected predictor rows.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logp, entropy = gather_logprobs_entropy(
        logits,
        plan.labels,
        temperature,
        chunk_size=chunk_size,
        checkpoint_chunks=checkpoint_chunks,
    )
    if plan.rows.numel():
        local_logp = logp.new_zeros(plan.local_length).index_copy(0, plan.rows, logp)
        local_entropy = entropy.new_zeros(plan.local_length).index_copy(
            0, plan.rows, entropy
        )
    else:
        # Keep the head and backbone in autograd even on placeholder-only
        # ranks. Both SP gathers and the model backward still participate.
        local_logp = logp.new_zeros(plan.local_length) + logp.sum() * 0.0
        local_entropy = entropy.new_zeros(plan.local_length) + entropy.sum() * 0.0
    if int(plan.sp_info.get("sp_size", 1)) > 1:
        from .sp import gather_entropy_and_shift, gather_logprobs_and_shift

        logp = gather_logprobs_and_shift(
            local_logp,
            plan.sp_pad_size,
            plan.cu_seqlens,
            plan.sp_info,
            grad_scaler=True,
        )
        entropy = gather_entropy_and_shift(
            local_entropy,
            plan.sp_pad_size,
            plan.cu_seqlens,
            plan.sp_info,
            grad_scaler=True,
        )
    else:
        logp = local_logp.index_select(0, plan.valid_rows)
        entropy = local_entropy.index_select(0, plan.valid_rows)
    if logp.shape != plan.mask.shape or entropy.shape != logp.shape:
        raise ValueError("Sparse gathered outputs do not match the shifted loss mask.")
    logp = torch.where(plan.mask, logp, plan.old_logp)
    entropy = torch.where(plan.mask, entropy, 0.0)
    return logp, entropy
