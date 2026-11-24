"""Cross-entropy loss from materialized language-model logits."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from torchturbo.ops import ops

__all__ = ["logits_loss"]


def prepare_loss_scale(
    weights: Sequence[torch.Tensor] | None, num_inputs: int,
) -> tuple[torch.Tensor | None, ...]:
    """Return flat, non-differentiable coefficients for each target stream."""
    if weights is None:
        return (None,) * num_inputs
    return tuple(weight.detach().reshape(-1).float() for weight in weights)


def token_ce_sum(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None, ignore_index: int,
) -> torch.Tensor:
    """Sum token CE before the caller applies the valid-target denominator."""
    if weights is None:
        return ops.cross_entropy_loss(logits, labels, reduction="sum", ignore_index=ignore_index)
    losses = ops.cross_entropy_loss(logits.float(), labels, reduction="none", ignore_index=ignore_index)
    return (losses * weights).sum()


def logits_loss(
    logits: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    normalizers: Sequence[float | torch.Tensor],
    loss_scaling_factors: Sequence[float],
    loss_scale: Sequence[torch.Tensor] | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-weighted cross entropy from materialized logits."""
    logits = tuple(logits)
    labels = tuple(labels)
    num_inputs = len(logits)
    if num_inputs == 0:
        raise ValueError("at least one LM loss input is required")
    if not (
        len(labels) == len(normalizers) == len(loss_scaling_factors) == num_inputs
    ):
        raise ValueError(
            "logits, labels, normalizers and loss_scaling_factors must have the same length"
        )
    loss_scale_per_stream = prepare_loss_scale(loss_scale, num_inputs)

    for logits_i, label in zip(logits, labels):
        if logits_i.dim() != 3:
            raise ValueError(
                f"logits must be [B, S, V], got {tuple(logits_i.shape)}"
            )
        if label.shape != logits_i.shape[:2]:
            raise ValueError(
                f"labels {tuple(label.shape)} must match logits "
                f"{tuple(logits_i.shape[:2])}"
            )

    normalizer_scalars: list[torch.Tensor] = []
    for value in normalizers:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("normalizer values must be scalar tensors")
            scalar = value.detach().to(
                device=logits[0].device, dtype=torch.float32
            )
        else:
            scalar_value = float(value)
            if not math.isfinite(scalar_value):
                raise ValueError(
                    f"normalizer values must be finite, got {scalar_value}"
                )
            if scalar_value <= 0:
                raise ValueError(
                    f"normalizer values must be positive, got {scalar_value}"
                )
            scalar = torch.tensor(
                scalar_value, device=logits[0].device, dtype=torch.float32
            )
        normalizer_scalars.append(scalar.reshape(()))
    normalizer_tensor = torch.stack(normalizer_scalars)

    scaling_factor_values: list[float] = []
    for value in loss_scaling_factors:
        if torch.is_tensor(value):
            raise TypeError(
                "loss scale values must be host scalars so zero-scaled "
                "objectives can be excluded from backward"
            )
        scalar_value = float(value)
        if not math.isfinite(scalar_value):
            raise ValueError(
                f"loss scale values must be finite, got {scalar_value}"
            )
        scaling_factor_values.append(scalar_value)

    total_loss: torch.Tensor | None = None
    raw_losses: list[torch.Tensor] = []
    for logits_i, label, normalizer, loss_scaling_factor, weight_i in zip(
        logits, labels, normalizer_tensor, scaling_factor_values, loss_scale_per_stream
    ):
        loss_logits = logits_i if loss_scaling_factor != 0.0 else logits_i.detach()
        flat_logits = loss_logits.reshape(-1, loss_logits.shape[-1])
        flat_labels = label.reshape(-1).to(logits_i.device)
        if weight_i is not None:
            weight_i = weight_i.to(flat_logits.device)
        raw_loss = token_ce_sum(flat_logits, flat_labels, weight_i, ignore_index) / normalizer
        if loss_scaling_factor != 0.0:
            scaled_loss = raw_loss * loss_scaling_factor
            total_loss = (
                scaled_loss if total_loss is None else total_loss + scaled_loss
            )
        raw_losses.append(raw_loss.detach())

    if total_loss is None:
        total_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=logits[0].device,
            requires_grad=torch.is_grad_enabled()
            and any(logits_i.requires_grad for logits_i in logits),
        )
    return total_loss, torch.stack(raw_losses)
