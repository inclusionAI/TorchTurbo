"""Model-owned input and output contracts for vision balancing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class VisionBatch:
    """Packed media and the output row counts needed to restore them.

    Attributes:
        pixels: Input patches concatenated along dimension zero in media
            order. The remaining dimensions retain the model's patch layout.
            All ranks use the same trailing shape, dtype, and device type.
        grid_thw: One integer ``(t, h, w)`` row per medium. Each medium
            occupies ``t * h * w`` contiguous rows in ``pixels``.
        output_lengths: Per-medium row counts for each named output tensor.
            Names and their insertion order must agree on every rank,
            including ranks with no media. Lengths are known before forward.
    """

    pixels: torch.Tensor
    grid_thw: torch.Tensor
    output_lengths: dict[str, list[int]]


class BaseVisionBalanceIO(ABC):
    """Translate one model's vision boundary to packed media and back.

    Implementations describe independent media with contiguous input and
    output rows. They must account for every media-dependent argument and
    output, preserve unrelated arguments and the return structure, and keep
    output transformations differentiable. This contract currently exchanges
    pixels and grids; boundaries needing other media-dependent inputs must
    adapt those inputs explicitly before opting into balancing.

    Instances hold model-format behavior, not per-forward state. Each WORLD
    rank must enter the same boundary in the same order. An adapter cannot
    compensate for a rank that skips the tower call.

    Empty outputs must retain their gradient connection when rebuilding the
    return value. Media inputs are always present at the boundary: training
    frameworks give media-less batches a dummy image, so ``read_inputs`` must
    not invent media for missing inputs. ``make_dummy`` serves only the ranks
    the distribution leaves without media because the world pool holds fewer
    media than ranks; that dummy must exercise the same differentiable output
    streams as real media so every rank joins the backward exchanges.
    """

    @abstractmethod
    def read_inputs(
        self, tower: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> VisionBatch:
        """Extract packed inputs and output lengths, including empty media."""

    @abstractmethod
    def write_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        batch: VisionBatch,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Replace media inputs with the assigned batch in a new call."""

    @abstractmethod
    def read_outputs(self, output: Any) -> dict[str, torch.Tensor]:
        """Extract all media-dependent output streams in declared order."""

    @abstractmethod
    def write_outputs(
        self,
        output: Any,
        features: dict[str, torch.Tensor],
        original_batch: VisionBatch,
    ) -> Any:
        """Restore the return structure for the caller's original media."""

    @abstractmethod
    def make_dummy(self, tower: nn.Module) -> VisionBatch:
        """Build minimal valid media for a rank the distribution assigns no
        media, which happens when the world pool holds fewer media than
        ranks."""
