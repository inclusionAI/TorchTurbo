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


class PackedPatchVisionBalanceIO(BaseVisionBalanceIO):
    """Shared input layout for towers taking flat patches and THW grids."""

    @abstractmethod
    def output_lengths(
        self, tower: nn.Module, grid_thw: torch.Tensor
    ) -> dict[str, list[int]]:
        """Describe each model-owned output stream's rows per medium."""

    def read_inputs(
        self, tower: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> VisionBatch:
        pixels = args[0] if args else kwargs.get("hidden_states")
        grid_thw = args[1] if len(args) > 1 else kwargs.get("grid_thw")
        if pixels is None or grid_thw is None:
            raise ValueError(
                "Vision balance requires hidden_states and grid_thw together"
            )
        return VisionBatch(pixels, grid_thw, self.output_lengths(tower, grid_thw))

    def write_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        batch: VisionBatch,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        new_args = list(args)
        new_kwargs = dict(kwargs)
        if new_args:
            new_args[0] = batch.pixels
        else:
            new_kwargs["hidden_states"] = batch.pixels
        if len(new_args) > 1:
            new_args[1] = batch.grid_thw
        else:
            new_kwargs["grid_thw"] = batch.grid_thw
        return tuple(new_args), new_kwargs

    def make_dummy(self, tower: nn.Module) -> VisionBatch:
        parameter = next(tower.parameters())
        merge = tower.spatial_merge_size
        patch_embed = tower.patch_embed
        pixel_width = (
            patch_embed.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        grid_thw = torch.tensor(
            [[1, merge, merge]], dtype=torch.long, device=parameter.device
        )
        return VisionBatch(
            pixels=torch.zeros(
                merge**2, pixel_width, dtype=parameter.dtype, device=parameter.device
            ),
            grid_thw=grid_thw,
            output_lengths=self.output_lengths(tower, grid_thw),
        )
