"""Vision balance contract for the Qwen3.5 dense vision tower."""

from dataclasses import replace
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..modules.vision.io import BaseVisionBalanceIO, VisionBatch


class Qwen3_5VisionBalanceIO(BaseVisionBalanceIO):
    """Adapt Qwen patch inputs and restore patch-level and merged features."""

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

    def output_lengths(
        self, tower: nn.Module, grid_thw: torch.Tensor
    ) -> dict[str, list[int]]:
        patches = grid_thw.prod(dim=-1)
        return {
            "last_hidden_state": patches.tolist(),
            "pooler_output": (patches // tower.spatial_merge_size**2).tolist(),
        }

    def read_outputs(
        self, output: BaseModelOutputWithPooling
    ) -> dict[str, torch.Tensor]:
        return {
            "last_hidden_state": output.last_hidden_state,
            "pooler_output": output.pooler_output,
        }

    def write_outputs(
        self,
        output: BaseModelOutputWithPooling,
        features: dict[str, torch.Tensor],
        original_batch: VisionBatch,
    ) -> BaseModelOutputWithPooling:
        return replace(output, **features)
