"""Vision balance input and output contract for the Bailing vision tower."""

from typing import Any

import torch
from torch import nn

from ..modules.vision.io import BaseVisionBalanceIO, VisionBatch


class BailingVisionBalanceIO(BaseVisionBalanceIO):
    """Balance packed pixels and the tower's merged feature tensor."""

    def read_inputs(
        self, tower: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> VisionBatch:
        pixels = args[0] if args else kwargs.get("hidden_states")
        grid_thw = args[1] if len(args) > 1 else kwargs.get("grid_thw")
        # Every training rank enters the tower with media: the framework gives
        # media-less batches a dummy image, and the balance pass only supplies
        # its own dummy to ranks the distribution leaves without media.
        if pixels is None or grid_thw is None:
            raise ValueError(
                "Bailing vision balance requires hidden_states and grid_thw together"
            )
        lengths = (grid_thw.prod(dim=-1) // tower.spatial_merge_size**2).tolist()
        return VisionBatch(
            pixels=pixels,
            grid_thw=grid_thw,
            output_lengths={"features": lengths},
        )

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

    def read_outputs(self, output: Any) -> dict[str, torch.Tensor]:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Bailing vision balance requires a tensor tower output")
        return {"features": output}

    def write_outputs(
        self,
        output: Any,
        features: dict[str, torch.Tensor],
        original_batch: VisionBatch,
    ) -> Any:
        return features["features"]

    def make_dummy(self, tower: nn.Module) -> VisionBatch:
        parameter = next(tower.parameters())
        merge = tower.spatial_merge_size
        patch_embed = tower.patch_embed
        pixel_width = (
            patch_embed.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        return VisionBatch(
            pixels=torch.zeros(
                merge**2,
                pixel_width,
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            grid_thw=torch.tensor(
                [[1, merge, merge]], dtype=torch.long, device=parameter.device
            ),
            output_lengths={"features": [1]},
        )
