"""Vision balance input and output contract for the Bailing vision tower."""

from typing import Any

import torch
from torch import nn

from ..modules.vision.io import PackedPatchVisionBalanceIO, VisionBatch


class BailingVisionBalanceIO(PackedPatchVisionBalanceIO):
    """Balance packed pixels and the tower's merged feature tensor."""

    def output_lengths(
        self, tower: nn.Module, grid_thw: torch.Tensor
    ) -> dict[str, list[int]]:
        lengths = (grid_thw.prod(dim=-1) // tower.spatial_merge_size**2).tolist()
        return {"features": lengths}

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
