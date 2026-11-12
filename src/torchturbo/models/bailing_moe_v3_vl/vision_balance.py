"""Vision balance input and output contract for the Bailing vision tower."""

from typing import Any

import torch
from torch import nn
from typing_extensions import override

from ..modules.vision.io import VisionBatch
from ..qwen3_5.vision_balance import Qwen3_5VisionBalanceIO


class BailingVisionBalanceIO(Qwen3_5VisionBalanceIO):
    """Reuse Qwen vision inputs and restore Bailing's merged feature tensor."""

    @override
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
