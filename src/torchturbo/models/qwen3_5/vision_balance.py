"""Vision balance contract for the Qwen3.5 dense vision tower."""

from dataclasses import replace

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..modules.vision.io import PackedPatchVisionBalanceIO, VisionBatch


class Qwen3_5VisionBalanceIO(PackedPatchVisionBalanceIO):
    """Restore both patch-level and spatially merged vision features."""

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
