"""Online model and hardware FLOP utilization estimates."""

from typing import Protocol

import torch


class FlopsRecipe(Protocol):
    """Model-specific theoretical forward work for a full, unsliced input."""

    def forward_flops(self, inputs: dict[str, torch.Tensor]) -> int: ...


from .mfu_estimator import MFUEstimator

__all__ = ["FlopsRecipe", "MFUEstimator"]
