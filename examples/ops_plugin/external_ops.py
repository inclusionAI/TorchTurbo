"""External-style functions with no TorchTurbo dependency."""

import torch


def scaled_square(x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    return x.square() * scale
