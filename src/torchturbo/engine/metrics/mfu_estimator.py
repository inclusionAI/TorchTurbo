"""Windowed MFU/HFU estimates using model FLOPs and synchronized wall time."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from ...logging import get_logger
from ...runtime.base import DeviceRuntime
from ..config import EngineConfig
from . import FlopsRecipe

logger = get_logger(__name__)

# BF16/FP16 dense peak TFLOP/s per runtime device (one die on 910C).
PEAK_TFLOPS_TABLE: dict[str, float] = {
    "910b": 365.0,
    "910c": 376.0,
    "h20": 148.0,
    "a100": 312.0,
    "h100": 989.0,
    "h200": 989.0,
}


class MFUEstimator:
    """Accumulate per-device work between training log events."""

    def __init__(
        self,
        recipe: FlopsRecipe,
        config: EngineConfig,
        sp_size: int,
        runtime: DeviceRuntime,
    ) -> None:
        card = config.mfu.card
        if card is None:
            card = {"npu": "910c", "cuda": "h20"}.get(runtime.device.type)
        if card not in PEAK_TFLOPS_TABLE:
            raise ValueError(
                f"Unsupported MFU card {card!r}; choose from {list(PEAK_TFLOPS_TABLE)}."
            )
        if config.mixed_precision.value not in ("bf16", "fp16"):
            raise ValueError("MFU/HFU peak tables require BF16 or FP16 training.")
        self.recipe = recipe
        self.sp_size = sp_size
        self.runtime = runtime
        self.peak_tflops = PEAK_TFLOPS_TABLE[card]
        self.hfu_factor = 4 if config.activation_checkpointing.enabled else 3
        self.start_time: float | None = None
        self.window_fwd: float = 0.0
        logger.info_rank0(
            "MFU estimator ready: recipe=%s card=%s peak=%.0f TFLOP/s "
            "per runtime device, sp_size=%d checkpointing=%s",
            type(recipe).__name__, card, self.peak_tflops, sp_size,
            config.activation_checkpointing.enabled,
        )

    def record_micro(self, inputs: dict[str, torch.Tensor]) -> None:
        """Record a training micro-batch before the forward pass."""
        if self.start_time is None:
            self.runtime.synchronize()
            self.start_time = time.perf_counter()
        self.window_fwd += self.recipe.forward_flops(inputs) / self.sp_size

    def report(self, logs: dict, gather: Callable) -> None:
        """Append cross-rank mean ratios and reset the logging window."""
        if self.start_time is None:
            return
        self.runtime.synchronize()
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        if elapsed <= 0:
            return
        base = self.window_fwd / elapsed / (self.peak_tflops * 1e12)
        values = torch.tensor(
            [base * 3, base * self.hfu_factor],
            dtype=torch.float32,
            device=self.runtime.device,
        )
        means = gather(values).reshape(-1, 2).mean(dim=0).tolist()
        logs.update(mfu=round(means[0], 8), hfu=round(means[1], 8))
        self.window_fwd = 0.0
        self.start_time = None
