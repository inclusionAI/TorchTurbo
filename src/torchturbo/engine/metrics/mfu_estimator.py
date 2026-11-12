"""Online MFU/HFU estimation.
"""

from __future__ import annotations

import time
from typing import Any
import torch
from torchturbo.engine.config import EngineConfig
from torchturbo.runtime.base import DeviceRuntime
from torchturbo.runtime.runtime import is_cuda_available, is_npu_available
from transformers.configuration_utils import PretrainedConfig

from .flops import ComputeProfile
from ...logging import get_logger

# Per-device peak bf16 TFLOP/s
PEAK_TFLOPS_TABLE: dict[str, float] = {
    "910b": 365.0,
    "910c": 376.0,
    "h20": 148.0,
    "a100": 312.0,
    "h100": 989.0,
    "h200": 989.0,
}

class MFUEstimator:
    """Per-micro flops accumulator that estimates windowed MFU/HFU for the engine report."""

    def __init__(
        self,
        flops_cls: type,
        engine_config: EngineConfig,
        model_config: PretrainedConfig,
        parallel_context: dict[str, Any],
        runtime: DeviceRuntime,
    ) -> None:
        assert flops_cls, "must provide a valid model flops class"
        self.flops_cls = flops_cls
        self.model_config = model_config
        self.sp_size = parallel_context["sp"]["sp_size"]
        self.device = runtime.current_device()
        self.gradient_checkpointing = engine_config.activation_checkpointing.enabled
        card = engine_config.mfu.card
        if card is None:
            if is_npu_available():
                card = "910c"
            elif is_cuda_available():
                card = "h20"

        # Resolve peak TFLOPs
        device_type = runtime.turbo_device.value
        assert card in PEAK_TFLOPS_TABLE, f"unsupported card {card}"
        self.hardware_label = f"{device_type}/{card}"
        self.peak_tflops = PEAK_TFLOPS_TABLE[card]

        self.start_time: float = time.perf_counter()
        self.window_fwd: float = 0.0
        self.window_tokens: float = 0.0
        self.window_micros = 0

        self.logger = get_logger(__name__)
        self.logger.info_rank0(
            "MFU estimator ready: recipe=%s hardware=%s sp_size=%d gc=%s peak=%.0f TFLOP/s",
            flops_cls.__name__, self.hardware_label, self.sp_size,
            self.gradient_checkpointing, self.peak_tflops,
        )

    # ------------------------------------------------------------------ #
    # Per-micro recording
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_dims(inputs: dict) -> tuple[int, int]:
        """(batch, seq) from input_ids / inputs_embeds shape."""
        for key in ("input_ids", "inputs_embeds"):
            t = inputs.get(key)
            if t is None:
                continue
            try:
                shape = tuple(t.shape)
            except AttributeError:
                continue
            if len(shape) >= 2:
                return int(shape[0]), int(shape[1])
            if len(shape) == 1:
                return 1, int(shape[0])
        return 0, 0

    @staticmethod
    def _read_segments(inputs: dict) -> list[int]:
        """Per-sample attention lengths from the packed cu boundaries."""
        return inputs["cu_seq_lens_q"].diff().tolist()

    @staticmethod
    def _read_vision_grids(inputs: dict) -> tuple[list, list]:
        """Per-batch image/video ``grid_thw`` as list[(t,h,w)] (empty if absent)."""
        def _to_grid(key: str) -> list:
            t = inputs.get(key)
            if t is None:
                return []
            try:
                rows = t.detach().reshape(-1, 3).tolist()
            except Exception:
                return []
            return [tuple(int(x) for x in r) for r in rows]

        return _to_grid("image_grid_thw"), _to_grid("video_grid_thw")

    def forward_flops(self, inputs: dict) -> int:
        """This device's forward FLOPs for one micro-batch's ``inputs``.

        The recipe returns the full-input upper bound; the SP group cooperates on
        one input, so this device's share is the total divided by ``sp_size``.
        """
        batch, seq = self._read_dims(inputs)
        if batch <= 0 or seq <= 0:
            return 0
        segments = self._read_segments(inputs)
        s_eff = sum(segments)
        if s_eff <= 0:
            return 0
        image_grids, video_grids = self._read_vision_grids(inputs)
        profile = ComputeProfile(
            batch_size=batch,
            seq_length=max(s_eff, 1 if segments else 0),
            seq_segments=list(segments) if segments else None,
            gradient_checkpointing=self.gradient_checkpointing,
            image_grid_thw=image_grids or None,
            video_grid_thw=video_grids or None,
        )
        recipe = self.flops_cls(self.model_config, profile)
        total = int(sum(c.fwd_flops for c in recipe.components()))
        return total // self.sp_size

    def record_micro(self, inputs: dict) -> None:
        """Accumulate one micro's per-device FLOPs + tokens."""
        batch, seq = self._read_dims(inputs)
        if batch <= 0 or seq <= 0:
            return

        micro_fwd = self.forward_flops(inputs)
        self.window_fwd += float(micro_fwd)
        self.window_tokens += float(batch * seq)
        self.window_micros += 1

    def report(self) -> dict[str, "torch.Tensor"]:
        """Per-rank {mfu, hfu} for the window since the last report (advances the clock)."""
        end_time = time.perf_counter()
        if self.window_micros == 0:
            return {}

        dt = end_time - self.start_time
        if dt <= 0:
            return {}
        fwd = self.window_fwd
        hfactor = 4 if self.gradient_checkpointing else 3
        model_flops = fwd * 3
        tflops = model_flops / 1e12 / dt
        peak = self.peak_tflops
        mfu = tflops / peak
        hfu = (fwd * hfactor / 1e12 / dt) / peak
        self.logger.debug(
            "MFU window: mfu=%.4f hfu=%.4f tflops_per_gpu=%.2f window_time=%.3f window_flops=%d hfu_factor=%d",
            mfu, hfu, tflops, dt, int(model_flops), hfactor,
        )

        metrics: dict[str, "torch.Tensor"] = {
            "mfu": torch.tensor(mfu, device=self.device),
            "hfu": torch.tensor(hfu, device=self.device),
        }

        self.window_fwd = 0.0
        self.window_tokens = 0.0
        self.window_micros = 0
        self.start_time = end_time

        return metrics