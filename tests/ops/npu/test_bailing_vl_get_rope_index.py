"""NPU alignment tests for the bailing_vl_get_rope_index op.

Compares the fused Triton implementation selected through the dispatcher
against the native PyTorch baseline for packed VL sequences.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.ops.providers.native.bailing_vl_get_rope_index import (
    torch_bailing_vl_get_rope_index,
)
from torchturbo.runtime.npu import NpuRuntime

pytest.importorskip("torch_npu")
pytest.importorskip("triton")

IMAGE_TOKEN = 60
VIDEO_TOKEN = 61
ROPE_KWARGS = dict(
    image_token_id=IMAGE_TOKEN,
    video_token_id=VIDEO_TOKEN,
    spatial_merge_size=2,
    use_interleaved_frame_timestamp=False,
)


@pytest.fixture
def npu_dispatch(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    runtime_impl.is_device_available.cache_clear()
    return get_or_create_ops(
        device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO
    )


def _reference(input_ids, grids, rope_kwargs):
    return torch_bailing_vl_get_rope_index(
        input_ids.cpu(),
        grids["image_grid_thw"].cpu() if "image_grid_thw" in grids else None,
        grids["video_grid_thw"].cpu() if "video_grid_thw" in grids else None,
        grids["segment_ids"].cpu() if "segment_ids" in grids else None,
        **rope_kwargs,
    )


def _assert_aligned(dispatcher, input_ids, grids, rope_kwargs=None):
    rope_kwargs = ROPE_KWARGS if rope_kwargs is None else rope_kwargs
    expected_positions, expected_deltas = _reference(
        input_ids, grids, rope_kwargs
    )

    actual_positions, actual_deltas = dispatcher.bailing_vl_get_rope_index(
        input_ids.to("npu"),
        **{
            key: value.to("npu") if value is not None else None
            for key, value in grids.items()
        },
        **rope_kwargs,
    )

    assert actual_positions.device.type == "npu"
    assert actual_deltas.device.type == "npu"
    torch.testing.assert_close(
        actual_positions.cpu(), expected_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_deltas.cpu(), expected_deltas, rtol=0, atol=0
    )


def test_npu_auto_policy_selects_triton_impl(npu_dispatch):
    input_ids = torch.tensor([[7, 8, 9]])

    npu_dispatch.bailing_vl_get_rope_index(
        input_ids.to("npu"), **ROPE_KWARGS
    )

    assert "triton_bailing_vl_get_rope_index" in npu_dispatch._impls


@pytest.mark.parametrize(
    "case_name, build_case",
    [
        ("text-only", lambda: (torch.tensor([[5, 6, 7, 8]]), {})),
        (
            "single-image",
            lambda: (
                torch.tensor([[2] + [IMAGE_TOKEN] * 64 + [3]]),
                {"image_grid_thw": torch.tensor([[1, 16, 16]])},
            ),
        ),
        (
            "multi-image-mixed",
            lambda: (
                torch.tensor(
                    [[2] + [IMAGE_TOKEN] * 16 + [3, 4] + [IMAGE_TOKEN] * 64 + [5, 6]]
                ),
                {
                    "image_grid_thw": torch.tensor(
                        [[1, 8, 8], [1, 16, 16]]
                    )
                },
            ),
        ),
        (
            "dense-video",
            lambda: (
                torch.tensor([[9] + [VIDEO_TOKEN] * 32 + [10]]),
                {"video_grid_thw": torch.tensor([[2, 8, 8]])},
            ),
        ),
        (
            "interleaved-video",
            lambda: (
                torch.tensor([[9] + [VIDEO_TOKEN] * 32 + [10]]),
                {"video_grid_thw": torch.tensor([[2, 8, 8]])},
            ),
        ),
        (
            "segments",
            lambda: (
                torch.tensor([[5, 5, 5] + [IMAGE_TOKEN] * 16 + [5, 6]]),
                {
                    "image_grid_thw": torch.tensor([[1, 8, 8]]),
                    "segment_ids": torch.tensor([[0, 0, 0] + [1] * 18]),
                },
            ),
        ),
    ],
)
def test_npu_triton_matches_native_baseline(
    npu_dispatch, case_name, build_case
):
    input_ids, grids = build_case()
    rope_kwargs = dict(ROPE_KWARGS)
    if case_name == "interleaved-video":
        rope_kwargs["use_interleaved_frame_timestamp"] = True
    _assert_aligned(npu_dispatch, input_ids, grids, rope_kwargs)


def test_npu_decode_survives_foreign_current_device(npu_dispatch):
    """Tensor device and current device may differ in pipeline contexts."""
    if torch.npu.device_count() < 2:
        pytest.skip("requires at least two NPU devices")
    input_ids = torch.tensor([[2] + [IMAGE_TOKEN] * 64 + [3]], device="npu:0")
    grids = {"image_grid_thw": torch.tensor([[1, 16, 16]], device="npu:0")}

    previous = torch.npu.current_device()
    try:
        torch.npu.set_device(1)
        positions, deltas = npu_dispatch.bailing_vl_get_rope_index(
            input_ids, **grids, **ROPE_KWARGS
        )
        torch.npu.synchronize(0)
    finally:
        torch.npu.set_device(previous)

    expected_positions, expected_deltas = _reference(
        input_ids, grids, ROPE_KWARGS
    )
    torch.testing.assert_close(positions.cpu(), expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(deltas.cpu(), expected_deltas, rtol=0, atol=0)
