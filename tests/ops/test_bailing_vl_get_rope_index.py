"""Semantic contract tests for the bailing_vl_get_rope_index op.

Runs on CPU against the native PyTorch baseline through the process
dispatcher; hardware alignment lives in tests/ops/npu/.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.ops import OpPolicy, get_ops, get_or_create_ops
from torchturbo.ops.providers.native.bailing_vl_get_rope_index import (
    torch_bailing_vl_get_rope_index,
)
from torchturbo.runtime.cpu import CpuRuntime

IMAGE_TOKEN = 60
VIDEO_TOKEN = 61
ROPE_KWARGS = dict(
    image_token_id=IMAGE_TOKEN,
    video_token_id=VIDEO_TOKEN,
    spatial_merge_size=2,
    use_interleaved_frame_timestamp=False,
)


@pytest.fixture(autouse=True)
def _cpu_dispatch(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    runtime_impl.is_device_available.cache_clear()
    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.AUTO)
    try:
        yield
    finally:
        runtime_impl.is_device_available.cache_clear()


def _dispatched(input_ids, **kwargs):
    return get_ops().bailing_vl_get_rope_index(
        input_ids, **{**ROPE_KWARGS, **kwargs}
    )


def _expected_single_image() -> torch.Tensor:
    # [text, image(1,4,4) -> merged 2x2, text]; image run starts at base 1.
    return torch.tensor(
        [
            [0, 1, 1, 1, 1, 2],
            [0, 1, 1, 2, 2, 2],
            [0, 1, 2, 1, 2, 2],
        ],
        dtype=torch.int64,
    )


def test_text_only_positions_and_zero_delta():
    input_ids = torch.tensor([[7, 8, 9]])

    positions, deltas = _dispatched(input_ids)

    assert positions.shape == (3, 1, 3)
    assert positions.dtype == input_ids.dtype
    assert deltas.dtype == input_ids.dtype
    assert positions[0].tolist() == [[0, 1, 2]]
    assert (positions[0] == positions[1]).all() and (positions[0] == positions[2]).all()
    assert deltas.tolist() == [[0]]


def test_image_run_geometry_matches_reference():
    input_ids = torch.tensor([[2] + [IMAGE_TOKEN] * 4 + [3]])
    image_grid_thw = torch.tensor([[1, 4, 4]])

    positions, deltas = _dispatched(input_ids, image_grid_thw=image_grid_thw)

    expected = _expected_single_image()
    assert positions.view(3, -1).tolist() == expected.tolist()
    # Two text advances plus one media delta (2*t - 1 = 1) minus six tokens.
    assert deltas.tolist() == [[-3]]


def test_dispatch_matches_native_baseline():
    input_ids = torch.tensor(
        [[2] + [IMAGE_TOKEN] * 4 + [3, 4] + [VIDEO_TOKEN] * 8 + [5]]
    )
    image_grid_thw = torch.tensor([[1, 4, 4]])
    video_grid_thw = torch.tensor([[2, 4, 4]])

    positions, deltas = _dispatched(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    expected_positions, expected_deltas = torch_bailing_vl_get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        **ROPE_KWARGS,
    )

    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(deltas, expected_deltas)


def test_explicit_impl_selection_uses_native_baseline():
    input_ids = torch.tensor([[7, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
    image_grid_thw = torch.tensor([[1, 4, 4]])

    positions, deltas = get_ops().bailing_vl_get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        impl="torch_bailing_vl_get_rope_index",
        **ROPE_KWARGS,
    )

    assert positions.view(3, -1)[1:, 1:].tolist() == [[1, 1, 2, 2], [1, 2, 1, 2]]
    assert deltas.tolist() == [[-3]]


def test_interleaved_video_expands_frames():
    input_ids = torch.tensor([[VIDEO_TOKEN] * 8])
    video_grid_thw = torch.tensor([[2, 4, 4]])

    dense, dense_deltas = _dispatched(input_ids, video_grid_thw=video_grid_thw)
    interleaved, interleaved_deltas = _dispatched(
        input_ids,
        video_grid_thw=video_grid_thw,
        use_interleaved_frame_timestamp=True,
    )

    # Dense video: one run advances time by 2*t - 1 = 3 across 8 tokens.
    assert dense.view(3, -1)[0].tolist() == [0, 0, 0, 0, 2, 2, 2, 2]
    assert dense_deltas.tolist() == [[-5]]
    # Interleaved: two one-frame runs, second frame starts at position 1.
    assert interleaved.view(3, -1)[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert interleaved_deltas.tolist() == [[-6]]


def test_segments_reset_positions_and_zero_segment():
    input_ids = torch.tensor([[5, 5, 5] + [IMAGE_TOKEN] * 4 + [5]])
    image_grid_thw = torch.tensor([[1, 4, 4]])
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])

    positions, deltas = _dispatched(
        input_ids, image_grid_thw=image_grid_thw, segment_ids=segment_ids
    )

    # Segment 0 keeps position 0 everywhere.
    assert positions.view(3, -1)[:, :3].tolist() == [[0, 0, 0]] * 3
    expected = _expected_single_image()
    # Segment 1 restarts the counter: the image run starts at base 0, one
    # below the single-image reference whose leading text token raised it.
    assert positions.view(3, -1)[:, 3:7].tolist() == (expected[:, 1:5] - 1).tolist()
    assert positions.view(3, -1)[:, 7].tolist() == [1, 1, 1]
    assert deltas.tolist() == [[-3]]


def test_multiple_batch_rows():
    input_ids = torch.tensor(
        [
            [7, 8, 9, 10],
            [2] + [IMAGE_TOKEN] * 2 + [3],
        ]
    )
    # Grid merges to h=1, w=2: a two-token image run with a width spread.
    image_grid_thw = torch.tensor([[1, 2, 4]])

    positions, deltas = _dispatched(input_ids, image_grid_thw=image_grid_thw)

    assert positions.shape == (3, 2, 4)
    assert positions[0, 0].tolist() == [0, 1, 2, 3]
    assert positions[0, 1].tolist() == [0, 1, 1, 2]
    assert positions[1, 1].tolist() == [0, 1, 1, 2]
    assert positions[2, 1].tolist() == [0, 1, 2, 2]
    assert deltas.tolist() == [[0], [-1]]


def test_empty_input_returns_zeros():
    input_ids = torch.zeros((1, 0), dtype=torch.int64)

    positions, deltas = _dispatched(input_ids)

    assert positions.shape == (3, 1, 0)
    assert deltas.shape == (1, 1)
    assert deltas.tolist() == [[0]]


@pytest.mark.parametrize(
    "input_ids, grids, message",
    [
        (
            torch.tensor([[IMAGE_TOKEN] * 4]),
            {"image_grid_thw": None},
            "Media patch tokens require a matching grid_thw entry.",
        ),
        (
            torch.tensor([[IMAGE_TOKEN] * 3]),
            {"image_grid_thw": torch.tensor([[1, 4, 4]])},
            "Media patch token count does not match grid_thw.",
        ),
        (
            torch.tensor([[5, 6]]),
            {"image_grid_thw": torch.tensor([[1, 2, 2]])},
            "Unused media grid entries do not match the token batch.",
        ),
    ],
    ids=["missing-grid", "count-mismatch", "unused-grid"],
)
def test_error_messages_match_cpu_decoder(input_ids, grids, message):
    with pytest.raises(ValueError, match=message):
        _dispatched(input_ids, **grids)
