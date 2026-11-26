"""Native PyTorch baseline for the Bailing VL mRoPE index op.

Decodes packed text/media runs on CPU and uploads the finished T/H/W
position tensor in a single copy, mirroring the upstream ``get_rope_index``
semantics including segment handling and greedy media grid consumption.
"""

from __future__ import annotations

from typing import Any

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _decode_mrope_runs(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    segment_ids: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    use_interleaved_frame_timestamp: bool,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Decode packed Bailing T/H/W positions into flat run records.

    Every contiguous text run becomes a linear run; every media run keeps
    its temporal, height and width geometry. The returned run records are
    consumed by the CPU position builder below.
    """
    batch_size, seq_length = input_ids.shape
    grids = {
        image_token_id: (
            image_grid_thw.cpu() if image_grid_thw is not None else None
        ),
        video_token_id: (
            video_grid_thw.cpu() if video_grid_thw is not None else None
        ),
    }
    if video_grid_thw is not None and use_interleaved_frame_timestamp:
        expanded_video_grids = [
            (1, int(h), int(w))
            for t, h, w in grids[video_token_id].tolist()
            for _ in range(int(t))
        ]
        grids[video_token_id] = torch.tensor(
            expanded_video_grids, dtype=video_grid_thw.dtype
        ).reshape(-1, 3)
    grid_indices = {token: 0 for token in grids}

    input_cpu = input_ids.detach().to("cpu")
    segment_cpu = (
        torch.ones_like(input_cpu)
        if segment_ids is None
        else segment_ids.detach().to("cpu")
    )
    deltas_cpu = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
    runs: list[dict[str, Any]] = []

    for batch_index in range(batch_size):
        row = segment_cpu[batch_index]
        changes = (row[1:] != row[:-1]).nonzero().flatten() + 1
        boundaries = torch.cat(
            (changes.new_tensor([0]), changes, changes.new_tensor([seq_length]))
        )
        for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            tokens = input_cpu[batch_index, start:end].tolist()
            cursor, position = 0, 0
            while cursor < len(tokens):
                text_end = cursor
                while text_end < len(tokens) and tokens[text_end] not in grids:
                    text_end += 1
                if text_end > cursor:
                    text_length = text_end - cursor
                    runs.append({
                        "batch": batch_index,
                        "start": start + cursor,
                        "num": text_length,
                        "base": position,
                        "is_text": True,
                        "t": 1,
                        "h": 1,
                        "w": 1,
                    })
                    cursor = text_end
                    position += text_length
                if cursor == len(tokens):
                    break
                token = tokens[cursor]
                grid = grids[token]
                grid_index = grid_indices[token]
                if grid is None or grid_index >= len(grid):
                    raise ValueError(
                        "Media patch tokens require a matching grid_thw entry."
                    )
                t, h, w = grid[grid_index].tolist()
                grid_indices[token] += 1
                h, w = int(h) // spatial_merge_size, int(w) // spatial_merge_size
                t = int(t)
                num_tokens = t * h * w
                if (
                    num_tokens == 0
                    or tokens[cursor : cursor + num_tokens] != [token] * num_tokens
                ):
                    raise ValueError(
                        "Media patch token count does not match grid_thw."
                    )
                runs.append({
                    "batch": batch_index,
                    "start": start + cursor,
                    "num": num_tokens,
                    "base": position,
                    "is_text": False,
                    "t": t,
                    "h": h,
                    "w": w,
                })
                cursor += num_tokens
                position += 2 * t - 1
            if row[start] != 0:
                deltas_cpu[batch_index, 0] = position - (end - start)

    for token, grid in grids.items():
        if grid is not None and grid_indices[token] != len(grid):
            raise ValueError(
                "Unused media grid entries do not match the token batch."
            )
    return runs, deltas_cpu, segment_cpu


def torch_bailing_vl_get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    use_interleaved_frame_timestamp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Bailing T/H/W mRoPE positions on CPU and upload the result once.

    Args:
        input_ids: Token ids of shape [batch_size, seq_length]; only the
            dtype, shape and device matter, values are scanned for media
            tokens.
        image_grid_thw: Optional [num_images, 3] temporal/height/width patch
            grids, merged by spatial_merge_size.
        video_grid_thw: Optional [num_videos, 3] patch grids. With
            use_interleaved_frame_timestamp each frame becomes its own
            single-frame entry.
        segment_ids: Optional [batch_size, seq_length] packed-sequence
            segment labels. Segment boundaries reset the position counter;
            tokens in segment 0 get position 0.
        image_token_id: Token id that opens an image patch run.
        video_token_id: Token id that opens a video patch run.
        spatial_merge_size: Vision merger spatial merge factor.
        use_interleaved_frame_timestamp: Expand video grids per frame.

    Returns:
        position_ids of shape [3, batch_size, seq_length] and rope_deltas of
        shape [batch_size, 1], both with the dtype and device of input_ids.

    Raises:
        ValueError: When media patch tokens and grid_thw entries disagree,
            with the same messages as the upstream implementation.
    """
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length
    if total_tokens == 0:
        return (
            torch.zeros(
                (3, batch_size, seq_length),
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            torch.zeros(
                (batch_size, 1), dtype=input_ids.dtype, device=input_ids.device
            ),
        )
    runs, deltas_cpu, segment_cpu = _decode_mrope_runs(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        segment_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        spatial_merge_size=spatial_merge_size,
        use_interleaved_frame_timestamp=use_interleaved_frame_timestamp,
    )
    positions_cpu = torch.zeros(
        3, batch_size, seq_length, dtype=input_ids.dtype, device="cpu"
    )
    for run in runs:
        batch_index = run["batch"]
        start = run["start"]
        num = run["num"]
        base = run["base"]
        if run["is_text"]:
            positions_cpu[:, batch_index, start : start + num] = torch.arange(
                base, base + num, dtype=input_ids.dtype, device="cpu"
            )
        else:
            t = run["t"]
            h = run["h"]
            w = run["w"]
            temporal = (
                torch.arange(
                    t, dtype=input_ids.dtype, device="cpu"
                ).repeat_interleave(h * w)
                * 2
                + base
            )
            height = (
                torch.arange(h, dtype=input_ids.dtype, device="cpu")
                .repeat_interleave(w)
                .repeat(t)
                - (h - 1) // 2
                + temporal
            )
            width = (
                torch.arange(w, dtype=input_ids.dtype, device="cpu").repeat(t * h)
                - (w - 1) // 2
                + temporal
            )
            positions_cpu[
                :, batch_index, start : start + num
            ] = torch.stack((temporal, height, width))
    positions_cpu[:, segment_cpu == 0] = 0
    return (
        positions_cpu.to(input_ids.device),
        deltas_cpu.to(input_ids.device),
    )


@register_op(
    op_name="bailing_vl_get_rope_index",
    impl_name="torch_bailing_vl_get_rope_index",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchBailingVlGetRopeIndex(OpImpl):
    def __call__(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        *,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
        use_interleaved_frame_timestamp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_bailing_vl_get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            segment_ids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            spatial_merge_size=spatial_merge_size,
            use_interleaved_frame_timestamp=use_interleaved_frame_timestamp,
        )
