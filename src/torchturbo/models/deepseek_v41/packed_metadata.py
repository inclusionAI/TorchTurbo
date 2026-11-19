"""Forward-local packed geometry shared by fused attention and indexer KL."""

from functools import cached_property
from itertools import accumulate
from typing import NamedTuple

import torch

from ..deepseek_v4.packing_utils import _segment_spans


def checked_block_info(row: torch.Tensor, ratio: int) -> dict[int, tuple[int, int, int, int]]:
    """Build complete-block offsets and reject repeated sample IDs."""
    starts, ends, ids = _segment_spans(row)
    info = {}
    offset = 0
    for start, end, sample_id in zip(starts.tolist(), ends.tolist(), ids.tolist()):
        if sample_id in info:
            raise ValueError("packed sequence IDs must be unique within a row")
        count = (end - start) // ratio
        info[sample_id] = (start, end, offset, count)
        offset += count
    return info


class PackedSegment(NamedTuple):
    batch: int
    start: int
    end: int
    sample_start: int
    block_start: int
    block_count: int


def cumulative_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Exclusive cumulative sums with a leading 0, int32 on ``device``."""
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


class PackedAttentionMetadata:
    """Parse each global mask row once, then intersect with the local shard.

    Instances belong to one attention forward, never to a module or global cache.
    Checkpoint recomputation builds a new instance. Only mask geometry and index
    tensors are retained; projected activations are gathered by the caller.
    """

    def __init__(self, mask: torch.Tensor, local_length: int, sp_rank: int, ratio: int) -> None:
        self.mask = mask
        self.local_length = local_length
        self.local_start = sp_rank * local_length
        self.ratio = ratio

    @cached_property
    def local_mask(self) -> torch.Tensor:
        return self.mask[:, self.local_start:self.local_start + self.local_length]

    @cached_property
    def block_infos(self) -> list[dict[int, tuple[int, int, int, int]]]:
        return [checked_block_info(row, self.ratio) for row in self.mask]

    @cached_property
    def compressed_capacity(self) -> int:
        """Largest complete-block count among the packed batch rows."""
        return max((sum(info[3] for info in row.values())
                    for row in self.block_infos), default=0)

    @cached_property
    def segments(self) -> list[PackedSegment]:
        segments = []
        for batch, info in enumerate(self.block_infos):
            for start, end, offset, count in info.values():
                left = max(start, self.local_start) - self.local_start
                right = min(end, self.local_start + self.local_length) - self.local_start
                if left < right:
                    segments.append(PackedSegment(batch, left, right, start, offset, count))
        return segments

    def check_padding(self, indices: torch.Tensor) -> None:
        if ((indices != -1) & (self.local_mask <= 0).unsqueeze(-1)).any().item():
            raise ValueError("Padding queries must not select compressed indices.")

    def local_indices(
        self, indices: torch.Tensor, segment: PackedSegment,
        positions: torch.Tensor | None = None,
        *, validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate one sample's selection and preserve the Top-K slot order.

        ``validate=False`` skips the legality sync (device-to-host check);
        callers that already validated this indices tensor on the same packed
        geometry (the prepare step of the fused attention path) may opt out."""
        batch, start, end, sample_start, offset, count = segment
        selected = indices[batch, start:end]
        relative = selected - offset
        query_positions = torch.arange(start, end, device=indices.device)
        query_positions = query_positions + self.local_start - sample_start
        if positions is not None and not torch.equal(positions[batch, start:end], query_positions):
            raise ValueError("Query positions must match their packed sample positions.")
        visible = (query_positions + 1) // self.ratio
        if validate:
            legal = (selected == -1) | (
                (relative >= 0) & (relative < count) & (relative < visible[:, None])
            )
            if not legal.all().item():
                raise ValueError("Compressed indices are invalid, cross-sample or non-causal (violate causality).")
        return relative.masked_fill_(selected == -1, -1), visible

    def _indices(self, parts: list[torch.Tensor]) -> torch.Tensor:
        if parts:
            return torch.cat(parts)
        return torch.empty(0, dtype=torch.long, device=self.mask.device)

    def _cumulative(self, lengths: list[int]) -> torch.Tensor:
        return cumulative_lengths(lengths, self.mask.device)

    @cached_property
    def query_lengths(self) -> list[int]:
        return [s.end - s.start for s in self.segments]

    @cached_property
    def query_indices(self) -> torch.Tensor:
        return self._indices([
            torch.arange(s.batch * self.local_length + s.start,
                         s.batch * self.local_length + s.end, device=self.mask.device)
            for s in self.segments
        ])

    @cached_property
    def cu_query(self) -> torch.Tensor:
        return self._cumulative(self.query_lengths)

    @cached_property
    def kv_lengths(self) -> list[int]:
        return [self.local_start + s.end - s.sample_start for s in self.segments]

    @cached_property
    def kv_indices(self) -> torch.Tensor:
        global_length = self.mask.shape[1]
        return self._indices([
            torch.arange(s.batch * global_length + s.sample_start,
                         s.batch * global_length + self.local_start + s.end, device=self.mask.device)
            for s in self.segments
        ])

    @cached_property
    def _compressed_plan(self) -> list[tuple[PackedSegment, int, int]]:
        """Per-segment (segment, complete-block count, residual prefix rows)."""
        return [
            (s, *divmod(prefix, self.ratio))
            for s, prefix in zip(self.segments, self.kv_lengths)
        ]

    @cached_property
    def block_indices(self) -> torch.Tensor:
        """Global block ID for each packed row of ``compressed_keys``.

        Row i of the gathered output is block ``block_indices[i]`` of its
        batch row; the fused-indexer path maps segment-local op outputs back
        to global packed-block rows through it (the v4 ``tnd_to_block``).
        """
        parts = [
            torch.arange(s.block_start, s.block_start + count, device=self.mask.device)
            for s, count, _ in self._compressed_plan
        ]
        if parts:
            return torch.cat(parts)
        return torch.empty(0, dtype=torch.long, device=self.mask.device)

    def compressed_keys(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather visible compressed prefixes with gradients to the input keys."""
        parts, lengths, residuals = [], [], []
        for s, count, residual in self._compressed_plan:
            parts.append(keys[s.batch, s.block_start:s.block_start + count])
            lengths.append(count)
            residuals.append(residual)
        packed = torch.cat(parts) if parts else keys.reshape(-1, keys.shape[-1])[:0]
        return (packed, self._cumulative(lengths),
                torch.tensor(residuals, dtype=torch.int32, device=self.mask.device))
