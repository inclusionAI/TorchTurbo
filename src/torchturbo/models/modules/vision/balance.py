"""World-pooled media rebalancing for VL vision encoders.

By default every rank encodes its own DP shard's media: SP members recompute
the same shard, and per-shard patch counts skew the vision stage. Instead, the
whole images published by each SP group's representative (``sp_rank == 0``) are
LPT-packed over all ranks, encoded through the untouched tower, and fanned out
to every member of the owning group. The hooks ride the tower's forward, so a
balanced step needs every rank to enter the tower exactly once; training
frameworks keep that invariant by giving media-less batches a dummy image.
When the world pool holds fewer media than ranks, the packing leaves some
ranks empty-handed, and those encode a model-defined dummy whose rows no
restore range touches. Under ``torch.no_grad`` both hooks step aside and the
local path runs unchanged.

Each output's fan-out is two ``index_select`` calls around comm.py's differentiable
``all_to_all``, so backward rides the same graph: the transposed all-to-all
carries each member's gradient share to the computing rank, where index_add
sums the ``sp_size`` copies per row. Those are exactly the summands the
baseline hands to the FSDP reduce-scatter, only braced earlier, so no scale
correction is needed. ``VisionBalanceMode`` picks the LPT weight: ``patch``
counts pixel rows, ``patch_sq`` squares them for attention-dominant profiles.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchturbo.logging import get_logger
from torchturbo.parallel import VisionBalanceMode, VisionBalanceSpec

from ..comm.comm import all_to_all
from .io import BaseVisionBalanceIO, VisionBatch

logger = get_logger(__name__)


@dataclass(slots=True)
class _MediaMeta:
    """One medium of the world pool, identical on every rank after the gather."""

    src: int  # contributing representative rank
    index: int  # position in the rep's merged media order
    grid: tuple[int, int, int]
    patches: int  # pixel rows: t * h * w
    tokens: dict[str, int]  # row count in each output stream
    weight: int  # LPT packing weight
    owner: int = -1  # computing rank, filled by assign_owners

    @staticmethod
    def lpt_weight(patches: int, mode: VisionBalanceMode) -> int:
        """Packing weight of one medium under the configured balance mode."""
        if mode is VisionBalanceMode.PATCH_SQ:
            return patches * patches
        return patches


@dataclass(frozen=True, slots=True)
class _MediaPool:
    """The world's media for one step, in ``(source rank, merged index)`` order.

    Every rank rebuilds the pool from the same gathered metadata, so the
    packing lands identically everywhere without being exchanged.
    """

    media: list[_MediaMeta]

    @classmethod
    def from_metadata(
        cls,
        metadata: list[list[list[int]]],
        output_names: tuple[str, ...],
        mode: VisionBalanceMode,
    ) -> _MediaPool:
        """Flatten gathered grids and output lengths into the world pool.

        Args:
            metadata: Rows of ``(t, h, w, *output_lengths)`` per rank,
                empty for members that are not a representative.
            output_names: Output fields in the order of the length columns.
            mode: Weight used to assign each medium.
        """
        media: list[_MediaMeta] = []
        for src, rows in enumerate(metadata):
            for index, row in enumerate(rows):
                t, h, w = row[:3]
                patches = t * h * w
                media.append(
                    _MediaMeta(
                        src=src,
                        index=index,
                        grid=(t, h, w),
                        patches=patches,
                        tokens=dict(zip(output_names, row[3:], strict=True)),
                        weight=_MediaMeta.lpt_weight(patches, mode),
                    )
                )
        return cls(media)

    def assign_owners(self, world_size: int) -> None:
        """LPT packing, tied weights resolved to the lowest rank index so
        every rank reproduces the same assignment."""
        order = sorted(range(len(self.media)), key=lambda i: (-self.media[i].weight, i))
        loads = [(0, rank) for rank in range(world_size)]
        heapq.heapify(loads)
        for i in order:
            load, rank = heapq.heappop(loads)
            self.media[i].owner = rank
            heapq.heappush(loads, (load + self.media[i].weight, rank))

    def assigned_to(self, rank: int) -> list[_MediaMeta]:
        """Media this rank computes, in pooled order: the order its
        arriving pixel rows and tower output rows follow."""
        return [media for media in self.media if media.owner == rank]

    def contributed_by(self, rank: int) -> list[_MediaMeta]:
        """Media this rank published, in its merged order."""
        return [media for media in self.media if media.src == rank]

    def in_group(self, group: int, sp_size: int) -> list[_MediaMeta]:
        """One SP group's media, in its representative's merged order: the
        order the group's restored rows come back in."""
        return [media for media in self.media if media.src // sp_size == group]


@dataclass(frozen=True, slots=True)
class _PixelPlan:
    """Outbound pixel exchange, as row counts and row ranges.

    ``all_to_all_single`` fills the receive buffer source-major, so the send
    buffer is destination-major and one rank's media form contiguous ranges
    into the contributor's pixel tensor.
    """

    splits: list[int]  # pixel rows sent to each rank
    ranges: list[tuple[int, int]]  # destination-major ranges into my pixels

    @classmethod
    def build(cls, contributed: list[_MediaMeta], world_size: int) -> _PixelPlan:
        """Plan the pixel rows one representative sends to each rank.

        Args:
            contributed: the media this rank published, pixel rows contiguous
                in merged order.
            world_size: number of ranks in the exchange.
        """
        ranges_by_dest: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
        splits = [0] * world_size
        row = 0
        for media in contributed:
            end = row + media.patches
            ranges_by_dest[media.owner].append((row, end))
            splits[media.owner] += media.patches
            row = end
        return cls(splits, [span for ranges in ranges_by_dest for span in ranges])


@dataclass(frozen=True, slots=True)
class _FanoutPlan:
    """Index tables for the post hook's fan-out and the restore after it."""

    send_splits: list[int]  # output rows I send to each rank
    recv_splits: list[int]  # output rows I receive from each rank
    send_ranges: list[tuple[int, int]]  # my output rows, destination-major
    restore_ranges: list[tuple[int, int]]  # received rows, in merged order

    @classmethod
    def build(
        cls,
        mine: list[_MediaMeta],
        own_group: list[_MediaMeta],
        sp_size: int,
        world_size: int,
        output_name: str,
    ) -> _FanoutPlan:
        """Plan who gets my encoded rows, and how my received rows reorder.

        Args:
            mine: media assigned to me, in pooled order; my tower output
                holds this stream's rows in that order.
            own_group: my SP group's media, in merged order.
            sp_size: SP group size; a group is the consecutive rank block
                starting at ``group * sp_size``.
            world_size: number of ranks.
            output_name: Output field whose rows this exchange restores.

        Both directions are span walks, because ordering alone fixes the
        layout: a group's media all come from one representative, so they are
        contiguous in the pool and form one span of my output rows, which
        every member of the group consumes whole. Inbound rows arrive
        source-major, in merged order inside each segment, so one cursor per
        computing rank walks them back.
        """
        # Row span of each source group inside my output. Only a group's first
        # medium anchors the start; the rest push the end forward.
        spans: dict[int, tuple[int, int]] = {}
        row = 0
        for media in mine:
            group = media.src // sp_size
            start = spans[group][0] if group in spans else row
            spans[group] = (start, row + media.tokens[output_name])
            row += media.tokens[output_name]

        send_splits = [0] * world_size
        send_ranges: list[tuple[int, int]] = []
        for dest in range(world_size):
            span = spans.get(dest // sp_size)
            if span is None:
                continue
            send_splits[dest] = span[1] - span[0]
            send_ranges.append(span)

        recv_splits = [0] * world_size
        for media in own_group:
            recv_splits[media.owner] += media.tokens[output_name]

        # Cursor into the receive buffer, one per computing rank.
        cursors: dict[int, int] = {}
        segment_start = 0
        for src_rank, count in enumerate(recv_splits):
            cursors[src_rank] = segment_start
            segment_start += count

        restore_ranges: list[tuple[int, int]] = []
        for media in own_group:
            first = cursors[media.owner]
            restore_ranges.append((first, first + media.tokens[output_name]))
            cursors[media.owner] = first + media.tokens[output_name]

        return cls(send_splits, recv_splits, send_ranges, restore_ranges)


@dataclass(frozen=True, slots=True)
class _OutputPlan:
    """Exchange sizes and row indices for one output stream."""

    send_index: torch.Tensor
    restore_index: torch.Tensor
    fanout: _FanoutPlan


@dataclass(frozen=True, slots=True)
class _StepPlan:
    """Per-forward inputs and output plans shared by the two hooks."""

    original_batch: VisionBatch
    assigned_batch: VisionBatch
    outputs: dict[str, _OutputPlan]


class _MaterializeOutputGradients(torch.autograd.Function):
    """Keep every output exchange in backward when any stream is consumed."""

    @staticmethod
    def forward(ctx: Any, *outputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ctx.mark_non_differentiable(
            *(output for output in outputs if not output.requires_grad)
        )
        return outputs

    @staticmethod
    def backward(
        ctx: Any, *gradients: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        # PyTorch materializes unused output gradients as zeros. All streams
        # therefore enter their backward exchanges in the same graph order,
        # even when different ranks consume different output fields.
        return gradients


class _VisionBalanceState:
    """Shared configuration and per-forward state for the vision hooks."""

    def __init__(
        self,
        tower: nn.Module,
        mode: VisionBalanceMode,
        io: BaseVisionBalanceIO,
        sp_size: int,
        sp_rank: int,
    ) -> None:
        self.tower = tower
        self.mode = mode
        self.io = io
        # SP groups are consecutive rank blocks in the FSDP2 backend.
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self._plan: _StepPlan | None = None

    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not torch.is_grad_enabled():
            return None
        batch = self.io.read_inputs(module, args, kwargs)
        self._validate_batch(batch)
        plan = self._build_plan(module, batch)
        call = self.io.write_inputs(args, kwargs, plan.assigned_batch)
        self._plan = plan
        return call

    def _post_forward(
        self, module: nn.Module, args: tuple[Any, ...], output: Any
    ) -> Any:
        plan = self._plan
        if plan is None:
            return None
        self._plan = None
        features = self.io.read_outputs(output)
        if features.keys() != plan.outputs.keys():
            raise ValueError(
                "vision balance output fields do not match output_lengths"
            )
        restored: dict[str, torch.Tensor] = {}
        for name, output_plan in plan.outputs.items():
            feature = features[name]
            expected_rows = sum(plan.assigned_batch.output_lengths[name])
            if feature.size(0) != expected_rows:
                raise ValueError(
                    f"vision balance output {name!r} has {feature.size(0)} rows; "
                    f"expected {expected_rows}"
                )
            # Both index_select operations and the exchange preserve autograd.
            # Backward sums gradient shares from every member of the SP group.
            send = feature.index_select(0, output_plan.send_index)
            received = all_to_all(
                dist.group.WORLD,
                send,
                output_plan.fanout.recv_splits,
                output_plan.fanout.send_splits,
            )
            restored[name] = received.index_select(0, output_plan.restore_index)
        if len(restored) > 1:
            outputs = _MaterializeOutputGradients.apply(*restored.values())
            restored = dict(zip(restored, outputs, strict=True))
        return self.io.write_outputs(output, restored, plan.original_batch)

    @staticmethod
    def _validate_batch(batch: VisionBatch) -> None:
        """Check the row boundaries used by every subsequent exchange."""
        grid = batch.grid_thw
        if grid.ndim != 2 or grid.size(1) != 3:
            raise ValueError("vision balance requires grid_thw with shape (num_media, 3)")
        expected_rows = int(grid.prod(-1).sum())
        if batch.pixels.size(0) != expected_rows:
            raise ValueError("vision balance pixel rows do not match grid_thw")
        if not batch.output_lengths:
            raise ValueError("vision balance requires at least one output stream")
        for name, lengths in batch.output_lengths.items():
            if len(lengths) != grid.size(0) or any(length < 0 for length in lengths):
                raise ValueError(
                    f"vision balance output_lengths[{name!r}] must contain "
                    "one nonnegative row count per medium"
                )

    def _build_plan(self, module: nn.Module, batch: VisionBatch) -> _StepPlan:
        """Assign whole media and build a restore plan for each output stream."""
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = batch.pixels.device
        output_names = tuple(batch.output_lengths)
        metadata = self._gather_metadata(batch)
        pool = _MediaPool.from_metadata(metadata, output_names, self.mode)
        pool.assign_owners(world)

        mine = pool.assigned_to(rank)
        own_group = pool.in_group(rank // self.sp_size, self.sp_size)
        contributed = pool.contributed_by(rank)
        assigned_batch = self._exchange_pixels(
            module, mine, contributed, batch, world
        )
        outputs: dict[str, _OutputPlan] = {}
        for name in output_names:
            fanout = _FanoutPlan.build(mine, own_group, self.sp_size, world, name)
            outputs[name] = _OutputPlan(
                send_index=self._ranges_index(fanout.send_ranges, device),
                restore_index=self._ranges_index(fanout.restore_ranges, device),
                fanout=fanout,
            )
        logger.debug(
            "[rank %d] vision balance mode=%s pooled_media=%d assigned_media=%d",
            rank,
            self.mode.value,
            len(pool.media),
            len(mine),
        )
        return _StepPlan(batch, assigned_batch, outputs)

    def _gather_metadata(self, batch: VisionBatch) -> list[list[list[int]]]:
        """Gather grids and per-stream row counts from SP representatives.

        The first collective exchanges media counts, and the second exchanges
        rows of ``(t, h, w, *output_lengths)`` padded to the largest count.
        All ranks must declare the same output names in the same order.
        """
        world = dist.get_world_size()
        device = batch.pixels.device
        is_rep = self.sp_rank == 0
        num_local = batch.grid_thw.size(0)
        num_columns = 3 + len(batch.output_lengths)
        local_count = torch.tensor(
            [num_local if is_rep else 0], dtype=torch.long, device=device
        )
        all_counts = torch.empty(world, dtype=torch.long, device=device)
        dist.all_gather_into_tensor(all_counts, local_count, group=dist.group.WORLD)
        counts = all_counts.tolist()
        max_media = max(counts)
        if max_media == 0:
            return [[] for _ in range(world)]

        rows = torch.zeros(max_media, num_columns, dtype=torch.long, device=device)
        if is_rep and num_local:
            rows[:num_local, :3] = batch.grid_thw.to(device=device, dtype=torch.long)
            rows[:num_local, 3:] = torch.tensor(
                list(batch.output_lengths.values()), dtype=torch.long, device=device
            ).T
        gathered = torch.empty(
            world * max_media * num_columns, dtype=torch.long, device=device
        )
        dist.all_gather_into_tensor(gathered, rows.reshape(-1), group=dist.group.WORLD)
        padded = gathered.view(world, max_media, num_columns).cpu()
        return [
            padded[src, :count].tolist() if count else []
            for src, count in enumerate(counts)
        ]

    def _exchange_pixels(
        self,
        module: nn.Module,
        mine: list[_MediaMeta],
        contributed: list[_MediaMeta],
        batch: VisionBatch,
        world: int,
    ) -> VisionBatch:
        """Exchange raw patches and rebuild the assigned media description.

        Pixels are raw model inputs, so this exchange carries no autograd.
        A rank assigned no media still runs the model-defined dummy to join
        the tower's FSDP collectives and every output exchange.
        """
        send = _PixelPlan.build(contributed, world)
        pixels = batch.pixels
        device = pixels.device
        send_pixels = pixels.index_select(0, self._ranges_index(send.ranges, device))
        recv_splits = [0] * world
        for media in mine:
            recv_splits[media.src] += media.patches
        local_pixels = pixels.new_empty((sum(recv_splits), *pixels.shape[1:]))
        dist.all_to_all_single(
            local_pixels,
            send_pixels,
            output_split_sizes=recv_splits,
            input_split_sizes=send.splits,
            group=dist.group.WORLD,
        )
        if not mine:
            dummy = self.io.make_dummy(module)
            self._validate_batch(dummy)
            if tuple(dummy.output_lengths) != tuple(batch.output_lengths):
                raise ValueError("vision balance dummy must declare the same output fields")
            return dummy
        return VisionBatch(
            pixels=local_pixels,
            grid_thw=torch.tensor(
                [media.grid for media in mine], dtype=torch.long, device=device
            ),
            output_lengths={
                name: [media.tokens[name] for media in mine]
                for name in batch.output_lengths
            },
        )

    @staticmethod
    def _ranges_index(
        ranges: list[tuple[int, int]], device: torch.device
    ) -> torch.Tensor:
        """Concatenate ``(start, end)`` row ranges into one index tensor."""
        if not ranges:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(
            [torch.arange(start, end, device=device) for start, end in ranges]
        )


def register_vision_balance(
    tower: nn.Module,
    *,
    spec: VisionBalanceSpec,
    sp_size: int,
    sp_rank: int,
) -> None:
    """Register redistribution hooks using a model-owned I/O implementation.

    Every WORLD rank must enter the tower in the same order with matching
    gradient tracking state. SP groups occupy consecutive rank blocks and
    their members supply identical media. The I/O instance describes packed
    patch inputs and every output stream through ``VisionBatch``; the hooks
    distribute whole media and restore each stream to its originating group.
    Output transformations must preserve autograd. With gradient tracking
    disabled, both hooks leave the original call unchanged.

    Register after FSDP sharding so its input casting and parameter hooks run
    before these hooks. Checkpointing the tower's blocks leaves the media
    exchange outside their recomputation. A spec with no mode installs no
    hooks; enabling a mode requires an I/O instance.

    Args:
        tower: Module at the model-declared vision boundary.
        spec: Balancing mode and model I/O implementation.
        sp_size: Number of ranks in each sequence parallel group.
        sp_rank: This rank's index within its sequence parallel group.
    """
    if spec.mode is None:
        return
    if spec.io is None:
        raise ValueError("vision balance requires a model-owned I/O implementation")
    state = _VisionBalanceState(tower, spec.mode, spec.io, sp_size, sp_rank)
    # Bound hook methods retain the per-module state for the tower's lifetime.
    tower.register_forward_pre_hook(state._pre_forward, with_kwargs=True)
    tower.register_forward_hook(state._post_forward)
