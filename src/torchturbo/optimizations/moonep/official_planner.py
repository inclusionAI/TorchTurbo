# SPDX-License-Identifier: Apache-2.0
"""TorchTurbo-owned official planning shared by native and fused experts.

The caller owns the EP runtime and scratch lifetime. No MegaMoE engine, weight
bank, routing collective or model parameter is created by this module.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import torch

from hyper_multicore.moonep.placement import ExpertPlacement

_PLAN_CACHE_ENTRIES = 8
_PLAN_CACHE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class _PlanGeometry:
    """Owned immutable count geometry; token order and parameters are never cached."""

    sorted_ids: torch.Tensor
    counts: torch.Tensor
    copies: torch.Tensor
    borrowed: int
    versions: tuple[int, int, int]
    size_bytes: int
    stream: object = None
    ready: object = None

    def unchanged(self):
        return (self.sorted_ids._version == self.versions[0]
                and self.counts._version == self.versions[1]
                and self.copies._version == self.versions[2])

    def wait(self, stream):
        if self.ready is not None and stream != self.stream:
            stream.wait_event(self.ready)
            for tensor in (self.sorted_ids, self.counts, self.copies):
                tensor.record_stream(stream)


def normalize_official_plan(
    counts_storage: torch.Tensor, virtual_ids: torch.Tensor, status: torch.Tensor,
    *, tokens: int, top_k: int, experts: int, ranks: int, full_capacity: int, blocks: int,
) -> ExpertPlacement | None:
    """Normalize one completed device plan using its single validated summary.

    The native exporter writes tightly packed counts for the logical width into
    a larger allocation. Allocation shape is deliberately not a route shape.
    All outputs belong to this invocation, including the copies snapshot.
    """
    if ranks < 1 or experts % ranks or full_capacity < 0 or blocks < 1:
        raise ValueError("Invalid official planner geometry")
    local = experts // ranks
    size = ranks * full_capacity
    flag_start = (size + 7) // 8 * 8
    if status.dtype != torch.int32 or status.ndim != 1 or status.numel() < flag_start + 8 * blocks:
        raise ValueError("Official planner returned an invalid status extent")
    summary = status.cpu().tolist()
    flags = 0
    for block in range(blocks):
        flags |= summary[flag_start + 8 * block]
    rows = tuple(tuple(summary[r * full_capacity:(r + 1) * full_capacity]) for r in range(ranks))
    used = max((slot + 1 for row in rows for slot, expert in enumerate(row) if expert >= 0), default=0)
    # A no-copy plan need not publish destinations or the virtual lookup. Input
    # IDs and copy ownership must still be valid, even when we retain the router.
    if flags & (1 | 8):
        raise ValueError(f"Official route validation failed (flags={flags})")
    if not used:
        return None
    if flags:
        raise ValueError(f"Official route validation failed (flags={flags})")
    borrowed = min(local, 1 << (used - 1).bit_length())
    if used > borrowed:
        raise ValueError("Official planner exceeds the physical borrowed-expert budget")
    host_copies = tuple(row[:borrowed] + (-1,) * max(0, borrowed - len(row)) for row in rows)
    if any(expert >= experts for row in host_copies for expert in row):
        raise ValueError("Official planner returned an out-of-range expert copy")
    copies = status[:size].view(ranks, full_capacity)
    if borrowed > full_capacity:
        copies = torch.nn.functional.pad(copies, (0, borrowed - full_capacity), value=-1)
    else:
        copies = copies[:, :borrowed].contiguous()
    width = local + borrowed
    count_size = ranks * ranks * width
    if counts_storage.dtype != torch.int32 or counts_storage.ndim != 1 or counts_storage.numel() < count_size:
        raise ValueError("Official planner returned an invalid count allocation")
    if virtual_ids.dtype != torch.int32 or virtual_ids.numel() != tokens * top_k:
        raise ValueError("Official planner returned an invalid virtual route allocation")
    counts = counts_storage[:count_size].view(ranks, ranks * width)
    # Official dst = destination * NvS + row, with validated 0 <= row < NvS.
    return ExpertPlacement(virtual_ids.view(tokens, top_k), counts, copies,
                           host_copies, borrowed, tokens * top_k)


class OfficialPlanner:
    """Official algorithm plus one renderer, independent of execution backend.

    workspace is (local INT32 scratch, symmetric pointer, symmetric byte size).
    source_rows is the existing validated source-slot summary, not token capacity.
    Uniform nonempty sources are an explicit current official API capability.
    """

    def __init__(self, experts: int, top_k: int, ranks: int, rank: int, blocks: int) -> None:
        if ranks < 1 or experts < 1 or experts % ranks or not 0 <= rank < ranks:
            raise ValueError("Invalid official planner EP geometry")
        if not 0 < top_k <= experts or blocks < 1:
            raise ValueError("Invalid official planner Top-K or worker count")
        self.experts, self.top_k = experts, top_k
        self.ranks, self.rank, self.blocks = ranks, rank, blocks
        self._geometry_cache = OrderedDict()
        self._geometry_bytes = 0

    @torch.inference_mode(False)
    @torch.no_grad()
    def _geometry(self, ids, counts_by_source, scratch, pointer, symmetric_bytes):
        """Use the official allocator once for an exact validated count table."""
        from npu_patch import _C
        from hyper_multicore.moonep.render import render_expert_major

        counts_by_source = counts_by_source.contiguous()
        cumulative, copies = _C.plan_from_counts(
            ids.to(torch.int32).contiguous(), counts_by_source, scratch,
            pointer, symmetric_bytes, self.rank, self.blocks)
        needed = int((copies >= 0).sum(1).max().item())
        if not needed:
            return None
        local = self.experts // self.ranks
        if needed > local:
            raise ValueError("Official planner exceeds the physical borrowed-expert budget")
        borrowed = min(local, 1 << (needed - 1).bit_length())
        sorted_ids, counts = render_expert_major(
            counts_by_source, cumulative, copies, self.rank, borrowed, ids.numel())
        copies = copies[:, :borrowed].contiguous()
        tensors = (sorted_ids, counts, copies)
        stream = ready = None
        if ids.device.type == "npu":
            stream = torch.npu.current_stream(ids.device)
            ready = torch.npu.Event()
            ready.record(stream)
        return _PlanGeometry(sorted_ids, counts, copies, borrowed,
                             tuple(t._version for t in tensors),
                             sum(t.numel() * t.element_size() for t in tensors), stream, ready)

    def workspace_sizes(self, token_capacity: int) -> tuple[int, int]:
        from npu_patch.launchers import planning_workspace_sizes
        return planning_workspace_sizes(token_capacity, self.top_k, self.experts,
                                        token_capacity * self.top_k, self.ranks, self.blocks)

    @torch.no_grad()
    def __call__(
        self, hidden: torch.Tensor, ids: torch.Tensor, counts_by_source: torch.Tensor,
        target: int, rank: int, *, resources: tuple[torch.Tensor, int, int],
        source_rows: tuple[int, ...], source_inverse: torch.Tensor | None = None,
        source_counts: tuple[tuple[int, ...], ...] | None = None,
    ) -> ExpertPlacement | None:
        del hidden, target
        from npu_patch.launchers import compute_route_plan

        if rank != self.rank or tuple(counts_by_source.shape) != (self.ranks, self.experts):
            raise ValueError("Official planner inputs do not match the ordered EP group")
        if ids.ndim != 2 or ids.shape[1] != self.top_k or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Official routes must be integer [tokens, Top-K]")
        if len(source_rows) != self.ranks or any(not isinstance(n, int) or n < 0 for n in source_rows):
            raise ValueError("Official planner requires validated source-slot counts")
        tokens = ids.shape[0]
        if not source_rows[0] or len(set(source_rows)) != 1:
            return None
        if source_rows[self.rank] != tokens * self.top_k:
            raise ValueError("Official source-slot summary does not match the local router")
        scratch, pointer, symmetric_bytes = resources
        if not pointer or symmetric_bytes <= 0 or scratch.dtype != torch.int32 or scratch.device != ids.device:
            raise ValueError("Official planner scratch must be bound on the source device")
        local = self.experts // self.ranks
        if source_inverse is not None and source_counts is not None:
            # Only the caller's current, already validated all-rank table may key
            # reuse. Changed counts/topology/device force the official allocator.
            # Every invocation still restores its own token order below.
            if (len(source_counts) != self.ranks
                    or any(len(row) != self.experts for row in source_counts)):
                raise ValueError("Validated source counts do not match planner geometry")
            key = (ids.device, tokens, source_counts)
            geometry = self._geometry_cache.get(key)
            if geometry is not None and not geometry.unchanged():
                self._geometry_bytes -= geometry.size_bytes
                del self._geometry_cache[key]
                geometry = None
            if geometry is None:
                geometry = self._geometry(ids, counts_by_source, scratch, pointer, symmetric_bytes)
                if geometry is None:
                    return None
                if geometry.size_bytes <= _PLAN_CACHE_BYTES:
                    self._geometry_cache[key] = geometry
                    self._geometry_bytes += geometry.size_bytes
                    while (len(self._geometry_cache) > _PLAN_CACHE_ENTRIES
                           or self._geometry_bytes > _PLAN_CACHE_BYTES):
                        _, retired = self._geometry_cache.popitem(last=False)
                        self._geometry_bytes -= retired.size_bytes
            else:
                self._geometry_cache.move_to_end(key)
                if ids.device.type == "npu":
                    geometry.wait(torch.npu.current_stream(ids.device))
            virtual = geometry.sorted_ids.index_select(0, source_inverse.flatten()).reshape_as(ids)
            return ExpertPlacement(virtual, geometry.counts, geometry.copies,
                                   None, geometry.borrowed, tokens * self.top_k)
        if source_inverse is not None:
            from npu_patch import _C
            from hyper_multicore.moonep.render import render_route

            counts_by_source = counts_by_source.contiguous()
            cumulative, copies = _C.plan_from_counts(
                ids.to(torch.int32).contiguous(), counts_by_source, scratch,
                pointer, symmetric_bytes, rank, self.blocks)
            # Like PR19, read back only the execution width. Copies stay on
            # device; native all-to-all consumers retain the full host plan API.
            needed = int((copies >= 0).sum(1).max().item())
            if not needed:
                return None
            if needed > local:
                raise ValueError("Official planner exceeds the physical borrowed-expert budget")
            borrowed = min(local, 1 << (needed - 1).bit_length())
            virtual, counts = render_route(ids, source_inverse, counts_by_source,
                                          cumulative, copies, rank, borrowed)
            return ExpertPlacement(virtual, counts, copies[:, :borrowed].contiguous(),
                                   None, borrowed, tokens * self.top_k)
        counts, virtual, status = compute_route_plan(
            ids.to(torch.int32).contiguous(), scratch, pointer, symmetric_bytes,
            self.experts, self.ranks, self.rank, self.blocks)
        return normalize_official_plan(counts, virtual, status, tokens=tokens,
                                       top_k=self.top_k, experts=self.experts,
                                       ranks=self.ranks, full_capacity=local, blocks=self.blocks)


class PlanningScratch:
    @torch.inference_mode(False)
    def __init__(self, planner: OfficialPlanner, token_capacity: int, device: torch.device, *, allocator: Any) -> None:
        self.allocator = allocator
        local, self.symmetric_bytes = planner.workspace_sizes(token_capacity)
        self.local = torch.empty(local, dtype=torch.int32, device=device)
        self.source = allocator.empty((self.symmetric_bytes,), dtype=torch.uint8, alignment=512)
        self.source.zero_()
        self.pointer = self.source.data_ptr()

    def inputs(self) -> tuple[torch.Tensor, int, int]:
        if not self.pointer:
            raise RuntimeError("Bind planner scratch after initializing the official EP root")
        return self.local, self.pointer, self.symmetric_bytes

    def close(self) -> None:
        self.allocator.free(self.source)
        self.source = None
        self.pointer = 0
