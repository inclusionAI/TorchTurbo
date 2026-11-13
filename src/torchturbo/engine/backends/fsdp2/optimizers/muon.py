# The packed bucket exchange and the transfer/compute stream pipeline follow
# the approach taken by TorchTitan's DistributedMuon.

"""FSDP2 Distributed Muon construction, scheduling, and execution."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast, overload

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer

from .....logging import get_logger
from .....runtime import DeviceRuntime
from .....runtime.cpu import CpuRuntime
from .....utils.torch_utils import torch_version
from ....optimizers.muon import (
    MuonPlan,
    Owned,
    ResolvedBatchedMatrixView,
)
from ....optimizers.plan import ParamGroup
from ..config import CPUOffloadConfig, CPUOffloadMode
from .adamw import prepare_fsdp_adamw_optimizer
from .muon_layout import (
    _MatrixRoute,
    _MuonLayoutPlanner,
    _MuonParameterLayout,
    _StorageToComputePlan,
)
from .state import wrap_cpu_state

__all__ = [
    "FSDPDistributedMuon",
    "prepare_fsdp_muon_optimizers",
]

logger = get_logger(__name__)

# Default communication-to-compute cost ratio for owner assignment, in FLOP/byte.
_COLLECTIVE_FLOPS_PER_BYTE = 2000


def prepare_fsdp_muon_optimizers(
    plan: MuonPlan,
    *,
    device_runtime: DeviceRuntime,
    cpu_offload: CPUOffloadConfig,
    split_param_groups: Callable[
        [Sequence[Mapping[str, Any]]],
        list[dict[str, Any]],
    ],
) -> dict[str, Optimizer]:
    """Build the FSDP Muon and AdamW optimizer instances."""
    optimizers_by_name: dict[str, Optimizer] = {}
    for optimizer_name, groups in plan.get_param_groups_by_optimizer_name().items():
        if optimizer_name == "Muon":
            optimizers_by_name[optimizer_name] = FSDPDistributedMuon(
                plan,
                device_runtime=device_runtime,
                cpu_offload=cpu_offload,
            )
        elif optimizer_name == "AdamW":
            optimizers_by_name[optimizer_name] = prepare_fsdp_adamw_optimizer(
                groups,
                plan.adamw_kwargs,
                device_runtime=device_runtime,
                cpu_offload=cpu_offload,
                split_param_groups=split_param_groups,
            )
        else:
            raise ValueError(f"FSDP2 optimizer {optimizer_name!r} is not registered")
    return optimizers_by_name


@dataclass(slots=True, init=False)
class _FSDPStorageLayout:
    """The distributed storage layout shared by a parameter and its state."""

    global_ranks_in_mesh_order: tuple[int, ...]
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    placements: tuple[Any, ...]
    local_shape: torch.Size
    local_stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def __init__(self, fqn: str, tensor: DTensor) -> None:
        mesh = tensor.device_mesh
        mesh_names = mesh.mesh_dim_names
        if mesh.ndim != 1 or mesh_names is None or mesh_names[0] not in {"dp", "edp"}:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} requires a one-dimensional dp or edp "
                "FSDP mesh"
            )
        if len(tensor.placements) != 1 or type(tensor.placements[0]) is not Shard:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} requires exact Shard(0) storage"
            )
        placement = cast(Shard, tensor.placements[0])
        if placement.dim not in (0, -tensor.ndim):
            raise NotImplementedError(
                f"Muon parameter {fqn!r} requires Shard(0) storage"
            )

        local = tensor.to_local()
        if not local.is_contiguous():
            raise ValueError(
                f"Muon parameter {fqn!r} requires contiguous local storage"
            )
        self.global_ranks_in_mesh_order = _global_ranks_in_mesh_order(tensor)
        self.global_shape = torch.Size(tensor.shape)
        self.global_stride = tuple(tensor.stride())
        self.placements = tuple(tensor.placements)
        self.local_shape = torch.Size(local.shape)
        self.local_stride = tuple(local.stride())
        self.dtype = local.dtype
        self.device = local.device

    def matches(self, tensor: DTensor) -> bool:
        """Return whether another DTensor has the same distributed layout."""
        local = tensor.to_local()
        return (
            self.global_ranks_in_mesh_order == _global_ranks_in_mesh_order(tensor)
            and self.global_shape == tensor.shape
            and self.global_stride == tensor.stride()
            and self.placements == tuple(tensor.placements)
            and self.matches_local(local)
            and local.is_contiguous()
        )

    def matches_local(self, tensor: Tensor) -> bool:
        """Return whether a local tensor retains the captured local metadata."""
        return (
            self.local_shape == tensor.shape
            and self.local_stride == tensor.stride()
            and self.dtype == tensor.dtype
            and self.device == tensor.device
        )


@dataclass(slots=True, init=False)
class _LocalStorageIdentity:
    """Identity of the backing storage referenced by a parameter."""

    backing_data_ptr: int
    storage_offset: int

    def __init__(self, tensor: Tensor) -> None:
        self.backing_data_ptr = tensor.untyped_storage().data_ptr()
        self.storage_offset = tensor.storage_offset()

    def matches(self, tensor: Tensor) -> bool:
        return (
            self.backing_data_ptr == tensor.untyped_storage().data_ptr()
            and self.storage_offset == tensor.storage_offset()
        )


@dataclass(frozen=True, slots=True)
class _BoundMuonParameter:
    fqn: str
    param: DTensor
    group_index: int
    compute_placement: Owned | Shard
    global_compute_shape: torch.Size
    resolved_compute_view: ResolvedBatchedMatrixView | None
    compute_view_key: tuple[Any, ...]
    storage_layout: _FSDPStorageLayout
    local_storage_identity: _LocalStorageIdentity


@dataclass(slots=True, init=False)
class _MuonCommGroup:
    """Collective group with mesh indices listed in group-rank order."""

    process_group: dist.ProcessGroup
    global_ranks_in_group_order: tuple[int, ...]
    mesh_indices_in_group_order: tuple[int, ...]
    current_group_rank: int

    def __init__(
        self,
        item: _BoundMuonParameter,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        mesh = item.param.device_mesh
        if process_group is None:
            process_group = mesh.get_group()
        group_size = dist.get_world_size(process_group)
        global_ranks_in_group_order = tuple(
            dist.get_global_rank(process_group, group_rank)
            for group_rank in range(group_size)
        )
        global_ranks_in_mesh_order = (
            item.storage_layout.global_ranks_in_mesh_order
        )
        if (
            len(set(global_ranks_in_group_order)) != group_size
            or set(global_ranks_in_group_order) != set(global_ranks_in_mesh_order)
        ):
            raise RuntimeError(
                f"Muon parameter {item.fqn!r} mesh and process group do not match"
            )

        mesh_index_by_global_rank = {
            global_rank: mesh_index
            for mesh_index, global_rank in enumerate(global_ranks_in_mesh_order)
        }
        self.process_group = process_group
        self.global_ranks_in_group_order = global_ranks_in_group_order
        self.mesh_indices_in_group_order = tuple(
            mesh_index_by_global_rank[global_rank]
            for global_rank in global_ranks_in_group_order
        )
        self.current_group_rank = dist.get_rank(process_group)


@dataclass(frozen=True, slots=True)
class _TransferSpan:
    """Match a strided tensor rectangle to a contiguous exchange segment."""

    buffer_offset: int
    tensor_offset: int
    num_rows: int
    num_columns: int
    tensor_row_stride: int

    @property
    def numel(self) -> int:
        return self.num_rows * self.num_columns

    def tensor_view(self, tensor: Tensor) -> Tensor:
        """View this rectangle relative to the supplied tensor slice."""
        return tensor.as_strided(
            (self.num_rows, self.num_columns),
            (self.tensor_row_stride, 1),
            storage_offset=tensor.storage_offset() + self.tensor_offset,
        )

    def buffer_view(self, buffer: Tensor) -> Tensor:
        return buffer.narrow(0, self.buffer_offset, self.numel).view(
            self.num_rows, self.num_columns
        )


@dataclass(frozen=True, slots=True)
class _PackedTransferSchedule:
    """Per-rank packed metadata for one transfer direction."""

    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    input_spans_by_parameter: tuple[tuple[_TransferSpan, ...], ...]
    output_spans_by_parameter: tuple[tuple[_TransferSpan, ...], ...]
    has_remote_transfers: bool

    def __post_init__(self) -> None:
        if len(self.input_splits) != len(self.output_splits):
            raise RuntimeError("Muon transfer splits have different group sizes")
        if len(self.input_spans_by_parameter) != len(
            self.output_spans_by_parameter
        ):
            raise RuntimeError("Muon transfer spans have different parameter counts")
        if any(count < 0 for count in (*self.input_splits, *self.output_splits)):
            raise RuntimeError("Muon transfer splits contain a negative size")

        for direction, spans_by_parameter, buffer_numel in (
            ("input", self.input_spans_by_parameter, self.input_numel),
            ("output", self.output_spans_by_parameter, self.output_numel),
        ):
            spans = sorted(
                (
                    span
                    for parameter_spans in spans_by_parameter
                    for span in parameter_spans
                ),
                key=lambda span: span.buffer_offset,
            )
            cursor = 0
            for span in spans:
                if (
                    span.num_rows <= 0
                    or span.num_columns <= 0
                    or span.tensor_row_stride < span.num_columns
                    or span.tensor_offset < 0
                    or span.buffer_offset != cursor
                ):
                    raise RuntimeError(
                        f"Muon transfer {direction} spans do not form a packed buffer"
                    )
                cursor += span.numel
            if cursor != buffer_numel:
                raise RuntimeError(
                    f"Muon transfer {direction} spans do not match split sizes"
                )

        if not self.has_remote_transfers and self.input_numel != self.output_numel:
            raise RuntimeError("local Muon exchange has mismatched buffers")

    @property
    def input_numel(self) -> int:
        return sum(self.input_splits)

    @property
    def output_numel(self) -> int:
        return sum(self.output_splits)


@dataclass(frozen=True, slots=True)
class _CompiledMuonBucket:
    name: str
    items: tuple[_BoundMuonParameter, ...]
    local_items: tuple[_BoundMuonParameter, ...]
    local_item_compute_shapes: tuple[torch.Size, ...]
    redistributed_items: tuple[_BoundMuonParameter, ...]
    redistributed_plans: tuple[_StorageToComputePlan, ...]
    group: _MuonCommGroup | None
    forward: _PackedTransferSchedule | None
    reverse: _PackedTransferSchedule | None
    dtype: torch.dtype | None
    # Where this bucket's Muon arithmetic and buffers live.
    device: torch.device

    @property
    def has_redistribution(self) -> bool:
        return self.forward is not None

    @classmethod
    def compile(
        cls,
        *,
        name: str,
        items: tuple[_BoundMuonParameter, ...],
        parameter_plans: tuple[_StorageToComputePlan, ...],
        parameter_groups: tuple[_MuonCommGroup, ...],
        execution_device: torch.device | None = None,
    ) -> _CompiledMuonBucket:
        """Build this rank's execution metadata for one Muon bucket."""
        if not (len(items) == len(parameter_plans) == len(parameter_groups)):
            raise ValueError(
                f"Muon bucket {name!r} requires one plan and group per parameter"
            )

        local_entries = tuple(
            (item, plan, parameter_group)
            for item, plan, parameter_group in zip(
                items, parameter_plans, parameter_groups, strict=True
            )
            if not plan.requires_redistribution
        )
        redistributed_entries = tuple(
            (item, plan, parameter_group)
            for item, plan, parameter_group in zip(
                items,
                parameter_plans,
                parameter_groups,
                strict=True,
            )
            if plan.requires_redistribution
        )
        local_items = tuple(item for item, _plan, _group in local_entries)
        local_item_compute_shapes = tuple(
            plan.compute_partition(parameter_group.current_group_rank).tensor_shape
            for _item, plan, parameter_group in local_entries
        )
        # Without an explicit execution device, Muon runs where shards live.
        device = (
            items[0].storage_layout.device
            if execution_device is None
            else execution_device
        )

        if not redistributed_entries:
            return cls(
                name=name,
                items=items,
                local_items=local_items,
                local_item_compute_shapes=local_item_compute_shapes,
                redistributed_items=(),
                redistributed_plans=(),
                group=None,
                forward=None,
                reverse=None,
                dtype=None,
                device=device,
            )
        redistributed_items = tuple(
            item for item, _plan, _group in redistributed_entries
        )
        redistributed_plans = tuple(
            plan for _item, plan, _group in redistributed_entries
        )
        redistributed_groups = tuple(
            parameter_group
            for _item, _plan, parameter_group in redistributed_entries
        )
        group = redistributed_groups[0]
        for candidate in redistributed_groups[1:]:
            if (
                candidate.global_ranks_in_group_order
                != group.global_ranks_in_group_order
                or candidate.mesh_indices_in_group_order
                != group.mesh_indices_in_group_order
            ):
                raise NotImplementedError(
                    f"invalid Muon bucket schedule for {name!r}: "
                    "redistributed parameters span more than one process group"
                )

        dtype = redistributed_items[0].storage_layout.dtype
        storage_device = redistributed_items[0].storage_layout.device
        if any(
            item.storage_layout.dtype != dtype
            or item.storage_layout.device != storage_device
            for item in redistributed_items[1:]
        ):
            raise ValueError(
                f"invalid Muon bucket schedule for {name!r}: "
                "redistributed parameters mix dtype or device"
            )

        forward = cls._compile_transfer_schedule(
            name=name,
            routes_by_parameter=tuple(
                plan.storage_to_compute_routes
                for plan in redistributed_plans
            ),
            group=group,
        )
        reverse = cls._compile_transfer_schedule(
            name=name,
            routes_by_parameter=tuple(
                plan.compute_to_storage_routes
                for plan in redistributed_plans
            ),
            group=group,
        )
        if (
            forward.input_numel != reverse.output_numel
            or forward.output_numel != reverse.input_numel
        ):
            raise RuntimeError(
                f"invalid Muon bucket schedule for {name!r}: forward and "
                "reverse schedules disagree on exchange buffer sizes"
            )

        return cls(
            name=name,
            items=items,
            local_items=local_items,
            local_item_compute_shapes=local_item_compute_shapes,
            redistributed_items=redistributed_items,
            redistributed_plans=redistributed_plans,
            group=group,
            forward=forward,
            reverse=reverse,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def _compile_transfer_schedule(
        cls,
        *,
        name: str,
        routes_by_parameter: tuple[tuple[_MatrixRoute, ...], ...],
        group: _MuonCommGroup,
    ) -> _PackedTransferSchedule:
        current_group_rank = group.current_group_rank
        group_size = len(group.global_ranks_in_group_order)
        routes_by_pair: dict[
            tuple[int, int],
            list[tuple[int, _MatrixRoute]],
        ] = {}
        counts = [[0] * group_size for _ in range(group_size)]
        for parameter_index, routes in enumerate(routes_by_parameter):
            for route in routes:
                pair = (
                    route.source_group_rank,
                    route.destination_group_rank,
                )
                routes_by_pair.setdefault(pair, []).append(
                    (parameter_index, route)
                )
                counts[pair[0]][pair[1]] += route.numel

        for indexed_routes in routes_by_pair.values():
            indexed_routes.sort(
                key=lambda entry: (
                    entry[0],
                    entry[1].matrix_index,
                    entry[1].global_row_offset,
                    entry[1].global_column_offset,
                )
            )

        num_parameters = len(routes_by_parameter)
        input_spans: list[list[_TransferSpan]] = [
            [] for _ in range(num_parameters)
        ]
        input_cursor = 0
        for destination_rank in range(group_size):
            split_start = input_cursor
            for parameter_index, route in routes_by_pair.get(
                (current_group_rank, destination_rank),
                (),
            ):
                input_spans[parameter_index].append(
                    _TransferSpan(
                        buffer_offset=input_cursor,
                        tensor_offset=route.source_flat_offset,
                        num_rows=route.num_rows,
                        num_columns=route.num_columns,
                        tensor_row_stride=route.source_row_stride,
                    )
                )
                input_cursor += route.numel
            if (
                input_cursor - split_start
                != counts[current_group_rank][destination_rank]
            ):
                raise RuntimeError(
                    f"invalid Muon bucket schedule for {name!r}: input spans "
                    "do not match split sizes"
                )

        output_spans: list[list[_TransferSpan]] = [
            [] for _ in range(num_parameters)
        ]
        output_cursor = 0
        for source_rank in range(group_size):
            split_start = output_cursor
            for parameter_index, route in routes_by_pair.get(
                (source_rank, current_group_rank),
                (),
            ):
                output_spans[parameter_index].append(
                    _TransferSpan(
                        buffer_offset=output_cursor,
                        tensor_offset=route.destination_flat_offset,
                        num_rows=route.num_rows,
                        num_columns=route.num_columns,
                        tensor_row_stride=route.destination_row_stride,
                    )
                )
                output_cursor += route.numel
            if (
                output_cursor - split_start
                != counts[source_rank][current_group_rank]
            ):
                raise RuntimeError(
                    f"invalid Muon bucket schedule for {name!r}: output spans "
                    "do not match split sizes"
                )

        return _PackedTransferSchedule(
            input_splits=tuple(counts[current_group_rank]),
            output_splits=tuple(
                counts[source_rank][current_group_rank]
                for source_rank in range(group_size)
            ),
            input_spans_by_parameter=tuple(
                tuple(spans) for spans in input_spans
            ),
            output_spans_by_parameter=tuple(
                tuple(spans) for spans in output_spans
            ),
            has_remote_transfers=any(
                source_rank != destination_rank and count > 0
                for source_rank, row in enumerate(counts)
                for destination_rank, count in enumerate(row)
            ),
        )


_BufferKey = tuple[torch.device, torch.dtype]


@dataclass(slots=True)
class _MuonBuffers:
    """Reusable temporary tensors for local and redistributed Muon execution.

    "Storage" is the FSDP local-shard side, while "compute" is the tensor shape
    on which Muon runs. Exchange buffers pack a whole bucket for communication;
    scratch buffers are reshaped and reused one parameter at a time.

    Each pool is keyed by ``(device, dtype)`` and grows on demand. A workspace
    owns one instance for local-only buckets and one inside each pipeline slot.
    """

    # Bucket-wide gather input and scatter output on the FSDP-storage side.
    storage_exchange: dict[_BufferKey, Tensor] = field(default_factory=dict)
    # Bucket-wide gather output and scatter input on the matrix-compute side.
    compute_exchange: dict[_BufferKey, Tensor] = field(default_factory=dict)
    # Per-parameter workspace in the shape on which Muon computes.
    compute_scratch: dict[_BufferKey, Tensor] = field(default_factory=dict)
    # Per-parameter workspace in the local FSDP-shard shape.
    storage_scratch: dict[_BufferKey, Tensor] = field(default_factory=dict)
    # Compute streams already registered for the current tensor allocations.
    _recorded_compute_streams: set[Any] = field(default_factory=set)

    def reserve(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        device_runtime: DeviceRuntime,
        compute_stream: Any = None,
        transfer_stream: Any = None,
        storage_exchange_numel: int | None = None,
        compute_exchange_numel: int | None = None,
        compute_scratch_numel: int | None = None,
        storage_scratch_numel: int | None = None,
    ) -> None:
        """Ensure the requested buffer capacities exist for a device and dtype.

        A ``None`` size leaves that pool untouched. Existing tensors are reused
        when large enough; otherwise they are replaced by larger flat tensors.
        Buffers are allocated on the stream that first uses them.
        """
        key = (device, dtype)
        # These buffers are first used while preparing or exchanging FSDP shards.
        with device_runtime.stream(transfer_stream):
            if storage_exchange_numel is not None:
                self.storage_exchange[key] = self._reserve_tensor(
                    self.storage_exchange.get(key),
                    storage_exchange_numel,
                    dtype=dtype,
                    device=device,
                )
            if compute_exchange_numel is not None:
                self.compute_exchange[key] = self._reserve_tensor(
                    self.compute_exchange.get(key),
                    compute_exchange_numel,
                    dtype=dtype,
                    device=device,
                )
            if storage_scratch_numel is not None:
                self.storage_scratch[key] = self._reserve_tensor(
                    self.storage_scratch.get(key),
                    storage_scratch_numel,
                    dtype=dtype,
                    device=device,
                )

        # Muon computation first uses this scratch space on the compute stream.
        with device_runtime.stream(compute_stream):
            if compute_scratch_numel is not None:
                self.compute_scratch[key] = self._reserve_tensor(
                    self.compute_scratch.get(key),
                    compute_scratch_numel,
                    dtype=dtype,
                    device=device,
                )

        # Re-register after a possible reallocation so the allocator observes
        # the current tensors' use on the compute stream.
        self._recorded_compute_streams.clear()
        self.record_compute_stream(compute_stream, device_runtime=device_runtime)

    def record_compute_stream(
        self,
        stream: Any,
        *,
        device_runtime: DeviceRuntime,
    ) -> None:
        """Register pooled tensors' compute-stream use with the allocator.

        This protects tensor lifetime; events separately enforce execution
        ordering between compute and transfer streams.
        """
        if stream is None or stream in self._recorded_compute_streams:
            return
        for pool in (
            self.storage_exchange,
            self.compute_exchange,
            self.compute_scratch,
            self.storage_scratch,
        ):
            for tensor in pool.values():
                device_runtime.record_stream(tensor, stream)
        self._recorded_compute_streams.add(stream)

    def exchange_buffers(self, bucket: _CompiledMuonBucket) -> tuple[Tensor, Tensor]:
        """Return the active storage- and compute-side slices for one bucket."""
        assert bucket.forward is not None
        assert bucket.dtype is not None
        key = (bucket.device, bucket.dtype)
        storage = self.storage_exchange[key][: bucket.forward.input_numel]
        compute = self.compute_exchange[key][: bucket.forward.output_numel]
        return storage, compute

    def compute_tensor(
        self,
        shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Return a shaped view of reusable per-parameter compute scratch."""
        numel = math.prod(shape)
        return self.compute_scratch[(device, dtype)][:numel].view(shape)

    def storage_tensor(
        self,
        shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Return a shaped view of reusable local FSDP-shard scratch."""
        numel = math.prod(shape)
        return self.storage_scratch[(device, dtype)][:numel].view(shape)

    @staticmethod
    def _reserve_tensor(
        tensor: Tensor | None,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Reuse a large-enough flat tensor or replace it with a larger one."""
        if tensor is None or tensor.numel() < numel:
            return torch.empty(numel, dtype=dtype, device=device)
        return tensor


@dataclass(slots=True)
class _MuonSlot:
    """A reusable pipeline slot: temporary buffers and their execution state.

    The buffers hold one in-flight bucket, while the events track their handoff
    between the shared transfer and compute streams.
    """

    buffers: _MuonBuffers
    input_ready: Any = None
    compute_done: Any = None
    finished: Any = None
    active_bucket: _CompiledMuonBucket | None = None


class _MuonWorkspace:
    """Own reusable buffers and optional pipeline state for Muon execution.

    Local-only buckets share ``local_buffers``. Redistributed buckets use
    one reusable slot, or two slots when asynchronous
    communication and computation overlap. Each slot combines temporary buffers
    with the events that track their execution state.

    ``configure()`` derives this runtime layout from compiled buckets and grows
    existing allocations when necessary; it does not execute communication or
    Muon.
    """

    def __init__(
        self,
        device_runtime: DeviceRuntime,
        *,
        overlap_comm_compute: bool,
    ) -> None:
        if not isinstance(device_runtime, DeviceRuntime):
            raise TypeError("device_runtime must be a DeviceRuntime")
        self.device_runtime = device_runtime
        # Async execution is a device capability; overlap is a user policy that
        # determines whether the asynchronous path needs one slot or two.
        self.overlap_comm_compute = bool(overlap_comm_compute)
        self.async_enabled = device_runtime.supports_async_streams
        # The transfer stream and slots are created lazily for redistribution.
        self.transfer_stream: Any = None
        self._compute_stream: Any = None
        self.slots: tuple[_MuonSlot, ...] = ()
        # Local-only buckets do not participate in the slot pipeline.
        self.local_buffers = _MuonBuffers()
        # One entry per compiled bucket; None denotes a local-only bucket.
        self.slot_index_by_bucket: tuple[int | None, ...] = ()
        # Whether buffers cover the buckets last passed to configure().
        self.configured = False

    @property
    def num_slots(self) -> int:
        """Return the pipeline slots used by redistributed buckets."""
        return 2 if self.async_enabled and self.overlap_comm_compute else 1

    @staticmethod
    def _include_bucket_capacities(
        bucket: _CompiledMuonBucket,
        capacities: dict[_BufferKey, dict[str, int]],
    ) -> None:
        for item, shape in zip(
            bucket.local_items,
            bucket.local_item_compute_shapes,
            strict=True,
        ):
            key = (bucket.device, item.storage_layout.dtype)
            required = capacities.setdefault(key, {})
            required["compute_scratch_numel"] = max(
                required.get("compute_scratch_numel", 0),
                math.prod(shape),
            )

        if not bucket.has_redistribution:
            return

        assert bucket.group is not None
        assert bucket.forward is not None
        assert bucket.dtype is not None
        rank = bucket.group.current_group_rank
        required = capacities.setdefault((bucket.device, bucket.dtype), {})
        sizes = {
            "storage_exchange_numel": bucket.forward.input_numel,
            "compute_exchange_numel": bucket.forward.output_numel,
            "storage_scratch_numel": max(
                plan.storage_partition(rank).numel
                for plan in bucket.redistributed_plans
            ),
            "compute_scratch_numel": max(
                plan.compute_partition(rank).numel
                for plan in bucket.redistributed_plans
            ),
        }
        for name, numel in sizes.items():
            required[name] = max(required.get(name, 0), numel)

    @staticmethod
    def _plan_capacities(
        buckets: Sequence[_CompiledMuonBucket],
        num_slots: int,
    ) -> tuple[
        dict[_BufferKey, dict[str, int]],
        tuple[dict[_BufferKey, dict[str, int]], ...],
        tuple[int | None, ...],
    ]:
        """Return local and per-slot buffer capacities and each bucket's slot.

        Capacities are aggregated before allocating: a pool grows once per
        device and dtype, even when many differently sized buckets share its
        slot. Only redistributed buckets rotate through the slots; local-only
        buckets neither use a slot nor advance the rotation.
        """
        local_capacities: dict[_BufferKey, dict[str, int]] = {}
        slot_capacities: tuple[dict[_BufferKey, dict[str, int]], ...] = tuple(
            {} for _ in range(num_slots)
        )
        slot_indices: list[int | None] = []
        redistributed_bucket_index = 0
        for bucket in buckets:
            if not bucket.has_redistribution:
                slot_indices.append(None)
                _MuonWorkspace._include_bucket_capacities(bucket, local_capacities)
                continue

            slot_index = redistributed_bucket_index % num_slots
            slot_indices.append(slot_index)
            redistributed_bucket_index += 1
            _MuonWorkspace._include_bucket_capacities(
                bucket,
                slot_capacities[slot_index],
            )
        return local_capacities, slot_capacities, tuple(slot_indices)

    def configure(self, buckets: Sequence[_CompiledMuonBucket]) -> None:
        """Prepare reusable runtime resources for the compiled buckets.

        Streams and slots are created lazily when redistribution is needed.
        Local-only buckets use ``local_buffers``; other buckets rotate through
        pipeline slots whose buffers grow to cover every bucket assigned to them.
        Existing streams, slots, and sufficiently large tensors are reused.
        """
        # A scheduled bucket requires the communication path on every rank,
        # even when this rank has no complete compute tensors.
        has_redistribution = any(bucket.has_redistribution for bucket in buckets)
        compute_stream = (
            self.device_runtime.current_stream() if self.async_enabled else None
        )
        if self.async_enabled and compute_stream is None:
            raise RuntimeError(
                f"{type(self.device_runtime).__name__} did not provide a current stream"
            )
        if (
            compute_stream is not None
            and self._compute_stream is not None
            and compute_stream != self._compute_stream
        ):
            compute_stream.wait_stream(self._compute_stream)

        if has_redistribution and not self.slots:
            if self.async_enabled:
                self.transfer_stream = self.device_runtime.new_stream(priority=0)
                if self.transfer_stream is None:
                    raise RuntimeError(
                        f"{type(self.device_runtime).__name__} did not create a "
                        "transfer stream"
                    )
                # One slot serializes transfers; two slots double-buffer the
                # next bucket's communication while the current bucket computes.
                self.slots = tuple(
                    _MuonSlot(
                        buffers=_MuonBuffers(),
                        input_ready=self.device_runtime.new_event(),
                        compute_done=self.device_runtime.new_event(),
                        finished=self.device_runtime.new_event(),
                    )
                    for _ in range(self.num_slots)
                )
            else:
                self.slots = (_MuonSlot(buffers=_MuonBuffers()),)

        local_capacities, slot_capacities, slot_indices = self._plan_capacities(
            buckets, len(self.slots)
        )
        for (device, dtype), required in local_capacities.items():
            self.local_buffers.reserve(
                device=device,
                dtype=dtype,
                device_runtime=self.device_runtime,
                compute_stream=compute_stream,
                **required,
            )
        for slot, capacities in zip(self.slots, slot_capacities, strict=True):
            for (device, dtype), required in capacities.items():
                slot.buffers.reserve(
                    device=device,
                    dtype=dtype,
                    device_runtime=self.device_runtime,
                    compute_stream=compute_stream,
                    transfer_stream=self.transfer_stream,
                    **required,
                )

        self.slot_index_by_bucket = slot_indices
        self._compute_stream = compute_stream
        if self.async_enabled and self.slots:
            assert self.transfer_stream is not None
            for slot in self.slots:
                # Prime each event on the stream that produces it at runtime.
                slot.input_ready.record(self.transfer_stream)
                slot.compute_done.record(compute_stream)
                slot.finished.record(self.transfer_stream)
        self.configured = True

    def record_compute_stream(self, stream: Any) -> None:
        """Register all workspace buffers with the current compute stream.

        This stream may differ from the one observed during ``configure()``.
        A changed caller waits for the previous compute stream before reusing
        scratch. Allocator registration separately protects tensor lifetime.
        """
        if (
            stream is not None
            and self._compute_stream is not None
            and stream != self._compute_stream
        ):
            stream.wait_stream(self._compute_stream)
        self._compute_stream = stream
        self.local_buffers.record_compute_stream(
            stream,
            device_runtime=self.device_runtime,
        )
        for slot in self.slots:
            slot.buffers.record_compute_stream(
                stream,
                device_runtime=self.device_runtime,
            )

    def slot_for_bucket(self, bucket_index: int) -> _MuonSlot:
        """Return the pipeline slot assigned to a redistributed bucket."""
        slot_index = self.slot_index_by_bucket[bucket_index]
        if slot_index is None:
            raise RuntimeError("local Muon bucket does not have a pipeline slot")
        return self.slots[slot_index]

    def release(self) -> None:
        """Wait for all users before releasing buffers and their stream records."""
        self.device_runtime.synchronize()
        self.local_buffers = _MuonBuffers()
        self.slots = ()
        self.slot_index_by_bucket = ()
        self.transfer_stream = None
        self._compute_stream = None
        self.configured = False

    def required_bytes(self, buckets: Sequence[_CompiledMuonBucket]) -> int:
        """Estimate buffers plus concurrently live staging and matrix temporaries."""
        local_capacities, slot_capacities, _ = self._plan_capacities(
            buckets, self.num_slots
        )
        buffers = sum(
            sum(sizes.values()) * dtype.itemsize
            for capacities in (local_capacities, *slot_capacities)
            for (_device, dtype), sizes in capacities.items()
        )
        max_staging = 0
        max_compute = 0
        for bucket in buckets:
            shapes = list(bucket.local_item_compute_shapes)
            if bucket.group is not None:
                shapes.extend(
                    plan.compute_partition(bucket.group.current_group_rank).tensor_shape
                    for plan in bucket.redistributed_plans
                )
            for shape in shapes:
                rows, columns = shape[-2:]
                matrices = math.prod(shape[:-2])
                # BF16 iterates and Gram matrices can coexist across addmm calls.
                required = (
                    6 * matrices * (rows * columns + min(rows, columns) ** 2)
                    + 4 * matrices
                )
                max_compute = max(max_compute, required)
            for item in bucket.items:
                max_staging = max(
                    max_staging,
                    2
                    * math.prod(item.storage_layout.local_shape)
                    * item.param.element_size(),
                )
        # The transfer stream may stage while the caller runs a local item.
        return buffers + max_compute + self.num_slots * max_staging


def _global_ranks_in_mesh_order(tensor: DTensor) -> tuple[int, ...]:
    return tuple(int(rank) for rank in tensor.device_mesh.mesh.flatten().tolist())


def _estimate_muon_compute_cost(shape: torch.Size, ns_steps: int) -> int:
    """Estimate Newton-Schulz GEMM FLOPs with two FLOPs per multiply-add."""
    rows, columns = shape
    short_dim, long_dim = sorted((rows, columns))
    return 2 * ns_steps * short_dim * short_dim * (2 * long_dim + short_dim)


@dataclass(slots=True)
class _MuonRankLoad:
    """Matrix-multiply FLOPs and forward traffic assigned to one group rank."""

    compute: int = 0
    send_bytes: int = 0
    receive_bytes: int = 0


@dataclass(slots=True)
class _MuonGroupSummary:
    """Transient plan totals reported by one representative of each group."""

    current_group_rank: int
    num_buckets: int = 0
    num_matrices: int = 0
    num_refined_buckets: int = 0
    assignment_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _MuonMatrixTask:
    """An indivisible matrix and its storage bytes on each group rank."""

    parameter_index: int
    matrix_index: int
    compute: int
    num_bytes: int
    held_bytes: tuple[int, ...]


class _MuonOwnerAssignment:
    """A candidate assignment with independent bucket and cumulative loads."""

    def __init__(self, previous: Sequence[_MuonRankLoad]) -> None:
        self.owners: list[int] = []
        self.loads = [_MuonRankLoad() for _ in previous]
        self.cumulative = [
            _MuonRankLoad(load.compute, load.send_bytes, load.receive_bytes)
            for load in previous
        ]

    def update(self, task: _MuonMatrixTask, owner: int, *, sign: int) -> None:
        """Add or remove a matrix's complete compute and forward traffic."""
        remote_bytes = task.num_bytes - task.held_bytes[owner]
        for loads in (self.loads, self.cumulative):
            loads[owner].compute += sign * task.compute
            loads[owner].receive_bytes += sign * remote_bytes
            for rank, held in enumerate(task.held_bytes):
                if rank != owner:
                    loads[rank].send_bytes += sign * held

    def choices(self, task: _MuonMatrixTask) -> list[tuple[int, int, int, int]]:
        """Score every owner, with the matrix absent from the current loads.

        Other matrices' total remote bytes are constant across choices, so
        only this matrix's remote bytes are needed for the second score key.
        """
        sender_peaks = []
        receiver_peaks = []
        compute_peaks = []
        for loads in (self.loads, self.cumulative):
            # Two largest send totals exclude the candidate's own storage
            # contribution in O(1), rather than scanning peers per candidate.
            sender_peaks.append(
                heapq.nlargest(
                    2,
                    (
                        (load.send_bytes + held, rank)
                        for rank, (load, held) in enumerate(
                            zip(loads, task.held_bytes, strict=True)
                        )
                    ),
                )
            )
            receiver_peaks.append(max(load.receive_bytes for load in loads))
            compute_peaks.append(max(load.compute for load in loads))

        choices = []
        for rank, local_bytes in enumerate(task.held_bytes):
            remote_bytes = task.num_bytes - local_bytes
            costs = []
            for loads, senders, receive_peak, compute_peak in zip(
                (self.loads, self.cumulative),
                sender_peaks,
                receiver_peaks,
                compute_peaks,
                strict=True,
            ):
                if senders[0][1] != rank:
                    other_send_peak = senders[0][0]
                else:
                    other_send_peak = senders[1][0] if len(senders) > 1 else 0
                traffic_peak = max(
                    other_send_peak,
                    loads[rank].send_bytes,
                    receive_peak,
                    loads[rank].receive_bytes + remote_bytes,
                )
                projected_compute_peak = max(
                    compute_peak, loads[rank].compute + task.compute
                )
                costs.append(
                    projected_compute_peak
                    + 2 * _COLLECTIVE_FLOPS_PER_BYTE * traffic_peak
                )
            bucket_cost, cumulative_cost = costs
            choices.append((bucket_cost, 2 * remote_bytes, cumulative_cost, rank))
        return choices

    def score(self) -> tuple[int, int, int, tuple[int, ...]]:
        """Compare complete assignments, including total remote traffic."""
        costs = []
        for loads in (self.loads, self.cumulative):
            compute_peak = max(load.compute for load in loads)
            traffic_peak = max(
                max(load.send_bytes, load.receive_bytes) for load in loads
            )
            costs.append(
                compute_peak + 2 * _COLLECTIVE_FLOPS_PER_BYTE * traffic_peak
            )
        remote_bytes = sum(load.receive_bytes for load in self.loads)
        return costs[0], 2 * remote_bytes, costs[1], tuple(self.owners)


class FSDPDistributedMuon(Optimizer):
    """Muon with complete-matrix compute on one-dimensional FSDP storage."""

    def __init__(
        self,
        plan: MuonPlan,
        *,
        device_runtime: DeviceRuntime,
        cpu_offload: CPUOffloadConfig | None = None,
    ) -> None:
        # PyTorch 2.9 may replace an FSDP parameter's local storage when
        # reset_sharded_param() re-pads an uneven shard. This optimizer caches
        # local storage identities, so it requires the idempotent reset from 2.10.
        if torch_version() < (2, 10, 0):
            raise RuntimeError("FSDPDistributedMuon requires PyTorch 2.10 or newer")
        if not isinstance(plan, MuonPlan):
            raise TypeError("plan must be a MuonPlan")
        muon_groups = plan.get_param_groups_by_optimizer_name().get("Muon", ())
        if not muon_groups:
            raise ValueError("FSDPDistributedMuon requires a Muon parameter group")

        prepared_groups = [
            group.get_optimizer_param_group() for group in muon_groups
        ]

        defaults = dict(plan.muon_kwargs)
        self._param_groups_frozen = False
        self._first_step_validated = False
        self._failed = False
        super().__init__(prepared_groups, defaults)
        self._validate_groups()
        self._param_groups_frozen = True

        self._bucket_definitions = tuple(plan.buckets)
        self._parameter_specs = dict(plan.parameter_specs)
        self._offload = cpu_offload or CPUOffloadConfig(enabled=False)
        self._device_runtime = device_runtime
        self._checkpoint_meshes: dict[DeviceMesh, DeviceMesh] = {}
        self._bound_parameters = self._bind_parameters(muon_groups)
        parameter_device = self._bound_parameters[0].storage_layout.device
        full_offload = (
            self._offload.enabled and self._offload.mode is CPUOffloadMode.FULL
        )
        expected_device = "cpu" if full_offload else device_runtime.device.type
        if parameter_device.type != expected_device:
            raise ValueError(
                f"Muon requires parameter storage on {expected_device!r} for "
                f"the configured offload policy, got {parameter_device.type!r}"
            )
        cpu_execution = (
            self._offload.enabled and self._offload.optimizer_memory_fraction == 0
        )
        execution_runtime = CpuRuntime() if cpu_execution else device_runtime
        # Muon runs on the shards' own device when its type matches; full
        # offload instead stages CPU shards on the training device.
        self._execution_device = (
            parameter_device
            if parameter_device.type == execution_runtime.device.type
            else execution_runtime.device
        )
        # Offloaded device execution checks its budget on acquisition and keeps
        # buffers after successful steps; explicit release still frees them.
        self._step_scoped_workspace = (
            self._offload.enabled and self._execution_device.type != "cpu"
        )
        self._cpu_groups = self._prepare_cpu_groups() if cpu_execution else {}
        self._compiled_buckets = self._compile_buckets(self._bound_parameters)
        self._validate_compiled_buckets_across_ranks(self._compiled_buckets)
        self._workspace = _MuonWorkspace(
            execution_runtime,
            overlap_comm_compute=plan.overlap_comm_compute,
        )
        if not self._step_scoped_workspace:
            self._workspace.configure(self._compiled_buckets)
        # Like AdamWSwap, materialize state up front so DCP finds it initialized
        # and never runs a zero-gradient step to create it.
        self._initialize_state()
        # Keep the registered callback unbound; it forwards to the instance
        # method without creating a self -> hook -> bound-self reference cycle.
        self.register_load_state_dict_post_hook(
            _load_state_dict_post_hook,
            prepend=True,
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if self._failed:
            raise RuntimeError(
                "FSDPDistributedMuon cannot retry a failed update; rebuild the "
                "optimizer and restore model and optimizer state from a checkpoint"
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._preflight_step()
        self._acquire_workspace()
        try:
            if self._workspace.async_enabled and any(
                bucket.has_redistribution for bucket in self._compiled_buckets
            ):
                self._run_async()
            else:
                self._run_synchronous()
        except BaseException:
            # Momentum or parameters may already be updated. Retrying cannot
            # safely resume this step, even if queued streams have been joined.
            self._failed = True
            if self._step_scoped_workspace:
                self.release_workspace()
            raise
        return loss

    def release_workspace(self) -> None:
        """Release execution buffers; the next step acquires them again."""
        self._workspace.release()

    def _acquire_workspace(self) -> None:
        """Allocate missing execution buffers with one decision across ranks.

        When any rank lacks budget or memory, every rank fails before changing
        momentum or parameters, so no rank enters Muon collectives alone.
        """
        if self._workspace.configured:
            return
        # Allocate the status first so it exists even when the workspace
        # allocation itself runs out of memory.
        status = torch.ones((), dtype=torch.int32, device=self._device_runtime.device)
        error = None
        if self._step_scoped_workspace:
            error = self._workspace_budget_error()
        if error is None:
            try:
                self._workspace.configure(self._compiled_buckets)
            except RuntimeError as allocation_error:
                # Its frames would keep the partially allocated buffers alive.
                error = allocation_error.with_traceback(None)
        if error is not None:
            status.zero_()
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if not status.item():
            self.release_workspace()
            raise RuntimeError(
                "Muon workspace acquisition failed on a training rank before any "
                "update; release device memory, reduce Muon bucket size or "
                "overlap, or select CPU execution"
            ) from error

    def _workspace_budget_error(self) -> RuntimeError | None:
        """Compare the planned workspace with this rank's offload budget.

        Freed activations usually stay cached by the device allocator after
        backward and can serve the workspace, so the budget counts them in
        addition to free driver memory.
        """
        runtime = self._device_runtime
        free, _ = runtime.memory_info()
        cached = runtime.memory_reserved() - runtime.memory_allocated()
        fraction = self._offload.optimizer_memory_fraction
        budget = int((free + cached) * fraction)
        required = self._workspace.required_bytes(self._compiled_buckets)
        if required <= budget:
            return None
        largest = max(
            self._bound_parameters,
            key=lambda item: math.prod(item.global_compute_shape),
        )
        return RuntimeError(
            f"Muon workspace requires {required} bytes, but "
            f"optimizer_memory_fraction={fraction} allows {budget} bytes; "
            f"largest parameter={largest.fqn!r}"
        )

    def _prepare_cpu_groups(self) -> dict[tuple[int, ...], dist.ProcessGroup]:
        """Reuse CPU-capable groups or create matching Gloo groups collectively."""
        groups = {}
        missing = set()
        for item in self._bound_parameters:
            group = item.param.device_mesh.get_group()
            ranks = tuple(sorted(item.storage_layout.global_ranks_in_mesh_order))
            if "cpu:" in dist.get_backend_config(group):
                groups[ranks] = group
            else:
                missing.add(ranks)
        requested: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(requested, sorted(missing))
        all_missing = sorted({ranks for entries in requested for ranks in entries})
        if all_missing and not dist.is_gloo_available():
            raise RuntimeError("CPU Muon execution requires a Gloo backend")
        for ranks in all_missing:
            group = dist.new_group(list(ranks), backend="gloo")
            if dist.get_rank() in ranks:
                groups[ranks] = group
        return groups

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._param_groups_frozen:
            raise RuntimeError("FSDPDistributedMuon parameter groups are frozen")
        super().add_param_group(param_group)

    def _after_load_state_dict(self) -> None:
        """Refresh state derived from checkpointed parameter-group options.

        ``Optimizer.load_state_dict`` replaces the group dictionaries, and
        restored values such as ``ns_steps`` participate in owner balancing.
        """
        self._validate_groups()
        self._compiled_buckets = self._compile_buckets(self._bound_parameters)
        self._validate_compiled_buckets_across_ranks(self._compiled_buckets)
        if self._step_scoped_workspace:
            self.release_workspace()
        else:
            self._workspace.configure(self._compiled_buckets)
        self._first_step_validated = False

    def _validate_ns_coefficients(self, coefficients: Any, ns_steps: int) -> bool:
        """Whether Newton-Schulz coefficients are a flat triple or per-step triples.
        Per-step triples must provide exactly one triple per ``ns_steps`` iteration.
        """
        if not isinstance(coefficients, (tuple, list)) or not coefficients:
            return False
        if not any(isinstance(entry, (tuple, list)) for entry in coefficients):
            return len(coefficients) == 3 and all(
                isinstance(value, (int, float)) for value in coefficients
            )
        return len(coefficients) == ns_steps and all(
            isinstance(entry, (tuple, list))
            and len(entry) == 3
            and all(isinstance(value, (int, float)) for value in entry)
            for entry in coefficients
        )

    def _validate_groups(self) -> None:
        if len(self.param_groups) != 1:
            raise ValueError("FSDPDistributedMuon requires exactly one parameter group")
        group = self.param_groups[0]
        ns_steps = group["ns_steps"]
        coefficients = group["ns_coefficients"]
        if (
            group.get("fused")
            or group.get("foreach")
            or any(
                not 0 <= group[name]
                for name in ("lr", "weight_decay", "momentum", "eps")
            )
            or not isinstance(ns_steps, int)
            or not 0 <= ns_steps < 100
            or not self._validate_ns_coefficients(coefficients, ns_steps)
            or group["adjust_lr_fn"]
            not in (None, "original", "match_rms_adamw", "spectral_unclamped")
        ):
            raise ValueError("unsupported FSDPDistributedMuon parameter group")

    def _bind_parameters(
        self,
        param_groups: Sequence[ParamGroup],
    ) -> tuple[_BoundMuonParameter, ...]:
        bound_parameters = []
        seen_names: set[str] = set()
        seen_parameters: set[int] = set()
        devices: set[torch.device] = set()
        for group_index, group in enumerate(param_groups):
            for fqn, param in group.params_by_fqn.items():
                if fqn in seen_names or id(param) in seen_parameters:
                    raise ValueError(f"duplicate Muon parameter {fqn!r}")
                seen_names.add(fqn)
                seen_parameters.add(id(param))
                if fqn not in self._parameter_specs:
                    raise ValueError(f"missing Muon parameter spec for {fqn!r}")
                if not isinstance(param, DTensor):
                    raise TypeError("FSDPDistributedMuon requires DTensor parameters")
                if param.ndim != 2:
                    raise ValueError(
                        f"Muon parameter {fqn!r} storage shape must be two-dimensional"
                    )
                storage_layout = _FSDPStorageLayout(fqn, param)
                local_tensor = param.to_local()
                local_storage_identity = _LocalStorageIdentity(local_tensor)
                devices.add(storage_layout.device)
                spec = self._parameter_specs[fqn]
                resolved_view = (
                    None
                    if spec.compute_view is None
                    else spec.compute_view.resolve(torch.Size(param.shape))
                )
                if resolved_view is not None:
                    global_compute_shape = resolved_view.global_compute_shape
                    compute_view_key = resolved_view.signature
                else:
                    global_compute_shape = torch.Size(param.shape)
                    compute_view_key = ("identity",)

                if type(spec.compute_placement) not in (Owned, Shard):
                    raise TypeError(
                        f"Muon parameter {fqn!r} has an unsupported compute placement"
                    )

                bound_parameters.append(
                    _BoundMuonParameter(
                        fqn=fqn,
                        param=param,
                        group_index=group_index,
                        compute_placement=spec.compute_placement,
                        global_compute_shape=global_compute_shape,
                        resolved_compute_view=resolved_view,
                        compute_view_key=compute_view_key,
                        storage_layout=storage_layout,
                        local_storage_identity=local_storage_identity,
                    )
                )

        if seen_names != set(self._parameter_specs):
            missing = sorted(set(self._parameter_specs) - seen_names)
            raise ValueError(f"Muon parameter group omitted specs: {missing[:8]}")
        if len(devices) != 1:
            raise ValueError(
                "FSDPDistributedMuon requires one local tensor device per process"
            )
        return tuple(bound_parameters)

    def _assign_matrix_owners(
        self,
        items: tuple[_BoundMuonParameter, ...],
        layouts: tuple[_MuonParameterLayout, ...],
        parameter_groups: tuple[_MuonCommGroup, ...],
        cumulative_loads_by_group: dict[tuple[int, ...], list[_MuonRankLoad]],
        *,
        summaries_by_group: dict[tuple[int, ...], _MuonGroupSummary] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Compare greedy placement with refinement of a complete local seed.

        Each communication group is planned against its own prior loads.
        Only the winning candidate contributes to subsequent buckets.
        """
        assignments = [[] for _ in items]
        tasks_by_group: dict[tuple[int, ...], list[_MuonMatrixTask]] = {}
        for parameter_index, (item, layout, group) in enumerate(
            zip(items, layouts, parameter_groups, strict=True)
        ):
            group_key = group.global_ranks_in_group_order
            tasks = tasks_by_group.setdefault(group_key, [])
            assignments[parameter_index] = [0] * len(layout.matrices)
            ns_steps = self._group(item)["ns_steps"]
            element_size = item.param.element_size()
            for matrix in layout.matrices:
                matrix_end = matrix.global_row_offset + matrix.num_rows
                held_bytes = []
                for partition in layout.storage_partitions:
                    row_start = max(
                        matrix.global_row_offset, partition.global_row_offset
                    )
                    row_end = min(matrix_end, partition.global_row_end)
                    local_rows = max(0, row_end - row_start)
                    held_bytes.append(local_rows * matrix.num_columns * element_size)
                tasks.append(
                    _MuonMatrixTask(
                        parameter_index=parameter_index,
                        matrix_index=matrix.matrix_index,
                        compute=_estimate_muon_compute_cost(
                            matrix.tensor_shape, ns_steps
                        ),
                        num_bytes=matrix.numel * element_size,
                        held_bytes=tuple(held_bytes),
                    )
                )

        for group_key, tasks in tasks_by_group.items():
            assignment_start = perf_counter()
            tasks.sort(
                key=lambda task: (
                    -task.compute,
                    -task.num_bytes,
                    items[task.parameter_index].fqn,
                    task.matrix_index,
                )
            )
            if group_key not in cumulative_loads_by_group:
                cumulative_loads_by_group[group_key] = [
                    _MuonRankLoad() for _ in group_key
                ]
            previous = cumulative_loads_by_group[group_key]
            greedy = _MuonOwnerAssignment(previous)
            local = _MuonOwnerAssignment(previous)
            for task in tasks:
                owner = min(greedy.choices(task))[-1]
                greedy.owners.append(owner)
                greedy.update(task, owner, sign=1)

                # Reserve every matrix's preferred storage location before
                # refinement, including tasks visited later in the pass.
                owner = min(
                    local.choices(task),
                    key=lambda choice: (-task.held_bytes[choice[-1]], choice),
                )[-1]
                local.owners.append(owner)
                local.update(task, owner, sign=1)

            # One bounded pass evaluates moves against the complete bucket.
            # The old owner remains a candidate, so each replacement improves
            # the full lexicographic score or leaves the assignment unchanged.
            # Retaining greedy also covers some improvements requiring joint moves
            # that a single-matrix pass cannot reach from the local seed.
            for task_index, task in enumerate(tasks):
                local.update(task, local.owners[task_index], sign=-1)
                owner = min(local.choices(task))[-1]
                local.owners[task_index] = owner
                local.update(task, owner, sign=1)

            winner = min((greedy, local), key=lambda candidate: candidate.score())
            assignment_seconds = perf_counter() - assignment_start
            if summaries_by_group is not None:
                group = parameter_groups[tasks[0].parameter_index]
                summary = summaries_by_group.setdefault(
                    group_key, _MuonGroupSummary(group.current_group_rank)
                )
                summary.num_buckets += 1
                summary.num_matrices += len(tasks)
                summary.num_refined_buckets += int(winner is local)
                summary.assignment_seconds += assignment_seconds
            for task, owner in zip(tasks, winner.owners, strict=True):
                assignments[task.parameter_index][task.matrix_index] = owner
            for total, added in zip(previous, winner.loads, strict=True):
                total.compute += added.compute
                total.send_bytes += added.send_bytes
                total.receive_bytes += added.receive_bytes
        return tuple(tuple(owners) for owners in assignments)

    def _compile_buckets(
        self,
        bound_parameters: tuple[_BoundMuonParameter, ...],
    ) -> tuple[_CompiledMuonBucket, ...]:
        """Compile logical bucket definitions into execution-ready buckets.

        Assign all logical matrices before resolving storage-to-compute plans.
        Identity mappings remain local; other items share packed schedules.
        This stage only builds metadata and does not execute collectives.
        """
        # Resolve bucket FQNs and require every bound parameter exactly once.
        by_fqn = {item.fqn: item for item in bound_parameters}
        bucket_fqns = tuple(
            fqn for bucket in self._bucket_definitions for fqn in bucket.parameter_fqns
        )
        if len(bucket_fqns) != len(set(bucket_fqns)) or set(bucket_fqns) != set(by_fqn):
            raise ValueError(
                "Muon buckets must contain every optimizer parameter exactly once"
            )

        cumulative_loads_by_group: dict[tuple[int, ...], list[_MuonRankLoad]] = {}
        summaries_by_group: dict[tuple[int, ...], _MuonGroupSummary] = {}
        compiled_buckets = []
        for bucket in self._bucket_definitions:
            items = tuple(by_fqn[fqn] for fqn in bucket.parameter_fqns)
            parameter_groups = tuple(
                _MuonCommGroup(
                    item,
                    self._cpu_groups.get(
                        tuple(sorted(item.storage_layout.global_ranks_in_mesh_order))
                    ),
                )
                for item in items
            )
            planners = tuple(
                _MuonLayoutPlanner(
                    parameter_group.mesh_indices_in_group_order,
                    parameter_group.current_group_rank,
                )
                for parameter_group in parameter_groups
            )
            layouts = tuple(
                planner.resolve_parameter(
                    fqn=item.fqn,
                    global_storage_shape=item.storage_layout.global_shape,
                    local_storage_shape=item.storage_layout.local_shape,
                    compute_view=item.resolved_compute_view,
                )
                for item, planner in zip(items, planners, strict=True)
            )
            matrix_owners = self._assign_matrix_owners(
                items,
                layouts,
                parameter_groups,
                cumulative_loads_by_group,
                summaries_by_group=summaries_by_group,
            )
            parameter_plans = tuple(
                planner.build_plan(layout, owners)
                for planner, layout, owners in zip(
                    planners, layouts, matrix_owners, strict=True
                )
            )
            compiled_buckets.append(
                _CompiledMuonBucket.compile(
                    name=bucket.name,
                    items=items,
                    parameter_plans=parameter_plans,
                    parameter_groups=parameter_groups,
                    execution_device=self._execution_device,
                )
            )
        for group_ranks, summary in summaries_by_group.items():
            if summary.current_group_rank != 0:
                continue
            remote_bytes = sum(
                load.receive_bytes for load in cumulative_loads_by_group[group_ranks]
            )
            logger.info(
                "Muon plan: group_leader=%d ranks=%d buckets=%d matrices=%d "
                "forward_remote=%.2f MiB refined_buckets=%d assignment_ms=%.2f",
                group_ranks[0],
                len(group_ranks),
                summary.num_buckets,
                summary.num_matrices,
                remote_bytes / (1024 * 1024),
                summary.num_refined_buckets,
                summary.assignment_seconds * 1000,
            )
        return tuple(compiled_buckets)

    def _validate_compiled_buckets_across_ranks(
        self,
        compiled_buckets: tuple[_CompiledMuonBucket, ...],
    ) -> None:
        """Check plans after rank-stable bucket and transfer-group selection.

        Bucket membership and redistribution requirements must already agree
        across each potential communication group, as in TorchTitan. This hash
        detects plan differences among ranks entering the same collectives.
        """
        buckets_by_group: dict[tuple[int, ...], list[_CompiledMuonBucket]] = {}
        for bucket in compiled_buckets:
            if not bucket.has_redistribution:
                continue
            group = bucket.group
            assert group is not None
            group_ranks = group.global_ranks_in_group_order
            buckets_by_group.setdefault(group_ranks, []).append(bucket)

        for group_ranks in sorted(buckets_by_group):
            buckets = buckets_by_group[group_ranks]
            first_bucket = buckets[0]
            assert first_bucket.group is not None
            group = first_bucket.group
            descriptions = []
            for bucket in buckets:
                assert bucket.group is not None
                descriptions.append(
                    (
                        bucket.name,
                        bucket.group.global_ranks_in_group_order,
                        bucket.group.mesh_indices_in_group_order,
                        tuple(
                            self._parameter_signature(item, bucket)
                            for item in bucket.items
                        ),
                    )
                )
            digest = hashlib.sha256(repr(descriptions).encode()).digest()
            group_hash = int.from_bytes(digest[:7], "little")
            local_hash = torch.tensor(
                group_hash,
                dtype=torch.int64,
                device=first_bucket.device,
            )
            gathered = [torch.empty_like(local_hash) for _ in group_ranks]
            dist.all_gather(gathered, local_hash, group=group.process_group)
            if any(value.item() != group_hash for value in gathered):
                raise RuntimeError("compiled Muon buckets differ across ranks")

    def _parameter_signature(
        self,
        item: _BoundMuonParameter,
        bucket: _CompiledMuonBucket,
    ) -> tuple[Any, ...]:
        plan_signature = None
        parameter_index = next(
            (
                index
                for index, candidate in enumerate(bucket.redistributed_items)
                if candidate is item
            ),
            None,
        )
        if parameter_index is not None:
            plan = bucket.redistributed_plans[parameter_index]
            plan_signature = (
                item.storage_layout.global_ranks_in_mesh_order,
                plan.matrices,
                plan.storage_partitions,
                plan.compute_partitions,
                plan.storage_to_compute_routes,
            )
        # Local items may use different EDP subgroups within a dense DP bucket.
        # Their physical group identities do not govern this bucket's transfer.
        return (
            item.fqn,
            tuple(item.param.shape),
            tuple(item.param.stride()),
            str(item.param.dtype),
            item.param.to_local().device.type,
            item.compute_view_key,
            type(item.compute_placement).__name__,
            tuple(str(placement) for placement in item.storage_layout.placements),
            plan_signature,
            self._group_signature(item),
        )

    def _group(self, item: _BoundMuonParameter) -> dict[str, Any]:
        return self.param_groups[item.group_index]

    def _group_signature(self, item: _BoundMuonParameter) -> tuple[Any, ...]:
        group = self._group(item)
        return tuple(
            group[key]
            for key in (
                "lr",
                "weight_decay",
                "momentum",
                "nesterov",
                "ns_coefficients",
                "eps",
                "ns_steps",
                "adjust_lr_fn",
            )
        )

    def _preflight_step(self) -> None:
        """Validate every input before mutating state or entering collectives.

        Gradient presence/type and cached parameter storage are checked every
        step. Full gradient-layout and momentum validation runs only on the
        first step and after loading optimizer state.
        """
        initialize_state = not self._first_step_validated
        missing_gradients = []
        changed_parameter = None
        changed_gradient = None
        for item in self._bound_parameters:
            local_parameter = item.param.to_local()
            if (
                changed_parameter is None
                and (
                    not item.storage_layout.matches_local(local_parameter)
                    or not item.local_storage_identity.matches(local_parameter)
                )
            ):
                changed_parameter = item.fqn
            grad = item.param.grad
            if grad is None:
                missing_gradients.append(item.fqn)
            elif not isinstance(grad, DTensor):
                if changed_gradient is None:
                    changed_gradient = item.fqn
            elif initialize_state and not item.storage_layout.matches(grad):
                if changed_gradient is None:
                    changed_gradient = item.fqn

        if missing_gradients:
            raise RuntimeError(
                "FSDPDistributedMuon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )
        if changed_parameter is not None:
            raise RuntimeError(
                f"parameter local storage changed for {changed_parameter!r}; "
                "moving FSDP parameters, for example with module.to(), replaces "
                "their storage, so keep them in place or rebuild FSDPDistributedMuon"
            )
        if changed_gradient is not None:
            raise RuntimeError(
                f"gradient storage layout changed for {changed_gradient!r}"
            )

        if initialize_state:
            self._initialize_state()
            self._first_step_validated = True

    def _validate_momentum(self, item: _BoundMuonParameter) -> None:
        momentum = self.state.get(item.param, {}).get("momentum_buffer")
        if momentum is None:
            return
        if self._offload.enabled:
            valid = (
                not isinstance(momentum, DTensor)
                and momentum.device.type == "cpu"
                and momentum.shape == item.storage_layout.local_shape
                and momentum.dtype == item.storage_layout.dtype
                and momentum.is_contiguous()
            )
        else:
            valid = isinstance(momentum, DTensor) and item.storage_layout.matches(
                momentum
            )
        if not valid:
            raise RuntimeError(f"momentum storage layout changed for {item.fqn!r}")

    def _initialize_state(self) -> None:
        """Validate existing momentum, then create zero momentum where missing.

        Validating every entry first keeps invalid restored state from leaving
        the optimizer partially initialized.
        """
        for item in self._bound_parameters:
            self._validate_momentum(item)
        for item in self._bound_parameters:
            state = self.state[item.param]
            if "momentum_buffer" in state:
                continue
            if self._offload.enabled:
                state["momentum_buffer"] = self._empty_cpu_momentum(item).zero_()
            else:
                state["momentum_buffer"] = torch.zeros_like(
                    item.param,
                    memory_format=torch.preserve_format,
                )

    def _empty_cpu_momentum(self, item: _BoundMuonParameter) -> Tensor:
        """Allocate an offloaded momentum shard, pinned when a device stages it."""
        return torch.empty_like(
            item.param.to_local(),
            device="cpu",
            pin_memory=self._execution_device.type != "cpu",
        )

    def state_dict(self) -> dict[str, Any]:
        """Export CPU momentum with the parameter's distributed shard metadata."""
        result = super().state_dict()
        if not self._offload.enabled:
            return result
        for group, saved_group in zip(
            self.param_groups, result["param_groups"], strict=True
        ):
            for param, param_id in zip(
                group["params"], saved_group["params"], strict=True
            ):
                if param_id in result["state"]:
                    entry = dict(result["state"][param_id])
                    entry["momentum_buffer"] = wrap_cpu_state(
                        entry["momentum_buffer"],
                        param,
                        self._checkpoint_meshes,
                    )
                    result["state"][param_id] = entry
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load offloaded momentum without the default parameter-device casting.

        Existing CPU buffers are reused: DCP restores into the live buffers
        exposed by ``state_dict()``, and reuse avoids a second pinned copy.
        """
        if not self._offload.enabled:
            super().load_state_dict(state_dict)
            return
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            replacement = hook(self, state_dict)
            if replacement is not None:
                state_dict = replacement
        saved_groups = deepcopy(state_dict["param_groups"])
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, self.param_groups)
        ):
            raise ValueError("loaded Muon state has different parameter groups")
        items_by_param = {item.param: item for item in self._bound_parameters}
        by_id = {
            saved_id: items_by_param[param]
            for saved, current in zip(saved_groups, self.param_groups)
            for saved_id, param in zip(saved["params"], current["params"])
        }
        # Validate every entry and create every missing destination before
        # copying, so a rejected checkpoint leaves the reused buffers intact.
        pending = []
        for saved_id, entry in state_dict["state"].items():
            if saved_id not in by_id or set(entry) != {"momentum_buffer"}:
                raise ValueError(
                    "loaded Muon state has an unknown parameter or state key"
                )
            item = by_id[saved_id]
            value = entry["momentum_buffer"]
            if isinstance(value, DTensor):
                if (
                    value.shape != item.storage_layout.global_shape
                    or value.placements != item.storage_layout.placements
                    or _global_ranks_in_mesh_order(value)
                    != item.storage_layout.global_ranks_in_mesh_order
                ):
                    raise ValueError(f"momentum layout mismatch for {item.fqn!r}")
                value = value.to_local()
            if (
                value.shape != item.storage_layout.local_shape
                or value.dtype != item.storage_layout.dtype
            ):
                raise ValueError(f"momentum shape or dtype mismatch for {item.fqn!r}")
            target = self.state.get(item.param, {}).get("momentum_buffer")
            if target is None:
                target = self._empty_cpu_momentum(item)
            pending.append((item, target, value))
        restored = defaultdict(dict)
        for item, target, value in pending:
            target.copy_(value)
            restored[item.param]["momentum_buffer"] = target
        for current, saved in zip(self.param_groups, saved_groups, strict=True):
            saved["params"] = current["params"]
        self.state = restored
        self.param_groups = saved_groups
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)

    def _local_gradient_and_momentum(
        self,
        item: _BoundMuonParameter,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Return local gradient/state views and this item's parameter group."""
        grad = cast(DTensor, item.param.grad)
        momentum_state = self.state[item.param]["momentum_buffer"]
        local_grad = grad.to_local().view_as(reference)
        local_momentum = (
            momentum_state.to_local()
            if isinstance(momentum_state, DTensor)
            else momentum_state
        ).view_as(reference)
        return local_grad, local_momentum, momentum_state, self._group(item)

    def _prepare_local(self, item: _BoundMuonParameter, prepared: Tensor) -> None:
        """Update local momentum and write the prepared Muon input."""
        gradient, momentum_buffer, momentum_state, group = (
            self._local_gradient_and_momentum(item, prepared)
        )
        stored_momentum = momentum_buffer
        gradient = gradient.to(prepared.device)
        momentum_buffer = momentum_buffer.to(prepared.device)
        momentum_coefficient = group["momentum"]
        momentum_buffer.lerp_(gradient, 1 - momentum_coefficient)
        if group["nesterov"]:
            torch.lerp(
                gradient,
                momentum_buffer,
                momentum_coefficient,
                out=prepared,
            )
        else:
            prepared.copy_(momentum_buffer)
        if stored_momentum.device != momentum_buffer.device:
            stored_momentum.copy_(momentum_buffer)
        if isinstance(momentum_state, DTensor):
            torch.autograd.graph.increment_version(momentum_state)

    def _ns_step_coefficients(
        self,
        coefficients: Sequence[float] | Sequence[Sequence[float]],
        ns_steps: int,
    ) -> tuple[Sequence[float], ...]:
        """Return one Newton-Schulz (a, b, c) triple per iteration.
        A flat triple applies to every one of the ``ns_steps`` iterations;
        per-step triples already carry one triple per iteration (their count is
        validated against ``ns_steps``).
        """
        if not any(isinstance(entry, (tuple, list)) for entry in coefficients):
            return (tuple(coefficients),) * ns_steps
        return tuple(coefficients)

    def _compute_update(self, item: _BoundMuonParameter, compute: Tensor) -> None:
        """Replace ``compute`` with Muon's approximate orthogonal direction."""
        group = self._group(item)
        step_coefficients = self._ns_step_coefficients(
            group["ns_coefficients"], group["ns_steps"]
        )
        result = compute.to(dtype=torch.bfloat16, copy=True)
        # Newton-Schulz operates on the orientation with rows <= columns.
        transposed = result.shape[-2] > result.shape[-1]
        if transposed:
            result = result.transpose(-2, -1)
        result.div_(
            result.norm(dim=(-2, -1), keepdim=True).clamp_min(group["eps"])
        )

        # Use matrix kernels for one matrix and batched kernels for compute views.
        if result.ndim == 2:
            for a, b, c in step_coefficients:
                gram = result @ result.T
                gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
                result = torch.addmm(result, gram_update, result, beta=a)
        else:
            original_shape = result.shape
            matrices = result.reshape(-1, *original_shape[-2:])
            for a, b, c in step_coefficients:
                gram = matrices @ matrices.transpose(-2, -1)
                gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
            result = matrices.reshape(original_shape)

        direction = result.transpose(-2, -1) if transposed else result
        compute.copy_(direction)

    def _apply_update(self, item: _BoundMuonParameter, direction: Tensor) -> None:
        """Apply weight decay and a computed direction to local parameter data."""
        group = self._group(item)
        local_param = item.param.to_local().view_as(direction)
        stored_param = local_param
        local_param = local_param.to(direction.device)
        if item.resolved_compute_view is not None:
            matrix_shape = item.resolved_compute_view.matrix_shape
        else:
            matrix_shape = item.global_compute_shape

        # The direction step is shape-adjusted; decoupled weight decay continues
        # to use the configured base learning rate.
        rows, columns = matrix_shape[-2:]
        adjust_lr_fn = group["adjust_lr_fn"]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            ratio = math.sqrt(max(1.0, rows / columns))
        elif adjust_lr_fn == "match_rms_adamw":
            ratio = 0.2 * math.sqrt(max(rows, columns))
        elif adjust_lr_fn == "spectral_unclamped":
            ratio = math.sqrt(rows / columns)
        else:
            raise ValueError(f"unsupported adjust_lr_fn {adjust_lr_fn!r}")
        adjusted_lr = group["lr"] * ratio

        local_param.mul_(1 - group["lr"] * group["weight_decay"])
        local_param.add_(direction, alpha=-adjusted_lr)
        if stored_param.device != local_param.device:
            stored_param.copy_(local_param)
        torch.autograd.graph.increment_version(item.param)

    def _compute_local_items(
        self,
        bucket: _CompiledMuonBucket,
        buffers: _MuonBuffers,
    ) -> None:
        """Prepare, compute, and apply local items using shared scratch space."""
        for item, shape in zip(
            bucket.local_items,
            bucket.local_item_compute_shapes,
            strict=True,
        ):
            prepared = buffers.compute_tensor(
                shape,
                dtype=item.storage_layout.dtype,
                device=bucket.device,
            )
            self._prepare_local(item, prepared)
            if math.prod(shape):
                self._compute_update(item, prepared)
            self._apply_update(item, prepared)

    def _prepare_redistributed(
        self,
        bucket: _CompiledMuonBucket,
        buffers: _MuonBuffers,
    ) -> None:
        """Prepare local storage partitions and pack transfer input."""
        assert bucket.group is not None and bucket.forward is not None
        assert bucket.dtype is not None
        storage_buffer, _compute_buffer = buffers.exchange_buffers(bucket)
        rank = bucket.group.current_group_rank
        for index, (item, plan) in enumerate(
            zip(
                bucket.redistributed_items,
                bucket.redistributed_plans,
                strict=True,
            )
        ):
            prepared = buffers.storage_tensor(
                plan.storage_partition(rank).tensor_shape,
                dtype=bucket.dtype,
                device=bucket.device,
            )
            self._prepare_local(item, prepared)
            for span in bucket.forward.input_spans_by_parameter[index]:
                span.buffer_view(storage_buffer).copy_(
                    span.tensor_view(prepared)
                )

    def _compute_redistributed(
        self,
        bucket: _CompiledMuonBucket,
        buffers: _MuonBuffers,
    ) -> None:
        """Rebuild, compute, and repack this rank's compute partitions."""
        assert bucket.group is not None
        assert bucket.forward is not None and bucket.reverse is not None
        assert bucket.dtype is not None
        _storage_buffer, compute_buffer = buffers.exchange_buffers(bucket)
        rank = bucket.group.current_group_rank
        for index, (item, plan) in enumerate(
            zip(
                bucket.redistributed_items,
                bucket.redistributed_plans,
                strict=True,
            )
        ):
            shape = plan.compute_partition(rank).tensor_shape
            if not math.prod(shape):
                continue
            compute = buffers.compute_tensor(
                shape,
                dtype=bucket.dtype,
                device=bucket.device,
            )
            for span in bucket.forward.output_spans_by_parameter[index]:
                span.tensor_view(compute).copy_(
                    span.buffer_view(compute_buffer)
                )
            self._compute_update(item, compute)
            for span in bucket.reverse.input_spans_by_parameter[index]:
                span.buffer_view(compute_buffer).copy_(
                    span.tensor_view(compute)
                )

    def _finalize_redistributed(
        self,
        bucket: _CompiledMuonBucket,
        buffers: _MuonBuffers,
    ) -> None:
        """Unpack returned directions and update local parameter storage."""
        assert bucket.group is not None and bucket.reverse is not None
        assert bucket.dtype is not None
        storage_buffer, _compute_buffer = buffers.exchange_buffers(bucket)
        rank = bucket.group.current_group_rank
        for index, (item, plan) in enumerate(
            zip(
                bucket.redistributed_items,
                bucket.redistributed_plans,
                strict=True,
            )
        ):
            update = buffers.storage_tensor(
                plan.storage_partition(rank).tensor_shape,
                dtype=bucket.dtype,
                device=bucket.device,
            )
            for span in bucket.reverse.output_spans_by_parameter[index]:
                span.tensor_view(update).copy_(
                    span.buffer_view(storage_buffer)
                )
            self._apply_update(item, update)

    @staticmethod
    def _execute_transfer(
        schedule: _PackedTransferSchedule,
        *,
        output_buffer: Tensor,
        input_buffer: Tensor,
        process_group: dist.ProcessGroup,
    ) -> None:
        """Execute one packed transfer, including the rank-local copy path."""
        if not schedule.has_remote_transfers:
            output_buffer[: schedule.output_numel].copy_(
                input_buffer[: schedule.input_numel]
            )
            return
        dist.all_to_all_single(
            output_buffer[: schedule.output_numel],
            input_buffer[: schedule.input_numel],
            output_split_sizes=list(schedule.output_splits),
            input_split_sizes=list(schedule.input_splits),
            group=process_group,
        )

    def _gather(self, bucket: _CompiledMuonBucket, buffers: _MuonBuffers) -> None:
        """Exchange prepared storage shards into complete compute partitions."""
        assert bucket.group is not None and bucket.forward is not None
        storage_buffer, compute_buffer = buffers.exchange_buffers(bucket)
        self._execute_transfer(
            bucket.forward,
            output_buffer=compute_buffer,
            input_buffer=storage_buffer,
            process_group=bucket.group.process_group,
        )

    def _scatter(self, bucket: _CompiledMuonBucket, buffers: _MuonBuffers) -> None:
        """Exchange computed directions back to their persistent storage."""
        assert bucket.group is not None and bucket.reverse is not None
        storage_buffer, compute_buffer = buffers.exchange_buffers(bucket)
        self._execute_transfer(
            bucket.reverse,
            output_buffer=storage_buffer,
            input_buffer=compute_buffer,
            process_group=bucket.group.process_group,
        )

    def _run_synchronous(self) -> None:
        """Run every bucket to completion without a communication pipeline."""
        self._workspace.record_compute_stream(
            self._workspace.device_runtime.current_stream()
        )
        for bucket_index, bucket in enumerate(self._compiled_buckets):
            if not bucket.has_redistribution:
                # Local-only buckets need no exchange buffers or pipeline slot.
                self._compute_local_items(
                    bucket,
                    self._workspace.local_buffers,
                )
                continue
            slot = self._workspace.slot_for_bucket(bucket_index)
            # Complete the storage -> compute -> storage round trip before
            # advancing to the next bucket.
            self._prepare_redistributed(bucket, slot.buffers)
            self._gather(bucket, slot.buffers)
            self._compute_local_items(bucket, slot.buffers)
            self._compute_redistributed(bucket, slot.buffers)
            self._scatter(bucket, slot.buffers)
            self._finalize_redistributed(bucket, slot.buffers)

    def _run_async(self) -> None:
        """Run redistributed buckets through a TorchTitan-style stream pipeline.

        A local-only bucket stops look-ahead, but does not wait for the previous
        redistributed bucket to finish. The next transfer bucket retires it.
        With overlap disabled, the same machinery uses one slot and remains
        deliberately serialized.
        """
        runtime = self._workspace.device_runtime
        caller = runtime.current_stream()
        transfer = self._workspace.transfer_stream
        if caller is None or transfer is None:
            raise RuntimeError("Muon asynchronous workspace is not configured")
        self._workspace.record_compute_stream(caller)

        overlap = self._workspace.overlap_comm_compute
        # With two slots, this contains at most the next consecutive transfer
        # bucket. A local-only bucket deliberately leaves it empty.
        prefetched: _MuonSlot | None = None
        previous: _MuonSlot | None = None
        transfer_started = False
        completed = False
        try:
            for bucket_index, bucket in enumerate(self._compiled_buckets):
                if not bucket.has_redistribution:
                    if prefetched is not None:
                        raise RuntimeError(
                            "Muon pipeline prefetched across a local-only bucket"
                        )
                    if not overlap and previous is not None:
                        # The opt-out path intentionally completes each transfer
                        # bucket before running a local-only bucket.
                        caller.wait_event(previous.finished)
                        previous = None
                    with runtime.stream(caller):
                        self._compute_local_items(
                            bucket,
                            self._workspace.local_buffers,
                        )
                    continue

                # TorchTitan establishes the caller dependency once, before the
                # first gather. Later gathers are independent bucket work and
                # must be allowed to overlap caller-side local computation.
                if not transfer_started or not overlap:
                    transfer.wait_stream(caller)
                    transfer_started = True

                previous_slot = previous
                slot = prefetched
                prefetched = None
                if slot is not None:
                    if not overlap or previous_slot is None:
                        raise RuntimeError("invalid Muon prefetch state")
                    if slot.active_bucket is not bucket:
                        raise RuntimeError("Muon pipeline prefetched the wrong bucket")
                    # The current gather is already queued. Return the previous
                    # bucket before its slot can be used again.
                    assert previous_slot.active_bucket is not None
                    self._enqueue_scatter(previous_slot.active_bucket, previous_slot)
                else:
                    slot = self._workspace.slot_for_bucket(bucket_index)
                    if (
                        overlap
                        and previous_slot is not None
                        and slot is previous_slot
                    ):
                        # Same-slot reuse must retire the old contents first.
                        assert previous_slot.active_bucket is not None
                        self._enqueue_scatter(previous_slot.active_bucket, previous_slot)
                    self._enqueue_gather(bucket, slot)
                    if (
                        overlap
                        and previous_slot is not None
                        and slot is not previous_slot
                    ):
                        # Different slots preserve the overlap window: gather the
                        # current bucket before returning the previous one.
                        assert previous_slot.active_bucket is not None
                        self._enqueue_scatter(previous_slot.active_bucket, previous_slot)

                with runtime.stream(caller):
                    # Local items use a separate scratch pool. In the overlap
                    # path they can run while this bucket's gather completes.
                    if not overlap:
                        caller.wait_event(slot.input_ready)
                    self._compute_local_items(bucket, slot.buffers)
                    if overlap:
                        caller.wait_event(slot.input_ready)
                    self._compute_redistributed(bucket, slot.buffers)
                    slot.compute_done.record(caller)

                if not overlap:
                    self._enqueue_scatter(bucket, slot)
                elif previous_slot is not None:
                    # Make the previous parameter update visible before later
                    # caller work, without blocking the host here.
                    caller.wait_event(previous_slot.finished)
                previous = slot

                if overlap:
                    next_index = bucket_index + 1
                    if (
                        next_index < len(self._compiled_buckets)
                        and self._compiled_buckets[next_index].has_redistribution
                    ):
                        # Stop at a local-only bucket: it is a prefetch boundary.
                        next_bucket = self._compiled_buckets[next_index]
                        next_slot = self._workspace.slot_for_bucket(next_index)
                        self._enqueue_gather(next_bucket, next_slot)
                        prefetched = next_slot

            if prefetched is not None:
                raise RuntimeError("Muon pipeline prefetched past the final bucket")
            # The final compute result has not been returned in overlap mode.
            if previous is not None:
                if overlap:
                    assert previous.active_bucket is not None
                    self._enqueue_scatter(previous.active_bucket, previous)
                caller.wait_event(previous.finished)
            completed = True
        finally:
            if not completed:
                # Preserve cross-stream ordering for work already queued before
                # propagating a fatal pipeline exception.
                transfer.wait_stream(caller)
                caller.wait_stream(transfer)

    def _enqueue_gather(
        self,
        bucket: _CompiledMuonBucket,
        slot: _MuonSlot,
    ) -> None:
        """Queue local preparation and gather, then signal input readiness."""
        runtime = self._workspace.device_runtime
        transfer = self._workspace.transfer_stream
        assert transfer is not None
        if slot.active_bucket is not None:
            raise RuntimeError("Muon pipeline slot is still occupied")
        slot.active_bucket = bucket
        with runtime.stream(transfer):
            self._prepare_redistributed(bucket, slot.buffers)
            self._gather(bucket, slot.buffers)
            slot.input_ready.record(transfer)

    def _enqueue_scatter(
        self,
        bucket: _CompiledMuonBucket,
        slot: _MuonSlot,
    ) -> None:
        """Wait for matrix compute, scatter, and signal slot completion."""
        runtime = self._workspace.device_runtime
        transfer = self._workspace.transfer_stream
        assert transfer is not None
        with runtime.stream(transfer):
            transfer.wait_event(slot.compute_done)
            self._scatter(bucket, slot.buffers)
            self._finalize_redistributed(bucket, slot.buffers)
            slot.finished.record(transfer)
        # Later reuse runs on the same transfer stream, after this return.
        slot.active_bucket = None


def _load_state_dict_post_hook(optimizer: Optimizer) -> None:
    muon = cast(FSDPDistributedMuon, optimizer)
    muon._after_load_state_dict()
