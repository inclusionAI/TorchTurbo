from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.distributed.tensor import Shard

from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import (
    _MatrixPartition,
    _MatrixRoute,
    _MuonLayoutPlanner,
    _MuonMatrix,
    _MuonParameterLayout,
    _RowPartition,
    _StorageToComputePlan,
)
from torchturbo.engine.optimizers.muon import (
    BatchedMatrixComputeView,
    ResolvedBatchedMatrixView,
)
from torchturbo.utils.torch_utils import torch_version


pytestmark = pytest.mark.skipif(
    torch_version() < (2, 10, 0),
    reason="Muon layout planning requires PyTorch 2.10 or newer",
)


def _matrix_layout(
    batch_size: int,
    matrix_rows: int,
    columns: int,
    mesh_indices: tuple[int, ...],
    *,
    current_group_rank: int = 0,
    column_groups: int = 1,
) -> tuple[_MuonLayoutPlanner, _MuonParameterLayout]:
    global_shape = torch.Size((batch_size * matrix_rows, columns))
    local_rows, _ = Shard.local_shard_size_and_offset(
        global_shape[0], len(mesh_indices), mesh_indices[current_group_rank]
    )
    planner = _MuonLayoutPlanner(mesh_indices, current_group_rank)
    layout = planner.resolve_parameter(
        fqn="projection.weight",
        global_storage_shape=global_shape,
        local_storage_shape=torch.Size((local_rows, columns)),
        compute_view=BatchedMatrixComputeView(batch_size, column_groups).resolve(
            global_shape
        ),
    )
    return planner, layout


def _matrix_plan(
    batch_size: int,
    matrix_rows: int,
    columns: int,
    mesh_indices: tuple[int, ...],
    *,
    current_group_rank: int = 0,
    column_groups: int = 1,
    owner_group_ranks: tuple[int, ...] | None = None,
) -> _StorageToComputePlan:
    planner, layout = _matrix_layout(
        batch_size,
        matrix_rows,
        columns,
        mesh_indices,
        current_group_rank=current_group_rank,
        column_groups=column_groups,
    )
    if owner_group_ranks is None:
        owners = [-1] * (batch_size * column_groups)
        for group_rank, mesh_index in enumerate(mesh_indices):
            local_batch, batch_offset = Shard.local_shard_size_and_offset(
                batch_size, len(mesh_indices), mesh_index
            )
            for batch_index in range(batch_offset, batch_offset + local_batch):
                for column_group in range(column_groups):
                    owners[batch_index * column_groups + column_group] = group_rank
        owner_group_ranks = tuple(owners)
    return planner.build_plan(layout, owner_group_ranks)


def _apply_routes(
    sources: tuple[torch.Tensor, ...],
    destination_partitions: tuple[_RowPartition | _MatrixPartition, ...],
    routes: tuple[_MatrixRoute, ...],
) -> tuple[torch.Tensor, ...]:
    destinations = tuple(
        torch.full(partition.tensor_shape, -1, dtype=torch.int64)
        for partition in destination_partitions
    )
    source_coverage = tuple(torch.zeros_like(source) for source in sources)
    destination_coverage = tuple(
        torch.zeros_like(destination) for destination in destinations
    )
    for route in routes:
        shape = (route.num_rows, route.num_columns)
        source_stride = (route.source_row_stride, 1)
        destination_stride = (route.destination_row_stride, 1)
        source = sources[route.source_group_rank].view(-1).as_strided(
            shape, source_stride, route.source_flat_offset
        )
        destination = destinations[route.destination_group_rank].view(-1).as_strided(
            shape, destination_stride, route.destination_flat_offset
        )
        destination.copy_(source)
        source_coverage[route.source_group_rank].view(-1).as_strided(
            shape, source_stride, route.source_flat_offset
        ).add_(1)
        destination_coverage[route.destination_group_rank].view(-1).as_strided(
            shape, destination_stride, route.destination_flat_offset
        ).add_(1)

    for coverage in (*source_coverage, *destination_coverage):
        assert torch.all(coverage == 1)
    return destinations


def _assert_round_trip(
    plan: _StorageToComputePlan, global_shape: torch.Size
) -> tuple[torch.Tensor, ...]:
    original = torch.arange(global_shape.numel()).view(global_shape)
    storage = tuple(
        original.narrow(0, partition.global_row_offset, partition.num_rows).clone()
        for partition in plan.storage_partitions
    )
    if plan.requires_redistribution:
        compute = _apply_routes(
            storage, plan.compute_partitions, plan.storage_to_compute_routes
        )
    else:
        compute = tuple(
            tensor.view(partition.tensor_shape)
            for tensor, partition in zip(storage, plan.compute_partitions, strict=True)
        )
    for actual, partition in zip(compute, plan.compute_partitions, strict=True):
        logical_matrices = []
        for matrix_index in partition.matrix_indices:
            matrix = plan.matrices[matrix_index]
            logical_matrices.append(
                original.narrow(0, matrix.global_row_offset, matrix.num_rows).narrow(
                    1, matrix.global_column_offset, matrix.num_columns
                )
            )
        expected = (
            torch.stack(logical_matrices).view(partition.tensor_shape)
            if logical_matrices
            else torch.empty(partition.tensor_shape, dtype=torch.int64)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if plan.requires_redistribution:
        restored = _apply_routes(
            compute, plan.storage_partitions, plan.compute_to_storage_routes
        )
    else:
        restored = tuple(
            tensor.view(partition.tensor_shape)
            for tensor, partition in zip(compute, plan.storage_partitions, strict=True)
        )
    for actual, expected in zip(restored, storage, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(route.inverse() for route in plan.compute_to_storage_routes) == (
        plan.storage_to_compute_routes
    )
    return compute


def test_resolve_parameter_describes_independent_matrices_before_assignment():
    planner, layout = _matrix_layout(2, 3, 8, (0, 1, 2, 3), column_groups=2)

    assert layout.fqn == "projection.weight"
    assert layout.global_storage_shape == (6, 8)
    assert layout.is_batched
    assert layout.matrices == (
        _MuonMatrix(0, 0, 0, 3, 4),
        _MuonMatrix(1, 0, 4, 3, 4),
        _MuonMatrix(2, 3, 0, 3, 4),
        _MuonMatrix(3, 3, 4, 3, 4),
    )
    assert all(matrix.numel == 12 for matrix in layout.matrices)
    assert all(matrix.tensor_shape == (3, 4) for matrix in layout.matrices)
    owned = planner.resolve_parameter(
        fqn="linear.weight",
        global_storage_shape=torch.Size((6, 8)),
        local_storage_shape=torch.Size((2, 8)),
    )
    assert not owned.is_batched
    assert owned.matrices == (_MuonMatrix(0, 0, 0, 6, 8),)


@pytest.mark.parametrize("num_ranks", [1, 4, 8])
def test_aligned_matrix_batches_remain_local(num_ranks):
    plan = _matrix_plan(8, 128, 3, tuple(range(num_ranks)))

    assert not plan.requires_redistribution
    assert plan.storage_to_compute_routes == ()
    assert plan.compute_to_storage_routes == ()
    for rank in range(num_ranks):
        assert plan.storage_partition(rank).num_rows == 1024 // num_ranks
        compute = plan.compute_partition(rank)
        assert compute.tensor_shape == (8 // num_ranks, 128, 3)
        assert compute.matrix_indices == tuple(
            range(rank * 8 // num_ranks, (rank + 1) * 8 // num_ranks)
        )
    _assert_round_trip(plan, torch.Size((1024, 3)))


def test_aligned_empty_partitions_remain_local():
    plan = _matrix_plan(3, 3, 5, (0, 1, 2, 3), current_group_rank=3)

    assert not plan.requires_redistribution
    assert plan.storage_partition(3).tensor_shape == (0, 5)
    assert plan.compute_partition(3).tensor_shape == (0, 3, 5)
    assert plan.storage_partition(3).global_row_offset == 9
    assert plan.compute_partition(3).matrix_indices == ()
    _assert_round_trip(plan, torch.Size((9, 5)))


@pytest.mark.parametrize(
    ("batch_size", "matrix_rows", "storage_rows", "compute_rows"),
    [
        (5, 3, (4, 4, 4, 3), (6, 6, 3, 0)),
        (2, 2, (1, 1, 1, 1, 0, 0, 0, 0), (2, 2, 0, 0, 0, 0, 0, 0)),
        (8, 128, (64,) * 16, (128,) * 8 + (0,) * 8),
    ],
)
def test_uneven_and_empty_partitions_round_trip(
    batch_size, matrix_rows, storage_rows, compute_rows
):
    plan = _matrix_plan(batch_size, matrix_rows, 3, tuple(range(len(storage_rows))))

    assert plan.requires_redistribution
    assert tuple(partition.num_rows for partition in plan.storage_partitions) == (
        storage_rows
    )
    for rank, rows in enumerate(compute_rows):
        assert plan.compute_partition(rank).tensor_shape == (
            rows // matrix_rows, matrix_rows, 3
        )
    _assert_round_trip(plan, torch.Size((batch_size * matrix_rows, 3)))


@pytest.mark.parametrize(
    ("batch_size", "matrix_rows", "num_ranks", "storage_ranks_per_matrix"),
    [(1, 8, 4, 4), (1, 16, 8, 8), (2, 8, 8, 4), (3, 9, 7, 3)],
)
def test_matrix_spanning_more_than_two_storage_ranks_round_trips(
    batch_size, matrix_rows, num_ranks, storage_ranks_per_matrix
):
    plan = _matrix_plan(batch_size, matrix_rows, 3, tuple(range(num_ranks)))

    assert plan.requires_redistribution
    sources_by_matrix: dict[int, set[int]] = {}
    for route in plan.storage_to_compute_routes:
        sources_by_matrix.setdefault(route.matrix_index, set()).add(
            route.source_group_rank
        )
    assert max(len(sources) for sources in sources_by_matrix.values()) == (
        storage_ranks_per_matrix
    )
    non_empty_compute = [
        partition for partition in plan.compute_partitions if partition.matrix_indices
    ]
    assert len(non_empty_compute) == batch_size
    assert all(p.tensor_shape == (1, matrix_rows, 3) for p in non_empty_compute)
    _assert_round_trip(plan, torch.Size((batch_size * matrix_rows, 3)))


def test_matrix_ownership_uses_group_ranks_independently_of_mesh_coordinates():
    mesh_indices = (2, 0, 3, 1)
    plan = _matrix_plan(3, 4, 5, mesh_indices, owner_group_ranks=(2, 0, 3))

    assert tuple(p.global_row_offset for p in plan.storage_partitions) == (6, 0, 9, 3)
    assert tuple(p.matrix_indices for p in plan.compute_partitions) == (
        (1,), (), (0,), (2,)
    )
    for group_rank in range(len(mesh_indices)):
        assert plan == _matrix_plan(
            3, 4, 5, mesh_indices,
            current_group_rank=group_rank,
            owner_group_ranks=(2, 0, 3),
        )
    _assert_round_trip(plan, torch.Size((12, 5)))


@pytest.mark.parametrize(
    ("mesh_indices", "owner_rank", "rows", "requires_redistribution"),
    [((0,), 0, 5, False), ((2, 0, 3, 1), 2, 5, True), ((2, 0, 3, 1), 1, 1, False)],
)
def test_owned_layout_uses_two_dimensions_and_elides_identity_routes(
    mesh_indices, owner_rank, rows, requires_redistribution
):
    global_shape = torch.Size((rows, 7))
    local_rows, _ = Shard.local_shard_size_and_offset(
        rows, len(mesh_indices), mesh_indices[0]
    )
    planner = _MuonLayoutPlanner(mesh_indices, 0)
    layout = planner.resolve_parameter(
        fqn="linear.weight",
        global_storage_shape=global_shape,
        local_storage_shape=torch.Size((local_rows, 7)),
    )
    plan = planner.build_plan(layout, (owner_rank,))

    assert plan.requires_redistribution == requires_redistribution
    assert plan.compute_partition(owner_rank).tensor_shape == global_shape
    assert plan.compute_partition(owner_rank).matrix_indices == (0,)
    for rank in range(len(mesh_indices)):
        if rank != owner_rank:
            assert plan.compute_partition(rank).tensor_shape == (0, 7)
            assert plan.compute_partition(rank).matrix_indices == ()
    _assert_round_trip(plan, global_shape)


def test_arbitrary_owners_pack_noncontiguous_matrix_indices():
    plan = _matrix_plan(
        5, 3, 4, (0, 1, 2, 3), owner_group_ranks=(3, 1, 3, 0, 1)
    )

    assert tuple(p.matrix_indices for p in plan.compute_partitions) == (
        (3,), (1, 4), (), (0, 2)
    )
    _assert_round_trip(plan, torch.Size((15, 4)))


def test_empty_storage_ranks_can_own_complete_matrices():
    plan = _matrix_plan(2, 2, 4, tuple(range(8)), owner_group_ranks=(6, 7))

    for rank in (6, 7):
        assert plan.storage_partition(rank).numel == 0
        assert plan.compute_partition(rank).tensor_shape == (1, 2, 4)
    _assert_round_trip(plan, torch.Size((4, 4)))


def test_column_groups_can_have_independent_owners():
    plan = _matrix_plan(
        3, 4, 8, (0, 1, 2, 3),
        column_groups=2,
        owner_group_ranks=(3, 1, 0, 2, 3, 0),
    )

    assert tuple(p.matrix_indices for p in plan.compute_partitions) == (
        (2, 5), (1,), (3,), (0, 4)
    )
    assert all(p.tensor_shape[1:] == (4, 4) for p in plan.compute_partitions)
    assert all(route.source_row_stride == 8 for route in plan.storage_to_compute_routes)
    assert all(
        route.destination_row_stride == 4 for route in plan.storage_to_compute_routes
    )
    _assert_round_trip(plan, torch.Size((12, 8)))


def test_local_column_groups_retain_self_routes_for_reordering():
    plan = _matrix_plan(4, 3, 8, (0, 1, 2, 3), column_groups=2)

    assert plan.requires_redistribution
    assert len(plan.storage_to_compute_routes) == 8
    assert all(
        route.source_group_rank == route.destination_group_rank
        for route in plan.storage_to_compute_routes
    )
    _assert_round_trip(plan, torch.Size((12, 8)))


def test_single_row_column_groups_need_no_reordering():
    plan = _matrix_plan(4, 1, 8, (0, 1, 2, 3), column_groups=2)

    assert not plan.requires_redistribution
    _assert_round_trip(plan, torch.Size((4, 8)))


def test_route_inverse_swaps_both_offsets_and_row_strides():
    route = _MatrixRoute(3, 6, 4, 2, 4, 2, 1, 4, 12, 8, 4)

    assert route.numel == 8
    assert route.inverse() == _MatrixRoute(3, 6, 4, 2, 4, 1, 2, 12, 4, 4, 8)
    assert route.inverse().inverse() == route


def test_128_heads_on_256_ranks_has_complete_compute_matrices():
    columns = 8192
    plan = _matrix_plan(128, 128, columns, tuple(range(256)))

    assert len(plan.storage_to_compute_routes) == 256
    assert all(partition.num_rows == 64 for partition in plan.storage_partitions)
    assert tuple(len(p.matrix_indices) for p in plan.compute_partitions) == (
        (1,) * 128 + (0,) * 128
    )
    assert plan.storage_to_compute_routes[:4] == (
        _MatrixRoute(0, 0, 0, 64, columns, 0, 0, 0, 0, columns, columns),
        _MatrixRoute(0, 64, 0, 64, columns, 1, 0, 0, 64 * columns, columns, columns),
        _MatrixRoute(1, 128, 0, 64, columns, 2, 1, 0, 0, columns, columns),
        _MatrixRoute(1, 192, 0, 64, columns, 3, 1, 0, 64 * columns, columns, columns),
    )
    assert sum(route.numel for route in plan.storage_to_compute_routes) == (
        16384 * columns
    )


@pytest.mark.parametrize("matrix_rows", [8, 8192])
def test_column_group_route_count_does_not_grow_with_matrix_rows(matrix_rows):
    plan = _matrix_plan(2, matrix_rows, 8, (0, 1, 2, 3), column_groups=2)

    assert len(plan.storage_to_compute_routes) == 8


@pytest.mark.parametrize(
    ("batch_size", "matrix_rows", "mesh_indices", "current_group_rank", "local_rows"),
    [
        (2, 4, (0, 1), 0, 3),
        (5, 3, (0, 1, 2, 3), 3, 4),
        (2, 2, tuple(range(8)), 7, 1),
    ],
)
def test_planner_checks_actual_local_shape_before_owner_assignment(
    batch_size, matrix_rows, mesh_indices, current_group_rank, local_rows
):
    global_shape = torch.Size((batch_size * matrix_rows, 3))
    planner = _MuonLayoutPlanner(mesh_indices, current_group_rank)

    with pytest.raises(
        RuntimeError, match="logical local shard does not match its mesh"
    ):
        planner.resolve_parameter(
            fqn="projection.weight",
            global_storage_shape=global_shape,
            local_storage_shape=torch.Size((local_rows, 3)),
            compute_view=BatchedMatrixComputeView(batch_size).resolve(global_shape),
        )


@pytest.mark.parametrize(
    "view",
    [ResolvedBatchedMatrixView(2, 2, 8), ResolvedBatchedMatrixView(2, 4, 4, 3)],
)
def test_resolved_view_must_preserve_storage_rows_and_columns(view):
    planner = _MuonLayoutPlanner((0, 1), 0)
    with pytest.raises(ValueError, match="compute view does not match its storage"):
        planner.resolve_parameter(
            fqn="projection.weight",
            global_storage_shape=torch.Size((8, 4)),
            local_storage_shape=torch.Size((4, 4)),
            compute_view=view,
        )


@pytest.mark.parametrize("owners", [(), (0,), (0, 1, 2), (-1, 0), (0, 4)])
def test_owner_assignment_must_cover_each_matrix_with_a_valid_group_rank(owners):
    planner, layout = _matrix_layout(2, 3, 4, (0, 1, 2, 3))

    with pytest.raises(ValueError, match="one owner per matrix|outside its group"):
        planner.build_plan(layout, owners)


@pytest.mark.parametrize(
    ("corruption", "value"),
    [
        ("missing", None),
        ("duplicate", None),
        ("matrix_index", -1),
        ("matrix_index", 10),
        ("source_group_rank", -1),
        ("source_group_rank", 4),
        ("destination_group_rank", -1),
        ("destination_group_rank", 4),
        ("destination_group_rank", 1),
        ("global_row_offset", 1),
        ("global_column_offset", 1),
        ("num_rows", 0),
        ("num_rows", 4),
        ("num_columns", 3),
        ("source_flat_offset", 1),
        ("destination_flat_offset", 1),
        ("source_row_stride", 4),
        ("destination_row_stride", 8),
    ],
)
def test_route_validation_rejects_incomplete_or_misplaced_transfers(corruption, value):
    planner, layout = _matrix_layout(5, 3, 8, (0, 1, 2, 3), column_groups=2)
    plan = planner.build_plan(layout, (0, 1, 2, 3, 0, 1, 2, 3, 0, 1))
    routes = plan.storage_to_compute_routes
    if corruption == "missing":
        routes = routes[1:]
    elif corruption == "duplicate":
        routes = (*routes, routes[0])
    else:
        routes = (replace(routes[0], **{corruption: value}), *routes[1:])

    with pytest.raises(RuntimeError, match="invalid Muon redistribution plan:"):
        _MuonLayoutPlanner._validate_routes(layout, plan.compute_partitions, routes)


def test_route_validation_requires_compute_partitions_in_group_rank_order():
    planner, layout = _matrix_layout(2, 3, 4, (0, 1))
    plan = planner.build_plan(layout, (1, 0))

    with pytest.raises(RuntimeError, match="do not follow group-rank order"):
        _MuonLayoutPlanner._validate_routes(
            layout,
            tuple(reversed(plan.compute_partitions)),
            plan.storage_to_compute_routes,
        )


@pytest.mark.parametrize("num_ranks", [8, 128])
def test_route_validation_does_not_rescan_routes_for_each_rank(num_ranks):
    class CountedRoutes(tuple):
        def __new__(cls, routes):
            instance = super().__new__(cls, routes)
            instance.num_visits = 0
            return instance

        def __iter__(self):
            for route in super().__iter__():
                self.num_visits += 1
                yield route

    planner, layout = _matrix_layout(
        num_ranks // 2, 8, 8, tuple(range(num_ranks)), column_groups=2
    )
    plan = planner.build_plan(layout, tuple(range(num_ranks)))
    routes = CountedRoutes(plan.storage_to_compute_routes)

    _MuonLayoutPlanner._validate_routes(layout, plan.compute_partitions, routes)

    assert len(routes) <= routes.num_visits <= 4 * len(routes)
