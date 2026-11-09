"""Placement conservation and virtual expert route identity contracts."""

import importlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.ops.providers.native.plan_virtual_expert_routes import (
    plan_placement, count_routes, build_remap_table,
)
from torchturbo.ops import OpPolicy, get_ops, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime



class _NoScalarTensorReads(TorchDispatchMode):
    """Reject tensor-to-Python scalar reads in the CPU planning helpers."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, (
            "CPU planning must read tensor data in bulk."
        )
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("helper", ["placement", "counts", "remap"])
def test_cpu_planning_reads_tensor_data_in_bulk(helper):
    counts = torch.tensor([[3, 0, 0, 1], [2, 0, 0, 0]])
    allocations = torch.tensor([[3, 2], [0, 0], [0, 0], [0, 1]])
    copies = torch.tensor([[-1, -1], [0, -1]])
    with _NoScalarTensorReads():
        if helper == "placement":
            plan_placement(counts)
        elif helper == "counts":
            count_routes(counts, allocations, copies, 1)
        else:
            build_remap_table(counts, allocations, copies, 1)


def test_sparse_replica_slots_keep_int64_route_boundaries():
    num_routes = 2**31 + 3
    counts = torch.tensor([[num_routes, 0, 0, 1], [2, 0, 0, 0]])
    allocations = torch.tensor([[num_routes, 2], [0, 0], [0, 0], [0, 1]])
    copies = torch.tensor([[-1, -1], [-1, 0]])
    sent, received = count_routes(counts, allocations, copies, 1)
    torch.testing.assert_close(sent, torch.tensor([0, 2]))
    torch.testing.assert_close(received, torch.tensor([[0, 1, 0, 0], [0, 0, 0, 2]]))
    torch.testing.assert_close(build_remap_table(counts, allocations, copies, 1), torch.tensor([
        [num_routes, num_routes, num_routes + 2, 0, 3],
        [-2, 0, 0, 1, -1], [-2, 0, 0, -1, 0], [-1, 0, 1, -1, 1],
    ]))


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)


def _remap(indices, counts, allocations, copies, rank):
    """Map logical IDs through the dispatcher's ops, aligned with input columns."""
    positions, local_counts = get_ops().moe_route_metadata(indices, counts.shape[1])
    torch.testing.assert_close(local_counts, counts[rank])
    table = build_remap_table(counts, allocations, copies, rank)
    mapped, order = get_ops().moe_remap_routes(indices, positions, table)
    return torch.empty_like(mapped).scatter_(1, order, mapped)


def test_full_deficit_transfer_can_exceed_sender_excess():
    counts = torch.tensor([
        [6, 4, 5, 4, 2, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [1, 0, 0, 5], [4, 0, 0, 0], [0, 3, 2, 0], [1, 3, 0, 0],
        [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([
        [3, -1], [-1, -1], [2, -1], [0, -1],
    ]))


def test_rank_and_expert_ties_choose_lowest_id():
    counts = torch.tensor([[2, 2, 2, 2, 0, 0], [0] * 6, [0] * 6])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [0, 0, 2], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([[2, -1], [-1, -1], [0, -1]]))


def test_replica_slot_ties_choose_highest_global_expert():
    counts = torch.tensor([[2, 2, 2, 2, 0, 0, 0, 0], [0] * 8])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [0, 2], [0, 2], [2, 0], [2, 0], [0, 0], [0, 0], [0, 0], [0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1, -1, -1], [1, 0, -1, -1]]))


def test_floor_ceil_targets_and_empty_experts():
    counts = torch.tensor([[5, 0, 0, 0], [0, 0, 0, 0]])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 2], [0, 0], [0, 0], [0, 0]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [0, -1]]))


@pytest.mark.parametrize("num_ranks,num_local_experts", [(1, 3), (2, 2), (4, 1)])
def test_all_empty_placement(num_ranks, num_local_experts):
    num_experts = num_ranks * num_local_experts
    counts = torch.zeros((num_ranks, num_experts), dtype=torch.long)
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.zeros((num_experts, num_ranks), dtype=torch.long))
    torch.testing.assert_close(copies, torch.full((num_ranks, num_local_experts), -1))
    for rank in range(num_ranks):
        indices = torch.empty((0, 2), dtype=torch.long)
        mapped = _remap(indices, counts, allocations, copies, rank)
        assert mapped.shape == indices.shape


def test_remap_preserves_source_rank_and_flattened_topk_order():
    counts = torch.tensor([[2, 0, 1, 1], [1, 0, 0, 1]])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 0], [0, 0], [0, 1], [0, 2]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [-1, -1]]))
    first = torch.tensor([[0, 3], [2, 0]])
    second = torch.tensor([[3, 0]])
    torch.testing.assert_close(
        _remap(first, counts, allocations, copies, 0),
        torch.tensor([[0, 5], [4, 0]]),
    )
    torch.testing.assert_close(
        _remap(second, counts, allocations, copies, 1),
        torch.tensor([[5, 0]]),
    )


def test_remap_splits_expert_occurrences_across_sources():
    counts = torch.tensor([[3, 0, 0, 1], [2, 0, 0, 0]])
    allocations, copies = plan_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 2], [0, 0], [0, 0], [0, 1]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [0, -1]]))
    first = torch.tensor([[0, 3], [0, 0]])
    second = torch.tensor([[0, 0]])
    torch.testing.assert_close(
        _remap(first, counts, allocations, copies, 0),
        torch.tensor([[0, 5], [0, 0]]),
    )
    torch.testing.assert_close(
        _remap(second, counts, allocations, copies, 1),
        torch.tensor([[6, 6]]),
    )


def test_remap_table_encodes_source_prefix_and_replica_slots():
    counts = torch.tensor([[3, 0, 0, 1], [2, 0, 0, 0]])
    allocations = torch.tensor([[3, 2], [0, 0], [0, 0], [0, 1]])
    copies = torch.tensor([[-1, -1], [0, -1]])
    expected = torch.tensor([
        [0, 3, 5, 0, 2], [-3, 0, 0, 1, -1],
        [-3, 0, 0, -1, 0], [-3, 0, 1, -1, 1],
    ])
    first = build_remap_table(counts, allocations, copies, 0)
    second = build_remap_table(counts, allocations, copies, 1)
    torch.testing.assert_close(first, expected)
    expected[:, 0] = torch.tensor([3, -2, -2, -1])
    torch.testing.assert_close(second, expected)


@pytest.mark.parametrize("num_ranks,num_local_experts", [(1, 3), (2, 1), (3, 2), (4, 3), (8, 2)])
def test_random_placement_conserves_routes_and_bounds_replica_slots(num_ranks, num_local_experts):
    generator = torch.Generator().manual_seed(781)
    num_experts = num_ranks * num_local_experts
    for _ in range(50):
        counts = torch.randint(0, 13, (num_ranks, num_experts), generator=generator)
        counts[:, torch.randperm(num_experts, generator=generator)[:num_experts // 2]] = 0
        allocations, copies = plan_placement(counts)
        torch.testing.assert_close(allocations.sum(dim=1), counts.sum(dim=0))
        assert (allocations >= 0).all()
        total = int(counts.sum())
        targets = torch.full((num_ranks,), total // num_ranks)
        targets[:total % num_ranks] += 1
        torch.testing.assert_close(allocations.sum(dim=0), targets)
        for rank in range(num_ranks):
            remote = torch.nonzero(allocations[:, rank] > 0).flatten()
            remote = remote[remote // num_local_experts != rank]
            selected = copies[rank][copies[rank] >= 0]
            assert len(selected) <= num_local_experts
            torch.testing.assert_close(selected.sort().values, remote.sort().values)
            assert len(selected.unique()) == len(selected)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_random_remap_keeps_logical_identity_and_exact_assignment_counts(topk):
    generator = torch.Generator().manual_seed(97 + topk)
    num_ranks, num_local_experts = 4, 2
    num_experts = num_ranks * num_local_experts
    for _ in range(20):
        source_indices = [
            torch.rand((num_tokens, num_experts), generator=generator).argsort(dim=1)[:, :topk]
            for num_tokens in (0, 1, 7, 3)
        ]
        counts = torch.stack([
            torch.bincount(indices.flatten(), minlength=num_experts) for indices in source_indices
        ])
        allocations, copies = plan_placement(counts)
        assigned = torch.zeros_like(allocations)
        runtime_to_logical = torch.cat([
            torch.cat((torch.arange(rank * num_local_experts, (rank + 1) * num_local_experts), copies[rank]))
            for rank in range(num_ranks)
        ])
        for rank, indices in enumerate(source_indices):
            mapped = _remap(indices, counts, allocations, copies, rank)
            assert mapped.shape == indices.shape
            assert mapped.device == indices.device
            assert (mapped >= 0).all()
            assert (mapped < num_ranks * 2 * num_local_experts).all()
            assert (mapped.sort(dim=1).values.diff(dim=1) > 0).all()
            torch.testing.assert_close(runtime_to_logical[mapped], indices)
            destinations = mapped // (2 * num_local_experts)
            cells = indices.flatten() * num_ranks + destinations.flatten()
            assigned += torch.bincount(cells, minlength=num_experts * num_ranks).reshape_as(assigned)
        torch.testing.assert_close(assigned, allocations)


@pytest.mark.parametrize("num_ranks,num_local_experts", [(1, 2), (2, 2), (4, 3)])
def test_route_counts_match_remapped_indices(num_ranks, num_local_experts):
    generator = torch.Generator().manual_seed(53 + num_ranks)
    num_experts = num_ranks * num_local_experts
    num_runtime_experts = 2 * num_local_experts
    for _ in range(20):
        source_indices = [
            torch.rand((num_tokens, num_experts), generator=generator).argsort(dim=1)[:, :2]
            for num_tokens in torch.randint(0, 9, (num_ranks,), generator=generator).tolist()
        ]
        counts = torch.stack([
            torch.bincount(indices.flatten(), minlength=num_experts) for indices in source_indices
        ])
        allocations, copies = plan_placement(counts)
        mapped = [
            _remap(indices, counts, allocations, copies, rank).flatten()
            for rank, indices in enumerate(source_indices)
        ]
        for rank in range(num_ranks):
            sent, received = count_routes(counts, allocations, copies, rank)
            torch.testing.assert_close(
                sent, torch.bincount(mapped[rank] // num_runtime_experts, minlength=num_ranks),
            )
            local = [
                runtime[runtime // num_runtime_experts == rank] % num_runtime_experts
                for runtime in mapped
            ]
            torch.testing.assert_close(received, torch.stack([
                torch.bincount(slots, minlength=num_runtime_experts) for slots in local
            ]))
            assert (received[:, num_local_experts + int((copies[rank] >= 0).sum()):] == 0).all()
