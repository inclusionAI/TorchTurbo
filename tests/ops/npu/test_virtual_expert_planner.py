"""NPU virtual-expert planner parity, also runnable with torchrun."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.moe import VirtualExpertPlannerWorkspace
from torchturbo.ops.registry import registry
from torchturbo.runtime import TurboDevice, get_device_runtime, is_npu_available

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(scope="module")
def planner_env(tmp_path_factory):
    import torch_npu  # noqa: F401

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime = get_device_runtime()
    runtime.set_device(local_rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        if "RANK" in os.environ:
            dist.init_process_group("hccl", timeout=timedelta(minutes=5))
        else:
            rendezvous = tmp_path_factory.mktemp("planner") / "rendezvous"
            dist.init_process_group(
                "hccl", init_method=f"file://{rendezvous}", rank=0, world_size=1,
                timeout=timedelta(minutes=5),
            )
    dispatcher = OpDispatcher(
        device_runtime=runtime, registry=registry,
        default_policy=OpPolicy.AUTO,
    )
    yield dispatcher, dist.group.WORLD
    runtime.shmem.release_unused()
    if owns_group:
        dist.destroy_process_group()


def _assert_plan_equal(actual, expected):
    torch.testing.assert_close(actual.virtual_experts, expected.virtual_experts)
    torch.testing.assert_close(actual.experts_to_copy.cpu(), expected.experts_to_copy.cpu())
    assert actual.has_global_replicas == expected.has_global_replicas
    assert actual.num_local_replicas == expected.num_local_replicas
    assert (actual.dispatch_context is None) == (expected.dispatch_context is None)
    if actual.dispatch_context is not None:
        a, b = actual.dispatch_context, expected.dispatch_context
        assert a.input_splits == b.input_splits
        assert a.output_splits == b.output_splits
        torch.testing.assert_close(a.received_counts, b.received_counts)
        torch.testing.assert_close(a.tokens_per_expert, b.tokens_per_expert)
    a, b = actual.weight_context, expected.weight_context
    assert a.ep_group is b.ep_group
    assert a.send_splits == b.send_splits
    assert a.receive_splits == b.receive_splits
    assert a.has_global_replicas == b.has_global_replicas
    torch.testing.assert_close(a.send_indices, b.send_indices)


@contextmanager
def _planner_workspace(runtime, num_experts, group, *, peer_access=True):
    workspace = VirtualExpertPlannerWorkspace(
        num_experts=num_experts, device=runtime.device, group=group, runtime=runtime,
        peer_access=peer_access,
    )
    try:
        yield workspace
    finally:
        workspace.destroy()
        if peer_access:
            runtime.shmem.release_unused()


@contextmanager
def _planner_workspaces(runtime, num_experts, group):
    with (
        _planner_workspace(runtime, num_experts, group) as workspace,
        _planner_workspace(runtime, num_experts, group, peer_access=False) as reference,
    ):
        yield workspace, reference


@pytest.fixture(scope="module", params=[2, 3])
def parity_workspaces(planner_env, request):
    dispatcher, group = planner_env
    num_experts = dist.get_world_size(group) * request.param
    with _planner_workspaces(dispatcher.device_runtime, num_experts, group) as workspaces:
        yield workspaces


def _check_plan(dispatcher, routes, workspace, reference, *, async_finish=False):
    actual = dispatcher.plan_virtual_expert_routes(
        routes, workspace=workspace, async_finish=async_finish,
        impl="npu_triton_plan_virtual_expert_routes",
    ).wait()
    expected = dispatcher.plan_virtual_expert_routes(
        routes, workspace=reference, impl="torch_plan_virtual_expert_routes",
    )
    assert actual.ep_group is workspace.ep_group
    assert actual.weight_context.ep_group is workspace.ep_group
    _assert_plan_equal(actual, expected)
    return actual


def test_candidate_publishes_histograms_without_all_gather(planner_env, monkeypatch):
    dispatcher, group = planner_env
    if dist.get_world_size(group) == 1:
        pytest.skip("Requires distributed histogram exchange")
    num_experts = 2 * dist.get_world_size(group)
    routes = torch.zeros((113 + dist.get_rank(group), 1), dtype=torch.int64, device="npu")
    with _planner_workspaces(dispatcher.device_runtime, num_experts, group) as (workspace, reference):
        expected = dispatcher.plan_virtual_expert_routes(
            routes, workspace=reference, impl="torch_plan_virtual_expert_routes",
        )

        def reject_all_gather(*args, **kwargs):
            pytest.fail("The NPU planner must publish device histograms through SHMEM")

        with monkeypatch.context() as guarded:
            guarded.setattr(dist, "all_gather_into_tensor", reject_all_gather)
            actual = dispatcher.plan_virtual_expert_routes(
                routes, workspace=workspace, async_finish=True,
                impl="npu_triton_plan_virtual_expert_routes",
            )
            assert actual.experts_to_copy.device == routes.device
            assert actual.num_local_replicas is None
            _assert_plan_equal(actual.wait(), expected)


@pytest.mark.parametrize("inference_warmup", [False, True])
def test_async_plans_keep_metadata_when_workspace_is_reused(planner_env, inference_warmup):
    dispatcher, group = planner_env
    runtime = dispatcher.device_runtime
    ep_size = dist.get_world_size(group)
    num_experts = ep_size * 2
    routes = [
        torch.zeros((7, 1), dtype=torch.int64, device=runtime.device),
        torch.full((11, 1), num_experts - 1, dtype=torch.int64, device=runtime.device),
    ]
    with ExitStack() as cleanup:
        with torch.inference_mode(inference_warmup):
            workspace, reference = cleanup.enter_context(_planner_workspaces(runtime, num_experts, group))
            dispatcher.plan_virtual_expert_routes(
                routes[0], workspace=workspace, async_finish=True,
                impl="npu_triton_plan_virtual_expert_routes",
            ).wait()
        scratch = workspace.scratch
        assert (scratch is not None) == (ep_size > 1)
        stream = runtime.new_stream()
        other_stream = runtime.new_stream()
        stream.wait_stream(runtime.current_stream())
        other_stream.wait_stream(runtime.current_stream())
        plans = []
        for ids, planner_stream in zip(routes, (stream, other_stream), strict=True):
            with runtime.stream(planner_stream):
                plans.append(dispatcher.plan_virtual_expert_routes(
                    ids, workspace=workspace, async_finish=True,
                    impl="npu_triton_plan_virtual_expert_routes",
                ))
        for plan, ids in zip(plans, routes, strict=True):
            expected = dispatcher.plan_virtual_expert_routes(
                ids, workspace=reference, impl="torch_plan_virtual_expert_routes",
            )
            assert plan.wait() is plan
            _assert_plan_equal(plan, expected)
            assert plan.wait() is plan
        assert workspace.scratch is scratch


def test_managed_weight_push_consumes_device_plan_before_wait(planner_env, monkeypatch):
    from torchturbo.ops.moe import VirtualExpertPlan

    dispatcher, group = planner_env
    runtime = dispatcher.device_runtime
    ep_size = dist.get_world_size(group)
    if ep_size == 1:
        pytest.skip("Requires remote replicas")
    rank = dist.get_rank(group)
    num_local_experts = 2
    num_experts = ep_size * num_local_experts
    routes = torch.zeros((113 + rank, 1), dtype=torch.int64, device=runtime.device)
    with _planner_workspaces(runtime, num_experts, group) as (workspace, reference):
        expected = dispatcher.plan_virtual_expert_routes(
            routes, workspace=reference, impl="torch_plan_virtual_expert_routes",
        )
        weights = tuple(
            torch.full((num_local_experts, 3, 4), rank + offset, device=runtime.device)
            for offset in (1.0, 2.0)
        )
        arena, handle = runtime.allocate_communication_buffer(
            sum(weight.numel() for weight in weights), dtype=weights[0].dtype,
            device=weights[0].device, group=group,
        )
        out = tuple(part.view_as(weight) for part, weight in zip(arena.chunk(2), weights, strict=True))
        plan = dispatcher.plan_virtual_expert_routes(
            routes, workspace=workspace, async_finish=True,
            impl="npu_triton_plan_virtual_expert_routes",
        )

        def reject_wait(self):
            pytest.fail("Managed SHMEM weight push must consume device copies before plan.wait")

        with monkeypatch.context() as guarded, torch.no_grad():
            guarded.setattr(VirtualExpertPlan, "wait", reject_wait)
            result = dispatcher.virtual_expert_weight_exchange(
                *weights, None, None, plan, out=out, buffer_handles=(handle, None),
                impl="npu_shmem_virtual_expert_weight_exchange",
            )
        assert result[0] is out[0] and result[1] is out[1]
        assert result[0].shape[0] == num_local_experts
        plan.wait()
        _assert_plan_equal(plan, expected)
        for slot, expert in enumerate(expected.experts_to_copy[rank].tolist()):
            if expert < 0:
                break
            owner = expert // num_local_experts
            for value, offset in zip(result[:2], (1.0, 2.0), strict=True):
                torch.testing.assert_close(value[slot], torch.full_like(value[slot], owner + offset))


def _routes(case, rank, num_experts, dtype):
    generator = torch.Generator().manual_seed(713 + rank)
    if case == "empty":
        ids = torch.empty((0, 3), dtype=dtype)
    elif case == "zero_topk":
        ids = torch.empty((3 + rank, 0), dtype=dtype)
    elif case == "ties":
        ids = torch.arange(num_experts, dtype=dtype).repeat(3).reshape(-1, 3)
    elif case == "single_hot":
        ids = torch.zeros((173 + 19 * rank, 3), dtype=dtype)
    elif case == "mixed_hot":
        ids = torch.tensor([1, 0, num_experts - 1], dtype=dtype).repeat(193 + rank, 1)
    elif case == "repeated":
        ids = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=dtype).repeat(151 + rank, 1)
    else:
        length = 0 if case == "one_empty_rank" and rank == 0 else 269 + 37 * rank
        ids = torch.randint(num_experts, (length, 3), generator=generator, dtype=dtype)
    ids = ids.to(device="npu")
    if case == "strided":
        storage = torch.empty((ids.shape[0] * 2, 6), device="npu", dtype=dtype)
        storage[::2, ::2] = ids
        ids = storage[::2, ::2]
        assert not ids.is_contiguous()
    return ids


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("case", [
    "ties", "single_hot", "mixed_hot", "empty", "zero_topk",
    "one_empty_rank", "uneven_random", "strided", "repeated",
])
def test_plan_matches_native(planner_env, parity_workspaces, dtype, case):
    dispatcher, group = planner_env
    workspace, reference = parity_workspaces
    routes = _routes(case, dist.get_rank(group), workspace.num_experts, dtype)
    _check_plan(dispatcher, routes, workspace, reference)


def test_auto_selects_npu_planner(planner_env):
    dispatcher, _ = planner_env
    selected = dispatcher._candidate_impl("plan_virtual_expert_routes", TurboDevice.NPU)
    assert selected.op_spec().impl_name == "npu_triton_plan_virtual_expert_routes"


def test_ep1_keeps_input_column_order(planner_env):
    dispatcher, world = planner_env
    rank = dist.get_rank(world)
    groups = [dist.new_group([member]) for member in range(dist.get_world_size(world))]
    group = groups[rank]
    try:
        routes = torch.tensor([[2, 0, 2], [1, 1, 0]], device="npu", dtype=torch.int32)
        with _planner_workspaces(dispatcher.device_runtime, 3, group) as (workspace, reference):
            plan = _check_plan(dispatcher, routes, workspace, reference)
        assert plan.dispatch_context is None
        torch.testing.assert_close(plan.virtual_experts, routes.long())
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("case", ["full_deficit", "copy_slot_tie"])
def test_reference_placement_edge_cases(planner_env, case):
    dispatcher, world = planner_env
    group_size = 4 if case == "full_deficit" else 2
    if dist.get_world_size(world) % group_size:
        pytest.skip(f"Requires WORLD to partition into groups of {group_size} ranks")
    groups = [
        dist.new_group(list(range(start, start + group_size)))
        for start in range(0, dist.get_world_size(world), group_size)
    ]
    group = groups[dist.get_rank(world) // group_size]
    try:
        totals = [6, 4, 5, 4, 2, 2, 1, 0] if case == "full_deficit" else [2, 2, 2, 2, 0, 0, 0, 0]
        ids = torch.arange(8).repeat_interleave(torch.tensor(totals)).reshape(-1, 1)
        if dist.get_rank(group) != 0:
            ids = ids[:0]
        with _planner_workspaces(dispatcher.device_runtime, 8, group) as (workspace, reference):
            plan = _check_plan(dispatcher, ids.to("npu"), workspace, reference)
        copies = [[3, -1], [-1, -1], [2, -1], [0, -1]] if case == "full_deficit" else [
            [-1, -1, -1, -1], [1, 0, -1, -1],
        ]
        torch.testing.assert_close(plan.experts_to_copy.cpu(), torch.tensor(copies))
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("num_local_experts", [3, 5, 7])
@pytest.mark.parametrize("scale", [1, 2**32 + 1])
@pytest.mark.parametrize("ep_size", [3, 4, 5])
def test_allocation_with_padded_local_experts(planner_env, num_local_experts, scale, ep_size):
    import triton

    from torchturbo.ops.providers.native.plan_virtual_expert_routes import plan_placement
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _allocation_kernel

    dispatcher, _ = planner_env
    device = dispatcher.device_runtime.device
    num_experts = ep_size * num_local_experts
    counts = torch.zeros((ep_size, num_experts), dtype=torch.int64)
    counts[:, [0, 1, num_experts - 1]] = torch.arange(193, 193 + ep_size)[:, None] * scale
    expected, _ = plan_placement(counts)
    totals = counts.sum(0)
    loads = totals.reshape(ep_size, num_local_experts).sum(1).to(device)
    totals = totals.to(device)
    allocations = torch.empty((num_experts, ep_size), dtype=torch.int64, device=device)
    boundaries = torch.empty_like(allocations)
    for _ in range(3):
        allocations.fill_(-1)
        boundaries.fill_(-1)
        for _ in range(10):
            _allocation_kernel[(ep_size,)](
                totals, loads, allocations, boundaries,
                ep_size, num_local_experts,
                triton.next_power_of_2(ep_size), triton.next_power_of_2(num_local_experts),
            )
        torch.testing.assert_close(allocations.cpu(), expected)
        torch.testing.assert_close(boundaries.cpu(), expected.cumsum(1))


@pytest.mark.parametrize(
    "ep_size,num_experts,rank_skew",
    [(2, 256, None), (4, 256, None), (8, 256, None), (8, 512, 0), (8, 512, 1)],
)
def test_allocation_repeated_launches_with_many_experts(planner_env, ep_size, num_experts, rank_skew):
    import triton

    from torchturbo.ops.providers.native.plan_virtual_expert_routes import plan_placement
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _allocation_kernel

    dispatcher, _ = planner_env
    device = dispatcher.device_runtime.device
    num_local_experts = num_experts // ep_size
    if rank_skew is None:
        counts = torch.randint(
            0, 1024, (ep_size, num_experts), dtype=torch.int64,
            generator=torch.Generator().manual_seed(823 + ep_size),
        )
    else:
        counts = torch.ones(ep_size, num_experts, dtype=torch.int64)
        counts[0, 0] += rank_skew
        counts[0, -1] -= rank_skew
    expected, _ = plan_placement(counts)
    totals = counts.sum(0)
    loads = totals.reshape(ep_size, num_local_experts).sum(1).to(device)
    totals = totals.to(device)
    allocations = torch.empty((num_experts, ep_size), dtype=torch.int64, device=device)
    boundaries = torch.empty_like(allocations)
    for _ in range(3):
        # Queue launches before reading results to cover Ascend store-lock reuse.
        for _ in range(10):
            _allocation_kernel[(ep_size,)](
                totals, loads, allocations, boundaries, ep_size, num_local_experts,
                triton.next_power_of_2(ep_size), triton.next_power_of_2(num_local_experts),
            )
        torch.testing.assert_close(allocations.cpu(), expected)
        torch.testing.assert_close(boundaries.cpu(), expected.cumsum(1))


@pytest.mark.parametrize("num_local_experts", [2, 3])
def test_device_placement_keeps_int64_boundaries(planner_env, num_local_experts):
    from torchturbo.ops.providers.native.plan_virtual_expert_routes import (
        build_remap_table, count_routes, plan_placement,
    )
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _allocate_scratch, _place_routes

    dispatcher, group = planner_env
    num_ranks = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_experts = num_ranks * num_local_experts
    generator = torch.Generator().manual_seed(913)
    counts = torch.randint(0, 7, (num_ranks, num_experts), generator=generator)
    counts[:, 1::2] = 0
    counts = counts * (2**32 + 1)
    counts[0, 0] += 3
    scratch = _allocate_scratch(num_experts, num_ranks, dispatcher.device_runtime.device)
    allocations, copies, boundaries, slots, prefixes, statistics = _place_routes(
        counts.to("npu"), rank, scratch,
    )
    expected_allocations, expected_copies = plan_placement(counts)
    torch.testing.assert_close(allocations.cpu(), expected_allocations)
    torch.testing.assert_close(copies.cpu(), expected_copies)
    torch.testing.assert_close(boundaries.cpu(), expected_allocations.cumsum(1))
    torch.testing.assert_close(prefixes.cpu(), counts.cumsum(0) - counts)
    expected_table = build_remap_table(counts, expected_allocations, expected_copies, rank)
    replica_slots = expected_table[:, 1 + num_ranks:] - num_local_experts
    active = replica_slots >= 0
    torch.testing.assert_close(slots.cpu()[active], replica_slots[active])
    sent, received = count_routes(counts, expected_allocations, expected_copies, rank)
    torch.testing.assert_close(statistics[:num_ranks].cpu(), sent)
    torch.testing.assert_close(
        statistics[num_ranks:].cpu().reshape_as(received), received,
    )


def _expected_copy_slots(copies):
    ep_size, num_local_experts = copies.shape
    slots = torch.full((ep_size * num_local_experts, ep_size), -1, dtype=torch.int64)
    for destination, row in enumerate(copies.tolist()):
        for slot, expert in enumerate(row):
            if expert >= 0:
                slots[expert, destination] = slot
    return slots


@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("ep_size,num_local_experts,case", [
    (2, 2, "single_hot"), (2, 4, "multiple_hot"),
    (4, 3, "multiple_hot"), (8, 32, "multiple_hot"),
])
def test_copy_ids_survive_queued_launches(
    planner_env, side_stream, ep_size, num_local_experts, case,
):
    import triton

    from torchturbo.ops.providers.native.plan_virtual_expert_routes import plan_placement
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _copies_kernel

    dispatcher, _ = planner_env
    runtime = dispatcher.device_runtime
    num_experts = ep_size * num_local_experts
    counts = torch.zeros((ep_size, num_experts), dtype=torch.int64)
    if case == "single_hot":
        counts[:, 0] = torch.arange(3, 3 + ep_size)
    else:
        # Odd owners donate several positive expert IDs to preceding even ranks.
        hot = (torch.arange(num_experts) // num_local_experts) % 2 == 1
        counts[:, hot] = torch.arange(3, 3 + ep_size)[:, None]
    allocations, expected_copies = plan_placement(counts)
    expected_slots = _expected_copy_slots(expected_copies)
    allocations = allocations.contiguous().to(runtime.device)
    stream = runtime.new_stream() if side_stream else runtime.current_stream()
    stream.wait_stream(runtime.current_stream())
    outputs = []
    with runtime.stream(stream):
        for _ in range(50):
            copies = torch.empty(expected_copies.shape, dtype=torch.int64, device=runtime.device)
            slots = torch.empty_like(allocations)
            _copies_kernel[(ep_size,)](
                allocations, copies, slots, ep_size, num_experts,
                num_local_experts, triton.next_power_of_2(num_experts),
            )
            outputs.append((copies, slots))
    stream.synchronize()
    for copies, slots in outputs:
        torch.testing.assert_close(copies.cpu(), expected_copies)
        torch.testing.assert_close(slots.cpu(), expected_slots)


@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("ep_size,num_local_experts", [(2, 4), (3, 3), (4, 3), (8, 32)])
def test_device_placement_survives_queued_launches(
    planner_env, side_stream, ep_size, num_local_experts,
):
    from torchturbo.ops.providers.native.plan_virtual_expert_routes import count_routes, plan_placement
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _allocate_scratch, _place_routes

    dispatcher, _ = planner_env
    runtime = dispatcher.device_runtime
    num_experts = ep_size * num_local_experts

    cases = []
    counts = torch.arange(1, ep_size * num_experts + 1).reshape(ep_size, num_experts)
    for scale in (1, 2**32 + 1):
        scaled_counts = counts * scale
        counts_device = scaled_counts.contiguous().to(runtime.device)
        allocations, copies = plan_placement(scaled_counts)
        boundaries = allocations.cumsum(1)
        slots = _expected_copy_slots(copies)
        prefixes = scaled_counts.cumsum(0) - scaled_counts
        totals = scaled_counts.sum(0)
        loads = totals.reshape(ep_size, num_local_experts).sum(1)
        for rank in range(ep_size):
            sent, received = count_routes(scaled_counts, allocations, copies, rank)
            statistics = torch.cat((sent, received.flatten()))
            expected = (allocations, copies, boundaries, slots, prefixes, statistics, totals, loads)
            cases.append((counts_device, rank, expected))

    stream = runtime.new_stream() if side_stream else runtime.current_stream()
    stream.wait_stream(runtime.current_stream())
    outputs = []
    with runtime.stream(stream):
        for launch in range(50):
            counts_device, rank, expected = cases[launch % len(cases)]
            # Independent arenas retain every launch's intermediates until checking.
            scratch = _allocate_scratch(num_experts, ep_size, runtime.device)
            actual = _place_routes(counts_device, rank, scratch) + (
                scratch["global_tokens_per_expert"], scratch["balance"],
            )
            outputs.append((scratch, actual, expected))
    stream.synchronize()
    for _, actual, expected in outputs:
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result.cpu(), reference)


def test_launcher_uses_workspace_topology_and_raw_replica_slots(planner_env):
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import launch_virtual_expert_planner

    dispatcher, group = planner_env
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    with _planner_workspace(dispatcher.device_runtime, 2 * size, group) as workspace:
        assert workspace.scratch is None
        routes = torch.zeros((4, 1), dtype=torch.int64, device=workspace.device)
        mapped, copies = launch_virtual_expert_planner(routes, workspace)
        torch.testing.assert_close(mapped.cpu(), torch.full((4, 1), 0 if rank == 0 else 4 * rank + 2))
        expected_copies = torch.full((size, 2), -1)
        expected_copies[1:, 0] = 0
        torch.testing.assert_close(copies.cpu(), expected_copies)
        torch.testing.assert_close(workspace.field("virtual_expert_slots")[0, 1:].cpu(), torch.zeros(size - 1, dtype=torch.int64))
    assert workspace.scratch is None


def test_placement_reuses_scratch_and_retains_independent_copy_tables(planner_env):
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _allocate_scratch, _place_routes

    dispatcher, group = planner_env
    ep_size = dist.get_world_size(group)
    num_experts = 3 * ep_size
    scratch = _allocate_scratch(num_experts, ep_size, dispatcher.device_runtime.device)
    counts = torch.zeros((ep_size, num_experts), dtype=torch.int64, device="npu")
    counts[:, 0] = 12
    first = _place_routes(counts, dist.get_rank(group), scratch)
    first_copies = first[1].clone()
    allocations = []

    class RecordAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.empty.memory_format:
                allocations.append(result)
            return result

    counts.zero_()
    counts[:, -1] = 12
    with RecordAllocations():
        second = _place_routes(counts, dist.get_rank(group), scratch)
    # Only the copy table belongs to the new plan; placement intermediates reuse scratch.
    assert len(allocations) == 1 and allocations[0].data_ptr() == second[1].data_ptr()
    for index in (0, 2, 3, 4, 5):
        assert first[index].data_ptr() == second[index].data_ptr()
        assert second[index].untyped_storage().data_ptr() == scratch["program_histogram"].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()
    torch.testing.assert_close(first[1], first_copies)


def test_stable_occurrences_across_multiple_tiles_per_program(planner_env):
    dispatcher, group = planner_env
    rank = dist.get_rank(group)
    num_experts = 3 * dist.get_world_size(group)
    generator = torch.Generator().manual_seed(1171 + rank)
    routes = torch.randint(num_experts, (9001 + 37 * rank, 2), generator=generator)
    with _planner_workspaces(dispatcher.device_runtime, num_experts, group) as (workspace, reference):
        _check_plan(dispatcher, routes.to("npu"), workspace, reference)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_local_experts", [2, 3])
def test_varying_lengths_reuse_compiled_planner(
    planner_env, monkeypatch, dtype, num_local_experts,
):
    from triton.runtime.jit import JITFunction

    from torchturbo.ops.providers.npu._kernels import virtual_expert_planner

    dispatcher, group = planner_env
    ep_size, rank = dist.get_world_size(group), dist.get_rank(group)
    if ep_size == 1:
        pytest.skip("Run with torchrun EP2 to exercise the planner kernels")
    runtime = dispatcher.device_runtime
    num_experts = num_local_experts * ep_size
    generator = torch.Generator().manual_seed(1907 + rank)
    misses = []

    def cache_miss(**kwargs):
        if kwargs["fn"].module == virtual_expert_planner.__name__:
            misses.append(kwargs["fn"].name)
        return False

    with _planner_workspaces(runtime, num_experts, group) as (workspace, reference):
        def run(num_tokens):
            routes = torch.randint(
                num_experts, (num_tokens, 3), generator=generator, dtype=dtype,
            ).to(runtime.device)
            _check_plan(dispatcher, routes, workspace, reference, async_finish=True)

        run(32)
        with monkeypatch.context() as guarded:
            guarded.setattr(JITFunction, "cache_hook", cache_miss)
            # Cross tile and program boundaries, then reuse after an empty plan.
            for num_tokens in (1, 17, 511, 512, 513, 5461, 5462, 9001, 0, 32):
                run(num_tokens)
            run(0 if rank == 0 else 173 + rank)
            run(8192 + 37 * rank)
        assert misses == [], f"Token lengths created planner specializations: {misses}"


@pytest.mark.parametrize("source_prefix", [2**32 - 1, 2**48 + 2**31 + 1])
def test_remap_preserves_large_source_occurrence_ordinals(planner_env, source_prefix):
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _remap_kernel

    routes = torch.tensor([[0, 0]], device="npu")
    histograms = torch.tensor([[2, 0, 0, 0]], device="npu")
    prefixes = torch.tensor([[0, 0, 0, 0], [source_prefix, 0, 0, 0]], device="npu")
    boundaries = torch.tensor([
        [source_prefix, source_prefix + 2], [0, 0], [0, 0], [0, 0],
    ], device="npu")
    slots = torch.tensor([[-1, 0], [-1, -1], [-1, -1], [-1, -1]], device="npu")
    mapped = torch.empty_like(routes)
    _remap_kernel[(1,)](
        routes, histograms, prefixes, boundaries, slots, mapped,
        NUM_ROUTES=2, NUM_EXPERTS=4, EP_RANK=1, EP_SIZE=2,
        NUM_LOCAL_EXPERTS=2, LOG_EP=1, ROUTES_PER_PROGRAM=512,
        BLOCK_EXPERTS=4, BLOCK_PROGRAMS=1, ROUTE_TILE=128,
    )
    torch.testing.assert_close(mapped.cpu(), torch.tensor([[6, 6]]))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ep_size", [3, 4])
def test_rank_values_reuse_compiled_planner_kernels(planner_env, dtype, ep_size):
    import triton

    from torchturbo.ops.providers.native.plan_virtual_expert_routes import (
        _remap_routes, _route_metadata, build_remap_table, count_routes, plan_placement,
    )
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import (
        _remap_kernel, _statistics_kernel,
    )

    dispatcher, _ = planner_env
    device = dispatcher.device_runtime.device
    num_local_experts = 3
    num_experts = ep_size * num_local_experts
    routes = torch.randint(
        3, (ep_size, 193, 3), dtype=dtype, generator=torch.Generator().manual_seed(2017),
    )
    routes[routes == 2] = num_experts - 1
    counts = torch.stack([torch.bincount(row.flatten(), minlength=num_experts) for row in routes])
    allocations, copies = plan_placement(counts)
    counts_device = counts.to(device)
    prefixes = (counts.cumsum(0) - counts).to(device)
    boundaries = allocations.cumsum(1).to(device)
    copies_device = copies.contiguous().to(device)
    table = build_remap_table(counts, allocations, copies, 0)
    slots = (table[:, 1 + ep_size:] - num_local_experts).contiguous().to(device)
    statistics = torch.empty(ep_size * (1 + 2 * num_local_experts), dtype=torch.int64, device=device)
    statistics_hashes, remap_hashes = set(), set()
    for rank in range(ep_size):
        local_routes = routes[rank].to(device)
        histograms = torch.stack([
            torch.bincount(row, minlength=num_experts)
            for row in routes[rank].flatten().split(512)
        ])
        program_histograms = torch.zeros((32, num_experts), dtype=torch.int64)
        program_histograms[:histograms.shape[0]] = histograms
        program_histograms = program_histograms.to(device)
        mapped = torch.empty_like(local_routes, dtype=torch.int64)
        statistics_kernel = _statistics_kernel[(ep_size,)](
            counts_device, prefixes, boundaries, copies_device, statistics,
            rank, ep_size, num_experts, num_local_experts,
            triton.next_power_of_2(num_experts), triton.next_power_of_2(2 * num_local_experts),
        )
        remap_kernel = _remap_kernel[(32,)](
            local_routes, program_histograms, prefixes, boundaries, slots, mapped,
            local_routes.numel(), num_experts, rank, ep_size, num_local_experts,
            (ep_size - 1).bit_length(), 512, triton.next_power_of_2(num_experts), 32, 128,
        )
        sent, received = count_routes(counts, allocations, copies, rank)
        torch.testing.assert_close(statistics.cpu(), torch.cat((sent, received.flatten())))
        positions, _ = _route_metadata(routes[rank], num_experts)
        table = build_remap_table(counts, allocations, copies, rank)
        expected = _remap_routes(routes[rank], positions, table)
        torch.testing.assert_close(mapped.cpu(), expected.to(torch.int64))
        statistics_hashes.add(statistics_kernel.hash)
        remap_hashes.add(remap_kernel.hash)
    assert len(statistics_hashes) == 1, "EP_RANK created statistics specializations"
    assert len(remap_hashes) == 1, "EP_RANK created remap specializations"


@pytest.mark.parametrize("first", ["counts", "prefixes", "boundaries", "copies"])
def test_statistics_at_expandable_allocator_boundary(first):
    environment = os.environ.copy()
    environment["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    probe = Path(__file__).with_name("virtual_expert_statistics_boundary.py")
    result = subprocess.run(
        [sys.executable, str(probe), "--first", first], env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"STATISTICS_EXPANDABLE_BOUNDARY_PASS first={first}" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-s"]))
