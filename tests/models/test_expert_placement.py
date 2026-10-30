import torch
from torch.utils.data import Dataset

from torchturbo.models.modules.moe.expert_placement import (
    ExpertPlacementManager,
    _inverse_perm,
    solve_lpt,
)
from torchturbo.models.modules.moe.utils import RoutingResult


def make_manager(num_experts=8, ep_size=2, num_local=4):
    ExpertPlacementManager._reset_for_testing()
    manager = ExpertPlacementManager.activate(
        num_experts, ep_size, num_local, None, interval_steps=2, ema_decay=0.5
    )
    manager._ensure_device(torch.device("cpu"))
    return manager


def test_inverse_perm_roundtrip():
    placement = [3, 0, 2, 1]
    slot_of = _inverse_perm(placement)
    assert slot_of == [1, 3, 2, 0]
    assert _inverse_perm(slot_of) == placement


def test_solve_lpt_fills_grid_and_is_deterministic():
    loads = [float((i * 37) % 11) for i in range(16)]
    placement = solve_lpt(loads, ep_size=4, num_local_experts=4)
    assert sorted(placement) == list(range(16))
    assert solve_lpt(loads, 4, 4) == placement


def test_solve_lpt_spreads_hot_experts():
    # Expert 0-3 are hot; each should land on a different rank.
    loads = [100.0, 90.0, 80.0, 70.0] + [1.0] * 12
    placement = solve_lpt(loads, ep_size=4, num_local_experts=4)
    hot_ranks = {placement.index(e) // 4 for e in range(4)}
    assert len(hot_ranks) == 4


def test_solve_lpt_rejects_bad_grid():
    try:
        solve_lpt([1.0] * 8, ep_size=2, num_local_experts=3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_remap_is_identity_before_first_migration():
    manager = make_manager()
    routing_map = torch.zeros(3, 8, dtype=torch.bool)
    routing_map[0, 2] = True
    routing_map[1, 5] = True
    expert_indices = torch.tensor([[2], [5], [0]])
    routing = RoutingResult(
        probs=torch.rand(3, 8), routing_map=routing_map,
        expert_indices=expert_indices,
    )
    remapped = manager.remap_routing(routing)
    assert remapped is routing
    ExpertPlacementManager._reset_for_testing()


def test_remap_expresses_slot_space_after_migration():
    manager = make_manager()
    # Swap experts 0 and 4 across the two ranks.
    manager._set_placement([4, 1, 2, 3, 0, 5, 6, 7])
    assert not manager.is_identity

    routing_map = torch.zeros(2, 8, dtype=torch.bool)
    routing_map[0, 0] = True  # expert 0 now lives in slot 4
    routing_map[1, 4] = True  # expert 4 now lives in slot 0
    expert_indices = torch.tensor([[0], [4]])
    routing = RoutingResult(
        probs=torch.rand(2, 8), routing_map=routing_map,
        expert_indices=expert_indices,
    )
    remapped = manager.remap_routing(routing)
    # Slot-ordered routing map: slot 0 holds expert 4, slot 4 holds expert 0.
    assert remapped.routing_map[0, 4] and not remapped.routing_map[0, 0]
    assert remapped.routing_map[1, 0] and not remapped.routing_map[1, 4]
    assert remapped.expert_indices.tolist() == [[4], [0]]
    ExpertPlacementManager._reset_for_testing()


def test_accumulate_sums_true_expert_counts():
    manager = make_manager()
    counts = torch.zeros(8)
    counts[3] = 4.0
    manager.accumulate_counts(counts)
    manager.accumulate_counts(counts)
    assert manager._step_counts[3].item() == 8.0
    assert manager._step_counts[0].item() == 0.0
    ExpertPlacementManager._reset_for_testing()


def test_lean_remap_helpers():
    manager = make_manager()
    manager._set_placement([4, 1, 2, 3, 0, 5, 6, 7])
    counts = torch.arange(8, dtype=torch.long)
    torch.testing.assert_close(
        manager.slot_counts(counts), torch.tensor([4, 1, 2, 3, 0, 5, 6, 7])
    )
    idx = torch.tensor([[0], [4]])
    assert manager.remap_indices(idx).tolist() == [[4], [0]]
    ExpertPlacementManager._reset_for_testing()


def test_migrate_dtensor_self_permutation():
    """World=1 migration: permuted slots must receive the right expert rows."""
    import os

    import torch.distributed as dist
    from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        manager = make_manager(num_experts=4, ep_size=1, num_local=4)
        manager._rank_grid = {(0, 0): 0}
        # 4 experts x 3 rows x 2 cols; slot j starts with expert j's rows.
        base = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        mesh = DeviceMesh("cpu", [0])
        dt = distribute_tensor(base.clone(), mesh, [Shard(0)])

        manager._migrate_dtensor(dt, [0, 1, 2, 3], [2, 0, 1, 3])

        expected = base[[6, 7, 8, 0, 1, 2, 3, 4, 5, 9, 10, 11]]
        torch.testing.assert_close(dt.to_local(), expected)
    finally:
        dist.destroy_process_group()
        ExpertPlacementManager._reset_for_testing()
