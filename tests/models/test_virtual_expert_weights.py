"""CPU replica-plan decisions used by ordered EP weight communication."""

import pytest
import torch
import torch.distributed as dist

from torchturbo.ops.providers.native.virtual_expert_weight_exchange import (
    build_weight_exchange_context,
)


@pytest.mark.parametrize("rank", range(3))
def test_empty_replica_plan_is_global(monkeypatch, rank):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.full((3, 2), -1), None,
    )
    assert not plan.has_global_replicas
    assert plan.send_indices.numel() == sum(plan.receive_splits) == 0


@pytest.mark.parametrize("rank", range(3))
def test_idle_rank_retains_global_replica_participation(monkeypatch, rank):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.tensor([[-1, -1], [0, -1], [-1, -1]]), None,
    )
    assert plan.has_global_replicas
    if rank == 2:
        assert plan.send_indices.numel() == sum(plan.receive_splits) == 0


@pytest.mark.parametrize("rank,send_indices,send_splits,receive_splits", [
    (0, [1, 0], [0, 2, 0], [0, 0, 1]),
    (1, [], [0, 0, 0], [2, 0, 0]),
    (2, [1], [1, 0, 0], [0, 0, 0]),
])
def test_owners_send_replicas_in_destination_slot_order(
    monkeypatch, rank, send_indices, send_splits, receive_splits,
):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.tensor([[5, -1], [1, 0], [-1, -1]]), None,
    )
    assert plan.send_indices.tolist() == send_indices
    assert plan.send_splits == send_splits
    assert plan.receive_splits == receive_splits


@pytest.mark.parametrize("rank_copies", [[4, 2], [-1, 2]])
def test_rejects_replica_slots_out_of_owner_order(monkeypatch, rank_copies):
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    with pytest.raises(AssertionError, match="prefix ordered by owner"):
        build_weight_exchange_context(
            torch.tensor([rank_copies, [-1, -1], [-1, -1]]), None,
        )
