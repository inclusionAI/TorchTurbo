"""Unit tests for vision-balance packing and gather shapes.

The plan dataclasses are exercised without dist; the metadata gathers
run against a mocked collective, so no process group is needed.
"""

import pytest
import torch
import torch.nn as nn

from torchturbo.parallel import VisionBalanceMode
from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
from torchturbo.models.modules.vision import VisionBatch
from torchturbo.models.modules.vision.balance import (
    _FanoutPlan,
    _MediaMeta,
    _MediaPool,
    _PixelPlan,
    _VisionBalanceState,
)


def _meta(src, index, patches, mode, merge_unit=4):
    return _MediaMeta(
        src=src,
        index=index,
        grid=(1, patches, 1),
        patches=patches,
        tokens={"features": patches // merge_unit},
        weight=_MediaMeta.lpt_weight(patches, mode),
    )


def _owned(src, index, patches, tokens, owner):
    """A medium with its computing rank already decided."""
    return _MediaMeta(
        src=src,
        index=index,
        grid=(1, patches, 1),
        patches=patches,
        tokens={"features": tokens},
        weight=patches,
        owner=owner,
    )


def _pool(specs, mode):
    """specs: list of (src, index, patches) in pooled order."""
    return _MediaPool(
        [_meta(src, index, patches, mode) for src, index, patches in specs]
    )


def test_lpt_weight_mode_selection():
    assert _MediaMeta.lpt_weight(7, VisionBalanceMode.PATCH) == 7
    assert _MediaMeta.lpt_weight(7, VisionBalanceMode.PATCH_SQ) == 49


def test_assign_owners_covers_all_ranks_in_range():
    world = 8
    pool = _pool([(0, 0, 100), (0, 1, 50), (16, 0, 30), (16, 1, 60)], VisionBalanceMode.PATCH)
    pool.assign_owners(world)
    owners = {m.owner for m in pool.media}
    assert all(0 <= owner < world for owner in owners)


def test_assign_owners_balances_patch_mode():
    world = 4
    patches = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    pool = _pool([(0, i, p) for i, p in enumerate(patches)], VisionBalanceMode.PATCH)
    pool.assign_owners(world)
    loads = [0] * world
    for m in pool.media:
        loads[m.owner] += m.patches
    total = sum(patches)
    # LPT bound: max load cannot exceed ideal by more than one item.
    assert max(loads) - min(loads) <= max(patches)
    assert sum(loads) == total


def test_assign_owners_is_deterministic():
    specs = [(0, 0, 100), (0, 1, 100), (8, 0, 70), (8, 1, 70), (24, 0, 10)]
    pool_a = _pool(specs, VisionBalanceMode.PATCH)
    pool_b = _pool(specs, VisionBalanceMode.PATCH)
    pool_a.assign_owners(8)
    pool_b.assign_owners(8)
    assert [m.owner for m in pool_a.media] == [m.owner for m in pool_b.media]


def test_assign_owners_patch_sq_prefers_keeping_large_images_apart():
    # Squared weights push the two huge images apart; patch mode cannot
    # guarantee that.
    pool = _pool(
        [(0, 0, 900), (0, 1, 900), (16, 0, 100), (16, 1, 100)],
        VisionBalanceMode.PATCH_SQ,
    )
    pool.assign_owners(4)
    assert pool.media[0].owner != pool.media[1].owner


def test_assign_owners_empty_pool_is_noop():
    pool = _MediaPool([])
    pool.assign_owners(8)
    assert pool.media == []


def test_assign_owners_single_item_goes_to_rank_zero():
    pool = _pool([(0, 0, 42)], VisionBalanceMode.PATCH)
    pool.assign_owners(8)
    owners = [m.owner for m in pool.media]
    # First (heaviest) item lands on the empty least-loaded bin, rank 0.
    assert owners[0] == 0


def test_from_metadata_flattens_in_source_order():
    # Only representatives publish, so a rank that is not one stays empty.
    metadata = [[[1, 2, 4, 4], [2, 2, 2, 4]], [], [[1, 2, 2, 2]]]
    pool = _MediaPool.from_metadata(metadata, ("features",), VisionBalanceMode.PATCH)
    assert [(m.src, m.index, m.patches, m.tokens["features"]) for m in pool.media] == [
        (0, 0, 8, 4),
        (0, 1, 8, 4),
        (2, 0, 4, 2),
    ]


def test_fanout_matches_the_worked_example():
    # Design doc 5.5: world=4, sp_size=2, medium A of group 0 computed by
    # rank 2, medium B of group 1 computed by rank 0.
    a = _owned(src=0, index=0, patches=12, tokens=3, owner=2)
    b = _owned(src=2, index=0, patches=8, tokens=2, owner=0)

    plan_of_rank2 = _FanoutPlan.build(mine=[a], own_group=[b], sp_size=2, world_size=4, output_name="features")
    assert plan_of_rank2.send_splits == [3, 3, 0, 0]
    assert plan_of_rank2.recv_splits == [2, 0, 0, 0]
    assert plan_of_rank2.restore_ranges == [(0, 2)]

    plan_of_rank0 = _FanoutPlan.build(mine=[b], own_group=[a], sp_size=2, world_size=4, output_name="features")
    assert plan_of_rank0.send_splits == [0, 0, 2, 2]
    assert plan_of_rank0.recv_splits == [0, 0, 3, 0]


def test_fanout_one_span_per_source_group():
    # Rank 0 computes media of two groups: each group's rows stay contiguous,
    # and each of a group's members receives that group's whole span.
    a = _owned(src=0, index=0, patches=8, tokens=2, owner=0)  # group 0
    b = _owned(src=4, index=0, patches=4, tokens=1, owner=0)  # group 2
    plan = _FanoutPlan.build(mine=[a, b], own_group=[a], sp_size=2, world_size=6, output_name="features")
    assert plan.send_ranges == [(0, 2), (0, 2), (2, 3), (2, 3)]
    assert plan.send_splits == [2, 2, 0, 0, 1, 1]
    # My own group's medium was computed by myself, so it comes straight back.
    assert plan.recv_splits == [2, 0, 0, 0, 0, 0]
    assert plan.restore_ranges == [(0, 2)]


def test_fanout_restore_walks_source_major_segments():
    # One group's two media computed by different ranks: rows arrive ordered
    # by computing rank, while the restore has to hand back merged order.
    merged_first = _owned(src=2, index=0, patches=8, tokens=2, owner=3)
    merged_second = _owned(src=2, index=1, patches=4, tokens=1, owner=0)
    plan = _FanoutPlan.build(
        mine=[], own_group=[merged_first, merged_second], sp_size=2, world_size=4, output_name="features"
    )
    assert plan.recv_splits == [1, 0, 0, 2]  # segment per computing rank
    assert plan.restore_ranges == [(1, 3), (0, 1)]


def test_fanout_empty_assignment_exchanges_nothing():
    plan = _FanoutPlan.build(mine=[], own_group=[], sp_size=2, world_size=4, output_name="features")
    assert plan.send_splits == [0, 0, 0, 0]
    assert plan.send_ranges == []
    assert plan.recv_splits == [0, 0, 0, 0]
    assert plan.restore_ranges == []


def test_pixel_plan_is_destination_major():
    own = [
        _owned(src=0, index=0, patches=4, tokens=1, owner=2),
        _owned(src=0, index=1, patches=6, tokens=1, owner=1),
        _owned(src=0, index=2, patches=2, tokens=1, owner=2),
    ]
    plan = _PixelPlan.build(own, world_size=3)
    assert plan.splits == [0, 6, 6]
    # Rows for rank 1 first, then rank 2's media in merged order.
    assert plan.ranges == [(4, 10), (0, 4), (10, 12)]


def _world_plans(rep_grids, world, sp_size):
    """Plan every rank of a simulated step from one gathered metadata view."""
    published = [
        [[t, h, w, t * h * w // 4] for t, h, w in rep_grids.get(r, [])]
        if r % sp_size == 0 else []
        for r in range(world)
    ]
    pool = _MediaPool.from_metadata(published, ("features",), VisionBalanceMode.PATCH)
    pool.assign_owners(world)

    plans = []
    for rank in range(world):
        mine = pool.assigned_to(rank)
        own_group = pool.in_group(rank // sp_size, sp_size)
        own_pool = pool.contributed_by(rank)
        plans.append(
            {
                "mine": mine,
                "own_group": own_group,
                "own_pool": own_pool,
                "fanout": _FanoutPlan.build(mine, own_group, sp_size, world, "features"),
                "pixels": _PixelPlan.build(own_pool, world),
            }
        )
    return plans


def _all_to_all(buffers, send_splits, world):
    """Simulate all_to_all_single: send buffers are destination-major, so a
    receiver's rows come out source-major."""
    received = []
    for dest in range(world):
        rows = []
        for src in range(world):
            start = sum(send_splits[src][:dest])
            rows.extend(buffers[src][start : start + send_splits[src][dest]])
        received.append(rows)
    return received


def _exchange_embeds(plans, world):
    """Run every rank's post hook over fake tower output rows.

    Each row is labelled with the medium it belongs to, so a mistake in any
    layout table shows up as rows landing in the wrong place.
    """
    tower_rows = [
        [(m.src, m.index, k) for m in p["mine"] for k in range(m.tokens["features"])]
        for p in plans
    ]
    sends = [
        [row for start, end in p["fanout"].send_ranges for row in rows[start:end]]
        for p, rows in zip(plans, tower_rows)
    ]
    received = _all_to_all(sends, [p["fanout"].send_splits for p in plans], world)

    restored = []
    for rank, plan in enumerate(plans):
        assert len(received[rank]) == sum(plan["fanout"].recv_splits)
        restored.append(
            [
                received[rank][i]
                for start, end in plan["fanout"].restore_ranges
                for i in range(start, end)
            ]
        )
    return restored


def _merged_labels(plans):
    """What each rank should hold after the restore: its group's media."""
    return [
        [(m.src, m.index, k) for m in p["own_group"] for k in range(m.tokens["features"])]
        for p in plans
    ]


def test_world_round_trip_returns_each_group_its_merged_order():
    rep_grids = {
        0: [(1, 2, 2), (1, 4, 2)],
        2: [(2, 2, 4)],
        6: [(1, 2, 6), (1, 2, 2), (1, 4, 4)],
    }
    world, sp_size = 8, 2
    plans = _world_plans(rep_grids, world, sp_size)
    assert _exchange_embeds(plans, world) == _merged_labels(plans)
    # Six media over eight ranks leaves some rank empty-handed.
    unassigned = [p for p in plans if not p["mine"]]
    assert unassigned
    assert all(p["fanout"].send_ranges == [] for p in unassigned)


def test_world_round_trip_covers_sp_size_one():
    # Degenerate SP: every rank is its own representative and its own group,
    # so the protocol collapses to plain DP-shard balancing.
    rep_grids = {0: [(1, 2, 2)], 1: [(1, 4, 4), (1, 2, 2)], 3: [(2, 2, 2)]}
    world, sp_size = 4, 1
    plans = _world_plans(rep_grids, world, sp_size)
    assert _exchange_embeds(plans, world) == _merged_labels(plans)


def test_world_round_trip_delivers_assigned_pixels():
    rep_grids = {
        0: [(1, 2, 2), (1, 4, 2)],
        2: [(2, 2, 4)],
        6: [(1, 2, 6), (1, 2, 2)],
    }
    world, sp_size = 8, 2
    plans = _world_plans(rep_grids, world, sp_size)

    # A representative publishes its media's pixel rows in merged order.
    pixel_rows = [
        [(m.src, m.index, k) for m in p["own_pool"] for k in range(m.patches)]
        for p in plans
    ]
    sends = [
        [row for start, end in p["pixels"].ranges for row in rows[start:end]]
        for p, rows in zip(plans, pixel_rows)
    ]
    received = _all_to_all(sends, [p["pixels"].splits for p in plans], world)

    expected = [
        [(m.src, m.index, k) for m in p["mine"] for k in range(m.patches)]
        for p in plans
    ]
    assert received == expected


def test_gather_metadata_contributes_local_shards_only(monkeypatch):
    """Gather one count per rank followed by grids and all output lengths."""
    world, sp_size = 4, 2
    rep_rows = {
        0: [(1, 2, 2, 4, 1), (1, 4, 2, 8, 2)],
        2: [(2, 2, 4, 16, 4), (1, 2, 6, 12, 3), (1, 2, 2, 4, 1)],
    }
    counts = torch.tensor([len(rep_rows.get(src, [])) for src in range(world)])
    max_media = int(counts.max())
    padded = torch.zeros(world, max_media, 5, dtype=torch.long)
    for src, rows in rep_rows.items():
        padded[src, :len(rows)] = torch.tensor(rows)

    inputs = []

    def gather(output, inp, group=None):
        assert group is torch.distributed.group.WORLD
        assert output.numel() == world * inp.numel()
        output.copy_(counts if inp.numel() == 1 else padded.reshape(-1))
        inputs.append(inp.clone())

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    expected = [[list(row) for row in rep_rows.get(src, [])] for src in range(world)]
    for sp_rank in range(sp_size):
        balance = _VisionBalanceState(
            nn.Identity(), VisionBalanceMode.PATCH, BailingVisionBalanceIO(),
            sp_size=sp_size, sp_rank=sp_rank,
        )
        batch = VisionBatch(
            torch.empty(12, 2),
            torch.tensor([[1, 2, 2], [1, 4, 2]]),
            {"raw": [4, 8], "merged": [1, 2]},
        )
        assert balance._gather_metadata(batch) == expected
        assert [inp.numel() for inp in inputs] == [1, max_media * 5]
        assert inputs[0].item() == (2 if sp_rank == 0 else 0)
        if sp_rank == 0:
            torch.testing.assert_close(inputs[1].view(-1, 5)[:2], padded[0, :2])
        else:
            assert inputs[1].count_nonzero() == 0
        inputs.clear()


def test_fanout_uses_each_output_streams_own_lengths():
    first = _MediaMeta(0, 0, (2, 2, 2), 8, {"raw": 8, "pooled": 4}, 8, 1)
    second = _MediaMeta(0, 1, (1, 2, 2), 4, {"raw": 4, "pooled": 4}, 4, 0)
    raw = _FanoutPlan.build([second], [first, second], 1, 2, "raw")
    pooled = _FanoutPlan.build([second], [first, second], 1, 2, "pooled")
    assert raw.recv_splits == [4, 8]
    assert raw.restore_ranges == [(4, 12), (0, 4)]
    assert pooled.recv_splits == [4, 4]
    assert pooled.restore_ranges == [(4, 8), (0, 4)]
