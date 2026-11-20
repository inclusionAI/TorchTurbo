"""Contract tests for DP rebalance capability detection, re-packing and data conservation."""

import random
from collections import Counter

import pytest
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from torchturbo.engine.dp_rebalance import (
    CostModel,
    DPRebalanceConfig,
    DatasetAnalyzer,
    DatasetCapabilities,
    LayoutKind,
    PackedCost,
    RepackableView,
    StepRebalancePlanner,
    resolve_dp_rebalance_plan,
)
from torchturbo.engine.core import DPBlockDistributedSampler


class RecipePacks(Dataset):
    """Minimal recipe-driven packed dataset (no tensor loading)."""

    def __init__(self) -> None:
        self.dataset = {"lengths": [[10], [20], [30], [40], [50], [60]]}
        self.packed_idx = [[0, 1], [2], [3, 4], [5]]
        self.packing_length = 100

    def __len__(self) -> int:
        return len(self.packed_idx)

    def __getitem__(self, index: int) -> object:
        # Recipe-driven assembly, mirroring real cached packing datasets.
        return [self.dataset["lengths"][i] for i in self.packed_idx[index]]


class FiveSixties(Dataset):
    """Recipe dataset whose segments cannot re-bin into the same pack count."""

    def __init__(self) -> None:
        self.dataset = {"lengths": [[60]] * 5}
        self.packed_idx = [[0], [1], [2], [3, 4]]
        self.packing_length = 100

    def __len__(self) -> int:
        return len(self.packed_idx)

    def __getitem__(self, index: int) -> object:
        return [self.dataset["lengths"][i] for i in self.packed_idx[index]]


class MaterializedRows(Dataset):
    """Minimal materialized packed rows dataset (fixed-length padded rows)."""

    def __init__(self) -> None:
        self.features = {"input_ids": None, "labels": None, "packing_params": None}
        self._rows = [
            (128, [0, 40, 100, 128], 28),
            (128, [0, 128], 0),
            (128, [0, 10, 20, 128], 0),
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def _row(self, index):
        row_length, boundaries, pad = self._rows[index]
        return {
            "input_ids": [0] * row_length,
            "labels": [0] * row_length,
            "packing_params": {
                "sequence_boundaries": boundaries,
                "right_padding_length": pad,
            },
        }

    def __getitem__(self, index):
        # Mimic both HF row access (int) and column access (str).
        if isinstance(index, str):
            return [self._row(i)[index] for i in range(len(self))]
        return self._row(index)


def _config(**kwargs):
    return DPRebalanceConfig(**kwargs)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def test_detector_registry_is_structural():
    names = DatasetAnalyzer().known_detectors()
    assert "recipe_packed" in names
    assert "materialized_packed" in names
    # No framework-named detectors remain.
    assert not any("swift" in n or "llamafactory" in n for n in names)


def test_recipe_capabilities():
    caps = DatasetAnalyzer().resolve(RecipePacks())
    assert caps is not None
    assert caps.layout is LayoutKind.PACKED_RECIPE
    assert caps.recomposable
    assert list(caps.costs) == [
        PackedCost(tokens=30, squared_lengths=500, chunks=2, longest=20, capacity=100),
        PackedCost(tokens=30, squared_lengths=900, chunks=1, longest=30, capacity=100),
        PackedCost(tokens=90, squared_lengths=4100, chunks=2, longest=50, capacity=100),
        PackedCost(tokens=60, squared_lengths=3600, chunks=1, longest=60, capacity=100),
    ]


def test_materialized_capabilities_are_atomic():
    caps = DatasetAnalyzer().resolve(MaterializedRows())
    assert caps is not None
    assert caps.layout is LayoutKind.PACKED_MATERIALIZED
    assert not caps.recomposable  # slicing needs a re-assembly contract (v2)
    costs = list(caps.costs)
    assert len(costs) == 3
    first = costs[0]
    assert first.tokens == 100 and first.chunks == 2
    assert first.squared_lengths == 40 * 40 + 60 * 60
    assert first.longest == 60 and first.capacity == 128
    assert costs[1] == PackedCost(128, 128 * 128, 1, 128, 128)
    assert costs[2].chunks == 3 and costs[2].tokens == 128


def test_wrapped_costs_but_not_recomposable():
    caps = DatasetAnalyzer().resolve(ConcatDataset([RecipePacks(), RecipePacks()]))
    assert caps is not None and not caps.recomposable
    base = DatasetAnalyzer().resolve(RecipePacks())
    assert list(caps.costs) == list(base.costs) + list(base.costs)


def test_custom_detector_via_analyzer_extension():
    class Marker:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    analyzer = DatasetAnalyzer()
    analyzer.register(
        lambda dataset: DatasetCapabilities(
            costs=(PackedCost(7, 49, 1, 7, 8), PackedCost(1, 1, 1, 1, 8)),
            layout=LayoutKind.PACKED_MATERIALIZED,
        )
        if isinstance(dataset, Marker)
        else None,
        name="marker_detector",
        priority=90,
    )
    caps = analyzer.resolve(Marker())
    assert caps is not None and caps.layout is LayoutKind.PACKED_MATERIALIZED
    assert "marker_detector" in analyzer.known_detectors()
    # Other analyzer instances are unaffected by this extension.
    assert "marker_detector" not in DatasetAnalyzer().known_detectors()


# ---------------------------------------------------------------------------
# Re-packing primitive: fixed pack count under capacity
# ---------------------------------------------------------------------------


def test_repack_preserves_segments_and_capacity():
    rng = random.Random(5)
    sizes = [rng.randint(1, 100) for _ in range(200)]
    groups = [[i, i + 1] for i in range(0, len(sizes), 2)]
    capacity = 130
    cost_fn = lambda size: float(size * size)  # noqa: E731
    bins = RepackableView.repack_segments(len(groups), sizes, capacity, cost_fn)
    # Global segment multiset preserved.
    assert sorted(seg for group in bins for seg in group) == list(range(len(sizes)))
    # Pack count preserved, so per-rank microbatch counts stay unchanged.
    assert len(bins) == len(groups)
    # Capacity respected everywhere.
    assert all(sum(sizes[i] for i in group) <= capacity for group in bins)
    # Deterministic.
    assert bins == RepackableView.repack_segments(len(groups), sizes, capacity, cost_fn)
    # Cost evenness: max bin cost should not exceed the original worst pack.
    def pack_cost(members):
        return sum(sizes[i] ** 2 for i in members)
    assert max(pack_cost(group) for group in bins) <= max(pack_cost(g) for g in groups)


def test_repack_fixed_pack_count_and_overflow_rejection():
    cost_fn = lambda size: float(size * size)  # noqa: E731
    # A 4-pack dataset re-packs into exactly 4 packs.
    sizes = [60, 50, 40, 30, 20, 10]
    groups = [[0, 1], [2], [3, 4], [5]]
    bins = RepackableView.repack_segments(len(groups), sizes, 100, cost_fn)
    assert len(bins) == len(groups)
    assert sorted(seg for group in bins for seg in group) == list(range(len(sizes)))
    # Three length-6 segments cannot fit two packs of capacity 10.
    with pytest.raises(ValueError):
        RepackableView.repack_segments(2, [6, 6, 6], 10, cost_fn)
    # A segment longer than the packing capacity is rejected too.
    with pytest.raises(ValueError):
        RepackableView.repack_segments(1, [11], 10, cost_fn)
    # Fewer source rows than packs would leave empty bins; reject defensively.
    with pytest.raises(ValueError):
        RepackableView.repack_segments(3, [10, 10], 20, cost_fn)


# ---------------------------------------------------------------------------
# RepackableView: in-place window rewrites with restore
# ---------------------------------------------------------------------------


def _squares_model():
    return CostModel(_config(enabled=True, cost_model="ffd_squares"))


def test_repackable_view_repacks_window_in_place():
    base = RecipePacks()
    packs, sizes, capacity = DatasetAnalyzer().recipe_material(base)
    view = RepackableView(base, packs, sizes, capacity)
    model = _squares_model()
    window = [0, 1, 2, 3]
    new_scores = view.repack_window(
        window, segment_score=model.segment_score, pack_score=model.pack_score
    )
    assert new_scores is not None and set(new_scores) == set(window)
    # Pack count and dataset length never change.
    assert len(view.dataset) == len(base) == 4
    # Window segment multiset preserved.
    recipe = view.dataset.packed_idx
    assert sorted(seg for group in recipe for seg in group) == [0, 1, 2, 3, 4, 5]
    assert all(sum(sizes[i] for i in group) <= capacity for group in recipe)
    # getitem follows the rewritten recipe; collator path untouched.
    lengths = base.dataset["lengths"]
    for index in window:
        assert view.dataset[index] == [lengths[i] for i in recipe[index]]
    # Base dataset untouched.
    assert base.packed_idx == [[0, 1], [2], [3, 4], [5]]
    # Deterministic: re-packing the same window again gives the same recipe.
    again = view.repack_window(
        window, segment_score=model.segment_score, pack_score=model.pack_score
    )
    assert again == new_scores and view.dataset.packed_idx == recipe


def test_repackable_view_failure_restores_originals():
    # Five length-60 segments cannot fit into 4 packs of capacity 100.
    base = FiveSixties()
    packs, sizes, capacity = DatasetAnalyzer().recipe_material(base)
    assert packs is not None
    view = RepackableView(base, packs, sizes, capacity)
    model = _squares_model()
    failed = view.repack_window(
        [0, 1, 2, 3], segment_score=model.segment_score, pack_score=model.pack_score
    )
    assert failed is None
    assert view.dataset.packed_idx == [[0], [1], [2], [3, 4]]


def test_repackable_view_empty_pack_window_falls_back_cleanly():
    # Defensive: a window with more packs than unique rows must not IndexError.
    base = RecipePacks()
    base.dataset = {"lengths": [[10]]}
    base.packed_idx = [[0], [], []]
    packs, sizes, capacity = DatasetAnalyzer().recipe_material(base)
    assert packs == [[0], [], []] and capacity == 100
    view = RepackableView(base, packs, sizes, capacity)
    failed = view.repack_window(
        [0, 1, 2], segment_score=_squares_model().segment_score,
        pack_score=_squares_model().pack_score,
    )
    assert failed is None
    assert view.dataset.packed_idx == [[0], [], []]


# ---------------------------------------------------------------------------
# Plan resolution gates
# ---------------------------------------------------------------------------


class _NoCapsDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index):
        return index


def test_resolve_plan_atomic_rebalances_without_repack():
    dataset = MaterializedRows()
    resolved, planner = resolve_dp_rebalance_plan(
        dataset,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=32,
        dp_size=3,
        batch_size=1,
        rank_block=1,
    )
    assert resolved is dataset and planner is not None
    assert not planner.repacks  # atomic rows: rebalance only


def test_resolve_plan_gate_conditions():
    dataset = RecipePacks()
    base = dict(gbs=32, dp_size=4, batch_size=1, rank_block=4)
    # disabled / no cost model
    assert resolve_dp_rebalance_plan(dataset, _config(), **base) == (dataset, None)
    assert resolve_dp_rebalance_plan(
        dataset, _config(enabled=True, cost_model="ffd_squares"), gbs=None, dp_size=4,
        batch_size=1, rank_block=4,
    ) == (dataset, None)
    # single DP rank
    assert resolve_dp_rebalance_plan(
        dataset, _config(enabled=True, cost_model="ffd_squares"), gbs=32, dp_size=1,
        batch_size=1, rank_block=4,
    ) == (dataset, None)
    # rank_block must contain whole microbatches
    assert resolve_dp_rebalance_plan(
        dataset, _config(enabled=True, cost_model="ffd_squares"), gbs=32, dp_size=4,
        batch_size=3, rank_block=4,
    ) == (dataset, None)
    # dataset without recognized capabilities
    assert resolve_dp_rebalance_plan(
        _NoCapsDataset(), _config(enabled=True, cost_model="ffd_squares"), **base
    )[1] is None


def test_resolve_plan_recipe_returns_view_planner():
    dataset = RecipePacks()
    resolved, planner = resolve_dp_rebalance_plan(
        dataset,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=32,
        dp_size=4,
        batch_size=1,
        rank_block=1,
    )
    assert planner is not None and planner.repacks
    assert resolved is not dataset  # re-packable view
    assert len(resolved) == len(dataset)
    # Recipes are untouched until the planner runs on a window.
    assert resolved.packed_idx == dataset.packed_idx
    planned = planner([[0], [1], [2], [3]])
    assert sorted(x for row in planned for x in row) == [0, 1, 2, 3]
    # Window re-packed in place; step data preserved.
    assert sorted(seg for group in resolved.packed_idx for seg in group) == [
        0, 1, 2, 3, 4, 5,
    ]
    # base dataset untouched
    assert dataset.packed_idx == [[0, 1], [2], [3, 4], [5]]


def test_resolve_plan_capacity_margin_extends_repack_bins():
    dataset = FiveSixties()
    resolved, planner = resolve_dp_rebalance_plan(
        dataset,
        _config(enabled=True, cost_model="ffd_squares", capacity_margin=20),
        gbs=32,
        dp_size=4,
        batch_size=1,
        rank_block=1,
    )
    assert planner is not None and planner.repacks
    assert planner._repack_view.capacity == dataset.packing_length + 20
    # Five 60-token segments do not fit 4 packs at capacity 100, but do at 120.
    planned = planner([[0], [1], [2], [3]])
    assert sorted(x for row in planned for x in row) == [0, 1, 2, 3]


def test_resolve_plan_failed_window_still_rebalances():
    dataset = FiveSixties()
    resolved, planner = resolve_dp_rebalance_plan(
        dataset,
        _config(enabled=True, cost_model="ffd_squares", capacity_margin=0),
        gbs=32,
        dp_size=4,
        batch_size=1,
        rank_block=1,
    )
    assert planner is not None and planner.repacks
    planned = planner([[0], [1], [2], [3]])
    # Recipes restored to the originals.
    assert resolved.packed_idx == [[0], [1], [2], [3, 4]]
    # ...but the window is still reassigned across DP ranks.
    assert sorted(x for row in planned for x in row) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Guard/fallback contracts added by the re-packing hardening pass
# ---------------------------------------------------------------------------


class _DuplicateRowRecipe(Dataset):
    """Recipe that references the same source row twice."""

    def __init__(self):
        self.dataset = {"lengths": [[10], [20]]}
        self.packed_idx = [[0], [1, 0]]
        self.packing_length = 100

    def __len__(self):
        return len(self.packed_idx)

    def __getitem__(self, index):
        return [self.dataset["lengths"][i] for i in self.packed_idx[index]]


def test_recipe_material_rejects_duplicate_source_rows():
    base = _DuplicateRowRecipe()
    assert DatasetAnalyzer().recipe_material(base) is None
    resolved, planner = resolve_dp_rebalance_plan(
        base,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=32,
        dp_size=4,
        batch_size=1,
        rank_block=2,
    )
    assert resolved is base and planner is not None and not planner.repacks


@pytest.mark.parametrize(
    ("drop_last", "allow_repack", "expect_repack"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_resolve_plan_gates_repack_on_sampler_visibility(
    drop_last, allow_repack, expect_repack
):
    base = RecipePacks()
    resolved, planner = resolve_dp_rebalance_plan(
        base,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=32,
        dp_size=4,
        batch_size=1,
        rank_block=1,
        drop_last=drop_last,
        allow_repack=allow_repack,
    )
    assert planner is not None and planner.repacks is expect_repack
    assert (resolved is base) is not expect_repack
    assert len(resolved) == len(base)


def test_planner_rejects_empty_rank_block():
    costs = [PackedCost(1, 1, 1, 1, 8)] * 4
    planner = StepRebalancePlanner(costs, batch_size=1, pack_score=lambda c: 1.0)
    with pytest.raises(ValueError):
        planner([[], []])


def test_planner_requires_segment_score_for_repack_view():
    costs = [PackedCost(1, 1, 1, 1, 8)]
    view = _WorseRepackWindow()
    packs, sizes, capacity = DatasetAnalyzer().recipe_material(view)
    with pytest.raises(ValueError):
        StepRebalancePlanner(
            costs,
            batch_size=1,
            pack_score=lambda c: 1.0,
            segment_score=None,
            repack_view=RepackableView(view, packs, sizes, capacity),
        )


class _WorseMakespanView:
    """Fake view returning scores that tighten spread but raise rank makespan."""

    def __init__(self):
        self.restored = 0

    def repack_window(self, indices, *, segment_score, pack_score):
        del segment_score, pack_score
        return {0: 9.0, 1: 9.0, 2: 0.0, 3: 0.0}

    def restore(self, indices):
        self.restored += 1


def test_planner_restores_when_repack_makespan_worsens():
    costs = [
        PackedCost(10, 10, 1, 10, 16),
        PackedCost(0, 0, 1, 0, 16),
        PackedCost(0, 0, 1, 0, 16),
        PackedCost(0, 0, 1, 0, 16),
    ]
    view = _WorseMakespanView()
    planner = StepRebalancePlanner(
        costs,
        batch_size=1,
        pack_score=lambda c: float(c.tokens),
        segment_score=lambda size: float(size),
        repack_view=view,
    )
    planned = planner([[0, 1], [2, 3]])
    assert view.restored == 1
    assert planner.stats["windows"] == 1
    assert planner.stats["repack_windows"] == 0
    assert planner.stats["repack_fallbacks"] == 1
    assert sorted(x for row in planned for x in row) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Cross-DP reassignment: membership, counts, cost floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 137])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_membership_and_sp(seed, shuffle, batch_size):
    data = list(range(193))
    rng = random.Random(19)
    costs = [
        PackedCost(n, n * n, (n + 63) // 64, n, 131072)
        for n in [rng.randint(1, 131072) for _ in data]
    ]
    block = 4 * batch_size
    before = []
    after = []
    for dp_rank in range(4):
        options = dict(
            dataset=data,
            num_replicas=8,
            rank=dp_rank * 2,
            dp_size=4,
            dp_rank=dp_rank,
            rank_block=block,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        base = DPBlockDistributedSampler(**options)
        base.set_epoch(1)
        planner = StepRebalancePlanner(
            costs, batch_size=batch_size, pack_score=lambda c: float(c.squared_lengths)
        )
        candidate = DPBlockDistributedSampler(**options, step_planner=planner)
        candidate.set_epoch(1)
        before.append(list(base))
        after.append(list(candidate))
        assert len(candidate) == len(base)
        # SP replica (same dp_rank, different world rank) must match.
        replica = DPBlockDistributedSampler(
            **{**options, "rank": dp_rank * 2 + 1}, step_planner=planner
        )
        replica.set_epoch(1)
        assert list(replica) == after[-1]
    # Every optimizer step keeps exactly the same multiset of microbatch units.
    for start in range(0, len(before[0]), block):
        original = Counter(
            tuple(row[i : i + batch_size])
            for row in before
            for i in range(start, start + block, batch_size)
        )
        balanced = Counter(
            tuple(row[i : i + batch_size])
            for row in after
            for i in range(start, start + block, batch_size)
        )
        assert original == balanced


class _PairRecipe(Dataset):
    """Recipe dataset packing consecutive segment pairs."""

    def __init__(self, sizes):
        self.dataset = {"lengths": [[size] for size in sizes]}
        self.packed_idx = [[i, i + 1] for i in range(0, len(sizes), 2)]
        self.packing_length = 200

    def __len__(self):
        return len(self.packed_idx)

    def __getitem__(self, index):
        return [self.dataset["lengths"][i] for i in self.packed_idx[index]]


class _WorseRepackWindow(Dataset):
    """Window where FFD re-bin would worsen per-microstep spread without a guard."""

    def __init__(self) -> None:
        self.dataset = {"lengths": [[10], [3], [6], [2], [14], [3], [18]]}
        self.packed_idx = [[1, 2], [0, 4], [3], [5, 6]]
        self.packing_length = 20

    def __len__(self) -> int:
        return len(self.packed_idx)

    def __getitem__(self, index: int) -> object:
        return [self.dataset["lengths"][i] for i in self.packed_idx[index]]


def test_planner_restores_original_recipe_when_repack_spread_worsens():
    """The planner keeps the original pack recipes if re-bin widens microstep spread."""
    base = _WorseRepackWindow()
    caps = DatasetAnalyzer().resolve(base)
    assert caps is not None and caps.recomposable
    packs, sizes, capacity = DatasetAnalyzer().recipe_material(base)
    assert packs == [[1, 2], [0, 4], [3], [5, 6]]
    view = RepackableView(base, packs, sizes, capacity)
    model = _squares_model()
    planner = StepRebalancePlanner(
        caps.costs,
        batch_size=1,
        pack_score=model.pack_score,
        segment_score=model.segment_score,
        repack_view=view,
    )
    original_recipe = [list(pack) for pack in packs]
    planned = planner([[0, 1], [2, 3]])
    # The spread guard reverts to original recipes (and base pack scores).
    assert view.dataset.packed_idx == original_recipe
    assert planner.stats["windows"] == 1
    assert planner.stats["repack_windows"] == 0
    assert planner.stats["repack_fallbacks"] == 1
    # Window pack multiset is unchanged by the fallback.
    assert sorted(x for row in planned for x in row) == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", [0, 42])
def test_step_windows_consume_exactly_the_original_data(seed):
    """Verify each optimizer step keeps its base window's segment multiset through the sampler."""
    rng = random.Random(seed)
    sizes = [rng.randint(1, 100) for _ in range(120)]
    base_dataset = _PairRecipe(sizes)
    resolved, planner = resolve_dp_rebalance_plan(
        base_dataset,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=12,
        dp_size=4,
        batch_size=1,
        rank_block=3,
    )
    assert planner is not None and planner.repacks
    base_streams = []
    planned_streams = []
    for dp_rank in range(4):
        options = dict(
            dataset=base_dataset,
            num_replicas=8,
            rank=dp_rank * 2,
            dp_size=4,
            dp_rank=dp_rank,
            rank_block=3,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        base = DPBlockDistributedSampler(**options)
        base.set_epoch(2)
        base_streams.append(list(base))
        common = {k: v for k, v in options.items() if k != "dataset"}
        candidate = DPBlockDistributedSampler(resolved, step_planner=planner, **common)
        candidate.set_epoch(2)
        planned_streams.append(list(candidate))
        assert len(candidate) == len(base)
        # SP replica agreement.
        replica = DPBlockDistributedSampler(
            resolved, step_planner=planner, **{**common, "rank": dp_rank * 2 + 1}
        )
        replica.set_epoch(2)
        assert list(replica) == planned_streams[-1]
    # Per-step windows: identical segment multisets as the base layout.
    num_steps = len(base_streams[0]) // 3
    for step in range(num_steps):
        base_window = sorted(
            index for stream in base_streams for index in stream[step * 3 : step * 3 + 3]
        )
        planned_window = sorted(
            index for stream in planned_streams for index in stream[step * 3 : step * 3 + 3]
        )
        assert planned_window == base_window
        original_segments = Counter(
            segment for index in base_window for segment in base_dataset.packed_idx[index]
        )
        repacked_segments = Counter(
            segment for index in planned_window for segment in resolved.packed_idx[index]
        )
        assert repacked_segments == original_segments
    # Dataset length never changes.
    assert len(resolved) == len(base_dataset)


class SwiftStylePacked(Dataset):
    """Pack dataset that mimics ms-swift ``PackingDataset`` and materializes token ids."""

    def __init__(self, num_rows: int = 480, capacity: int = 512, seed: int = 7):
        rng = random.Random(seed)
        self.rows: list[list[int]] = []
        lengths: list[int] = []
        for row in range(num_rows):
            length = rng.randint(1, capacity)
            lengths.append(length)
            # Unique tokens per source row: any drop/duplicate is detectable.
            self.rows.append([row * capacity + k for k in range(length)])
        self.dataset = {"lengths": lengths}
        self.packing_length = capacity
        # swift-style greedy packing in shuffled row order.
        order = list(range(num_rows))
        rng.shuffle(order)
        self.packed_idx: list[list[int]] = []
        self.packed_length: list[int] = []
        current: list[int] = []
        current_len = 0
        for row in order:
            length = lengths[row]
            if current and current_len + length > capacity:
                self.packed_idx.append(current)
                self.packed_length.append(current_len)
                current, current_len = [], 0
            current.append(row)
            current_len += length
            if current_len >= capacity:
                self.packed_idx.append(current)
                self.packed_length.append(current_len)
                current, current_len = [], 0
        if current:
            self.packed_idx.append(current)
            self.packed_length.append(current_len)

    def __len__(self) -> int:
        return len(self.packed_idx)

    def __getitem__(self, index):
        return {
            "input_ids": [
                token for row in self.packed_idx[index] for token in self.rows[row]
            ]
        }


@pytest.mark.parametrize("epoch", [0, 1, 3])
def test_swift_recipe_step_data_invariant_under_repack(epoch):
    """Verify every optimizer step consumes the same tokens after re-packing and reassignment."""
    dataset = SwiftStylePacked()
    dp_size, sp_size, ga, batch_size = 4, 2, 2, 1
    rank_block = ga * batch_size
    resolved, planner = resolve_dp_rebalance_plan(
        dataset,
        _config(enabled=True, cost_model="ffd_squares"),
        gbs=dp_size * rank_block,
        dp_size=dp_size,
        batch_size=batch_size,
        rank_block=rank_block,
    )
    assert planner is not None and planner.repacks
    assert len(resolved) == len(dataset)

    def sampler_streams(ds, use_planner):
        streams = []
        for dp_rank in range(dp_size):
            sampler = DPBlockDistributedSampler(
                dataset=ds,
                num_replicas=dp_size * sp_size,
                rank=dp_rank * sp_size,
                dp_size=dp_size,
                dp_rank=dp_rank,
                rank_block=rank_block,
                shuffle=True,
                seed=42,
                drop_last=True,
                step_planner=planner if use_planner else None,
            )
            sampler.set_epoch(epoch)
            streams.append(list(sampler))
        return streams

    baseline = sampler_streams(dataset, False)
    planned = sampler_streams(resolved, True)

    # Every DP rank consumes the same number of microbatches per step.
    for dp_rank in range(dp_size):
        assert len(planned[dp_rank]) == len(baseline[dp_rank])
        assert len(planned[dp_rank]) % rank_block == 0

    base_tokens_epoch = Counter()
    planned_tokens_epoch = Counter()
    for start in range(0, len(baseline[0]), rank_block):
        base_step = [row[start : start + rank_block] for row in baseline]
        planned_step = [row[start : start + rank_block] for row in planned]
        # The step consumes the same pack indices, only reassigned across DPs.
        assert Counter(i for row in base_step for i in row) == Counter(
            i for row in planned_step for i in row
        )
        # Token-level data conservation across all DPs of this step.
        base_tokens = Counter(
            t for row in base_step for i in row for t in dataset[i]["input_ids"]
        )
        planned_tokens = Counter(
            t for row in planned_step for i in row for t in resolved[i]["input_ids"]
        )
        assert planned_tokens == base_tokens
        base_tokens_epoch.update(base_tokens)
        planned_tokens_epoch.update(planned_tokens)
        # Re-packed packs respect the capacity when materialized.
        assert all(
            len(resolved[i]["input_ids"]) <= resolved.packing_length
            for row in planned_step
            for i in row
        )

    # Whole-epoch conservation: nothing dropped, nothing duplicated.
    assert planned_tokens_epoch == base_tokens_epoch
    # At least one window was actually re-packed (recipe rewritten in place).
    assert resolved.packed_idx != dataset.packed_idx


def test_balancer_cost_floor():
    rng = random.Random(71)

    def waits(groups, cost_fn):
        return sum(
            max(cost_fn(costs[row[i]]) for row in groups) for i in range(4)
        )

    for _ in range(50):
        costs = [
            PackedCost(n, n * n, (n + 63) // 64, n, 131072)
            for n in [rng.randint(1, 131072) for _ in range(32)]
        ]
        cost_fn = lambda c: float(c.squared_lengths)  # noqa: E731
        original = [list(range(rank * 4, (rank + 1) * 4)) for rank in range(8)]
        planner = StepRebalancePlanner(costs, batch_size=1, pack_score=cost_fn)
        candidate = planner(original)
        assert waits(candidate, cost_fn) <= waits(original, cost_fn) + 1e-8


def test_optimizer_update_equivalence():
    """Verify DP microbatch reassignment leaves the optimizer update unchanged."""
    import torch

    torch.manual_seed(42)
    x = torch.randn(32, 5, dtype=torch.float64)
    target = torch.randn(32, 3, dtype=torch.float64)
    first = torch.nn.Linear(5, 3, dtype=torch.float64)
    second = torch.nn.Linear(5, 3, dtype=torch.float64)
    second.load_state_dict(first.state_dict())
    costs = [PackedCost(i + 1, (i + 1) ** 2, 1, i + 1, 64) for i in range(32)]
    original = [list(range(rank * 4, (rank + 1) * 4)) for rank in range(8)]
    planner = StepRebalancePlanner(costs, batch_size=1, pack_score=lambda c: float(c.squared_lengths))
    balanced = planner(original)
    for model, groups in [(first, original), (second, balanced)]:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for row in groups:
            for index in row:
                ((model(x[index]) - target[index]).square().sum() / 32).backward()
        optimizer.step()
    for a, z in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, z, rtol=1e-12, atol=1e-12)


def test_planner_reduces_skewed_step():
    # Rank 0 holds all expensive packs; balancing must lower its max load.
    costs = [PackedCost(n, n * n, 1, n, 1024) for n in range(1, 17)]
    original = [list(range(4)), [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    cost_fn = lambda c: float(c.squared_lengths)  # noqa: E731
    planned = StepRebalancePlanner(costs, batch_size=1, pack_score=cost_fn)(original)
    assert sorted(x for row in planned for x in row) == list(range(16))
    assert all(len(row) == 4 for row in planned)
    loads = [sum(cost_fn(costs[i]) for i in row) for row in planned]
    assert max(loads) < sum(cost_fn(costs[i]) for i in original[3])
    assert max(loads) <= max(sum(cost_fn(costs[i]) for i in row) for row in original)


def test_planner_balances_every_microstep():
    """Verify every gradient-accumulation microstep has balanced cost across DP ranks."""
    costs = [PackedCost(n, n * n, 1, n, 1024) for n in range(1, 17)]
    original = [list(range(4)), [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    cost_fn = lambda c: float(c.squared_lengths)  # noqa: E731
    planned = StepRebalancePlanner(costs, batch_size=1, pack_score=cost_fn)(original)

    def spread(blocks):
        return max(
            max(cost_fn(costs[row[k]]) for row in blocks)
            - min(cost_fn(costs[row[k]]) for row in blocks)
            for k in range(len(blocks[0]))
        )

    # Window multiset and per-rank microbatch counts unchanged.
    assert sorted(x for row in planned for x in row) == list(range(16))
    assert all(len(row) == 4 for row in planned)
    # Every microstep column is at least as tight as the input layout.
    assert spread(planned) <= spread(original)
    assert spread(planned) < spread(original)  # skewed input strictly improves
    # And the step makespan never worsens.
    loads = [sum(cost_fn(costs[i]) for i in row) for row in planned]
    assert max(loads) <= max(sum(cost_fn(costs[i]) for i in row) for row in original)


def test_planner_rejects_bad_blocks():
    costs = [PackedCost(1, 1, 1, 1, 8)] * 8
    planner = StepRebalancePlanner(costs, batch_size=1, pack_score=lambda c: 1.0)
    with pytest.raises(ValueError):
        planner([[0, 1], [2]])
    planner3 = StepRebalancePlanner(costs, batch_size=3, pack_score=lambda c: 1.0)
    with pytest.raises(ValueError):
        planner3([[0, 1], [2, 3]])


def test_planner_failure_keeps_base_blocks():
    data = list(range(64))

    def sampler_with(planner):
        sampler = DPBlockDistributedSampler(
            dataset=data,
            num_replicas=4,
            rank=0,
            dp_size=4,
            dp_rank=0,
            rank_block=4,
            shuffle=True,
            seed=3,
            drop_last=True,
        )
        sampler.step_planner = planner
        sampler.set_epoch(0)
        return list(sampler)

    base = sampler_with(None)

    def broken(blocks):
        raise RuntimeError("boom")

    assert sampler_with(broken) == base

    def wrong_shape(blocks):
        return [list(blocks[0])]

    assert sampler_with(wrong_shape) == base


class _WorkerVisiblePacks(Dataset):
    """Four packs, five segments; re-packing redistributes them into new packs."""

    def __init__(self):
        self.dataset = {"lengths": [[64], [34], [58], [66], [36]]}
        self.packed_idx = [[0, 1], [2], [3], [4]]
        self.packing_length = 100

    def __len__(self):
        return len(self.packed_idx)

    def __getitem__(self, index):
        return index, tuple(self.packed_idx[index])


def _read_with_workers(dataset, *, dp_rank, planner, num_workers):
    sampler = DPBlockDistributedSampler(
        dataset,
        num_replicas=2,
        rank=dp_rank,
        dp_size=2,
        dp_rank=dp_rank,
        rank_block=2,
        shuffle=False,
        seed=0,
        drop_last=True,
        step_planner=planner,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=lambda batch: batch[0],
    )
    return {index: recipe for index, recipe in loader}


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_workers_see_repacked_recipes(num_workers):
    """Worker processes must consume the recipes planned by the main sampler."""
    seen = {}
    expected = None
    for dp_rank in range(2):
        base_dataset = _WorkerVisiblePacks()
        resolved, planner = resolve_dp_rebalance_plan(
            base_dataset,
            _config(enabled=True, cost_model="ffd_squares"),
            gbs=4,
            dp_size=2,
            batch_size=1,
            rank_block=2,
        )
        assert planner is not None and planner.repacks
        seen.update(
            _read_with_workers(
                resolved, dp_rank=dp_rank, planner=planner, num_workers=num_workers
            )
        )
        if expected is None:
            expected = {
                index: tuple(resolved.packed_idx[index])
                for index in range(len(base_dataset))
            }
    assert seen == expected


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------


def test_cost_model_weights_and_scores():
    class ModelWithHook:
        def get_dp_rebalance_cost_profile(self):
            return {"alpha": 9.0, "beta": 8.0, "gamma": 7.0}

    hybrid = CostModel(_config(enabled=True, cost_model="ffd_hybrid", alpha=1.5), ModelWithHook())
    assert hybrid.weights() == (1.5, 8.0, 7.0)
    # no model hook: fall back to defaults
    assert CostModel(_config(enabled=True, cost_model="ffd_hybrid", alpha=1.5), None).weights() == (
        1.5,
        0.0,
        64.0,
    )
    squares = _squares_model()
    pack = PackedCost(tokens=10, squared_lengths=100, chunks=2, longest=5, capacity=32)
    assert squares.pack_score(pack) == 100.0
    assert squares.segment_score(5) == 25.0
    tokens = CostModel(_config(enabled=True, cost_model="ffd_tokens"))
    assert tokens.pack_score(pack) == 10.0
    assert tokens.segment_score(5) == 5.0
    # segment scores derive from the length directly (no cost object)
    assert hybrid.segment_score(5) == 1.5 * 25 + 8.0 * 5 + 7.0
