"""Attention-aware per-step DP rebalance for packed datasets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from torch.utils.data import ConcatDataset, Subset

from ..logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_COST_MODELS = ("none", "ffd_tokens", "ffd_squares", "ffd_hybrid")


@dataclass
class DPRebalanceConfig:
    """Configuration for attention-cost-aware DP rebalance."""

    enabled: bool = False
    cost_model: str = "none"
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    capacity_margin: int = 1024

    def __post_init__(self) -> None:
        self.enabled = self._parse_bool(self.enabled)
        self.cost_model = str(self.cost_model).strip().lower()
        if self.cost_model not in _COST_MODELS:
            raise ValueError(
                f"dp_rebalance.cost_model must be one of {_COST_MODELS}, got {self.cost_model!r}."
            )
        self.capacity_margin = self._parse_nonnegative_int(self.capacity_margin)
        for name in ("alpha", "beta", "gamma"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, float(value))

    @property
    def active(self) -> bool:
        """Balancing is on."""
        return self.enabled and self.cost_model != "none"

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _parse_nonnegative_int(value: Any) -> int:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"dp_rebalance capacity_margin must be >= 0, got {parsed}.")
        return parsed


# ---------------------------------------------------------------------------
# Cost domain: pack/segment cost values and scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedCost:
    """Main-model scheduling metadata of one pack; vision work excluded."""

    tokens: int
    squared_lengths: int
    chunks: int
    longest: int
    capacity: int

    def score(self, *, alpha: float, beta: float, gamma: float) -> float:
        """alpha * quadratic(FA/MLA) + beta * linear(KDA/FFN) + gamma * segment tails."""
        return (
            alpha * float(self.squared_lengths)
            + beta * float(self.tokens)
            + gamma * float(self.chunks)
        )


class CostModel:
    """Scores packs and segments under the configured cost model."""

    def __init__(self, dp_rebalance: DPRebalanceConfig, model: Any | None = None) -> None:
        if dp_rebalance.cost_model not in _COST_MODELS:
            raise ValueError(
                f"dp_rebalance.cost_model must be one of {_COST_MODELS}, "
                f"got {dp_rebalance.cost_model!r}."
            )
        self._name = dp_rebalance.cost_model
        profile: dict[str, Any] = {}
        hook = getattr(model, "get_dp_rebalance_cost_profile", None)
        if callable(hook):
            try:
                proposed = hook()
                if isinstance(proposed, dict):
                    profile = {k: v for k, v in proposed.items() if v is not None}
            except Exception:  # noqa: BLE001 - hooks must never break scheduling
                logger.warning_rank0(
                    "Model dp-rebalance cost profile hook failed; using config defaults.",
                    exc_info=True,
                )

        def pick(explicit: float | None, key: str, default: float) -> float:
            if explicit is not None:
                return float(explicit)
            try:
                return float(profile.get(key, default))
            except (TypeError, ValueError):
                return default

        self._alpha = pick(dp_rebalance.alpha, "alpha", 1.0)
        self._beta = pick(dp_rebalance.beta, "beta", 0.0)
        self._gamma = pick(dp_rebalance.gamma, "gamma", 64.0)

    @property
    def name(self) -> str:
        return self._name

    def weights(self) -> tuple[float, float, float]:
        """Resolved (alpha, beta, gamma) hybrid weights."""
        return (self._alpha, self._beta, self._gamma)

    def pack_score(self, cost: PackedCost) -> float:
        """Score of one pack under the configured cost model."""
        if self._name == "ffd_tokens":
            return float(cost.tokens)
        if self._name == "ffd_squares":
            return float(cost.squared_lengths)
        if self._name == "ffd_hybrid":
            return cost.score(alpha=self._alpha, beta=self._beta, gamma=self._gamma)
        raise ValueError(f"Unknown dp rebalance cost model for scoring: {self._name!r}")

    def segment_score(self, size: int) -> float:
        """Return the cost-model score of one segment length."""
        size = int(size)
        if self._name == "ffd_tokens":
            return float(size)
        if self._name == "ffd_squares":
            return float(size * size)
        if self._name == "ffd_hybrid":
            return (
                self._alpha * float(size * size)
                + self._beta * float(size)
                + self._gamma
            )
        raise ValueError(f"Unknown dp rebalance cost model for scoring: {self._name!r}")


# ---------------------------------------------------------------------------
# Dataset capability domain: structural detection and recipe material
# ---------------------------------------------------------------------------


class LayoutKind(Enum):
    """Form of one dataset item."""

    PACKED_RECIPE = "packed_recipe"
    PACKED_MATERIALIZED = "packed_materialized"
    UNPACKED = "unpacked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DatasetCapabilities:
    """What the scheduler may do with a dataset: costs, layout, recomposability."""

    costs: Sequence[PackedCost] | None
    layout: LayoutKind
    recomposable: bool = False


@runtime_checkable
class RecipePackingContract(Protocol):
    """Structural contract: one pack = a list of source-row references."""

    dataset: Any
    packed_idx: Sequence[Sequence[int]]
    packing_length: int


CapabilityDetector = Callable[[Any], "DatasetCapabilities | None"]


class DatasetAnalyzer:
    """Detects cost and layout capabilities of packed datasets."""

    def __init__(self) -> None:
        self._detectors: list[tuple[int, str, CapabilityDetector]] = []
        self.register(self._detect_recipe_packed, name="recipe_packed", priority=50)
        self.register(
            self._detect_materialized_packed, name="materialized_packed", priority=60
        )

    def register(
        self, detector: CapabilityDetector, *, name: str, priority: int = 100
    ) -> None:
        """Register a structural dataset-capability detector."""
        self._detectors.append((priority, name, detector))
        self._detectors.sort(key=lambda item: item[0])

    def known_detectors(self) -> list[str]:
        return [name for _, name, _ in self._detectors]

    def resolve(
        self, dataset: Any, override: "DatasetCapabilities | Sequence[PackedCost] | None" = None
    ) -> DatasetCapabilities | None:
        """Resolve capabilities: explicit override, then wrappers, then detectors."""
        if override is not None:
            if isinstance(override, DatasetCapabilities):
                return override
            return DatasetCapabilities(costs=tuple(override), layout=LayoutKind.UNKNOWN)
        caps = self._merge_wrapped(dataset) or self._detect(dataset)
        if (
            caps is not None
            and caps.costs is not None
            and len(caps.costs) != len(dataset)  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"Capability costs cover {len(caps.costs)} items but the dataset has {len(dataset)}."  # type: ignore[arg-type]
            )
        return caps

    def recipe_material(
        self, dataset: Any
    ) -> tuple[list[list[int]], list[int], int] | None:
        """Return the original recipe, source lengths and capacity when re-packable."""
        packed_idx = getattr(dataset, "packed_idx", None)
        if not isinstance(packed_idx, (list, tuple)):
            return None
        sizes = self._source_lengths(dataset)
        if sizes is None:
            return None
        capacity = int(getattr(dataset, "packing_length", 0) or 0)
        packed_length = getattr(dataset, "packed_length", None)
        if capacity <= 0 and isinstance(packed_length, (list, tuple)) and packed_length:
            capacity = int(max(packed_length))
        if capacity <= 0:
            return None
        groups: list[list[int]] = []
        referenced = [False] * len(sizes)
        seen: set[int] = set()
        for pack in packed_idx:
            members: list[int] = []
            for raw in pack:
                index = int(raw)
                if index < 0 or index >= len(sizes) or int(sizes[index]) <= 0:
                    return None
                if index in seen:
                    return None
                seen.add(index)
                members.append(index)
                referenced[index] = True
            groups.append(members)
        if not all(referenced):
            return None
        return groups, [int(size) for size in sizes], capacity

    # -- Detection internals. ------------------------------------------------

    def _detect(self, dataset: Any) -> DatasetCapabilities | None:
        for _, name, detector in self._detectors:
            try:
                caps = detector(dataset)
            except Exception:  # noqa: BLE001 - detectors must never break loading
                logger.warning_rank0(
                    "Capability detector %r raised on dataset %r; ignoring it.",
                    name,
                    type(dataset).__name__,
                    exc_info=True,
                )
                continue
            if caps is not None:
                return caps
        return None

    def _merge_wrapped(self, dataset: Any) -> DatasetCapabilities | None:
        """Merge costs through ConcatDataset/Subset; wrappers are never recomposable."""
        if isinstance(dataset, ConcatDataset):
            costs: list[PackedCost] = []
            layout = LayoutKind.UNKNOWN
            for child in dataset.datasets:
                caps = self.resolve(child)
                if caps is None or caps.costs is None:
                    return None
                costs.extend(caps.costs)
                if layout is LayoutKind.UNKNOWN:
                    layout = caps.layout
                elif caps.layout != layout:
                    layout = LayoutKind.UNKNOWN
            return DatasetCapabilities(costs=tuple(costs), layout=layout)
        if isinstance(dataset, Subset):
            caps = self.resolve(dataset.dataset)
            if caps is None or caps.costs is None:
                return None
            inner = caps.costs
            return DatasetCapabilities(
                costs=tuple(inner[int(i)] for i in dataset.indices), layout=caps.layout
            )
        return None

    @staticmethod
    def _normalize_length(value: Any) -> int:
        return int(sum(value)) if isinstance(value, (list, tuple)) else int(value)

    @staticmethod
    def _source_lengths(dataset: Any) -> Sequence[int] | None:
        """Normalized per-source-row token lengths backing the recipe."""
        source = getattr(dataset, "dataset", None)
        if source is None:
            return None
        try:
            raw = source["lengths"]
        except Exception:  # noqa: BLE001 - dict-style access may be absent
            raw = getattr(source, "lengths", None)
        if raw is None:
            return None
        try:
            return [DatasetAnalyzer._normalize_length(value) for value in raw]
        except Exception:  # noqa: BLE001 - unreadable length entries
            return None

    def _recipe_costs(self, dataset: Any) -> "tuple[list[PackedCost], list[int]] | None":
        """Per-pack costs plus flat source-row sizes (empty = re-packing unsupported)."""
        packed_idx = dataset.packed_idx
        source_lengths = self._source_lengths(dataset)
        capacity = int(getattr(dataset, "packing_length", 0) or 0)
        packed_length = getattr(dataset, "packed_length", None)
        if capacity <= 0 and isinstance(packed_length, (list, tuple)) and packed_length:
            capacity = int(max(packed_length))

        costs: list[PackedCost] = []
        referenced: dict[int, int] = {}
        for pack_index, indices in enumerate(packed_idx):
            sizes: list[int] = []
            if source_lengths is not None:
                try:
                    sizes = [int(source_lengths[int(i)]) for i in indices]
                except Exception:  # noqa: BLE001 - fall back to packed totals
                    sizes = []
            if not sizes and isinstance(packed_length, (list, tuple)):
                sizes = [int(packed_length[pack_index])]
            if not sizes or any(size <= 0 for size in sizes):
                return None
            if source_lengths is not None:
                for i, size in zip(indices, sizes):
                    referenced[int(i)] = size
            costs.append(
                PackedCost(
                    tokens=sum(sizes),
                    squared_lengths=sum(size * size for size in sizes),
                    chunks=len(sizes),
                    longest=max(sizes),
                    capacity=capacity or sum(sizes),
                )
            )
        if source_lengths is None:
            return costs, []
        flat = [0] * len(source_lengths)
        for index, size in referenced.items():
            if index < len(flat):
                flat[index] = size
        # Rows missing from any pack disable re-packing.
        if any(size <= 0 for size in flat):
            flat = []
        return costs, flat

    def _detect_recipe_packed(self, dataset: Any) -> DatasetCapabilities | None:
        """Detect recipe-driven packs; recomposable because getitem follows the recipe."""
        if isinstance(dataset, (ConcatDataset, Subset)):
            return None
        if not isinstance(dataset, RecipePackingContract):
            packed_idx = getattr(dataset, "packed_idx", None)
            if not isinstance(packed_idx, (list, tuple)):
                return None
            if not hasattr(dataset, "packing_length") and not hasattr(dataset, "packed_length"):
                return None

        resolved = self._recipe_costs(dataset)
        if resolved is None:
            return None
        costs, sizes = resolved
        return DatasetCapabilities(
            costs=tuple(costs), layout=LayoutKind.PACKED_RECIPE, recomposable=bool(sizes)
        )

    @staticmethod
    def _row_materialized_cost(row_length: int, params: dict[str, Any] | None) -> PackedCost:
        """Cost of one materialized row; boundary metadata refines it when present."""
        sizes: list[int] = []
        if params is not None:
            boundaries = params.get("sequence_boundaries")
            if boundaries and len(boundaries) >= 2:
                sizes = [
                    int(boundaries[i + 1]) - int(boundaries[i])
                    for i in range(len(boundaries) - 1)
                ]
                right_pad = int(params.get("right_padding_length", 0) or 0)
                if right_pad > 0 and sizes and sizes[-1] == right_pad:
                    sizes = sizes[:-1]
                sizes = [size for size in sizes if size > 0]
        if not sizes:
            sizes = [row_length]
        return PackedCost(
            tokens=sum(sizes),
            squared_lengths=sum(size * size for size in sizes),
            chunks=len(sizes),
            longest=max(sizes),
            capacity=row_length,
        )

    def _detect_materialized_packed(self, dataset: Any) -> DatasetCapabilities | None:
        """Detect atomic fixed-length packed rows from the input_ids column."""
        if isinstance(dataset, (ConcatDataset, Subset)):
            return None
        features = getattr(dataset, "features", None)
        if features is None:
            return None
        feature_names = set(features.keys())
        if "input_ids" not in feature_names or "labels" not in feature_names:
            return None

        try:
            probe = dataset
            select_columns = getattr(dataset, "select_columns", None)
            if callable(select_columns):
                probe = select_columns(["input_ids"])  # never decode media columns
            first = probe[0]
        except Exception:  # noqa: BLE001 - not decodable / empty dataset
            return None
        input_ids = first.get("input_ids")
        if not isinstance(input_ids, (list, tuple)) or not input_ids:
            return None
        row_length = int(len(input_ids))

        boundary_params: Sequence[Any] | None = None
        if "packing_params" in feature_names:
            try:
                boundary_params = dataset["packing_params"]
            except Exception:  # noqa: BLE001 - column access failed
                boundary_params = None

        costs: list[PackedCost] = []
        for index in range(len(dataset)):
            params = None
            if boundary_params is not None and index < len(boundary_params):
                params = boundary_params[index]
            costs.append(self._row_materialized_cost(row_length, params))
        return DatasetCapabilities(costs=tuple(costs), layout=LayoutKind.PACKED_MATERIALIZED)


# ---------------------------------------------------------------------------
# Re-packing domain: fixed-count re-bin primitive + per-step recipe view
# ---------------------------------------------------------------------------


class RepackableView:
    """Shallow dataset copy whose pack recipes are rewritten per step window."""

    def __init__(
        self,
        dataset: Any,
        packs: Sequence[Sequence[int]],
        sizes: Sequence[int],
        capacity: int,
    ) -> None:
        self._dataset = copy.copy(dataset)
        self._dataset.packed_idx = [list(pack) for pack in packs]
        packed_length = getattr(dataset, "packed_length", None)
        if isinstance(packed_length, (list, tuple)):
            self._dataset.packed_length = [int(length) for length in packed_length]
        self._original_packs = [list(pack) for pack in packs]
        self._sizes = [int(size) for size in sizes]
        self._capacity = int(capacity)
        self._last_error: str | None = None
        self._failed_window_logs = 0
        if getattr(dataset, "packing_length", None) is not None:
            self._dataset.packing_length = int(capacity)

    @property
    def dataset(self) -> Any:
        """The shallow-copied dataset handed to the dataloader."""
        return self._dataset

    @property
    def last_error(self) -> str | None:
        """Human-readable reason for the most recent failed re-pack."""
        return self._last_error

    @property
    def capacity(self) -> int:
        return self._capacity

    @staticmethod
    def pack_cost(member_sizes: Sequence[int], capacity: int) -> PackedCost:
        """Cost metadata of one pack given its per-segment lengths."""
        return PackedCost(
            tokens=int(sum(member_sizes)),
            squared_lengths=int(sum(size * size for size in member_sizes)),
            chunks=len(member_sizes),
            longest=int(max(member_sizes)),
            capacity=capacity,
        )

    @staticmethod
    def _fill_bins(
        order: Sequence[int],
        seg_sizes: Sequence[int],
        seg_costs: Sequence[float] | None,
        pack_count: int,
        capacity: int,
    ) -> "list[list[int]] | None":
        """Place segments into bins with cost-aware or first-fit placement."""
        bins_tokens = [0] * pack_count
        bins_cost = [0.0] * pack_count
        bins_members: list[list[int]] = [[] for _ in range(pack_count)]
        for segment in order:
            size = seg_sizes[segment]
            best = -1
            best_cost = 0.0
            best_tokens = 0
            for index in range(pack_count):
                if bins_tokens[index] + size > capacity:
                    continue
                if seg_costs is None:
                    best = index
                    break
                cost = bins_cost[index]
                if best < 0 or cost < best_cost or (
                    cost == best_cost and bins_tokens[index] < best_tokens
                ):
                    best = index
                    best_cost = cost
                    best_tokens = bins_tokens[index]
            if best < 0:
                return None
            bins_tokens[best] += size
            if seg_costs is not None:
                bins_cost[best] += seg_costs[segment]
            bins_members[best].append(segment)
        return bins_members

    @staticmethod
    def _bfd_dynamic(
        seg_sizes: Sequence[int],
        capacity: int,
        pack_count: int,
    ) -> "list[list[int]] | None":
        """Pack with dynamic best-fit-decreasing and pad to ``pack_count`` bins."""
        count = len(seg_sizes)
        order = sorted(range(count), key=lambda index: (-int(seg_sizes[index]), index))
        bins: list[list[int]] = []
        bin_tokens: list[int] = []
        for index in order:
            size = int(seg_sizes[index])
            candidates = [
                candidate
                for candidate in range(len(bins))
                if bin_tokens[candidate] + size <= capacity
            ]
            if candidates:
                target = max(candidates, key=lambda candidate: bin_tokens[candidate])
            else:
                bins.append([])
                bin_tokens.append(0)
                target = len(bins) - 1
            bins[target].append(index)
            bin_tokens[target] += size
        if len(bins) > pack_count:
            return None
        # Split multi-segment bins to reach the fixed pack count without empties.
        while len(bins) < pack_count:
            donor = next(
                (
                    candidate
                    for candidate, members in enumerate(bins)
                    if len(members) >= 2
                ),
                None,
            )
            if donor is None:
                return None
            member = min(bins[donor], key=lambda index: int(seg_sizes[index]))
            bins[donor].remove(member)
            bin_tokens[donor] -= int(seg_sizes[member])
            bins.append([member])
            bin_tokens.append(int(seg_sizes[member]))
        bins.sort(key=lambda members: members[0])
        return bins

    @staticmethod
    def repack_segments(
        pack_count: int,
        sizes: Sequence[int],
        capacity: int,
        cost_fn: Callable[[int], float],
    ) -> "list[list[int]]":
        """Pack ``sizes`` into exactly ``pack_count`` capacity-bounded bins."""
        if capacity <= 0:
            raise ValueError("Re-packing needs a positive packing capacity")
        if pack_count <= 0:
            raise ValueError("Re-packing needs a positive pack count")
        count = len(sizes)
        if count < pack_count:
            raise ValueError(
                "Re-packing needs at least one segment per pack "
                f"(segments={count}, packs={pack_count}, capacity={capacity})"
            )
        seg_sizes = [0] * count
        seg_costs = [0.0] * count
        for i in range(count):
            size = int(sizes[i])
            if size <= 0:
                raise ValueError(f"Re-packing needs positive segment lengths, got {size}")
            if size > capacity:
                raise ValueError(
                    f"Segment {i} of length {size} exceeds capacity {capacity}"
                )
            seg_sizes[i] = size
            seg_costs[i] = cost_fn(size)
        order = sorted(range(count), key=lambda i: (-seg_costs[i], i))
        # Cost-aware placement can dead-end; fall back to first-fit then dynamic BFD.
        bins_members = RepackableView._fill_bins(
            order, seg_sizes, seg_costs, pack_count, capacity
        )
        if bins_members is None:
            bins_members = RepackableView._fill_bins(
                order, seg_sizes, None, pack_count, capacity
            )
        if bins_members is None:
            bins_members = RepackableView._bfd_dynamic(
                seg_sizes, capacity, pack_count
            )
        if bins_members is None:
            raise ValueError(
                f"Segments do not fit {pack_count} packs of capacity {capacity}"
            )
        if any(not members for members in bins_members):
            raise ValueError("Re-packing produced empty packs")
        bins_members.sort(key=lambda members: members[0])
        return bins_members

    def repack_window(
        self,
        indices: Sequence[int],
        *,
        segment_score: Callable[[int], float],
        pack_score: Callable[[PackedCost], float],
    ) -> dict[int, float] | None:
        """Re-pack one step window in place, returning new scores or None after restoring."""
        if len(set(indices)) != len(indices):
            self._last_error = "window repeats the same pack index"
            return None
        # Collect the window's unique source segments in first-seen order.
        local_to_global: list[int] = []
        position: dict[int, int] = {}
        for index in indices:
            for raw in self._original_packs[index]:
                segment = int(raw)
                if segment in position:
                    self._last_error = (
                        f"segment {segment} appears more than once in the window"
                    )
                    return None
                position[segment] = len(local_to_global)
                local_to_global.append(segment)
        local_sizes = [self._sizes[segment] for segment in local_to_global]
        packed_length = getattr(self._dataset, "packed_length", None)
        try:
            bins = self.repack_segments(
                len(indices), local_sizes, self._capacity, segment_score
            )
            new_scores: dict[int, float] = {}
            updates: dict[int, list[int]] = {}
            packed_length_updates: dict[int, int] = {}
            for index, members in zip(sorted(indices), bins):
                segments = [local_to_global[member] for member in members]
                member_sizes = [self._sizes[segment] for segment in segments]
                updates[index] = segments
                if isinstance(packed_length, list):
                    packed_length_updates[index] = int(sum(member_sizes))
                new_scores[index] = pack_score(
                    self.pack_cost(member_sizes, self._capacity)
                )
            # Commit only after every new recipe was computed successfully.
            for index, segments in updates.items():
                self._dataset.packed_idx[index] = segments
            if isinstance(packed_length, list):
                for index, length in packed_length_updates.items():
                    packed_length[index] = length
            self._last_error = None
            return new_scores
        except ValueError as exc:
            self.restore(indices)
            self._last_error = str(exc)
            self._log_failed_window(local_sizes, len(indices), exc)
            return None

    def restore(self, indices: Sequence[int]) -> None:
        """Reset the recipes of the given packs to the original packing."""
        packed_length = getattr(self._dataset, "packed_length", None)
        for index in indices:
            pack = self._original_packs[index]
            self._dataset.packed_idx[index] = list(pack)
            if isinstance(packed_length, list):
                packed_length[index] = int(
                    sum(self._sizes[segment] for segment in pack)
                )

    def _log_failed_window(
        self, sizes: Sequence[int], pack_count: int, exc: Exception
    ) -> None:
        """Log the first failed windows for off-line repack failure analysis."""
        total = sum(sizes)
        budget = pack_count * self._capacity
        largest = max(sizes) if sizes else 0
        fill = total / budget if budget else 0.0
        logger.info_rank0(
            "TorchTurbo dp rebalance repack input for %d packs: "
            "segments=%d total=%d budget=%d fill=%.4f "
            "max_segment=%d over_budget=%s (%s)",
            pack_count,
            len(sizes),
            total,
            budget,
            fill,
            largest,
            total > budget,
            exc,
        )
        if self._failed_window_logs >= 3:
            return
        self._failed_window_logs += 1
        counts: dict[int, int] = {}
        for size in sizes:
            counts[size] = counts.get(size, 0) + 1
        top_sizes = sorted(counts, key=lambda size: (-counts[size], -size))[:8]
        dup_pairs = " ".join(f"{size}x{counts[size]}" for size in top_sizes)
        edges = [self._capacity * step // 8 for step in range(1, 9)]
        hist = [0] * len(edges)
        for size in sizes:
            for step, edge in enumerate(edges):
                if size <= edge:
                    hist[step] += 1
                    break
        logger.info_rank0(
            "TorchTurbo dp rebalance failed window sizes=%s", list(sizes)
        )
        logger.info_rank0(
            "TorchTurbo dp rebalance failed window shape: histogram(<=cap*i/8)=%s "
            "top_duplicates=[%s]",
            hist,
            dup_pairs,
        )


# ---------------------------------------------------------------------------
# Scheduling domain: per-step planner for the DP-block sampler
# ---------------------------------------------------------------------------


class StepRebalancePlanner:
    """Re-packs a step window and rebalances its microbatches across DP ranks."""

    def __init__(
        self,
        costs: Sequence[PackedCost],
        *,
        batch_size: int,
        pack_score: Callable[[PackedCost], float],
        segment_score: Callable[[int], float] | None = None,
        repack_view: RepackableView | None = None,
    ) -> None:
        self._costs = tuple(costs)
        self._batch_size = int(batch_size)
        self._pack_score = pack_score
        self._segment_score = segment_score
        self._repack_view = repack_view
        if self._repack_view is not None and self._segment_score is None:
            raise ValueError("Re-packing needs a segment_score function.")
        self._windows = 0
        self._repack_windows = 0
        self._repack_fallbacks = 0
        self._assignment_fallbacks = 0
        self._debug_logged = 0

    @property
    def repacks(self) -> bool:
        """Whether step windows are re-packed before reassignment."""
        return self._repack_view is not None

    @property
    def stats(self) -> dict[str, int]:
        """Aggregate step-window counters for debug observability."""
        return {
            "windows": self._windows,
            "repack_windows": self._repack_windows,
            "repack_fallbacks": self._repack_fallbacks,
            "assignment_fallbacks": self._assignment_fallbacks,
        }

    def _check_blocks(self, blocks: Sequence[Sequence[int]]) -> tuple[int, int]:
        """Validate one step window's DP block shape."""
        dp_size = len(blocks)
        rank_block = len(blocks[0]) if blocks else 0
        if (
            dp_size == 0
            or rank_block <= 0
            or self._batch_size <= 0
            or rank_block % self._batch_size
        ):
            raise ValueError("DP blocks must contain whole microbatches")
        if any(len(items) != rank_block for items in blocks):
            raise ValueError("All DP blocks must have the same length")
        return dp_size, rank_block

    def _window_metrics(
        self,
        blocks: Sequence[Sequence[int]],
        scores: dict[int, float] | None,
    ) -> tuple[list[float], float]:
        """Per-rank microbatch cost list and input-layout makespan."""
        unit_costs: list[float] = []
        append_cost = unit_costs.append
        makespan = 0.0
        get_score = scores.get if scores is not None else None
        for items in blocks:
            rank_load = 0.0
            for offset in range(0, len(items), self._batch_size):
                total = 0.0
                for position in range(offset, offset + self._batch_size):
                    index = items[position]
                    if get_score is not None:
                        score = get_score(index)
                        if score is not None:
                            total += score
                            continue
                    total += self._pack_score(self._costs[index])
                append_cost(total)
                rank_load += total
            makespan = max(makespan, rank_load)
        return unit_costs, makespan

    @staticmethod
    def _spread(unit_costs: Sequence[float], dp_size: int) -> float:
        """Worst max-minus-min microbatch cost spread in one step window."""
        microsteps = len(unit_costs) // dp_size
        if microsteps <= 0:
            return 0.0
        worst = 0.0
        for step in range(microsteps):
            values = [unit_costs[rank * microsteps + step] for rank in range(dp_size)]
            worst = max(worst, max(values) - min(values))
        return worst

    def _debug_window(
        self,
        *,
        reason: str,
        before: float,
        after: float | None,
        dp_size: int,
    ) -> None:
        """Emit bounded debug lines for the first windows; never per-window polluted."""
        if self._debug_logged >= 16:
            return
        self._debug_logged += 1
        logger.debug_rank0(
            "TorchTurbo dp rebalance window %d %s: spread %.3f -> %s (dp_size=%d)",
            self._windows,
            reason,
            before,
            f"{after:.3f}" if after is not None else "n/a",
            dp_size,
        )

    def __call__(self, blocks: Sequence[Sequence[int]]) -> list[list[int]]:
        """Re-pack (when possible) then rebalance one step window."""
        dp_size, _ = self._check_blocks(blocks)
        self._windows += 1
        new_scores: dict[int, float] | None = None
        if self._repack_view is not None:
            window = [index for items in blocks for index in items]
            base_units, base_makespan = self._window_metrics(blocks, None)
            base_spread = self._spread(base_units, dp_size)
            new_scores = self._repack_view.repack_window(
                window,
                segment_score=self._segment_score,
                pack_score=self._pack_score,
            )
            if new_scores is None:
                self._repack_fallbacks += 1
                self._debug_window(
                    reason=f"fallback (repack failed: {self._repack_view.last_error or 'n/a'})",
                    before=base_spread,
                    after=None,
                    dp_size=dp_size,
                )
            else:
                repacked_units, repacked_makespan = self._window_metrics(
                    blocks, new_scores
                )
                repacked_spread = self._spread(repacked_units, dp_size)
                if (
                    repacked_spread > base_spread + 1e-12
                    or repacked_makespan > base_makespan + 1e-12
                ):
                    self._repack_view.restore(window)
                    new_scores = None
                    self._repack_fallbacks += 1
                    self._debug_window(
                        reason="fallback (repack worse)",
                        before=base_spread,
                        after=repacked_spread,
                        dp_size=dp_size,
                    )
                    return self._assign(blocks, new_scores)
                self._repack_windows += 1
                self._debug_window(
                    reason="repacked",
                    before=base_spread,
                    after=repacked_spread,
                    dp_size=dp_size,
                )
                return self._assign(
                    blocks,
                    new_scores,
                    unit_costs=repacked_units,
                    base_makespan=repacked_makespan,
                )
        return self._assign(blocks, new_scores)

    def _assign(
        self,
        blocks: Sequence[Sequence[int]],
        new_scores: dict[int, float] | None,
        *,
        unit_costs: Sequence[float] | None = None,
        base_makespan: float | None = None,
    ) -> list[list[int]]:
        dp_size, rank_block = self._check_blocks(blocks)
        batch_size = self._batch_size
        ga = rank_block // batch_size
        if unit_costs is None or base_makespan is None:
            unit_costs, base_makespan = self._window_metrics(blocks, new_scores)
        unit_costs = list(unit_costs)

        # Sort cost-descending, cut one group per microstep, and deal each
        # group to the least-loaded ranks.
        order = sorted(range(len(unit_costs)), key=lambda unit: (-unit_costs[unit], unit))
        assigned: list[list[int]] = [[] for _ in range(dp_size)]
        loads = [0.0] * dp_size
        for group_start in range(0, len(order), dp_size):
            taken = 0  # bitmask: each rank gets exactly one unit per microstep
            for position in range(group_start, group_start + dp_size):
                unit = order[position]
                best = -1
                for rank in range(dp_size):
                    if taken & (1 << rank):
                        continue
                    if best < 0 or loads[rank] < loads[best]:
                        best = rank
                taken |= 1 << best
                assigned[best].append(unit)
                loads[best] += unit_costs[unit]

        # Fall back to the input blocks if balancing raises the makespan.
        assert base_makespan is not None
        if max(loads) > base_makespan:
            self._assignment_fallbacks += 1
            return [list(items) for items in blocks]

        result: list[list[int]] = []
        for rank_units in assigned:
            rank_items: list[int] = []
            extend = rank_items.extend
            for unit in rank_units:
                items = blocks[unit // ga]
                offset = (unit % ga) * batch_size
                extend(items[offset : offset + batch_size])
            result.append(rank_items)
        return result


def _log_repack_material(
    packs: Sequence[Sequence[int]],
    sizes: Sequence[int],
    capacity: int,
    *,
    margin: int = 0,
) -> None:
    """Log measured per-pack totals versus the re-packing capacity."""
    totals = sorted(sum(sizes[index] for index in pack) for pack in packs)
    if not totals:
        return
    sorted_sizes = sorted(sizes)
    over = sum(1 for total in totals if total > capacity)
    logger.debug_rank0(
        "TorchTurbo dp rebalance repack input: packs=%d segments=%d "
        "capacity=%d margin=%d pack totals min/median/max=%d/%d/%d "
        "over_capacity=%d max_fill=%.4f segment_max=%d",
        len(totals),
        len(sizes),
        capacity,
        margin,
        totals[0],
        totals[len(totals) // 2],
        totals[-1],
        over,
        totals[-1] / capacity,
        sorted_sizes[-1] if sorted_sizes else 0,
    )


def resolve_dp_rebalance_plan(
    dataset: Any,
    dp_rebalance: DPRebalanceConfig,
    *,
    model: Any | None = None,
    gbs: int | None,
    dp_size: int,
    batch_size: int,
    rank_block: int,
    drop_last: bool = True,
    allow_repack: bool = True,
) -> tuple[Any, StepRebalancePlanner | None]:
    """Build the dataset view and per-step planner used for DP rebalance."""
    if not dp_rebalance.active or gbs is None or dp_size <= 1 or batch_size <= 0:
        return dataset, None
    if rank_block % batch_size:
        logger.warning_rank0(
            "DP rebalance needs whole microbatches per rank block "
            "(rank_block=%d, batch_size=%d); keeping the base sampler.",
            rank_block,
            batch_size,
        )
        return dataset, None

    analyzer = DatasetAnalyzer()
    caps = analyzer.resolve(dataset)
    if caps is None or not caps.costs:
        logger.info_rank0(
            "DP rebalance enabled but no capability detector recognized dataset "
            "%r; keeping the original sampler order.",
            type(dataset).__name__,
        )
        return dataset, None

    cost_model = CostModel(dp_rebalance, model)
    material = analyzer.recipe_material(dataset)
    repack_view: RepackableView | None = None
    repack_skip_reason: str | None = None
    if material is not None:
        packs, sizes, base_capacity = material
        capacity = base_capacity + dp_rebalance.capacity_margin
        packed_length = getattr(dataset, "packed_length", None)
        if not drop_last:
            repack_skip_reason = "drop_last=False can repeat pack indices"
        elif not allow_repack:
            repack_skip_reason = "persistent workers do not see later recipe updates"
        elif packed_length is not None and not isinstance(packed_length, list):
            repack_skip_reason = (
                "packed_length must be a list for in-place recipe updates"
            )
        else:
            repack_view = RepackableView(dataset, packs, sizes, capacity)
            _log_repack_material(
                packs, sizes, capacity, margin=dp_rebalance.capacity_margin
            )

    if repack_view is not None:
        planner = StepRebalancePlanner(
            caps.costs,
            batch_size=batch_size,
            pack_score=cost_model.pack_score,
            segment_score=cost_model.segment_score,
            repack_view=repack_view,
        )
        logger.info_rank0(
            "TorchTurbo dp rebalance enabled: step re-packing + reassignment "
            "(cost_model=%s items=%d dp_size=%d rank_block=%d batch_size=%d "
            "capacity_margin=%d)",
            cost_model.name,
            len(caps.costs),
            dp_size,
            rank_block,
            batch_size,
            dp_rebalance.capacity_margin,
        )
        return repack_view.dataset, planner

    planner = StepRebalancePlanner(
        caps.costs, batch_size=batch_size, pack_score=cost_model.pack_score
    )
    skip = f"; repack skipped: {repack_skip_reason}" if repack_skip_reason else ""
    logger.info_rank0(
        "TorchTurbo dp rebalance enabled: whole-pack reassignment "
        "(cost_model=%s items=%d dp_size=%d rank_block=%d batch_size=%d%s)",
        cost_model.name,
        len(caps.costs),
        dp_size,
        rank_block,
        batch_size,
        skip,
    )
    return dataset, planner
