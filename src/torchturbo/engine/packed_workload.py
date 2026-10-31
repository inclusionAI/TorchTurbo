"""Balance existing packed samples without changing optimizer-batch membership."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING, Iterator, Sequence, Protocol, runtime_checkable

from torch.utils.data import ConcatDataset, Subset, Sampler

if TYPE_CHECKING:
    from .core import DPBlockDistributedSampler


@dataclass(frozen=True)
class PackedCost:
    """CPU scheduling metadata; score is a proxy, not measured FLOPs."""

    tokens: int
    squared_lengths: int
    chunks: int
    longest: int
    capacity: int

    @property
    def score(self) -> float:
        # Token work plus chunk-tail overhead and normalized quadratic work.
        return 64 * self.chunks + self.squared_lengths / self.capacity


@runtime_checkable
class CachedPackingMetadata(Protocol):
    """Metadata contract provided by cached packing datasets."""

    dataset: object
    packed_idx: Sequence[Sequence[int]]
    packing_length: int


def supports_packed_costs(dataset: object) -> bool:
    """Recognize cached pack metadata without reading sample tensors."""
    if isinstance(dataset, ConcatDataset):
        return all(supports_packed_costs(child) for child in dataset.datasets)
    if isinstance(dataset, Subset):
        return supports_packed_costs(dataset.dataset)
    return isinstance(dataset, CachedPackingMetadata)


def packed_costs(dataset: object) -> list[PackedCost]:
    """Read cached token lengths without encoding samples or decoding media."""
    if isinstance(dataset, ConcatDataset):
        cache = {}
        result = []
        for child in dataset.datasets:
            if id(child) not in cache:
                cache[id(child)] = packed_costs(child)
            result.extend(cache[id(child)])
        return result
    if isinstance(dataset, Subset):
        costs = packed_costs(dataset.dataset)
        return [costs[index] for index in dataset.indices]
    # Swift PackingDataset exposes these fields as its packing contract.
    lengths = dataset.dataset["lengths"]
    result = []
    for indices in dataset.packed_idx:
        # Swift stores encoded lengths as lists even for a single SFT row.
        encoded_lengths = [lengths[index] for index in indices]
        sizes = [operator.index(sum(value) if isinstance(value, list) else value)
                 for value in encoded_lengths]
        if any(size <= 0 for size in sizes):
            raise ValueError("Workload balancing requires positive encoded lengths")
        result.append(PackedCost(
            tokens=sum(sizes), squared_lengths=sum(size * size for size in sizes),
            chunks=sum((size + 63) // 64 for size in sizes), longest=max(sizes),
            capacity=dataset.packing_length,
        ))
    return result


def balance_step(
    original: Sequence[Sequence[int]], costs: Sequence[PackedCost],
    batch_size: int,
) -> list[list[int]]:
    """Group similar-cost packs in each wave, then balance cumulative rank work."""
    dp_size = len(original)
    rank_block = len(original[0])
    if batch_size <= 0 or rank_block % batch_size:
        raise ValueError("rank_block must contain whole microbatches")
    if any(len(items) != rank_block for items in original):
        raise ValueError("All DP ranks must have the same number of samples")
    # A unit preserves all samples that the original collator batches together.
    units = [list(items[offset:offset + batch_size])
             for items in original for offset in range(0, rank_block, batch_size)]
    unit_cost = lambda unit: sum(costs[index].score for index in unit)
    units.sort(key=lambda unit: (-unit_cost(unit), tuple(unit)))
    assigned = [[] for _ in range(dp_size)]
    loads = [0.0] * dp_size
    for offset in range(0, len(units), dp_size):
        wave = units[offset:offset + dp_size]
        ranks = sorted(range(dp_size), key=lambda rank: (loads[rank], rank))
        for rank, unit in zip(ranks, wave):
            assigned[rank].extend(unit)
            loads[rank] += unit_cost(unit)
    return assigned


class PackedWorkloadSampler(Sampler[int]):
    """Reassign whole microbatches within each existing optimizer batch."""

    def __init__(self, sampler: DPBlockDistributedSampler,
                 costs: Sequence[PackedCost], batch_size: int) -> None:
        if len(costs) != len(sampler.dataset):
            raise ValueError("Packed cost metadata must match the dataset length")
        if batch_size <= 0 or sampler.rank_block % batch_size:
            raise ValueError("Sampler blocks must contain whole microbatches")
        self.sampler = sampler
        self.costs = costs
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        streams = []
        for rank in range(self.sampler.dp_size):
            baseline = copy.copy(self.sampler)
            baseline.dp_rank = rank
            streams.append(list(baseline))
        block = self.sampler.rank_block
        for offset in range(0, len(streams[0]), block):
            original = [stream[offset:offset + block] for stream in streams]
            balanced = balance_step(original, self.costs, self.batch_size)
            yield from balanced[self.sampler.dp_rank]
