"""In-memory row cache for lazily encoded training datasets.

ms-swift's ``LazyLLMDataset`` runs ``template.encode`` (tokenize + media
preprocess) on every row fetch, so packing collators redo identical host work
on every epoch. ``RowCacheDataset`` memoizes encoded rows per rank up to a
byte budget; rows beyond the budget pass through uncached.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import Dataset

from ...logging import get_logger

logger = get_logger(__name__)


def _estimate_row_bytes(row: Any) -> int:
    """Rough per-row memory cost: tensors by storage, int lists by 8 bytes."""
    if isinstance(row, torch.Tensor):
        return row.numel() * row.element_size()
    if isinstance(row, dict):
        return sum(_estimate_row_bytes(value) for value in row.values())
    if isinstance(row, (list, tuple)):
        if row and all(isinstance(item, int) for item in row):
            return 8 * len(row)
        return sum(_estimate_row_bytes(item) for item in row)
    return 256


class RowCacheDataset(Dataset):
    """Memoize ``dataset[index]`` results until ``max_bytes`` is reached.

    Rows are deep-copied on store and again on every hit, so downstream
    in-place edits (collation, label masking, bridging) never leak into the
    cache or across epochs. Column access (``dataset["lengths"]``) and extra
    attributes pass through to the wrapped dataset.
    """

    def __init__(self, dataset: Dataset, max_bytes: int) -> None:
        self._dataset = dataset
        self._max_bytes = max_bytes
        self._cache: dict[int, Any] = {}
        self._cached_bytes = 0
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            return self._dataset[index]
        row = self._cache.get(index)
        if row is not None:
            self._hits += 1
            return copy.deepcopy(row)
        self._misses += 1
        row = self._dataset[index]
        row_bytes = _estimate_row_bytes(row)
        if self._cached_bytes + row_bytes <= self._max_bytes:
            self._cache[index] = copy.deepcopy(row)
            self._cached_bytes += row_bytes
        return row

    def __getattr__(self, name: str) -> Any:
        # Underscore lookups (e.g. during unpickling before __init__) must not
        # recurse through the wrapped dataset.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def log_stats(self) -> None:
        logger.info_rank0(
            "RowCacheDataset: hits=%d misses=%d cached_rows=%d cached=%.1f MiB",
            self._hits,
            self._misses,
            len(self._cache),
            self._cached_bytes / 2**20,
        )


def wrap_lazy_rows(dataset: Any, max_bytes: int) -> Any:
    """Wrap the lazily encoding dataset under a packing/collator chain.

    ``PackingDataset`` holds the encoder under ``.dataset``; wrap that inner
    dataset in place so the outer packing indices stay intact. A bare
    ``LazyLLMDataset`` (duck-typed by ``encode_func``) is wrapped directly.
    Returns the dataset to hand to the DataLoader.
    """
    if hasattr(dataset, "encode_func"):
        wrapped = RowCacheDataset(dataset, max_bytes)
        logger.info_rank0(
            "RowCacheDataset: caching lazily encoded rows (budget %.1f MiB)",
            max_bytes / 2**20,
        )
        return wrapped
    inner = getattr(dataset, "dataset", None)
    if inner is not None and hasattr(inner, "encode_func"):
        dataset.dataset = RowCacheDataset(inner, max_bytes)
        logger.info_rank0(
            "RowCacheDataset: caching lazily encoded rows under %s "
            "(budget %.1f MiB)",
            type(dataset).__name__,
            max_bytes / 2**20,
        )
        return dataset
    logger.info_rank0(
        "RowCacheDataset: dataset %s is not lazily encoded; cache not applied",
        type(dataset).__name__,
    )
    return dataset
