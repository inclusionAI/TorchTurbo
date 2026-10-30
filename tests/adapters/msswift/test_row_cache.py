import torch
from torch.utils.data import Dataset

from torchturbo.adapters.msswift.row_cache import RowCacheDataset, wrap_lazy_rows


class FakeLazyDataset(Dataset):
    """Mimics LazyLLMDataset: encode_func attr, per-fetch encoding cost."""

    def __init__(self, n=8):
        self.encode_func = lambda row: row
        self.fetch_count = 0
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        if isinstance(index, str):
            return [1] * self._n
        self.fetch_count += 1
        return {
            "input_ids": list(range(10)),
            "pixel_values": torch.zeros(4, 4),
        }


class FakePackingDataset(Dataset):
    """Mimics PackingDataset: holds the encoder under .dataset."""

    def __init__(self, inner):
        self.dataset = inner

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return [self.dataset[index]]


def test_row_cache_memoizes_rows():
    inner = FakeLazyDataset()
    cached = RowCacheDataset(inner, max_bytes=10 * 2**20)

    row0 = cached[2]
    row1 = cached[2]
    assert row0 is not row1
    assert row0["input_ids"] == row1["input_ids"]
    torch.testing.assert_close(row0["pixel_values"], row1["pixel_values"])
    assert inner.fetch_count == 1
    assert cached._hits == 1
    assert cached._misses == 1
    # Mutating a returned row must not leak into the cache.
    row1["pixel_values"].fill_(9)
    assert cached[2]["pixel_values"].abs().sum().item() == 0
    # Column access passes through.
    assert cached["lengths"] == [1] * 8
    assert len(cached) == 8


def test_row_cache_respects_budget():
    inner = FakeLazyDataset()
    # Budget fits exactly one row: 80 (ids) + 32 (tensor) + dict overhead 0.
    row_bytes = 8 * 10 + 16 * 4
    cached = RowCacheDataset(inner, max_bytes=row_bytes)

    cached[0]
    cached[1]  # over budget: served but not cached
    cached[1]
    assert inner.fetch_count == 3
    assert len(cached._cache) == 1


def test_wrap_lazy_rows_handles_both_layouts():
    inner = FakeLazyDataset()
    wrapped = wrap_lazy_rows(inner, max_bytes=2**20)
    assert isinstance(wrapped, RowCacheDataset)

    packed = FakePackingDataset(FakeLazyDataset())
    result = wrap_lazy_rows(packed, max_bytes=2**20)
    assert result is packed
    assert isinstance(packed.dataset, RowCacheDataset)


def test_wrap_lazy_rows_passes_through_eager_dataset():
    class Plain(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"input_ids": [index]}

    plain = Plain()
    assert wrap_lazy_rows(plain, max_bytes=2**20) is plain
