"""DPO sampling preserves the final preferences and keeps SP ranks aligned."""

import copy
import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch

from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import make_turbo_dpo_trainer
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.core import TorchTurboEngine


@pytest.mark.parametrize("drop_last", [False, True])
def test_dpo_sp_sampler_preserves_preferences_and_collator(monkeypatch, drop_last):
    dataset = [
        {"id": index, "chosen": [index, 1], "rejected": [index, 2]}
        for index in range(453)
    ]
    original_dataset = copy.deepcopy(dataset)

    def collate_pairs(rows):
        return tuple((row["id"], row["chosen"], row["rejected"]) for row in rows)

    engine = object.__new__(TorchTurboEngine)
    engine.config = EngineConfig(gbs=8)
    rank = 0
    engine.backend = SimpleNamespace(
        get_dp_info=lambda: {"dp_size": 4, "dp_rank": rank // 8},
        get_sp_info=lambda: {"sp_size": 8, "sp_rank": rank % 8},
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 32)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)

    trainer = object.__new__(make_turbo_dpo_trainer())
    trainer.train_dataset = dataset
    trainer.data_collator = collate_pairs
    trainer._train_batch_size = 2
    trainer.args = SimpleNamespace(
        train_dataloader_shuffle=False,
        data_seed=42,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=drop_last,
    )
    rank_ids = []
    for rank in range(32):
        dataloader = trainer.get_train_dataloader()
        assert trainer.data_collator is dataloader.collate_fn is collate_pairs
        assert dataloader.drop_last is dataloader.sampler.drop_last is drop_last
        assert dataloader.sampler.rank_block == 4
        batches = list(dataloader)
        assert len(batches) == (56 if drop_last else 58)
        assert all(len(batch) == 2 for batch in batches)
        rank_ids.append([pair[0] for batch in batches for pair in batch])

    for dp_rank in range(4):
        first_rank = dp_rank * 8
        assert all(
            rank_ids[rank] == rank_ids[first_rank]
            for rank in range(first_rank, first_rank + 8)
        )

    # Each update consumes two micro-batches per DP replica. Padding repeats
    # the epoch's first 11 pairs so all 453 pairs participate in 29 updates.
    epoch_ids = [
        pair_id
        for offset in range(0, len(rank_ids[0]), 4)
        for dp_rank in range(4)
        for pair_id in rank_ids[dp_rank * 8][offset:offset + 4]
    ]
    expected_ids = (
        list(range(448)) if drop_last else list(range(453)) + list(range(11))
    )
    assert epoch_ids == expected_ids
    assert dataset == original_dataset


@pytest.mark.parametrize("sp_size", [None, 1])
def test_dpo_without_sp_keeps_native_dataloader_and_collator(monkeypatch, sp_size):
    from swift.rlhf_trainers import DPOTrainer

    engine = None
    if sp_size is not None:
        engine = SimpleNamespace(get_sp_info=lambda: {"sp_size": sp_size})
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    native_dataloader = object()
    calls = []

    def get_native_dataloader(self, skip_batches=0):
        calls.append((self, skip_batches, self.data_collator))
        return native_dataloader

    monkeypatch.setattr(DPOTrainer, "get_train_dataloader", get_native_dataloader)
    trainer = object.__new__(make_turbo_dpo_trainer())
    collator = object()
    trainer.data_collator = collator

    assert trainer.get_train_dataloader(skip_batches=3) is native_dataloader
    assert calls == [(trainer, 3, collator)]
    assert trainer.data_collator is collator
