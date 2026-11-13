"""FSDP2-owned Ulysses input alignment tests."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import ActivationCheckpointingConfig, EngineConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.base import TurboModel, TurboVLMModel
from torchturbo.models.modules.attention.ulysses import (
    SequenceParallelInputSlicingSpec,
    patch_forward_for_sp_input_slicing,
)
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec


def _backend(sp_size: int) -> FSDP2Backend:
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {
        "sp_size": sp_size,
        "sp_group": None,
        "sp_rank": 0,
    }
    return backend


def _model():
    return _PostEmbeddingModel()


def test_prepare_inputs_aligns_all_sequence_fields_and_packed_metadata():
    seq_len = 5
    position_ids = torch.zeros(4, 1, seq_len, dtype=torch.long)
    position_ids[0, 0] = torch.tensor([0, 1, 0, 1, 2])
    batch = {
        "input_ids": torch.arange(1, seq_len + 1).reshape(1, -1),
        "labels": torch.arange(10, 10 + seq_len).reshape(1, -1),
        "position_ids": position_ids,
        "text_position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
        "mm_token_type_ids": torch.ones(1, seq_len, dtype=torch.long),
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 3,
        "pixel_values": torch.ones(2, 3, 4, 4),
    }

    result = _backend(sp_size=3).prepare_inputs(_model(), batch)

    assert result["input_ids"].shape == (1, 6)
    assert result["labels"].shape == (1, 6)
    assert result["position_ids"].shape == (4, 1, 6)
    assert result["text_position_ids"].shape == (1, 6)
    assert result["mm_token_type_ids"].shape == (1, 6)
    assert result["input_ids"][0, -1].item() == 0
    assert result["labels"][0, -1].item() == -100
    assert result["position_ids"][..., -1].eq(0).all()
    assert result["text_position_ids"][0, -1].item() == 0
    assert result["mm_token_type_ids"][0, -1].item() == 0
    # Preserve the valid prefix for attention and MoE padding handling.
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["cu_seq_lens_k"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == 3
    assert result["max_length_k"] == 3
    assert result["cu_seq_lens_q"] is batch["cu_seq_lens_q"]
    assert result["cu_seq_lens_k"] is batch["cu_seq_lens_k"]
    assert result["pixel_values"] is batch["pixel_values"]
    assert batch["input_ids"].shape == (1, seq_len)
    assert batch["position_ids"].shape == (4, 1, seq_len)
    assert batch["text_position_ids"].shape == (1, seq_len)
    assert batch["mm_token_type_ids"].shape == (1, seq_len)
    assert batch["labels"].shape == (1, seq_len)


def test_prepare_inputs_does_not_fabricate_attention_mask():
    result = _backend(sp_size=4).prepare_inputs(
        _model(),
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "cu_seq_lens_q": torch.tensor([0, 3], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 3], dtype=torch.int32),
            "max_length_q": 3,
            "max_length_k": 3,
        },
    )

    # Boundaries ride with cu; no padding mask is ever fabricated.
    assert "attention_mask" not in result
    assert result["input_ids"].shape == (1, 4)
    assert result["cu_seq_lens_q"].tolist() == [0, 3]


def test_prepare_inputs_preserves_optional_fields_without_mutating_batch():
    backend = _backend(sp_size=2)
    model = _model()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
    }

    result = backend.prepare_inputs(model, batch)

    assert result["input_ids"].tolist() == [[1, 2, 3, 4, 5, 0]]
    assert "attention_mask" not in result
    assert "labels" not in result
    assert "position_ids" not in result
    assert backend.prepare_inputs(model, result)["input_ids"] is result["input_ids"]
    assert batch["input_ids"].shape == (1, 5)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))
        self.calls = []

    def forward(self, inputs_embeds, position_ids=None, **kwargs):
        self.calls.append((inputs_embeds.shape[1], position_ids, kwargs))
        return (inputs_embeds * self.scale).sin()


class _PreEmbeddingModel(TurboModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.embedding = nn.Embedding(16, 4)
        self.language_model = _LanguageModel()
        self.embedding_lengths = []

    def forward(self, input_ids, position_ids=None, **kwargs):
        self.embedding_lengths.append(input_ids.shape[1])
        return self.language_model(
            inputs_embeds=self.embedding(input_ids),
            position_ids=position_ids,
            **kwargs,
        )


class _PostEmbeddingModel(TurboVLMModel, _PreEmbeddingModel):
    def get_sequence_parallel_input_slicing_spec(self):
        return SequenceParallelInputSlicingSpec(
            self.language_model,
            _LanguageModel.forward,
        )


@pytest.mark.parametrize("model_cls", [_PreEmbeddingModel, _PostEmbeddingModel])
@pytest.mark.parametrize("sp_size, sp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_backend_installs_slicing_with_checkpoint_recomputation(
    monkeypatch, model_cls, sp_size, sp_rank, checkpointed
):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: sp_size)
    model = model_cls()
    reference = copy.deepcopy(model)
    backend = _backend(sp_size)
    backend.get_sp_info = lambda: {
        "sp_size": sp_size, "sp_group": None, "sp_rank": sp_rank
    }
    backend.config = EngineConfig(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=checkpointed)
    )
    backend.prefetch_manager = SimpleNamespace(
        setup=lambda model: None,
        suspend_expert_forward_prefetch=nullcontext,
    )
    backend.get_ep_info = lambda: {"ep_size": 1}
    backend._init_mesh = lambda plan: None
    backend._replace_modules = lambda model, plan: model
    backend._shard_model = lambda model, plan: model
    target = "" if model_cls is _PreEmbeddingModel else "language_model"
    plan = FSDPParallelPlan(
        plan={target: FSDPPlanSpec(activation_checkpoint=checkpointed)},
        sp_size=sp_size,
    )
    backend.prepare_model(model, plan)

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    batch = backend.prepare_inputs(model, {
        "input_ids": ids,
        "position_ids": torch.arange(20).reshape(4, 1, 5),
        "cu_seq_lens_q": cu,
    })
    global_length = batch["input_ids"].shape[1]
    local_length = global_length // sp_size
    shard = slice(sp_rank * local_length, (sp_rank + 1) * local_length)
    local_positions = batch["position_ids"][..., shard]

    expected = reference(
        input_ids=batch["input_ids"][:, shard],
        position_ids=local_positions,
        cu_seq_lens_q=cu,
    )
    expected.sum().backward()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model.language_model.calls.clear()
        actual = model(**batch)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, ref_param.grad)
        assert len(model.language_model.calls) == (2 if checkpointed else 1)
        for length, positions, kwargs in model.language_model.calls:
            assert length == local_length
            torch.testing.assert_close(positions, local_positions)
            assert kwargs["cu_seq_lens_q"] is cu

    embedding_length = local_length if model_cls is _PreEmbeddingModel else global_length
    assert set(model.embedding_lengths) == {embedding_length}
    assert batch["input_ids"].shape[1] == global_length


@pytest.mark.parametrize("model_cls, media_keys", [
    (_PreEmbeddingModel, ()),
    (_PostEmbeddingModel, ()),
    (_PostEmbeddingModel, ("pixel_values", "image_grid_thw")),
    (_PostEmbeddingModel, ("pixel_values_videos", "video_grid_thw")),
])
@pytest.mark.parametrize("num_tokens, sp_size, sp_rank", [
    (5, 1, 0), (5, 2, 0), (5, 2, 1), (6, 2, 1), (2, 4, 3),
])
def test_rl_packed_inputs_are_sliced_once_by_the_model(
    monkeypatch, model_cls, num_tokens, sp_size, sp_rank, media_keys
):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: sp_size)
    sp_info = {"sp_size": sp_size, "sp_group": object(), "sp_rank": sp_rank}
    model = model_cls()
    patch_forward_for_sp_input_slicing(
        model.get_sequence_parallel_input_slicing_spec(), **sp_info
    )
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.model = model
    engine.backend = _backend(sp_size)
    engine.get_sp_info = lambda: sp_info
    split = min(2, num_tokens - 1)
    cu = torch.tensor([0, split, num_tokens], dtype=torch.int32)
    ids = list(range(1, num_tokens + 1))
    media = {key: torch.ones(3, 4) for key in media_keys}
    batch = engine.prepare_packed_micro_batch(
        {
            "input_ids": torch.tensor(ids),
            "cu_seqlens": cu,
            "ppo_loss_mask": torch.ones(num_tokens),
            **media,
        },
        [split, num_tokens - split],
    )

    pad_size = (-num_tokens) % sp_size
    local_length = (num_tokens + pad_size) // sp_size
    shard = slice(sp_rank * local_length, (sp_rank + 1) * local_length)
    expected_positions = (
        list(range(split)) + list(range(num_tokens - split)) + list(range(pad_size))
    )
    assert batch["input_ids"].tolist() == [ids + [0] * pad_size]
    assert batch["_sp_pad_size"] == pad_size
    assert "attention_mask" not in batch
    assert "cu_seqlens" not in batch
    assert "labels" not in batch
    assert "ppo_loss_mask" not in batch
    if sp_size > 1:
        expected_rolled_ids = ids[1:] + ids[:1] + [0] * pad_size
        assert batch["_input_ids_rolled"].tolist() == expected_rolled_ids[shard]
    else:
        assert "_input_ids_rolled" not in batch
    assert model(**batch).shape == (1, local_length, 4)
    _, positions, kwargs = model.language_model.calls[-1]
    assert positions.tolist() == [expected_positions[shard]]
    assert kwargs["cu_seq_lens_q"] is cu
    for key, value in media.items():
        assert kwargs[key] is value
    runtime_inputs = model.get_moe_runtime_inputs(batch, {"sp": sp_info})
    expected_padding = [False] * num_tokens + [True] * pad_size
    assert runtime_inputs.padding_mask.tolist() == [expected_padding[shard]]
@pytest.mark.parametrize("sp_rank", [0, 1])
def test_diffusion_model_inputs_and_loss_labels_share_one_slice(monkeypatch, sp_rank):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed.nn, "all_reduce", lambda value, **kwargs: value)

    class DiffusionModel(_PreEmbeddingModel):
        def forward(self, **kwargs):
            return SimpleNamespace(logits=super().forward(**kwargs))

    model = DiffusionModel()
    backend = _backend(2)
    sp_info = {"sp_size": 2, "sp_group": object(), "sp_rank": sp_rank}
    backend.get_sp_info = lambda: sp_info
    backend.config = EngineConfig()
    patch_forward_for_sp_input_slicing(
        model.get_sequence_parallel_input_slicing_spec(), **sp_info
    )
    block_mask = torch.zeros(1, 1, 6, 6)
    labels = torch.tensor([[1, 2, 3, -100, -100, -100]])
    loss, outputs = backend.compute_diffusion_loss(
        model,
        {
            "input_ids": torch.tensor([[1, 2, 3, 1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
            "attention_mask": block_mask,
            "labels": labels,
        },
        return_outputs=True,
    )

    assert outputs.logits.shape == (1, 3, 4)
    assert model.embedding_lengths == [3]
    assert model.language_model.calls[-1][2]["attention_mask"] is block_mask
    local_labels = labels[:, sp_rank * 3:(sp_rank + 1) * 3]
    expected = nn.functional.cross_entropy(
        outputs.logits.flatten(0, 1), local_labels.flatten(), reduction="sum"
    ) / 3
    torch.testing.assert_close(loss, expected)
