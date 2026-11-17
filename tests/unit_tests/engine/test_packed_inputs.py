"""Packed SFT inputs flow to the backend and model with cu/max metadata."""

from collections import UserDict
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.base import TurboModel
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids


class _NoScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("num_tokens", [5, 7])
def test_cu_mask_uses_true_lengths_without_scalar_reads(num_tokens):
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    with _NoScalarReads():
        mask = cu_seqlens_to_segment_ids(cu, num_tokens)
    assert mask.tolist() == [[1, 1, 2, 2, 2] + [0] * (num_tokens - 5)]


@pytest.mark.parametrize(("physical_length", "sp_size"), [(5, 1), (5, 2), (7, 4)])
@pytest.mark.parametrize("batch_type", [dict, UserDict])
def test_compute_loss_passes_cu_through_backend_and_model(
    physical_length, sp_size, batch_type
):
    calls = []
    padded_length = physical_length + (-physical_length) % sp_size

    class Model(TurboModel, torch.nn.Module):
        def forward(self, input_ids, position_ids, labels, cu_seq_lens_q, cu_seq_lens_k,
                    max_length_q, max_length_k):
            calls.append("model")
            # The model sees the packed boundaries directly; the SP padded
            # tail stays outside every cu segment.
            assert cu_seq_lens_q.tolist() == [0, 2, 5]
            assert input_ids.shape == (1, padded_length)
            return torch.tensor(0.0)

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {"sp_size": sp_size}
    original_prepare = backend.prepare_inputs

    def prepare(model, inputs):
        calls.append("prepare")
        assert inputs["cu_seq_lens_q"].tolist() == [0, 2, 5]
        return original_prepare(model, inputs)

    backend.prepare_inputs = prepare
    backend.compute_loss = lambda model, inputs, *args, **kwargs: model(**inputs)
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = backend

    def prepare_moe(model, inputs, gradient_accumulation_steps):
        calls.append("moe")
        assert inputs["cu_seq_lens_q"].tolist() == [0, 2, 5]

    engine._prepare_moe_runtime = prepare_moe
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    batch = batch_type({
        "input_ids": torch.arange(physical_length).unsqueeze(0),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2] + [0] * (physical_length - 5)]),
        "labels": torch.full((1, physical_length), -100),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": 3,
        "max_length_k": 3,
    })

    engine.compute_loss(Model().eval(), batch)

    assert calls == ["prepare", "moe", "model"]
    assert "attention_mask" not in batch
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 5]


def test_nonpacked_mask_is_rejected():
    class ContractModel(TurboModel):
        pass

    batch = {
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
    }
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = SimpleNamespace(
        prepare_inputs=lambda model, inputs: inputs,
        compute_loss=lambda model, inputs, *args, **kwargs: inputs,
    )
    engine._prepare_moe_runtime = lambda *args: None

    with pytest.raises(ValueError, match="not attention_mask"):
        engine.compute_loss(ContractModel(), batch)


def test_text_batch_without_positions_gets_contiguous_default():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    backend.config = SimpleNamespace(
        loss_chunk=SimpleNamespace(enabled=False),
        mtp=SimpleNamespace(loss_scaling_factor=0.0, log_loss=False, enabled=False),
    )
    captured = {}
    class _Outputs(SimpleNamespace):
        pass

    backend._forward_for_lm_loss = (
        lambda model, inputs, use_chunked_loss: captured.update(inputs) or (_Outputs(), None)
    )
    backend._compute_lm_loss_with_sp = (
        lambda **kw: (torch.tensor(0.0), (torch.tensor(0.0),))
    )
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=0)
            self.training = False

    inputs = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.full((1, 4), -100),
        "position_ids": None,
        "cu_seq_lens_q": torch.tensor([0, 4], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 4], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
    }
    backend.compute_loss(Model(), inputs)

    assert captured["position_ids"].tolist() == [[0, 1, 2, 3]]


def test_multimodal_batch_without_positions_keeps_none_for_mrope():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    backend.config = SimpleNamespace(
        loss_chunk=SimpleNamespace(enabled=False),
        mtp=SimpleNamespace(loss_scaling_factor=0.0, log_loss=False, enabled=False),
    )
    captured = {}
    class _Outputs(SimpleNamespace):
        pass

    backend._forward_for_lm_loss = (
        lambda model, inputs, use_chunked_loss: captured.update(inputs) or (_Outputs(), None)
    )
    backend._compute_lm_loss_with_sp = (
        lambda **kw: (torch.tensor(0.0), (torch.tensor(0.0),))
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=0)
            self.training = False

    inputs = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.full((1, 4), -100),
        "position_ids": None,
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        "cu_seq_lens_q": torch.tensor([0, 4], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 4], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
    }
    backend.compute_loss(Model(), inputs)

    # The model must see None so it derives mrope grid coordinates itself;
    # a linear default would silently corrupt image RoPE.
    assert captured.get("position_ids") is None


def test_sp_input_slicing_skips_absent_position_ids(monkeypatch):
    """VLM batches without positions must not crash the sequence-parallel slice."""
    import torch.distributed as dist

    from torchturbo.models.modules.attention.ulysses import (
        SequenceParallelInputSlicingSpec,
        patch_forward_for_sp_input_slicing,
    )

    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)

    class FakeLanguageModel(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    model = FakeLanguageModel()
    unpatched_model = FakeLanguageModel()
    patch_forward_for_sp_input_slicing(
        SequenceParallelInputSlicingSpec(model, FakeLanguageModel.forward),
        sp_group=object(),
        sp_size=2,
        sp_rank=0,
    )
    embeds = torch.zeros(1, 4, 2)
    result = model.forward(
        inputs_embeds=embeds,
        position_ids=None,
    )

    assert result["inputs_embeds"].shape == (1, 2, 2)
    assert result["position_ids"] is None
    assert unpatched_model.forward(inputs_embeds=embeds)["inputs_embeds"].shape == (
        1,
        4,
        2,
    )


@pytest.mark.parametrize("num_patches", [1, 2])
def test_sp_input_slicing_uses_model_input_contract(monkeypatch, num_patches):
    """Models can declare additional tensor and tensor-tuple inputs."""
    import torch.distributed as dist

    from torchturbo.models.modules.attention.ulysses import (
        SequenceParallelInputSlicingSpec,
        patch_forward_for_sp_input_slicing,
    )

    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)

    class FakeLanguageModel(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    model = FakeLanguageModel()
    sp_group = object()
    spec = SequenceParallelInputSlicingSpec(
        model,
        FakeLanguageModel.forward,
        sliced_inputs=(
            ("inputs_embeds", 1),
            ("position_ids", -1),
            ("text_position_ids", -1),
            ("mtp_shifted_input_ids", 1),
        ),
    )
    for _ in range(num_patches):
        patch_forward_for_sp_input_slicing(
            spec,
            sp_group=sp_group,
            sp_size=2,
            sp_rank=1,
        )
    result = model.forward(
        inputs_embeds=torch.arange(8).reshape(1, 4, 2),
        position_ids=torch.arange(12).reshape(3, 1, 4),
        text_position_ids=torch.arange(4).reshape(1, 4),
        mtp_shifted_input_ids=(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[2, 3, 4, 0]]),
        ),
    )

    assert result["inputs_embeds"].shape == (1, 2, 2)
    assert result["position_ids"].shape == (3, 1, 2)
    assert result["text_position_ids"].tolist() == [[2, 3]]
    assert result["mtp_shifted_input_ids"][0].tolist() == [[3, 4]]
    assert result["mtp_shifted_input_ids"][1].tolist() == [[4, 0]]
