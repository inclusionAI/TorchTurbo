"""Sequence parallel padding and slicing contract for model inputs."""

import pytest
import torch
from torch import nn

from torchturbo.models.modules.attention.ulysses import (
    SequencePadding,
    SequenceParallelInputSlicingSpec,
    patch_forward_for_sp_input_slicing,
    slice_input_tensor,
    ulysses_pad_inputs,
)


class _CaptureInputs(nn.Module):
    def forward(self, **kwargs):
        return kwargs


@pytest.mark.parametrize("sp_rank", range(4))
def test_padded_batch_is_sliced_once_at_the_boundary(monkeypatch, sp_rank):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    model = _CaptureInputs()
    spec = SequenceParallelInputSlicingSpec(
        model,
        _CaptureInputs.forward,
        sliced_inputs=(("input_ids", 1), ("coordinates", -1)),
        padded_inputs=(
            ("input_ids", SequencePadding.ZERO),
            ("coordinates", SequencePadding.POSITION),
            ("targets", SequencePadding.IGNORE_INDEX),
        ),
    )
    patch_forward_for_sp_input_slicing(spec, object(), 4, sp_rank)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "coordinates": torch.arange(5, dtype=torch.int32).expand(3, 1, -1),
        "targets": torch.tensor([[1, 2, 3, 4, 5]]),
        "pixel_values": torch.ones(1, 5, 2),
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_length_q": 3,
    }

    result = model(**ulysses_pad_inputs(inputs, spec, sp_size=4))

    shard = slice(2 * sp_rank, 2 * (sp_rank + 1))
    expected_coordinates = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int32)
    assert result["input_ids"].tolist() == [[1, 2, 3, 4, 5, 0, 0, 0][shard]]
    torch.testing.assert_close(
        result["coordinates"], expected_coordinates[shard].expand(3, 1, -1)
    )
    # Loss targets stay global; loss code slices them with the same shards.
    assert result["targets"].tolist() == [[1, 2, 3, 4, 5, -100, -100, -100]]
    assert result["pixel_values"] is inputs["pixel_values"]
    assert result["cu_seq_lens_q"] is inputs["cu_seq_lens_q"]
    assert result["max_length_q"] == 3
    assert inputs["input_ids"].shape == inputs["targets"].shape == (1, 5)
    assert inputs["coordinates"].shape == (3, 1, 5)


@pytest.mark.parametrize("seq_len, sp_size", [(5, 1), (4, 2)])
def test_aligned_batches_are_copied_without_padding(seq_len, sp_size):
    spec = SequenceParallelInputSlicingSpec(_CaptureInputs(), _CaptureInputs.forward)
    inputs = {"input_ids": torch.ones(1, seq_len, dtype=torch.long)}

    batch = ulysses_pad_inputs(inputs, spec, sp_size)

    assert batch is not inputs
    assert batch["input_ids"] is inputs["input_ids"]


def test_unaligned_inputs_are_rejected_instead_of_truncated(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    model = _CaptureInputs()
    patch_forward_for_sp_input_slicing(
        SequenceParallelInputSlicingSpec(model, _CaptureInputs.forward), object(), 2, 1
    )

    with pytest.raises(ValueError, match="not divisible by SP size 2"):
        model(inputs_embeds=torch.zeros(1, 5, 2))
    shard = slice_input_tensor(torch.arange(6).reshape(1, 6), dim=-1, group=None, rank=1)
    assert shard.tolist() == [[3, 4, 5]]
