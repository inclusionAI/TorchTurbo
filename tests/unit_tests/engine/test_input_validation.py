"""Checks for the engine's packed SFT input contract."""

import pytest
import torch

from torchturbo.engine.utils import assert_packed_inputs, get_packed_boundaries


def _batch():
    return {
        "input_ids": torch.tensor([[11, 12, 21, 22, 23, 99]]),
        "labels": torch.tensor([[-100, 12, -100, 22, 23, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0]]),
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 3,
    }


def test_valid_prefix_with_masked_tail():
    assert_packed_inputs(_batch())


def test_rejects_position_resets_inside_a_sample():
    batch = _batch()
    batch["position_ids"] = torch.tensor([[0, 0, 0, 1, 0, 0]])

    with pytest.raises(ValueError, match="resets must match"):
        assert_packed_inputs(batch)


@pytest.mark.parametrize("multimodal", [False, True])
def test_requires_position_ids(multimodal):
    batch = _batch()
    del batch["position_ids"]
    if multimodal:
        batch["mm_token_type_ids"] = torch.zeros(1, 6, dtype=torch.long)
    with pytest.raises(ValueError, match="require position_ids"):
        assert_packed_inputs(batch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_mask", torch.ones(1, 6), "without attention_mask"),
        ("input_ids", torch.ones(2, 3), "shape"),
        ("position_ids", torch.zeros(1, 5), "align"),
        ("position_ids", torch.tensor([[0, 1, 2, 0, 1, 0]]), "resets"),
        ("labels", torch.zeros(2, 3), "labels"),
        ("cu_seq_lens_k", torch.tensor([0, 3, 5], dtype=torch.int32), "identical"),
        ("cu_seq_lens_q", torch.tensor([0, 2, 5]), "int32"),
        ("max_length_q", 2, "longest"),
        ("max_length_k", torch.tensor(3), "as an int"),
    ],
)
def test_rejects_inconsistent_contract(field, value, message):
    batch = _batch()
    batch[field] = value
    with pytest.raises(ValueError, match=message):
        assert_packed_inputs(batch)


@pytest.mark.parametrize("boundaries", [[1, 2, 5], [0, 2, 2], [0, 2, 7]])
def test_rejects_invalid_sample_boundaries(boundaries):
    batch = _batch()
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = torch.tensor(boundaries, dtype=torch.int32)
    with pytest.raises(ValueError, match="start at zero and increase"):
        assert_packed_inputs(batch)


def test_mrope_coordinates_remain_independent_of_text_boundaries():
    batch = _batch()
    batch["text_position_ids"] = batch["position_ids"]
    positions = torch.tensor([[[0, 0, 1, 1, 1, 0]]]).expand(3, 1, 6).clone()
    batch["position_ids"] = positions

    assert_packed_inputs(batch)

    assert batch["position_ids"] is positions


@pytest.mark.parametrize(
    ("attention_mask", "text_positions", "expected_cu"),
    [
        # Neat-packing sample IDs with right padding.
        ([[1, 1, 2, 2, 2, 0]], None, [0, 2, 5]),
        # Binary masks with left padding give one sample per row.
        ([[0, 0, 1, 1], [0, 1, 1, 1]], None, [0, 2, 5]),
        # Packing with a binary mask restarts text positions per sample.
        ([[1, 1, 1, 1, 1, 1]], [[0, 1, 0, 1, 2, 0]], [0, 2, 5, 6]),
        # Rows without valid tokens contribute no sample.
        ([[1, 1, 0], [0, 0, 0], [1, 0, 0]], None, [0, 2, 3]),
    ],
)
def test_packed_boundaries_follow_mask_changes_and_position_resets(
    attention_mask, text_positions, expected_cu,
):
    mask = torch.tensor(attention_mask)
    positions = None if text_positions is None else torch.tensor(text_positions)

    valid_indices, cu, max_length = get_packed_boundaries(mask, positions)

    assert valid_indices.tolist() == torch.nonzero(mask.flatten()).flatten().tolist()
    assert cu.dtype == torch.int32
    assert cu.tolist() == expected_cu
    assert max_length == int(cu.diff().max())


def test_packed_boundaries_require_a_valid_token():
    with pytest.raises(ValueError, match="at least one valid token"):
        get_packed_boundaries(torch.zeros(1, 3, dtype=torch.long))
