"""CPU tests for Swift collation and the packed cu/max input contract."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch  # noqa: E402

from torchturbo.adapters.msswift.inputs_bridging import (  # noqa: E402
    SwiftInputCollator,
    bake_loss_scale_into_labels,
    bridge_packed_inputs,
    bridge_swift_inputs,
)


def _t(values):
    return torch.tensor(values)


def _packed_inputs(lengths):
    positions = [position for length in lengths for position in range(length)]
    return {
        "input_ids": torch.arange(len(positions)).unsqueeze(0),
        "position_ids": _t([positions]),
    }


@pytest.mark.parametrize("lengths,expected", [([3, 2], [0, 3, 5]), ([1], [0, 1]), ([2, 2, 3], [0, 2, 4, 7])])
def test_sample_lengths_produce_cu_max(lengths, expected):
    inputs = _packed_inputs(lengths)
    positions = inputs["position_ids"]
    out = bridge_packed_inputs(inputs, sequence_lengths=lengths)
    assert "attention_mask" not in out
    assert out["cu_seq_lens_q"].tolist() == expected
    assert out["cu_seq_lens_k"].tolist() == expected
    assert out["cu_seq_lens_q"].dtype == torch.int32
    assert out["max_length_q"] == out["max_length_k"] == max(lengths)
    assert out["position_ids"] is positions


def test_preserves_explicit_cu_and_mrope_coordinates():
    cu = _t([0, 3, 5]).int()
    positions = _t([[[0, 1, 1, 0, 1]], [[0, 1, 2, 0, 1]], [[0, 1, 1, 0, 1]]])
    inputs = {
        "input_ids": _t([[10, 11, 12, 13, 14]]),
        "position_ids": positions,
        "text_position_ids": _t([[0, 1, 2, 0, 1]]),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": _t(3),
        "max_length_k": _t(3),
    }
    out = bridge_packed_inputs(inputs, sequence_lengths=[3, 2])
    assert out["cu_seq_lens_q"] is cu
    assert out["cu_seq_lens_k"] is cu
    assert out["position_ids"] is positions
    assert type(out["max_length_q"]) is int
    assert "attention_mask" not in out


def test_longlora_tail_padding_is_not_counted_as_samples():
    inputs = {
        "input_ids": _t([[10, 11, 12, 13, 14, 0, 0]]),
        "position_ids": _t([[0, 1, 0, 1, 2, 0, 0]]),
        "cu_seq_lens_q": _t([0, 2, 5, 6, 7]).int(),
        "cu_seq_lens_k": _t([0, 2, 5, 6, 7]).int(),
        "max_length_q": _t(3),
        "max_length_k": _t(3),
    }
    out = bridge_packed_inputs(inputs, sequence_lengths=[2, 3])
    assert out["input_ids"].shape == (1, 7)
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert "attention_mask" not in out


def test_packed_left_padding_moves_to_tail_with_all_token_fields():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11, 12, 13, 14]]),
        "labels": _t([[-100, -100, 10, 11, 12, 13, 14]]),
        "position_ids": _t([[[0, 0, 0, 1, 0, 1, 2]]] * 3),
        "text_position_ids": _t([[0, 0, 0, 1, 0, 1, 2]]),
        "loss_scale": _t([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        "mm_token_type_ids": _t([[0, 0, 0, 1, 0, 0, 0]]),
        "cu_seq_lens_q": _t([0, 1, 2, 4, 7]).int(),
        "cu_seq_lens_k": _t([0, 1, 2, 4, 7]).int(),
        "max_length_q": _t(3),
        "max_length_k": _t(3),
        "pixel_values": torch.randn(4, 6),
    }
    pixels = inputs["pixel_values"]
    out = bridge_swift_inputs(inputs, sequence_lengths=[2, 3], padding_side="left")
    assert out["input_ids"].tolist() == [[10, 11, 12, 13, 14, 0, 0]]
    # Sample starts are unsupervised; loss_scale then drops token 11.
    assert out["labels"].tolist() == [[-100, -100, -100, 13, 14, -100, -100]]
    assert out["text_position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0]]
    assert out["position_ids"][0].tolist() == out["text_position_ids"].tolist()
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 0, 0, 0, 0, 0]]
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert out["pixel_values"] is pixels


def test_nonpacked_left_padding_flattens_to_cu_rows():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11], [0, 12, 13, 14]]),
        "position_ids": _t([[0, 0, 0, 1], [0, 0, 1, 2]]),
        "attention_mask": _t([[0, 0, 1, 1], [0, 1, 1, 1]]),
    }
    out = bridge_swift_inputs(inputs)
    # Left padding dropped; every batch row is one cu segment.
    assert out["input_ids"].tolist() == [[10, 11, 12, 13, 14]]
    assert out["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert out["cu_seq_lens_k"].tolist() == [0, 2, 5]
    assert out["max_length_q"] == out["max_length_k"] == 3
    assert "attention_mask" not in out


def test_nonpacked_mrope_left_padding_resets_text_positions_only():
    positions = _t([
        [[0, 0, 0, 1], [0, 1, 1, 2]],
        [[0, 0, 0, 1], [0, 1, 2, 3]],
        [[0, 0, 0, 1], [0, 1, 1, 2]],
    ])
    inputs = {
        "input_ids": _t([[0, 0, 10, 11], [20, 21, 22, 23]]),
        "labels": _t([[-100, -100, 10, 11], [20, 21, 22, 23]]),
        "attention_mask": _t([[0, 0, 1, 1], [1, 1, 1, 1]]),
        "position_ids": positions,
        "text_position_ids": torch.arange(4).expand(2, -1),
        "mm_token_type_ids": _t([[0, 0, 0, 0], [0, 1, 1, 0]]),
    }

    out = bridge_swift_inputs(inputs, padding_side="left")

    assert out["text_position_ids"].tolist() == [[0, 1, 0, 1, 2, 3]]
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 6]
    assert out["labels"].tolist() == [[-100, 11, -100, 21, 22, 23]]
    expected_mrope = torch.cat([positions[:, :1, 2:], positions[:, 1:]], dim=-1)
    torch.testing.assert_close(out["position_ids"], expected_mrope)


def test_rejects_native_cu_with_different_sample_boundaries():
    inputs = _packed_inputs([2, 3])
    inputs["cu_seq_lens_q"] = _t([0, 3, 5]).int()
    with pytest.raises(ValueError, match="original Swift sample boundaries"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


def test_packed_sample_starts_are_unsupervised():
    inputs = _packed_inputs([2, 3])
    inputs["labels"] = torch.arange(10, 15).unsqueeze(0)
    out = bridge_packed_inputs(inputs, sequence_lengths=[2, 3])
    # The trainer counts labels before the backend shift, which never
    # targets a sample start.
    assert out["labels"].tolist() == [[-100, 11, -100, 13, 14]]


def test_rejects_position_resets_inconsistent_with_samples():
    inputs = _packed_inputs([3, 2])
    with pytest.raises(ValueError, match="position resets"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


def test_rejects_incorrect_native_maximum_length():
    inputs = _packed_inputs([2, 3])
    inputs["max_length_q"] = _t(5)
    with pytest.raises(ValueError, match="longest packed sample"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


@pytest.mark.parametrize("nested", [False, True])
def test_collator_captures_lengths_before_native_batch_mutation(nested):
    samples = [{"input_ids": [10, 11]}, {"input_ids": [12, 13, 14]}]
    batch = [[samples[0]], [samples[1]]] if nested else samples
    template = SimpleNamespace(padding_free=True, padding_side="right", is_training=True)

    def native_collator(rows):
        rows[:] = [{"input_ids": [10, 11, 12, 13, 14]}]
        return {
            "input_ids": _t([[10, 11, 12, 13, 14]]),
            "position_ids": _t([[0, 1, 0, 1, 2]]),
            "labels": _t([[10, 11, 12, 13, 14]]),
            "loss_scale": _t([[1.0, 0.0, 1.0, 1.0, 1.0]]),
        }

    out = SwiftInputCollator(native_collator, template)(batch)
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert out["labels"].tolist() == [[-100, -100, -100, 13, 14]]
    assert "attention_mask" not in out
    assert "loss_scale" not in out
    assert out["cu_seq_lens_q"].device.type == "cpu"


def test_dpo_collator_captures_chosen_then_rejected_lengths():
    samples = [
        {"chosen_input_ids": [10, 11], "rejected_input_ids": [20, 21, 22]},
        {"chosen_input_ids": [30], "rejected_input_ids": [40, 41]},
    ]
    template = SimpleNamespace(
        mode="rlhf", padding_free=True, padding_side="right", is_training=True
    )
    pixel_values = torch.ones(1, 3, 2, 2)

    def native_collator(rows):
        return {
            "input_ids": _t([[10, 11, 30, 20, 21, 22, 40, 41]]),
            "position_ids": _t([[0, 1, 0, 0, 1, 2, 0, 1]]),
            "labels": _t([[10, 11, 30, 20, 21, 22, 40, 41]]),
            "loss_scale": _t([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
            "pixel_values": pixel_values,
        }

    out = SwiftInputCollator(native_collator, template)(samples)
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 3, 6, 8]
    assert out["max_length_q"] == out["max_length_k"] == 3
    # Only sample starts are masked; DPO does not fold loss_scale into labels.
    assert out["labels"].tolist() == [[-100, 11, -100, -100, 21, 22, -100, 41]]
    assert "loss_scale" in out
    assert out["pixel_values"] is pixel_values


# (input_ids, labels, mm_token_type_ids) in Swift's DPO collation order: every
# chosen sample, then every rejected sample.
_DPO_ROWS = [
    ([1, 2, 3, 4], [-100, -100, 3, 4], [0, 1, 1, 0]),
    ([5, 6, 7, 8, 9], [-100, 6, 7, 8, 9], [0, 0, 0, 0, 0]),
    ([1, 2, 9], [-100, -100, 9], [0, 1, 0]),
    ([5, 10, 3], [-100, 10, 3], [0, 0, 0]),
]


def _sample_mrope(index, length):
    # Distinct coordinates per sample; get_rope_index restarts at each sample.
    positions = torch.arange(length)
    return torch.stack((positions, positions + 10 * (index + 1), positions + 20 * (index + 1)))


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_dpo_collator_flattens_padded_rows_like_padding_free_rows(padding_side, mrope):
    num_pairs = len(_DPO_ROWS) // 2
    samples = [
        {"chosen_input_ids": _DPO_ROWS[i][0], "rejected_input_ids": _DPO_ROWS[i + num_pairs][0]}
        for i in range(num_pairs)
    ]
    lengths = [len(ids) for ids, _, _ in _DPO_ROWS]
    width = max(lengths)

    def pad(row, value):
        fill = [value] * (width - len(row))
        return row + fill if padding_side == "right" else fill + row

    # Padding-free collation concatenates the samples into one row.
    text_positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
    padding_free_batch = {
        "input_ids": _t([[t for ids, _, _ in _DPO_ROWS for t in ids]]),
        "labels": _t([[t for _, labels, _ in _DPO_ROWS for t in labels]]),
        "loss_scale": torch.ones(1, sum(lengths)),
        "position_ids": text_positions,
    }
    # Without padding-free, Swift pads every sample to the longest one.
    padded_batch = {
        "input_ids": _t([pad(ids, 0) for ids, _, _ in _DPO_ROWS]),
        "labels": _t([pad(labels, -100) for _, labels, _ in _DPO_ROWS]),
        "attention_mask": _t([pad([1] * len(ids), 0) for ids, _, _ in _DPO_ROWS]),
        "loss_scale": _t([pad([1.0] * len(ids), 0.0) for ids, _, _ in _DPO_ROWS]),
    }
    if padding_side == "left":
        padded_batch["position_ids"] = _t([pad(list(range(n)), 0) for n in lengths])
    if mrope:
        coordinates = [_sample_mrope(i, length) for i, length in enumerate(lengths)]
        padding_free_batch["position_ids"] = torch.cat(coordinates, dim=-1).unsqueeze(1)
        padding_free_batch["text_position_ids"] = text_positions
        padding_free_batch["mm_token_type_ids"] = _t(
            [[t for _, _, mm in _DPO_ROWS for t in mm]]
        )
        # get_rope_index fills padding with 1, and Swift numbers the text
        # positions of each padded row from zero.
        padded_positions = torch.ones(3, len(lengths), width, dtype=torch.long)
        for row, (coordinate, length) in enumerate(zip(coordinates, lengths)):
            start = 0 if padding_side == "right" else width - length
            padded_positions[:, row, start:start + length] = coordinate
        padded_batch["position_ids"] = padded_positions
        padded_batch["text_position_ids"] = torch.arange(width).expand(len(lengths), -1)
        padded_batch["mm_token_type_ids"] = _t([pad(mm, 0) for _, _, mm in _DPO_ROWS])

    def collate(padding_free, native_batch):
        template = SimpleNamespace(
            mode="rlhf", padding_free=padding_free, padding_side=padding_side,
            is_training=True,
        )
        return SwiftInputCollator(lambda rows: native_batch, template)(samples)

    expected = collate(True, padding_free_batch)
    out = collate(False, padded_batch)

    # The model sees identical inputs, without an attention mask.
    assert out.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(out[key], value), key
        else:
            assert out[key] == value, key
    assert out["cu_seq_lens_q"].tolist() == [0, 4, 9, 12, 15]


def _native_chatml_template(padding_free, padding_side):
    from swift.template import get_template
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    words = [
        "<|im_start|>", "<|im_end|>", "user", "assistant", "\n", "[UNK]", "[PAD]",
        "hi", "question", "good", "bad", "answer", "very", "a", "b", "c",
    ]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|im_end|>",
    )
    # Swift templates read the model metadata its loader attaches to the processor.
    tokenizer.model_info = SimpleNamespace(config=None, task_type="causal_lm", max_model_len=512)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    template = get_template(
        tokenizer, template_type="chatml", padding_free=padding_free, padding_side=padding_side,
    )
    template.set_mode("rlhf")
    return template


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_native_dpo_collator_flattens_padded_rows_like_padding_free(padding_side):
    pytest.importorskip("swift", minversion="4.0")
    pairs = [
        {
            "messages": [
                {"role": "user", "content": "hi question"},
                {"role": "assistant", "content": "good answer a b"},
            ],
            "rejected_response": "bad answer",
        },
        {
            "messages": [
                {"role": "user", "content": "question c"},
                {"role": "assistant", "content": "very good"},
            ],
            "rejected_response": "very bad answer c a",
        },
    ]

    def collate(padding_free):
        template = _native_chatml_template(padding_free, padding_side)
        encoded = [template.encode(dict(pair)) for pair in pairs]
        chosen_first_tokens = [
            token
            for prefix in ("chosen_", "rejected_")
            for sample in encoded
            for token in sample[f"{prefix}input_ids"]
        ]
        return chosen_first_tokens, SwiftInputCollator(template.data_collator, template)(encoded)

    _, expected = collate(True)
    chosen_first_tokens, out = collate(False)

    assert out.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(out[key], value), key
        else:
            assert out[key] == value, key
    # The DPO forward reads the first half of the segments as chosen samples.
    assert out["input_ids"][0].tolist() == chosen_first_tokens


def test_nonpacked_collator_flattens_binary_mask_batches():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11]]),
        "position_ids": _t([[0, 0, 0, 1]]),
        "labels": _t([[-100, -100, 10, 11]]),
        "attention_mask": _t([[0, 0, 1, 1]]),
    }
    template = SimpleNamespace(padding_free=False, padding_side="left", is_training=True)
    out = SwiftInputCollator(lambda batch: inputs, template)([{"input_ids": [10, 11]}])
    assert out["input_ids"].tolist() == [[10, 11]]
    # Each row's first token is not predictable: shifted loss keeps -100.
    assert out["labels"].tolist() == [[-100, 11]]
    assert out["cu_seq_lens_q"].tolist() == [0, 2]
    assert "attention_mask" not in out


def test_loss_scale_zeros_masked_into_labels():
    labels = _t([[10, 11, 12, 13, 14]])
    loss_scale = _t([[1.0, 0.0, 1.0, 0.0, 1.0]])  # ignore tokens at idx 1,3
    inputs = {"labels": labels.clone(), "loss_scale": loss_scale}
    out = bake_loss_scale_into_labels(inputs)
    assert out["labels"].tolist() == [[10, -100, 12, -100, 14]]
    assert "loss_scale" not in out  # removed after folding


def test_loss_scale_all_positive_passthrough_and_drops_loss_scale():
    labels = _t([[10, 11, 12]])
    inputs = {"labels": labels.clone(), "loss_scale": _t([[1.0, 1.0, 1.0]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out["labels"].tolist() == [[10, 11, 12]]
    assert "loss_scale" not in out


def test_loss_scale_absent_noop():
    inputs = {"labels": _t([[1, 2, 3]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out is inputs
    assert out["labels"].tolist() == [[1, 2, 3]]


def test_loss_scale_no_labels_noop():
    inputs = {"loss_scale": _t([[1.0, 0.0]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out is inputs  # nothing to fold into


def test_loss_scale_non_binary_warns_but_still_masks_zeros(caplog):
    labels = _t([[10, 11, 12, 13]])
    loss_scale = _t([[1.0, 0.5, 0.0, 0.3]])  # values in (0,1) -> non-binary
    inputs = {"labels": labels.clone(), "loss_scale": loss_scale}
    out = bake_loss_scale_into_labels(inputs)
    # zeros still masked; non-binary fractional weights dropped (warned).
    assert out["labels"].tolist() == [[10, 11, -100, 13]]
