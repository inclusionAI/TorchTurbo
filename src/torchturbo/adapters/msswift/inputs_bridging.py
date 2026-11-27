"""Normalize Swift batches on CPU before the trainer moves them to the device."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch

from ...engine.utils import assert_packed_inputs, get_packed_boundaries

_IGNORE_INDEX = -100


@dataclass
class SwiftInputCollator:
    """Keep Swift collation and normalize packed boundaries before device transfer."""

    collator: Callable[[list], dict[str, Any]]
    template: Any

    def __call__(self, batch: list) -> dict[str, Any]:
        is_dpo = getattr(self.template, "mode", None) == "rlhf"
        sequence_lengths = None
        if self.template.padding_free:
            # PackingDataset groups samples into lists; the native collator
            # subsequently flattens these groups and mutates the batch.
            samples = batch
            if isinstance(batch[0], list):
                samples = [sample for group in batch for sample in group]
            if is_dpo:
                sequence_lengths = [
                    len(sample[f"{prefix}input_ids"])
                    for prefix in ("chosen_", "rejected_")
                    for sample in samples
                ]
            else:
                sequence_lengths = [len(sample["input_ids"]) for sample in samples]
        padding_side = "left"
        if self.template.is_training:
            padding_side = self.template.padding_side
        inputs = self.collator(batch)
        if is_dpo:
            if sequence_lengths is None:
                # Swift pads the chosen rows, then the rejected rows; the
                # flattened segments keep that order.
                return bridge_unpacked_inputs(inputs)
            return bridge_packed_inputs(
                inputs,
                sequence_lengths=sequence_lengths,
                padding_side=padding_side,
            )
        return bridge_swift_inputs(
            inputs,
            sequence_lengths=sequence_lengths,
            padding_side=padding_side,
        )


def bridge_packed_inputs(
    inputs: dict[str, Any],
    *,
    sequence_lengths: list[int],
    padding_side: str = "right",
) -> dict[str, Any]:
    """Express native Swift packing with cu/max and preserve RoPE coordinates.

    Original sample lengths distinguish actual tokens from optional LongLoRA
    padding, whose zero position IDs otherwise resemble one-token samples.
    Physical padding is retained at the tail for the engine to mask.
    """
    input_ids = inputs["input_ids"]
    if input_ids.device.type != "cpu":
        raise ValueError("Packed Swift inputs must be normalized on CPU before device transfer.")
    num_tokens = input_ids.shape[-1]
    num_valid_tokens = sum(sequence_lengths)
    num_padding = num_tokens - num_valid_tokens
    if num_padding < 0 or not sequence_lengths or min(sequence_lengths) <= 0:
        raise ValueError("Packed Swift sample lengths must be positive and fit the collated input.")

    # The engine contract uses a valid prefix followed by optional tail padding.
    if num_padding and padding_side == "left":
        for key in (
            "input_ids",
            "labels",
            "loss_scale",
            "position_ids",
            "text_position_ids",
            "mm_token_type_ids",
        ):
            if key in inputs:
                inputs[key] = inputs[key].roll(-num_padding, dims=-1)

    expected_cu = torch.tensor([0, *accumulate(sequence_lengths)], dtype=torch.int32)
    for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
        cu = inputs.get(key)
        if cu is None:
            cu = expected_cu
        else:
            if cu.device.type != "cpu":
                raise ValueError("Packed Swift cumulative lengths must be normalized on CPU.")
            if num_padding:
                offset = num_padding if padding_side == "left" else 0
                cu = cu[(cu >= offset) & (cu <= offset + num_valid_tokens)] - offset
            if not torch.equal(cu, expected_cu):
                raise ValueError(f"{key} does not match the original Swift sample boundaries.")
        inputs[key] = cu

    max_length = max(sequence_lengths)
    for key in ("max_length_q", "max_length_k"):
        value = inputs.get(key, max_length)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Packed Swift maximum lengths must be normalized on CPU.")
            value = int(value)
        inputs[key] = value
    if inputs.get("labels") is not None:
        # Trainer counts supervision before the backend shifts labels. Sample
        # starts have no preceding prediction within their own sample.
        inputs["labels"][0, expected_cu[:-1]] = _IGNORE_INDEX
    inputs.pop("attention_mask", None)
    assert_packed_inputs(inputs)
    return inputs


def bridge_unpacked_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Flatten a padded Swift batch into one row with cu/max metadata.

    Swift without ``--packing`` collates padded [batch, seq] rows with a
    0/1 attention mask; every row is a separate sequence once flattened.
    """
    input_ids = inputs["input_ids"]
    if input_ids.device.type != "cpu":
        raise ValueError("Swift inputs must be normalized on CPU before device transfer.")
    attention_mask = inputs.pop("attention_mask")
    batch_size, seq_len = input_ids.shape
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError("Swift attention_mask must match input_ids.")

    valid_indices, cu_seqlens, max_length = get_packed_boundaries(attention_mask)
    text_positions = torch.cat([
        torch.arange(length) for length in cu_seqlens.diff().tolist()
    ]).unsqueeze(0)

    if "position_ids" in inputs:
        position_ids = inputs["position_ids"]
        if position_ids.dim() == 3:
            inputs["position_ids"] = (
                position_ids.flatten(1).index_select(1, valid_indices).unsqueeze(1)
            )
        else:
            inputs["position_ids"] = (
                position_ids.flatten().index_select(0, valid_indices).unsqueeze(0)
            )
    elif inputs.get("mm_token_type_ids") is None:
        # Text positions would be wrong for media tokens, so multimodal
        # batches must bring the template's RoPE coordinates.
        inputs["position_ids"] = text_positions

    if inputs.get("text_position_ids") is not None:
        # Swift's mRoPE collator numbers the padded row from zero; its text
        # coordinates must instead restart at each row's first valid token.
        inputs["text_position_ids"] = text_positions

    for key in (
        "input_ids",
        "labels",
        "loss_scale",
        "mm_token_type_ids",
    ):
        value = inputs.get(key)
        if value is not None:
            if value.shape[:2] != (batch_size, seq_len):
                raise ValueError(f"Swift {key} must align with the collated rows.")
            inputs[key] = value.flatten(0, 1).index_select(0, valid_indices).unsqueeze(0)

    if inputs.get("labels") is not None:
        # Trainer counts supervision before the backend shifts labels. A
        # row's first token has no preceding prediction within its row.
        inputs["labels"][0, cu_seqlens[:-1]] = _IGNORE_INDEX

    inputs["cu_seq_lens_q"] = cu_seqlens
    inputs["cu_seq_lens_k"] = cu_seqlens
    inputs["max_length_q"] = max_length
    inputs["max_length_k"] = max_length
    assert_packed_inputs(inputs)
    return inputs


def bridge_swift_inputs(
    inputs: dict[str, Any],
    *,
    sequence_lengths: list[int] | None = None,
    padding_side: str = "right",
) -> dict[str, Any]:
    """Normalize Swift batches while preserving per-token loss weights."""
    if sequence_lengths is not None:
        inputs = bridge_packed_inputs(
            inputs,
            sequence_lengths=sequence_lengths,
            padding_side=padding_side,
        )
    else:
        inputs = bridge_unpacked_inputs(inputs)
    return inputs
