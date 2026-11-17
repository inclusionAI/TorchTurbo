"""Tests for strict tied-weight checkpoint omission handling."""

from types import SimpleNamespace

import pytest
import torch

from torchturbo.models.weight_converter import BaseWeightConverter


SOURCE = "model.language_model.embed_tokens.weight"
TARGET = "lm_head.weight"


def make_converter(tensor_infos, tied_weight_keys=None):
    converter = BaseWeightConverter.__new__(BaseWeightConverter)
    converter.checkpoint_info = SimpleNamespace(
        get_tensor_infos=lambda: tensor_infos
    )
    converter.tied_weight_keys = tied_weight_keys or {}
    converter.moe_config = None
    return converter


def test_tied_target_may_be_omitted_when_source_exists():
    converter = make_converter(
        {SOURCE: ((8, 4), torch.bfloat16)},
        {TARGET: SOURCE},
    )

    assert converter.get_param_load_entries(TARGET, 32) == ([], None, {})


def test_tied_target_missing_with_missing_source_is_an_error():
    converter = make_converter({}, {TARGET: SOURCE})

    with pytest.raises(KeyError, match=TARGET):
        converter.get_param_load_entries(TARGET, 32)


def test_untied_missing_parameter_is_an_error():
    converter = make_converter({})

    with pytest.raises(KeyError, match=TARGET):
        converter.get_param_load_entries(TARGET, 32)


def test_present_tied_target_uses_original_load_path():
    converter = make_converter(
        {
            SOURCE: ((8, 4), torch.bfloat16),
            TARGET: ((8, 4), torch.bfloat16),
        },
        {TARGET: SOURCE},
    )

    assert converter.get_param_load_entries(TARGET, 32) == (
        [TARGET],
        None,
        {TARGET: 64},
    )


def test_tied_source_follows_converter_hf_name_mapping():
    converter = make_converter(
        {"hf.source": ((8, 4), torch.bfloat16)},
        {TARGET: SOURCE},
    )
    names = {TARGET: "hf.target", SOURCE: "hf.source"}
    converter.to_hf_name = lambda name: names.get(name, name)

    assert converter.get_param_load_entries(TARGET, 32) == ([], None, {})
