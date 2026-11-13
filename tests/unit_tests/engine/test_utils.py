"""Tests for dependency-light engine utilities."""

import pytest
import torch
import torch.nn as nn

from torchturbo.engine.utils import (
    has_meta_tensor,
    is_dora_magnitude_param,
    is_peft_adapter_param,
    strip_peft_prefix,
    validate_signature,
)


def test_validate_signature_requires_explicit_keyword_with_exact_default():
    def supported(*, flag=False):
        pass

    def kwargs_only(**kwargs):
        pass

    def wrong_default(*, flag=0):
        pass

    def positional_only(flag=False, /):
        pass

    validate_signature(supported, flag=False)
    for function in (kwargs_only, wrong_default, positional_only):
        with pytest.raises(TypeError, match="flag"):
            validate_signature(function, flag=False)


def test_strip_peft_prefix_normalizes_wrapped_names():
    name = "base_model.model.model.layers.0.mlp.experts.base_layer.weight1"

    assert strip_peft_prefix(name) == "model.layers.0.mlp.experts.weight1"


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.q_proj.lora_A.default.weight",
        "model.layers.0.q_proj.lora_B.default.weight",
        "model.layers.0.q_proj.lora_magnitude_vector.default.weight",
    ],
)
def test_is_peft_adapter_param_recognizes_lora_and_dora_state(name):
    assert is_peft_adapter_param(name)


def test_is_dora_magnitude_param_is_specific_to_dora_state():
    assert is_dora_magnitude_param(
        "model.layers.0.q_proj.lora_magnitude_vector.default.weight"
    )
    assert not is_dora_magnitude_param(
        "model.layers.0.q_proj.lora_A.default.weight"
    )


def test_has_meta_tensor_returns_parameters_and_buffers():
    model = nn.Module()
    model.register_parameter(
        "weight", nn.Parameter(torch.empty(1, device="meta"))
    )
    model.register_buffer("persistent_meta", torch.empty(1, device="meta"))
    model.register_buffer("persistent_cpu", torch.ones(1))

    child = nn.Module()
    child.register_buffer(
        "runtime", torch.empty(1, device="meta"), persistent=False
    )
    model.add_module("child", child)

    assert has_meta_tensor(model) == [
        "weight",
        "persistent_meta",
        "child.runtime",
    ]


def test_has_meta_tensor_ignores_none_buffers():
    model = nn.Module()
    model.register_buffer("runtime", None, persistent=False)

    assert has_meta_tensor(model) == []
