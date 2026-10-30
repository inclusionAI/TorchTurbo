"""A_log on_load/on_save round-trip for the Kimi-K3 weight converter.

Official K3 checkpoints store ``self_attn.A_log`` padded from num_heads (96
trained decay values) up to head_dim (128) with an all-zero tail. The load
hook narrows 128 -> 96; the save hook must invert it so exported checkpoints
stay byte-compatible with the official format.
"""

from types import SimpleNamespace

import pytest
import torch

from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.models.kimi_k3.model import KimiK3WeightConverter

NUM_HEADS = 96
HEAD_DIM = 128
A_LOG_NAME = "language_model.model.layers.0.self_attn.A_log"


def _k3_converter():
    converter = object.__new__(KimiK3WeightConverter)
    converter.config = SimpleNamespace(
        linear_attn_config={"num_heads": NUM_HEADS, "head_dim": HEAD_DIM}
    )
    return converter


def _official_a_log():
    tensor = torch.zeros(HEAD_DIM, dtype=torch.float32)
    tensor[:NUM_HEADS] = torch.rand(NUM_HEADS, dtype=torch.float32)
    return tensor


def test_load_narrows_zero_padded_a_log():
    official = _official_a_log()
    out = _k3_converter().hf_to_model_tensor(A_LOG_NAME, official.clone())
    assert out.shape == (NUM_HEADS,)
    assert torch.equal(out, official[:NUM_HEADS])


def test_load_refuses_nonzero_padding():
    official = _official_a_log()
    official[-1] = 1.0
    with pytest.raises(ValueError, match="non-zero padding"):
        _k3_converter().hf_to_model_tensor(A_LOG_NAME, official)


def test_save_pads_a_log_back_to_head_dim():
    trained = torch.rand(NUM_HEADS, dtype=torch.float32)
    out = _k3_converter().model_to_hf_tensor(A_LOG_NAME, trained)
    assert out.shape == (HEAD_DIM,)
    assert torch.equal(out[:NUM_HEADS], trained)
    assert torch.equal(out[NUM_HEADS:], torch.zeros(HEAD_DIM - NUM_HEADS))


def test_save_load_roundtrip_is_bitwise_identical():
    official = _official_a_log()
    converter = _k3_converter()
    narrowed = converter.hf_to_model_tensor(A_LOG_NAME, official.clone())
    restored = converter.model_to_hf_tensor(A_LOG_NAME, narrowed)
    assert torch.equal(restored, official)


def test_reverse_hook_leaves_other_tensors_untouched():
    weight = torch.rand(8, 4, dtype=torch.bfloat16)
    out = _k3_converter().model_to_hf_tensor(
        "language_model.model.layers.0.self_attn.q_proj.weight", weight
    )
    assert out is weight


def test_base_default_hooks_are_identity():
    base = object.__new__(BaseWeightConverter)
    t = torch.rand(4)
    assert base.hf_to_model_tensor("x", t) is t
    assert base.model_to_hf_tensor("x", t) is t


def test_normalize_tensor_runs_reverse_hook():
    from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import (
        ShardedParamManager,
    )

    calls = []

    class _FakeConverter:
        def model_to_hf_tensor(self, name, tensor):
            calls.append(name)
            return tensor

        def get_default_hf_dtype(self, name):
            return torch.float32

    saver = object.__new__(ShardedParamManager)
    saver.param_converter = _FakeConverter()
    saver.move_to_cpu = False
    tensor = torch.rand(4, dtype=torch.bfloat16)
    out = saver._normalize_tensor(A_LOG_NAME, tensor)
    assert calls == [A_LOG_NAME]
    assert out.dtype == torch.float32
