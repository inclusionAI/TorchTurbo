"""Mask-creation tests for the qwen3_moe registry migration."""

from __future__ import annotations

import torch

import pytest


def _mask_factory(use_sliding_window):
    import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_model

    # The registered mask entry maps "turbo_ulysses" to the flash-attention
    # mask function; the factory must not take the unrecognized-name
    # early-exit that drops the mask.
    if use_sliding_window:
        return qwen3_moe_model.create_sliding_window_causal_mask
    return qwen3_moe_model.create_causal_mask


def test_model_keeps_hf_mask_factory():
    """The model must not globally replace HF's mask factory."""
    import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_model
    from transformers.masking_utils import create_causal_mask

    assert qwen3_moe_model.create_causal_mask is create_causal_mask


def _config(use_sliding_window):
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        use_sliding_window=use_sliding_window,
        sliding_window=4 if use_sliding_window else None,
        max_position_embeddings=16,
    )
    config._attn_implementation = "turbo_ulysses"
    return config


def _call(factory, attention_mask, use_sliding_window):
    embeds = torch.zeros(1, attention_mask.size(-1), 8)
    return factory(
        config=_config(use_sliding_window),
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=None,
    )


@pytest.mark.parametrize("use_sliding_window", [False, True])
def test_mask_factory_keeps_padding_mask(use_sliding_window):
    """Unpacked padded batches keep a 2D bool mask so flash attention unpads."""
    from torchturbo.models.modules.attention.ulysses import (
        register_turbo_ulysses_attention,
    )

    register_turbo_ulysses_attention(_config(use_sliding_window), object(), 2)
    padding = torch.tensor([[1, 1, 1, 0, 0]])

    out = _call(_mask_factory(use_sliding_window), padding, use_sliding_window)

    torch.testing.assert_close(out, padding.bool())


@pytest.mark.parametrize("use_sliding_window", [False, True])
def test_mask_factory_returns_none_without_padding(use_sliding_window):
    """All-valid masks become None; flash attention runs causal without unpad."""
    from torchturbo.models.modules.attention.ulysses import (
        register_turbo_ulysses_attention,
    )

    register_turbo_ulysses_attention(_config(use_sliding_window), object(), 2)
    all_valid = torch.ones(1, 5, dtype=torch.long)

    out = _call(_mask_factory(use_sliding_window), all_valid, use_sliding_window)

    assert out is None
