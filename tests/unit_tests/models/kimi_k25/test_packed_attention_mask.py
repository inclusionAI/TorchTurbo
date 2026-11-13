from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from torchturbo.models.base import TurboModel
from torchturbo.models.kimi_k25 import modeling_deepseek
from torchturbo.models.kimi_k25.model import (
    TurboKimiK25ForConditionalGeneration,
)
from torchturbo.models.kimi_k25.modeling_deepseek import DeepseekV3FlashAttention2
from torchturbo.models.modules.attention.ulysses import (
    patch_forward_for_sp_input_slicing,
)
from torchturbo.utils.flash_attn_utils import get_unpad_data


def test_kimi_fa2_integer_mask_produces_packed_boundaries():
    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])

    indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)

    assert indices.tolist() == list(range(5))
    assert cu_seqlens.tolist() == [0, 2, 5]
    assert max_seqlen == 3


def test_kimi_custom_fa2_passes_integer_boundaries_to_varlen_kernel(monkeypatch):
    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])
    query = torch.arange(64, dtype=torch.float32).reshape(1, 8, 2, 4)
    captured = {}

    def fake_flash_attn_varlen_func(query_states, key_states, value_states, **kwargs):
        captured.update(kwargs)
        return query_states

    monkeypatch.setattr(
        modeling_deepseek,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )
    attention = object.__new__(DeepseekV3FlashAttention2)
    nn.Module.__init__(attention)
    attention.num_heads = 2

    output = attention._flash_attention_forward(
        query,
        query,
        query,
        attention_mask,
        query_length=8,
        is_causal=True,
    )

    assert captured["cu_seqlens_q"].tolist() == [0, 2, 5]
    assert captured["cu_seqlens_k"].tolist() == [0, 2, 5]
    assert captured["max_seqlen_q"] == 3
    assert captured["max_seqlen_k"] == 3
    assert output.shape == query.shape
    assert not output[:, 5:].any()


def test_kimi_native_fa2_consumes_cu_without_segment_mask(monkeypatch):
    """The native (non-turbo) flash path takes cu directly, not a rebuilt mask."""
    query = torch.arange(48, dtype=torch.float32).reshape(1, 6, 2, 4)
    captured = {}

    def fake_flash_attn_varlen_func(query_states, key_states, value_states, **kwargs):
        captured.update(kwargs)
        return query_states

    monkeypatch.setattr(
        modeling_deepseek,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )
    attention = object.__new__(DeepseekV3FlashAttention2)
    nn.Module.__init__(attention)
    attention.num_heads = 2

    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    output = attention._flash_attention_forward(
        query,
        query,
        query,
        None,
        query_length=6,
        is_causal=True,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=2,
        max_length_k=2,
    )

    assert captured["cu_seqlens_q"].tolist() == [0, 2, 4]
    assert captured["max_seqlen_q"] == 2
    # The padded tail (two tokens beyond cu[-1]) comes back as zeros.
    assert output.shape == query.shape
    assert not output[:, 4:].any()


def test_kimi_sp_slices_embeddings_but_keeps_global_packed_mask(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)

    class FakeLanguageModel(nn.Module):
        def forward(self, **kwargs):
            return kwargs

    model = FakeLanguageModel()
    wrapper = SimpleNamespace(language_model=model)
    patch_forward_for_sp_input_slicing(
        TurboKimiK25ForConditionalGeneration.get_sequence_parallel_input_slicing_spec(wrapper),
        sp_group=object(),
        sp_size=2,
        sp_rank=0,
    )
    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])
    result = model.forward(
        inputs_embeds=torch.zeros(1, 8, 4),
        position_ids=torch.tensor([[0, 1, 0, 1, 2, 0, 1, 2]]),
        attention_mask=attention_mask,
    )

    assert result["inputs_embeds"].shape == (1, 4, 4)
    assert result["position_ids"].shape == (1, 4)
    assert result["attention_mask"] is attention_mask


@pytest.mark.parametrize("sequence_length", [4, 6])
def test_kimi_forwards_cu_boundaries_and_padding_to_language_layers(sequence_length):
    captured = []

    class DecoderLayer(nn.Module):
        def forward(self, hidden_states, *, attention_mask, **kwargs):
            captured.append((attention_mask, kwargs))
            return (hidden_states,)

    config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        use_return_dict=True,
    )
    backbone = object.__new__(modeling_deepseek.DeepseekV3Model)
    nn.Module.__init__(backbone)
    backbone.config = config
    backbone.layers = nn.ModuleList([DecoderLayer()])
    backbone.norm = nn.Identity()
    backbone._use_flash_attention_2 = True
    backbone.gradient_checkpointing = False

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = backbone

        def forward(self, **kwargs):
            output = self.model(**kwargs)
            return CausalLMOutputWithPast(logits=output.last_hidden_state)

    model = object.__new__(TurboKimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.vision_tower = nn.Identity()
    model.language_model = LanguageModel()
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    output = model(
        inputs_embeds=torch.zeros(1, sequence_length, 4),
        position_ids=torch.tensor([[0, 1, 0, 1] + [0] * (sequence_length - 4)]),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=2,
        max_length_k=2,
    )

    # No segment mask is rebuilt: boundaries ride with cu to the layers and
    # the padding tail stays outside every boundary.
    layer_mask, layer_kwargs = captured[0]
    assert layer_mask is None
    assert layer_kwargs["cu_seq_lens_q"] is cu
    assert layer_kwargs["cu_seq_lens_k"] is cu
    assert layer_kwargs["max_length_q"] == 2
    assert layer_kwargs["max_length_k"] == 2
    assert output.logits.shape == (1, sequence_length, 4)
    assert cu.tolist() == [0, 2, 4]


def test_alignment_padding_is_excluded_from_moe_runtime_inputs():
    runtime_inputs = TurboModel.get_moe_runtime_inputs(
        None,
        {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        },
        {"sp": {"sp_size": 1}},
    )

    assert runtime_inputs.padding_mask.tolist() == [
        [False, False, False, False, False, True, True, True]
    ]


def test_kimi_multimodal_merge_preserves_integer_packed_mask():
    model = object.__new__(TurboKimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        media_placeholder_token_id=9,
        ignore_index=-100,
    )

    input_ids = torch.tensor([[9, 5, 9, 6, 0, 0]])
    inputs_embeds = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    image_features = [
        torch.full((1, 4), 101.0),
        torch.full((1, 4), 202.0),
    ]
    attention_mask = torch.tensor([[1, 1, 2, 2, 0, 0]])
    labels = torch.tensor([[9, 5, 9, 6, -100, -100]])

    merged, merged_mask, merged_labels = (
        model._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            labels,
        )
    )

    assert merged_mask is attention_mask
    assert merged_mask.tolist() == [[1, 1, 2, 2, 0, 0]]
    assert merged[0, 0].tolist() == [101.0] * 4
    assert merged[0, 2].tolist() == [202.0] * 4
    assert merged_labels.tolist() == [[-100, 5, -100, 6, -100, -100]]


def test_deepseek_attention_constructs_from_config():
    """Guards against attribute-name regressions in __init__ (e.g. qk_head_dim
    vs q_head_dim) — construction alone must succeed with a minimal MLA config."""
    cfg = SimpleNamespace(
        attention_dropout=0.0,
        hidden_size=64,
        num_attention_heads=4,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        q_lora_rank=None,
        qk_rope_head_dim=32,
        kv_lora_rank=128,
        v_head_dim=32,
        qk_nope_head_dim=64,
        rope_scaling=None,
        attention_bias=False,
    )
    attn = modeling_deepseek.DeepseekV3Attention(cfg, layer_idx=0)
    assert attn.softmax_scale == (64 + 32) ** -0.5
