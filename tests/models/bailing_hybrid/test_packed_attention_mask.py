import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import torchturbo.ops as turbo_ops
from torchturbo.models.bailing_hybrid import modeling_bailing_moe_v3 as modeling
from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM


class _CapturedKDA(Exception):
    pass


@pytest.mark.parametrize(
    ("cu_seq_lens", "num_tokens"),
    [
        ([0, 2, 4], 4),
        ([0, 2, 4], 5),  # one padding token beyond the last boundary
    ],
)
def test_model_preserves_packed_boundaries_for_kda(monkeypatch, cu_seq_lens, num_tokens):
    captured = []

    def capture_conv(**kwargs):
        packed = kwargs["packed_params"]
        assert packed.cu_seqlens_list[-1] == kwargs["x"].size(1)
        captured.append(packed.cu_seqlens_list)
        return kwargs["x"], None

    def capture_kda(**kwargs):
        cu_seqlens = kwargs["cu_seqlens"]
        captured.append(None if cu_seqlens is None else cu_seqlens.tolist())
        raise _CapturedKDA

    monkeypatch.setattr(
        turbo_ops, "ops", SimpleNamespace(causal_conv1d=capture_conv, kda=capture_kda)
    )
    monkeypatch.setitem(
        sys.modules, "fla.modules.fused_norm_gate", SimpleNamespace(rms_norm_gated=None)
    )
    attention_type = modeling.BailingMoeV3KimiDeltaAttention
    monkeypatch.setattr(attention_type, "forward", attention_type.forward)
    TurboBailingMoeV3ForCausalLM.patch_kda_forward(None, 1, 0)

    def identity(x):
        return x

    conv = SimpleNamespace(weight=torch.ones(2, 1, 2), bias=None, activation="silu")
    attention = SimpleNamespace(
        mode="chunk",
        q_proj=identity,
        k_proj=identity,
        v_proj=identity,
        q_conv1d=conv,
        k_conv1d=conv,
        v_conv1d=conv,
        no_kda_lora=True,
        f_proj=identity,
        b_proj=lambda x: x[..., :1],
        head_k_dim=2,
        head_dim=2,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        safe_gate=True,
        lower_bound=-5,
    )
    decoder = SimpleNamespace(
        input_layernorm=identity,
        attention_layer_type="linear_attention",
        attention=lambda **kwargs: attention_type.forward(attention, **kwargs),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            use_cache=False,
            use_return_dict=False,
            layer_group_size=2,
        ),
        gradient_checkpointing=False,
        training=True,
        _use_flash_attention_2=True,
        _use_sdpa=False,
        num_nextn_predict_layers=0,
        rotary_emb=lambda hidden, positions: None,
        layers=[
            lambda *args, **kwargs: modeling.BailingMoeV3DecoderLayer.forward(
                decoder, *args, **kwargs
            )
        ],
    )
    positions = [0, 1, 0, 1] + [0] * (num_tokens - 4)

    with pytest.raises(_CapturedKDA):
        modeling.BailingMoeV3Model.forward(
            model,
            inputs_embeds=torch.ones(1, num_tokens, 2),
            attention_mask=None,
            position_ids=torch.tensor([positions]),
            cu_seq_lens_q=torch.tensor(cu_seq_lens, dtype=torch.int32),
        )

    # Conv and KDA require complete coverage; padding is an isolated segment.
    expected = cu_seq_lens + ([num_tokens] if num_tokens > cu_seq_lens[-1] else [])
    assert captured == [expected] * 4


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_backbone_and_mtp_attention_keep_prefix_metadata(use_checkpoint):
    captured = []

    def attention(hidden_states, **kwargs):
        captured.append(kwargs)
        return hidden_states.sin(), None, None

    class Layer(torch.nn.Module):
        def __init__(self, mtp=False):
            super().__init__()
            self.mtp = mtp
            self.attention_layer_type = "attention"
            self.attention = attention
            self.input_layernorm = torch.nn.Identity()
            self.post_attention_layernorm = torch.nn.Identity()
            self.mlp = torch.nn.Identity()
            self.enorm = torch.nn.Identity()
            self.hnorm = torch.nn.Identity()
            self.eh_proj = torch.nn.Linear(4, 2)
            self.final_layernorm = torch.nn.Identity()

        def forward(self, *args, **kwargs):
            layer_cls = modeling.BailingMoeV3MTPLayer if self.mtp else modeling.BailingMoeV3DecoderLayer
            return layer_cls.forward(self, *args, **kwargs)

    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            use_cache=False,
            use_return_dict=True,
            layer_group_size=2,
        ),
        gradient_checkpointing=use_checkpoint,
        _gradient_checkpointing_func=partial(checkpoint, use_reentrant=True),
        training=True,
        _use_flash_attention_2=True,
        _use_sdpa=False,
        num_nextn_predict_layers=1,
        rotary_emb=lambda hidden, positions: None,
        norm=torch.nn.Identity(),
        word_embeddings=torch.nn.Embedding(16, 2),
        layers=[Layer(), Layer(mtp=True)],
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    output = modeling.BailingMoeV3Model.forward(
        model,
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 0]]),
        position_ids=torch.tensor([[0, 1, 0, 1, 2, 0]]),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=3,
        max_length_k=3,
    )
    (output.last_hidden_state.sum() + output.mtp_hidden_states[0].sum()).backward()

    assert len(captured) == (4 if use_checkpoint else 2)
    for kwargs in captured:
        torch.testing.assert_close(kwargs["cu_seq_lens_q"], cu)
        torch.testing.assert_close(kwargs["cu_seq_lens_k"], cu)
        assert kwargs["max_length_q"] == kwargs["max_length_k"] == 3
    assert cu.tolist() == [0, 2, 5]
