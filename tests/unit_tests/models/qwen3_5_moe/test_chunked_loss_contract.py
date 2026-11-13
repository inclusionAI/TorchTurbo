"""Qwen3.5 MoE contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.qwen3_5_moe.model import (
    TurboQwen3_5MoeForConditionalGeneration,
)
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModelOutputWithPast,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = SimpleNamespace(embed_tokens=nn.Embedding(8, 3))
        self.input_ids = None

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.input_ids = input_ids
        hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        return Qwen3_5MoeModelOutputWithPast(last_hidden_state=hidden)


class _MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ids = None

    def forward(self, *, input_ids, hidden_states, **kwargs):
        self.input_ids = input_ids
        return [hidden_states * 0.5]


def test_qwen_moe_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboQwen3_5MoeForConditionalGeneration.__dict__
    assert (
        TurboQwen3_5MoeForConditionalGeneration.forward
        is Qwen3_5MoeForConditionalGeneration.forward
    )


def test_qwen_moe_postprocess_merges_swift_position_rows():
    from torchturbo.models.qwen3_5.swift_template import _post_process_inputs

    inputs = {
        "position_ids": torch.tensor(
            [[[0, 1, 0, 1, 2]], [[9, 9, 9, 9, 9]], [[7, 7, 7, 7, 7]]]
        ),
        "text_position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
    }

    result = _post_process_inputs(inputs)

    assert result["position_ids"].shape == (4, 1, 5)
    assert result["position_ids"][0].tolist() == [[0, 1, 0, 1, 2]]
    assert result["position_ids"][1:].tolist() == [
        [[0, 1, 0, 1, 2]],
        [[9, 9, 9, 9, 9]],
        [[7, 7, 7, 7, 7]],
    ]


def test_qwen_moe_standard_forward_returns_main_and_mtp_hidden_states():
    model = object.__new__(TurboQwen3_5MoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        return_dict=True,
        text_config=SimpleNamespace(pad_token_id=0),
    )
    model.model = _Backbone()
    model.mtp = _MTP()
    model.lm_head = nn.Linear(3, 7, bias=False)
    model.train()
    input_ids = torch.ones(1, 4, dtype=torch.long)
    inputs_embeds = torch.randn(1, 4, 3)

    backend = FSDP2Backend.__new__(FSDP2Backend)
    outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"input_ids": input_ids, "inputs_embeds": inputs_embeds},
        use_chunked_loss=True,
    )

    assert model.model.input_ids is None
    assert model.mtp.input_ids is input_ids
    assert len(outputs.mtp_hidden_states) == 1
    assert outputs.logits is None
    assert outputs.mtp_logits is None
    assert outputs["logits"] is None
    assert outputs["mtp_logits"] is None
    assert hidden_states[1] is outputs.mtp_hidden_states[0]
    assert model.lm_head(hidden_states[0]).shape == (1, 4, 7)
