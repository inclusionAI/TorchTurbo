"""Qwen3.5 dense contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ids = None

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.input_ids = input_ids
        hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden)


def test_qwen_dense_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboQwen3_5ForConditionalGeneration.__dict__
    assert (
        TurboQwen3_5ForConditionalGeneration.forward
        is Qwen3_5ForConditionalGeneration.forward
    )


def test_qwen_dense_standard_forward_exposes_lm_head_input():
    model = object.__new__(TurboQwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(return_dict=True)
    model.model = _Backbone()
    model.mtp = None
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
    assert outputs.logits is None
    assert torch.equal(
        hidden_states[0],
        torch.arange(12, dtype=torch.float32).view(1, 4, 3),
    )
    assert model.lm_head(hidden_states[0]).shape == (1, 4, 7)
