"""Kimi K2.5 contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.kimi_k25.model import TurboKimiK25ForConditionalGeneration
from torchturbo.models.kimi_k25.modeling_kimi_k25 import (
    KimiK25ForConditionalGeneration,
)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(3, 7, bias=False, dtype=torch.bfloat16)

    def forward(self, *, inputs_embeds, **kwargs):
        return CausalLMOutputWithPast(logits=self.lm_head(inputs_embeds).float())


def test_kimi_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboKimiK25ForConditionalGeneration.__dict__
    assert (
        TurboKimiK25ForConditionalGeneration.forward
        is KimiK25ForConditionalGeneration.forward
    )


def test_kimi_standard_forward_exposes_nested_lm_head_input():
    model = object.__new__(TurboKimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )
    model.vision_tower = nn.Identity()
    model.mm_projector = nn.Identity()
    model.language_model = _LanguageModel()
    model.train()
    real_lm_head = model.language_model.lm_head
    hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3).bfloat16()

    backend = FSDP2Backend.__new__(FSDP2Backend)
    outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"inputs_embeds": hidden},
        use_chunked_loss=True,
    )

    assert outputs.logits is None
    assert hidden_states[0].dtype == torch.bfloat16
    assert torch.equal(hidden_states[0], hidden)
    assert model.get_lm_head() is real_lm_head
