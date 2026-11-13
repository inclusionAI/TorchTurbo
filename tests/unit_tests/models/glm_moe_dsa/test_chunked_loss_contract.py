"""GLM-MoE-DSA contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.glm_moe_dsa.model import TurboGlmMoeDsaForCausalLM
from torchturbo.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaForCausalLM,
)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)

    def forward(self, input_ids, **kwargs):
        return BaseModelOutputWithPast(last_hidden_state=self.hidden)


def test_glm_moe_dsa_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboGlmMoeDsaForCausalLM.__dict__
    assert TurboGlmMoeDsaForCausalLM.forward is GlmMoeDsaForCausalLM.forward


def test_glm_moe_dsa_standard_forward_exposes_lm_head_input():
    model = object.__new__(TurboGlmMoeDsaForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(use_return_dict=True)
    model.model = _Backbone()
    model.lm_head = nn.Linear(3, 7, bias=False)
    model.vocab_size = 7
    model.train()
    real_lm_head = model.lm_head

    backend = FSDP2Backend.__new__(FSDP2Backend)
    outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"input_ids": torch.ones(1, 4, dtype=torch.long)},
        use_chunked_loss=True,
    )

    assert outputs.logits is None
    assert hidden_states[0] is model.model.hidden
    assert model.lm_head is real_lm_head
    assert model.lm_head(hidden_states[0]).shape == (1, 4, 7)
