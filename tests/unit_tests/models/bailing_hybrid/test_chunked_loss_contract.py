"""Bailing Hybrid contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
    MoeV3ModelOutputWithPast,
)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3).bfloat16()
        self.mtp_hidden = self.hidden * 0.5

    def forward(self, input_ids, **kwargs):
        return MoeV3ModelOutputWithPast(
            last_hidden_state=self.hidden,
            mtp_hidden_states=[self.mtp_hidden],
        )


def test_bailing_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboBailingMoeV3ForCausalLM.__dict__
    assert TurboBailingMoeV3ForCausalLM.forward is BailingMoeV3ForCausalLM.forward


def test_bailing_standard_forward_returns_main_and_mtp_hidden_states():
    model = object.__new__(TurboBailingMoeV3ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
        vocab_size=7,
    )
    model.model = _Backbone()
    model.lm_head = nn.Linear(3, 7, bias=False, dtype=torch.bfloat16)
    model.num_nextn_predict_layers = 1
    model.mtp_loss_scaling_factor = 0.1
    model.train()
    real_lm_head = model.lm_head

    backend = FSDP2Backend.__new__(FSDP2Backend)
    outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"input_ids": torch.ones(1, 4, dtype=torch.long)},
        use_chunked_loss=True,
    )

    assert outputs.logits is None
    assert outputs.mtp_logits is None
    assert hidden_states[0].dtype == torch.bfloat16
    assert torch.equal(hidden_states[0], model.model.hidden)
    assert torch.equal(hidden_states[1], model.model.mtp_hidden)
    assert model.lm_head is real_lm_head
