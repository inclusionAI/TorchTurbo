"""Qwen3 MoE contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.qwen3_moe.model import TurboQwen3MoeForCausalLM


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)

    def forward(self, input_ids, **kwargs):
        return MoeModelOutputWithPast(last_hidden_state=self.hidden)


def test_qwen3_moe_uses_standard_forward_with_noop_lm_head():
    assert "forward" not in TurboQwen3MoeForCausalLM.__dict__
    assert TurboQwen3MoeForCausalLM.forward is Qwen3MoeForCausalLM.forward


def test_qwen3_moe_standard_forward_exposes_lm_head_input():
    model = object.__new__(TurboQwen3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(output_router_logits=False, return_dict=True)
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
    # transformers 5.x slices hidden_states (hidden_states[:, slice_indices, :])
    # before the lm_head, so the exposed tensor is a value-equal copy, not the
    # backbone's last_hidden_state object itself.
    assert torch.equal(hidden_states[0], model.model.hidden)
    assert model.lm_head is real_lm_head
    assert model.lm_head(hidden_states[0]).shape == (1, 4, 7)
