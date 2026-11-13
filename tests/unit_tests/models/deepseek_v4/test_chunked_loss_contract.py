"""DeepSeek V4 contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.deepseek_v4.model import TurboDeepseekV4ForCausalLM
from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Head,
    DeepseekV4RMSNorm,
)


class _IdentityLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class _FailingLayer(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("model forward failed")


def _make_model() -> TurboDeepseekV4ForCausalLM:
    torch.manual_seed(0)
    hidden_size = 4
    hc_mult = 2
    vocab_size = 7

    model = object.__new__(TurboDeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_return_dict=True,
        use_cache=False,
        hc_mult=hc_mult,
        num_hash_layers=0,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    model.vocab_size = vocab_size
    model.embed = nn.Embedding(11, hidden_size)
    model.layers = nn.ModuleList([_IdentityLayer()])
    model.norm = DeepseekV4RMSNorm(hidden_size, eps=model.config.rms_norm_eps)
    model.head = DeepseekV4Head(model.config)
    nn.init.normal_(model.head.weight)
    model.hc_head_fn = nn.Parameter(torch.randn(hc_mult, hc_mult * hidden_size))
    model.hc_head_base = nn.Parameter(torch.randn(hc_mult))
    model.hc_head_scale = nn.Parameter(torch.randn(1))
    model.gradient_checkpointing = False
    model.train()
    return model


def test_chunk_loss_prepares_post_hc_norm_lm_head_input():
    model = _make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    real_lm_head = model.head
    raw_head_inputs = []

    hook = real_lm_head.register_forward_pre_hook(
        lambda _module, args: raw_head_inputs.append(args[0])
    )
    regular_outputs = model(input_ids=input_ids, return_dict=True)
    hook.remove()

    backend = FSDP2Backend.__new__(FSDP2Backend)
    chunk_outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"input_ids": input_ids},
        use_chunked_loss=True,
    )
    hidden_state = hidden_states[0]

    assert len(raw_head_inputs) == 1
    assert chunk_outputs.logits is None
    assert raw_head_inputs[0].shape == (1, 4, 2, 4)
    assert hidden_state.shape == (1, 4, 4)
    assert hidden_state.is_contiguous()
    expected_hidden = real_lm_head._hc_head(
        raw_head_inputs[0],
        model.hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
    )
    expected_hidden = model.norm(expected_hidden).float()
    torch.testing.assert_close(hidden_state, expected_hidden)
    chunk_logits = nn.functional.linear(hidden_state, real_lm_head.weight)
    torch.testing.assert_close(regular_outputs.logits, chunk_logits)

    parameters = (
        model.embed.weight,
        model.norm.weight,
        model.head.weight,
        model.hc_head_fn,
        model.hc_head_base,
        model.hc_head_scale,
    )
    regular_grads = torch.autograd.grad(regular_outputs.logits.sum(), parameters)
    chunk_grads = torch.autograd.grad(chunk_logits.sum(), parameters)
    for regular_grad, chunk_grad in zip(regular_grads, chunk_grads):
        torch.testing.assert_close(regular_grad, chunk_grad)
    assert model.head is real_lm_head

    assert TurboDeepseekV4ForCausalLM.forward is not DeepseekV4ForCausalLM.forward
    assert model.get_lm_head() is real_lm_head
    assert model.get_output_embeddings() is real_lm_head
    assert not isinstance(real_lm_head, nn.Linear)
    assert "head.weight" in model.state_dict()


def test_chunk_loss_restores_deepseek_head_after_forward_error():
    model = _make_model()
    model.layers = nn.ModuleList([_FailingLayer()])
    real_lm_head = model.head
    backend = FSDP2Backend.__new__(FSDP2Backend)

    with pytest.raises(RuntimeError, match="model forward failed"):
        backend._forward_for_lm_loss(
            model,
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            use_chunked_loss=True,
        )

    assert "head" not in vars(model)
    assert model.head is real_lm_head
