"""Selected prediction positions must skip the unused vocabulary projection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss

from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
    MoeV3ModelOutputWithPast,
)


class _Backbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        assert "logits_to_keep" not in kwargs
        hidden = inputs_embeds.sin()
        return MoeV3ModelOutputWithPast(
            last_hidden_state=hidden,
            mtp_hidden_states=[hidden * 0.5],
        )


def _model(num_mtp=0, model_cls=TurboBailingMoeV3ForCausalLM):
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
        vocab_size=7,
    )
    model.model = _Backbone()
    model.lm_head = nn.Linear(3, 7, bias=False)
    model.num_nextn_predict_layers = num_mtp
    model.mtp_loss_scaling_factor = 0.1
    model.loss_function = ForCausalLMLoss
    return model


@pytest.mark.parametrize(
    "selector",
    [
        0,
        2,
        torch.tensor([False, True, False, True, True]),
        torch.tensor([4, 1, 3]),
        torch.zeros(5, dtype=torch.bool),
    ],
)
@pytest.mark.parametrize("num_mtp", [0, 1])
def test_selected_logits_and_gradients_match_full_projection(selector, num_mtp):
    torch.manual_seed(17)
    model = _model(num_mtp)
    hidden = torch.randn(2, 5, 3, requires_grad=True)
    indices = slice(-selector, None) if isinstance(selector, int) else selector
    full = model(inputs_embeds=hidden)
    expected_logits = full.logits[:, indices]
    objective = expected_logits.square().sum()
    if num_mtp:
        expected_mtp = full.mtp_logits[0][:, indices]
        objective = objective + expected_mtp.square().sum()
    expected_gradients = torch.autograd.grad(objective, (hidden, model.lm_head.weight))

    projected_shapes = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: projected_shapes.append(inputs[0].shape)
    )
    selected = model(inputs_embeds=hidden, logits_to_keep=selector)
    hook.remove()
    torch.testing.assert_close(selected.logits, expected_logits)
    objective = selected.logits.square().sum()
    if num_mtp:
        torch.testing.assert_close(selected.mtp_logits[0], expected_mtp)
        objective = objective + selected.mtp_logits[0].square().sum()
    gradients = torch.autograd.grad(objective, (hidden, model.lm_head.weight))
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    assert projected_shapes == [hidden[:, indices].shape] * (1 + num_mtp)


def test_default_full_logits_labeled_loss_and_tuple_output_are_unchanged():
    model = _model()
    hidden = torch.randn(2, 5, 3)
    labels = torch.tensor([[1, 2, -100, 4, 5], [1, 3, 2, 6, 4]])
    expected_logits = model.lm_head(hidden.sin()).float()
    expected_loss = ForCausalLMLoss(expected_logits, labels, model.config.vocab_size)
    outputs = model(inputs_embeds=hidden, labels=labels)
    torch.testing.assert_close(outputs.logits, expected_logits)
    torch.testing.assert_close(outputs.loss, expected_loss)
    as_tuple = model(inputs_embeds=hidden, labels=labels, return_dict=False)
    torch.testing.assert_close(as_tuple[0], expected_loss)
    torch.testing.assert_close(as_tuple[1], expected_logits)


@pytest.mark.parametrize("num_mtp", [0, 1])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_default_forward_matches_original_model(num_mtp, with_labels, return_dict):
    model = _model(num_mtp)
    original = _model(num_mtp, BailingMoeV3ForCausalLM)
    original.load_state_dict(model.state_dict())
    hidden = torch.randn(2, 5, 3)
    labels = torch.tensor([[1, 2, -100, 4, 5], [1, 3, 2, 6, 4]]) if with_labels else None
    inputs = dict(inputs_embeds=hidden, labels=labels, return_dict=return_dict)
    actual = model(**inputs)
    expected = original(**inputs)
    torch.testing.assert_close(
        dict(actual) if return_dict else actual,
        dict(expected) if return_dict else expected,
    )


def test_selected_logits_use_explicit_shifted_labels_without_realigning_them():
    model = _model()
    hidden = torch.randn(1, 5, 3)
    labels = torch.tensor([[-100, 1, -100, 4, 5]])
    shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    selector = shifted_labels[0] != -100
    full = model(inputs_embeds=hidden, labels=labels)
    selected = model(
        inputs_embeds=hidden,
        labels=labels,
        shift_labels=shifted_labels[:, selector],
        logits_to_keep=selector,
    )
    torch.testing.assert_close(selected.loss, full.loss)
