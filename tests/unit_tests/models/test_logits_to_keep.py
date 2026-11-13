"""Causal LM forwards project only the positions selected by logits_to_keep."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)

from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
    MoeV3ModelOutputWithPast,
)
from torchturbo.models.glm_moe_dsa.model import TurboGlmMoeDsaForCausalLM
from torchturbo.models.kimi_k25.model import TurboKimiK25ForConditionalGeneration
from torchturbo.models.kimi_k25.modeling_deepseek import DeepseekV3ForCausalLM
from torchturbo.models.llada2_moe.model import TurboLLaDA2MoeForCausalLM

_HIDDEN_SIZE = 3
_VOCAB_SIZE = 7


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB_SIZE, _HIDDEN_SIZE)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds.sin())


class _MoeBackbone(_Backbone):
    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        return MoeModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)


class _BailingBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds.sin()
        return MoeV3ModelOutputWithPast(
            last_hidden_state=hidden,
            mtp_hidden_states=[hidden * 0.5],
        )


def _inputs_embeds():
    return {"inputs_embeds": torch.randn(2, 5, _HIDDEN_SIZE)}


def _bailing_hybrid(num_mtp=0):
    model = object.__new__(TurboBailingMoeV3ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
        vocab_size=_VOCAB_SIZE,
    )
    model.model = _BailingBackbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    model.num_nextn_predict_layers = num_mtp
    model.mtp_loss_scaling_factor = 0.1
    return model, _inputs_embeds()


def _glm_moe_dsa():
    model = object.__new__(TurboGlmMoeDsaForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(use_return_dict=True)
    model.model = _Backbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    return model, _inputs_embeds()


def _kimi_k25():
    config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )
    language_model = object.__new__(DeepseekV3ForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = config
    language_model.model = _Backbone()
    language_model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)

    model = object.__new__(TurboKimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.vision_tower = nn.Identity()
    model.language_model = language_model
    return model, _inputs_embeds()


def _llada2_moe():
    model = object.__new__(TurboLLaDA2MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
    )
    model.model = _MoeBackbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    return model, _inputs_embeds()


def _kimi_language_model():
    model, inputs = _kimi_k25()
    return model.language_model, inputs


@pytest.mark.parametrize(
    "selector",
    [
        0,
        2,
        torch.tensor([False, True, False, True, True]),
        torch.tensor([4, 1, 3]),
        # A sequence-parallel shard may hold no target position.
        torch.zeros(5, dtype=torch.bool),
    ],
)
@pytest.mark.parametrize(
    "make_model",
    [_bailing_hybrid, _glm_moe_dsa, _kimi_k25, _llada2_moe],
)
def test_selected_logits_match_full_projection(make_model, selector):
    torch.manual_seed(0)
    model, inputs = make_model()
    indices = slice(-selector, None) if isinstance(selector, int) else selector
    expected_logits = model(**inputs).logits[:, indices]

    projected_lengths = []
    hook = model.get_lm_head().register_forward_hook(
        lambda module, args, output: projected_lengths.append(output.shape[1])
    )
    logits = model(**inputs, logits_to_keep=selector).logits
    hook.remove()

    torch.testing.assert_close(logits, expected_logits)
    # The head projects the selected positions only.
    assert projected_lengths == [expected_logits.shape[1]]


def test_bailing_hybrid_mtp_logits_follow_the_selection():
    torch.manual_seed(0)
    model, inputs = _bailing_hybrid(num_mtp=1)
    selector = torch.tensor([False, True, False, True, True])
    expected = model(**inputs)

    outputs = model(**inputs, logits_to_keep=selector)

    torch.testing.assert_close(outputs.logits, expected.logits[:, selector])
    torch.testing.assert_close(
        outputs.mtp_logits[0], expected.mtp_logits[0][:, selector]
    )


@pytest.mark.parametrize(
    ("selector", "use_inputs_embeds", "cache_length"),
    [
        pytest.param(None, False, 0, id="prefill-omitted"),
        pytest.param(0, True, 0, id="prefill-explicit-all"),
        pytest.param(2, False, 0, id="prefill-last-two"),
        pytest.param(torch.tensor([4, 1, 3]), True, 0, id="prefill-indices"),
        pytest.param(torch.zeros(1, dtype=torch.bool), False, 4, id="cached-empty"),
    ],
)
@pytest.mark.parametrize(
    "make_model", [_glm_moe_dsa, _kimi_k25, _kimi_language_model, _llada2_moe]
)
def test_generation_preparation_preserves_selected_projection(
    make_model, selector, use_inputs_embeds, cache_length
):
    torch.manual_seed(0)
    model, inputs = make_model()
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    past_key_values = None
    if cache_length:
        key = torch.zeros(2, 1, cache_length, 1)
        past_key_values = ((key, key.clone()),)
    selector_kwargs = {} if selector is None else {"logits_to_keep": selector}
    prepared = model.prepare_inputs_for_generation(
        input_ids,
        inputs_embeds=inputs["inputs_embeds"] if use_inputs_embeds else None,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=past_key_values,
        use_cache=True,
        **selector_kwargs,
    )
    if cache_length:
        assert prepared["input_ids"].shape[1] == 1
    full_inputs = dict(prepared)
    full_inputs.pop("logits_to_keep", None)
    full = model(**full_inputs).logits
    reference_selector = 0 if selector is None else selector
    indices = (
        slice(-reference_selector, None)
        if isinstance(reference_selector, int)
        else reference_selector
    )
    expected = full[:, indices]

    projected_lengths = []
    hook = model.get_output_embeddings().register_forward_hook(
        lambda module, args, output: projected_lengths.append(output.shape[1])
    )
    try:
        actual = model(**prepared).logits
    finally:
        hook.remove()

    torch.testing.assert_close(actual, expected)
    assert projected_lengths == [expected.shape[1]]
    if selector is None:
        assert "logits_to_keep" not in prepared
    else:
        assert prepared["logits_to_keep"] is selector
