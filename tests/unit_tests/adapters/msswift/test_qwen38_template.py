"""Exercise the Qwen3.8 template with the actual Swift text encoder."""

from copy import deepcopy
import json
import pickle
import sys

import pytest
import torch

swift = pytest.importorskip("swift")
if swift.__version__ != "4.4.2":
    pytest.skip("Qwen3.8 backport targets Swift 4.4.2", allow_module_level=True)

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config
from swift.model import ModelInfo, MODEL_MAPPING
from swift.template import get_template, TEMPLATE_MAPPING, get_last_user_round
from swift.template.template_inputs import StdTemplateInputs, TemplateInputs
from swift.template.templates.qwen import Qwen3_5Template
from torchturbo.models.qwen3_5.swift_qwen3_8 import Qwen3_8Template
from torchturbo.models.qwen3_5.swift_template import register


pytestmark = [pytest.mark.cpu, pytest.mark.integration]


@pytest.fixture
def tokenizer():
    vocab = {word: i for i, word in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
        "<|image_pad|>", "<|video_pad|>",
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
    )
    tokenizer.model_info = ModelInfo(
        "qwen3_5", ".", torch.bfloat16, 128000, None, 0,
        config=Qwen3_5Config(), task_type="causal_lm",
    )
    tokenizer.model_meta = MODEL_MAPPING["qwen3_5"]
    register()
    return tokenizer


def _template(tokenizer, name="qwen3_8", **kwargs):
    template = get_template(
        tokenizer, template_type=name,
        loss_scale=kwargs.pop("loss_scale", "last_round+ignore_empty_think"), **kwargs,
    )
    template.set_mode("train")
    return template


def _conversation():
    return [
        {"role": "user", "content": "old-query"},
        {"role": "assistant", "content": "old-call"},
        {"role": "tool", "content": "old-observation"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "new-query"},
        {"role": "assistant", "content": "new-call"},
        {"role": "tool", "content": "new-observation"},
        {"role": "assistant", "content": "new-answer"},
    ]


def _texts(template, messages):
    encoded = template.encode({"messages": messages})
    return (
        template.tokenizer.decode(encoded["input_ids"]),
        template.tokenizer.decode([v for v in encoded["labels"] if v != -100]),
    )


def test_qwen38_history_prefixes_and_last_tool_mask(tokenizer):
    messages = _conversation()
    original = deepcopy(messages)
    text, targets = _texts(_template(tokenizer), messages)
    assert text.count("<think>\n\n</think>\n\n") == 4
    assert "new-answer" in targets
    assert "new-call" not in targets
    assert "old-answer" not in targets
    assert "<think>" not in targets
    assert messages == original


def test_qwen35_and_global_last_round_behavior_are_unchanged(tokenizer, monkeypatch):
    native = deepcopy(TEMPLATE_MAPPING["qwen3_5"])
    native.template_cls = Qwen3_5Template
    monkeypatch.setitem(TEMPLATE_MAPPING, "qwen3_5", native)
    monkeypatch.delitem(TEMPLATE_MAPPING, "qwen3_8")
    register()
    # The Qwen3.5 registration also brings in the Qwen3.8 template.
    assert TEMPLATE_MAPPING["qwen3_8"].template_cls is Qwen3_8Template
    messages = _conversation()
    _texts(_template(tokenizer), messages)
    text, targets = _texts(_template(tokenizer, "qwen3_5"), messages)
    assert text.count("<think>\n\n</think>\n\n") == 2
    assert "new-call" in targets and "new-answer" in targets
    assert "old-answer" not in targets
    assert get_last_user_round(messages) == 4


def test_explicit_loss_overrides_last_round(tokenizer):
    messages = _conversation()
    messages[1]["loss"] = True
    messages[-1]["loss"] = False
    _, targets = _texts(_template(tokenizer), messages)
    assert "old-call" in targets
    assert "new-answer" not in targets
    assert "new-call" not in targets


@pytest.mark.parametrize("preserve", [None, False, True])
def test_history_thinking_override_and_no_duplicate_prefix(tokenizer, preserve):
    messages = _conversation()
    messages[3]["content"] = "<think>old-reason</think>old-answer"
    messages[5]["content"] = "<think>call-reason</think>new-call"
    messages[-1]["content"] = "<think>\n\n</think>\n\nnew-answer"
    text, targets = _texts(_template(tokenizer, preserve_thinking=preserve), messages)
    assert ("old-reason" in text) is (preserve is not False)
    assert ("call-reason" in text) is (preserve is not False)
    assert text.count("<think>\n\n</think>\n\nnew-answer") == 1
    assert "new-answer" in targets
    assert "old-reason" not in targets


@pytest.mark.parametrize("preserve", [False, True])
def test_per_input_thinking_override(tokenizer, preserve):
    template = _template(tokenizer, preserve_thinking=not preserve)
    messages = _conversation()
    messages[5]["content"] = "<think>call-reason</think>new-call"
    encoded = template.encode({
        "messages": messages,
        "chat_template_kwargs": {"preserve_thinking": preserve},
    })
    assert ("call-reason" in tokenizer.decode(encoded["input_ids"])) is preserve


def test_list_content_keeps_explicit_per_fragment_loss(tokenizer):
    template = _template(tokenizer)
    messages = _conversation()
    messages[5]["content"] = ["new-call", "extra-call"]
    messages[5]["loss"] = [True, False]
    inputs = StdTemplateInputs(messages=deepcopy(messages))
    template._add_non_thinking_prefix(inputs)
    assert inputs.messages[5]["content"] == ["<think>\n\n</think>\n\nnew-call", "extra-call"]
    from swift.template import ContextType

    contexts, scales = template.loss_scale(
        [messages[i]["content"] for i in (1, 3, 5, 7)],
        [ContextType.RESPONSE] * 4, messages,
    )
    included = [context for context, scale in zip(contexts, scales) if scale]
    assert included == ["new-call", "new-answer"]


def test_reasoning_instruction_is_inactive_unless_enabled(tokenizer):
    inputs = StdTemplateInputs(messages=[{"role": "user", "content": "query"}])
    assert _template(tokenizer)._get_reasoning_instructions(inputs) == ""
    template = _template(tokenizer, enable_thinking=True)
    assert template._get_reasoning_instructions(inputs).startswith("Reasoning effort is set to xhigh.")
    inputs.chat_template_kwargs["reasoning_effort"] = "low"
    assert template._get_reasoning_instructions(inputs).startswith("Reasoning effort is set to low.")


def test_registration_and_worker_pickle(tokenizer):
    cls = TEMPLATE_MAPPING["qwen3_8"].template_cls
    register()
    assert TEMPLATE_MAPPING["qwen3_8"].template_cls is cls
    template = _template(tokenizer)
    before = _texts(template, _conversation())
    restored = pickle.loads(pickle.dumps(template))
    assert _texts(restored, _conversation()) == before


@pytest.mark.parametrize("failure", [None, "import", "register"])
def test_adapter_registers_qwen38_template(tokenizer, monkeypatch, failure):
    from torchturbo.adapters.msswift import model_registry
    from torchturbo.adapters.msswift.loader import TurboModelMixin
    from torchturbo.models.qwen3_5 import swift_qwen3_8

    models = model_registry.get_all_models()
    monkeypatch.setattr(model_registry, "get_all_models", lambda: {"qwen3_5": models["qwen3_5"]})
    model_mapping = MODEL_MAPPING.copy()
    model_mapping.pop("turbo_qwen3_5", None)
    monkeypatch.setattr("swift.model.register.MODEL_MAPPING", model_mapping)
    native = deepcopy(TEMPLATE_MAPPING["qwen3_5"])
    native.template_cls = Qwen3_5Template
    monkeypatch.setitem(TEMPLATE_MAPPING, "qwen3_5", native)
    monkeypatch.delitem(TEMPLATE_MAPPING, "qwen3_8")

    if failure == "import":
        monkeypatch.setitem(sys.modules, swift_qwen3_8.__name__, None)
    elif failure == "register":
        def fail_registration(*args, **kwargs):
            raise RuntimeError("Qwen3.8 registration failed")

        monkeypatch.setattr(swift_qwen3_8, "Qwen3_8TemplateMeta", fail_registration)

    model_registry.register_turbo_models_to_swift()

    assert issubclass(model_mapping["turbo_qwen3_5"].loader, TurboModelMixin)
    wrapped = TEMPLATE_MAPPING["qwen3_5"].template_cls
    assert issubclass(wrapped, Qwen3_5Template)
    assert wrapped.__turbo_kwarg_preserve__ is True
    if failure is None:
        assert TEMPLATE_MAPPING["qwen3_8"].template_cls is Qwen3_8Template
    else:
        assert "qwen3_8" not in TEMPLATE_MAPPING


def test_dpo_collator_preserves_tokens_labels_and_mrope_with_cp_padding(tokenizer):
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from torchturbo.adapters.msswift.inputs_bridging import SwiftInputCollator

    template = _template(tokenizer, padding_free=True)
    template.set_mode("rlhf")
    # PackingDataset sets this flag before the trainer builds its collator.
    template.packing = True
    model = SimpleNamespace(config=tokenizer.model_info.config)
    model.get_rope_index = Qwen3_5Model.get_rope_index.__get__(model)
    template.model = model
    samples = [template.encode({"messages": _conversation(), "rejected_response": answer})
               for answer in ("bad-answer", "another-bad-answer")]

    packed = SwiftInputCollator(template.data_collator, template)([deepcopy(samples)])
    template.use_megatron = True
    template.sequence_parallel_size = 4
    cp_padded = template.data_collator([deepcopy(samples)], padding_to=2048)
    assert cp_padded["input_ids"].shape[-1] % 2048 == 0
    assert cp_padded["input_ids"].shape[-1] > packed["input_ids"].shape[-1]
    sequences = [
        (sample[f"{prefix}input_ids"], sample[f"{prefix}labels"])
        for prefix in ("chosen_", "rejected_")
        for sample in samples
    ]
    for batch in (packed, cp_padded):
        positions = batch["text_position_ids"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        assert len(starts) >= len(sequences)
        for start, (input_ids, labels) in zip(starts, sequences):
            end = start + len(input_ids)
            assert batch["input_ids"][0, start:end].tolist() == input_ids
            assert batch["labels"][0, start:end].tolist() == labels
            expected_positions = torch.arange(len(input_ids))
            assert torch.equal(positions[start:end], expected_positions)
            assert torch.equal(batch["position_ids"][:, 0, start:end], expected_positions.expand(3, -1))


@pytest.mark.parametrize("tool_role", ["tool", "tool_response"])
@pytest.mark.parametrize("rejected_response", [
    "bad-answer",
    [{"role": "assistant", "content": "bad-answer"}],
])
def test_rejected_response_replaces_the_whole_trajectory(tokenizer, tool_role, rejected_response):
    messages = _conversation()
    for message in messages:
        if message["role"] == "tool":
            message["role"] = tool_role
    template = _template(tokenizer)
    template.set_mode("rlhf")
    inputs = {"messages": messages, "rejected_response": rejected_response}
    original = deepcopy(inputs)
    encoded = template.encode(inputs)
    chosen = tokenizer.decode(encoded["chosen_input_ids"])
    rejected = tokenizer.decode(encoded["rejected_input_ids"])
    assert "new-call" in chosen and "new-observation" in chosen
    assert "new-call" not in rejected and "new-observation" not in rejected
    assert "old-call" in rejected and "old-observation" in rejected
    prompt_end = "new-query<|im_end|>\n"
    assert prompt_end in chosen and prompt_end in rejected
    assert chosen.split(prompt_end)[0] == rejected.split(prompt_end)[0]
    assert "bad-answer" in rejected
    assert "new-answer" not in rejected
    for prefix, answer in (("chosen", "new-answer"), ("rejected", "bad-answer")):
        targets = tokenizer.decode([token for token in encoded[f"{prefix}_labels"] if token != -100])
        assert targets == f"{answer}<|im_end|>\n"
    assert inputs == original


@pytest.mark.parametrize(("rejected_response", "error"), [
    (0, "must be a string or a list"),
    ([{"role": "user", "content": "query"}], "must not contain user messages"),
    ("new-answer", "chosen and rejected responses must differ"),
])
def test_rejected_response_validation(tokenizer, rejected_response, error):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    with pytest.raises(ValueError, match=error):
        template.encode({
            "messages": [{"role": "user", "content": "query"}, {"role": "assistant", "content": "new-answer"}],
            "rejected_response": rejected_response,
        })


def _tool_trajectory(prefix, *, tool_role="tool", openai=False, loss=None, content=None):
    calls = [
        {"id": f"call-{name}", "type": "function", "function": {
            "name": name, "arguments": {"flag": True, "empty": None, "items": [False]},
        }}
        for name in ("first", "second")
    ]
    if openai:
        for call in calls:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        messages = [{"role": "assistant", "tool_calls": calls}]
        if content != "missing":
            messages[0]["content"] = content
    else:
        messages = [{"role": "tool_call", "content": call["function"]} for call in calls]
        if content and content != "missing":
            messages.insert(0, {"role": "assistant", "content": content})
    if loss is not None:
        for message in messages:
            message["loss"] = loss
    results = [
        {"role": tool_role, "tool_call_id": f"call-{name}", "content": f"{prefix}-{name}-result"}
        for name in ("first", "second")
    ]
    # OpenAI result messages may arrive in a different order from the calls.
    messages.extend(reversed(results) if openai else results)
    messages.append({"role": "assistant", "content": f"{prefix}-answer"})
    return messages


@pytest.mark.parametrize("tool_role", ["tool", "tool_response"])
@pytest.mark.parametrize("field", ["rejected_response", "rejected_messages"])
@pytest.mark.parametrize("loss", [None, True, False])
@pytest.mark.parametrize("content", [None, "missing", "Working on it."])
def test_dpo_openai_calls_match_canonical_trajectories(tokenizer, tool_role, field, loss, content):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    prompt = [{"role": "system", "content": "shared-system"}, {"role": "user", "content": "shared-query"}]
    samples = []
    for openai in (False, True):
        rejected = _tool_trajectory("bad", tool_role=tool_role, openai=openai, loss=loss, content=content)
        inputs = {
            "messages": prompt + _tool_trajectory("good", tool_role=tool_role, openai=openai, loss=loss, content=content),
            field: prompt + rejected if field == "rejected_messages" else rejected,
        }
        original = deepcopy(inputs)
        samples.append(template.encode(inputs))
        assert inputs == original
    for prefix in ("chosen", "rejected"):
        for key in ("input_ids", "labels"):
            assert samples[0][f"{prefix}_{key}"] == samples[1][f"{prefix}_{key}"]
        text = tokenizer.decode(samples[1][f"{prefix}_input_ids"])
        answer_prefix = "good" if prefix == "chosen" else "bad"
        assert text.index(f"{answer_prefix}-first-result") < text.index(f"{answer_prefix}-second-result")
        targets = tokenizer.decode([token for token in samples[1][f"{prefix}_labels"] if token != -100])
        assert ("<tool_call>" in targets) is (loss is True)
        assert "-result" not in targets
        assert f"{answer_prefix}-answer" in targets


@pytest.mark.parametrize("tool_role", ["tool", "tool_response"])
@pytest.mark.parametrize("field", ["rejected_response", "rejected_messages", "template_inputs"])
def test_dpo_trajectory_input_forms_and_pickle(tokenizer, tool_role, field):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    prompt = [{"role": "user", "content": "shared-query"}]
    chosen = prompt + _tool_trajectory("good", tool_role=tool_role)
    rejected = _tool_trajectory("bad", tool_role=tool_role)
    expected = template.encode({"messages": chosen, "rejected_response": rejected})
    if field == "template_inputs":
        inputs = TemplateInputs(chosen={"messages": chosen}, rejected={"messages": prompt + rejected})
    else:
        inputs = {"messages": chosen, field: prompt + rejected if field == "rejected_messages" else rejected}
    assert template.encode(inputs) == expected
    assert pickle.loads(pickle.dumps(template)).encode(inputs) == expected


@pytest.mark.parametrize("tool_role", ["tool", "tool_response"])
def test_dpo_standalone_tool_results(tokenizer, tool_role):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    prompt = [{"role": "user", "content": "shared-query"}]
    chosen = [
        {"role": tool_role, "content": "good-result"},
        {"role": "assistant", "content": "good-answer"},
    ]
    rejected = [
        {"role": tool_role, "content": "bad-result"},
        {"role": "assistant", "content": "bad-answer"},
    ]
    encoded = template.encode({"messages": prompt + chosen, "rejected_response": rejected})
    for prefix, name in (("chosen", "good"), ("rejected", "bad")):
        text = tokenizer.decode(encoded[f"{prefix}_input_ids"])
        assert f"<tool_response>\n{name}-result\n</tool_response>" in text
        targets = tokenizer.decode([token for token in encoded[f"{prefix}_labels"] if token != -100])
        assert targets == f"{name}-answer<|im_end|>\n"


def test_dpo_observation_aliases_from_dataset_preprocessing(tokenizer):
    from swift.dataset.preprocessor import MessagesPreprocessor

    template = _template(tokenizer)
    template.set_mode("rlhf")
    prompt = [{"role": "user", "content": "shared-query"}]
    inputs = {
        "messages": prompt + _tool_trajectory("good", tool_role="observation"),
        "rejected_messages": prompt + _tool_trajectory("bad", tool_role="function_response"),
    }
    row = MessagesPreprocessor().preprocess(deepcopy(inputs))
    encoded = template.encode(row)
    expected = template.encode({
        "messages": prompt + _tool_trajectory("good"),
        "rejected_messages": prompt + _tool_trajectory("bad"),
    })
    assert encoded == expected


def test_dpo_shared_metadata_and_final_answer_after_tool(tokenizer):
    template = _template(tokenizer, enable_thinking=True)
    template.set_mode("rlhf")
    inputs = {
        "messages": _conversation(),
        "rejected_response": "new-answer",
        "margin": 0.25,
        "channel": "tool-dpo",
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    # Equal final strings still describe distinct trajectories when one branch
    # contains tool calls/results and the other directly answers the last user.
    encoded = template.encode(inputs)
    assert encoded["margin"] == 0.25 and encoded["channel"] == "tool-dpo"
    for prefix in ("chosen", "rejected"):
        assert "Reasoning effort is set to low." in tokenizer.decode(encoded[f"{prefix}_input_ids"])
    assert encoded["chosen_input_ids"] != encoded["rejected_input_ids"]


def test_dpo_identical_openai_tool_calls_are_rejected(tokenizer):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    calls = _tool_trajectory("good", openai=True)[0]
    calls.pop("content")
    rejected = deepcopy(calls)
    for call in rejected["tool_calls"]:
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    with pytest.raises(ValueError, match="chosen and rejected responses must differ"):
        template.encode({
            "messages": [{"role": "user", "content": "query"}, calls],
            "rejected_response": [rejected],
        })


def test_dpo_openai_calls_preserve_explicit_loss_scale(tokenizer):
    template = _template(tokenizer)
    template.set_mode("rlhf")
    samples = []
    for openai in (False, True):
        chosen = _tool_trajectory("good", openai=openai, loss=True)
        rejected = _tool_trajectory("bad", openai=openai, loss=True)
        for message in chosen + rejected:
            if "loss" in message:
                message["loss_scale"] = 0.0
        samples.append(template.encode({
            "messages": [{"role": "user", "content": "query"}] + chosen,
            "rejected_response": rejected,
        }))
    assert samples[0] == samples[1]
    for prefix in ("chosen", "rejected"):
        targets = tokenizer.decode([token for token in samples[1][f"{prefix}_labels"] if token != -100])
        assert "<tool_call>" not in targets


def test_dpo_left_truncation_keeps_each_branch_labels_aligned(tokenizer):
    inputs = {"messages": _conversation(), "rejected_response": _tool_trajectory("bad")}
    complete = _template(tokenizer)
    truncated = _template(tokenizer, max_length=128, truncation_strategy="left")
    complete.set_mode("rlhf")
    truncated.set_mode("rlhf")
    expected = complete.encode(inputs)
    encoded = truncated.encode(inputs)
    for prefix in ("chosen", "rejected"):
        for key in ("input_ids", "labels"):
            assert len(encoded[f"{prefix}_{key}"]) == 128
            assert encoded[f"{prefix}_{key}"] == expected[f"{prefix}_{key}"][-128:]


def test_tool_json_literals_do_not_change_qwen35(tokenizer):
    messages = [
        {"role": "user", "content": "query"},
        {"role": "tool_call", "content": {"name": "lookup", "arguments": {
            "yes": True, "no": False, "empty": None, "literal": "True",
            "nested": {"flag": True}, "items": [None, False],
        }}},
        {"role": "tool_response", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]
    text, _ = _texts(_template(tokenizer), messages)
    for name, value in (("yes", "true"), ("no", "false"), ("empty", "null"), ("literal", "True")):
        assert f"<parameter={name}>\n{value}\n</parameter>" in text
    assert '{"flag": true}' in text and '[null, false]' in text
    native, _ = _texts(_template(tokenizer, "qwen3_5"), messages)
    assert "<parameter=yes>\nTrue\n</parameter>" in native
    assert "<parameter=empty>\nNone\n</parameter>" in native
