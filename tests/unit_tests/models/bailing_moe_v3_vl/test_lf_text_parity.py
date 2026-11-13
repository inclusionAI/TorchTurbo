"""Compare LF and Swift SFT tokenization and supervision with real assets."""

import json
from copy import deepcopy
from itertools import product

import pytest

from .test_swift_integration import processor


_CALL = {"name": "lookup", "arguments": {"query": "value", "count": 2}}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


def _conversation(answer):
    return [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": answer},
    ]


def _cases():
    cases = []

    def add(name, messages, system=None, tools=None, swift_messages=None):
        cases.append(
            pytest.param(
                messages,
                messages if swift_messages is None else swift_messages,
                system,
                tools,
                id=name,
            )
        )

    answers = {
        "plain": "answer",
        "empty": "",
        "thinking": "<think>\n reasoning \n</think>\n answer\t",
        "empty-thinking": "<think></think>answer",
        "empty-thinking-whitespace": "<think>\n\n</think>\n answer",
        "leading-whitespace": " \n<think>reason</think>answer",
        "answer-whitespace": "\n answer\t ",
        "thinking-only": "<think>reason</think>",
    }
    for name, answer in answers.items():
        add(name, _conversation(answer))
    for name, system in {
        "empty": "",
        "plain": "system text",
        "newline": "system text\n",
        "newlines": "  system text\n\n",
        "on": "detailed thinking on",
        "off": "detailed thinking off",
    }.items():
        add("system-" + name, _conversation("answer"), system=system)
    for name in [
        "plain",
        "thinking",
        "leading-whitespace",
        "empty-thinking-whitespace",
        "thinking-only",
    ]:
        messages = _conversation(answers[name]) + _conversation("final answer")
        add("history-" + name, messages)
    add(
        "history-wrapped-observation",
        [
            *_conversation("<think>reason</think>answer"),
            {"role": "user", "content": "<tool_response>result</tool_response>"},
            {"role": "assistant", "content": "final answer"},
        ],
    )
    for name, thought in {
        "plain": "",
        "thinking": "<think>reason</think>",
        "whitespace": "<think>\nreason\n</think>\n",
    }.items():
        for multiple in [False, True]:
            calls = [_CALL]
            if multiple:
                calls += [
                    {
                        "name": "lookup",
                        "arguments": {"query": "<think>literal</think>"},
                    }
                ]
            messages = [
                {"role": "user", "content": "question"},
                {"role": "function", "content": thought + json.dumps(calls)},
                {"role": "observation", "content": "result"},
                {"role": "assistant", "content": "answer"},
            ]
            swift_messages = [messages[0]]
            if thought:
                swift_messages.append({"role": "assistant", "content": thought})
            swift_messages.extend(
                {"role": "tool_call", "content": json.dumps(call)} for call in calls
            )
            swift_messages += [
                {"role": "tool", "content": "result"},
                messages[-1],
            ]
            if multiple:
                messages += _conversation("final answer")
                swift_messages += _conversation("final answer")
            suffix = "multiple-history" if multiple else "single"
            add(
                f"tool-{name}-{suffix}",
                messages,
                system="system text\n\n",
                tools=_TOOLS,
                swift_messages=swift_messages,
            )
    return cases


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            (auto, enable, preserve),
            id=f"auto-{auto}-thinking-{enable}-preserve-{preserve}",
        )
        for auto, enable, preserve in product([False, True], repeat=3)
    ],
)
def encoders(request, processor):
    pytest.importorskip("llamafactory")
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import TEMPLATES
    from llamafactory.hparams import DataArguments

    from torchturbo.models.bailing_moe_v3_vl.lf_template import register
    from torchturbo.models.bailing_moe_v3_vl.swift_template import (
        BailingVLTemplate,
        _make_template_meta,
        _register_agent_template,
    )

    auto, enable, preserve = request.param
    register()
    template = deepcopy(TEMPLATES["bailing_moe_v3_vl"])
    template.auto_sft_thinking = auto
    template.enable_thinking = enable
    template.preserve_thinking = preserve
    data_args = DataArguments(template="bailing_moe_v3_vl", cutoff_len=32768)
    lf = SupervisedDatasetProcessor(template, processor.tokenizer, processor, data_args)
    _register_agent_template()
    swift = BailingVLTemplate(
        processor,
        _make_template_meta(),
        loss_scale="ignore_empty_think",
        enable_thinking=enable,
    )
    swift.set_mode("train")
    swift.auto_sft_thinking = auto
    return lf, swift, preserve


@pytest.mark.parametrize("messages,swift_messages,system,tools", _cases())
def test_sft_tokens_and_labels_match(encoders, messages, swift_messages, system, tools):
    lf, swift, preserve = encoders
    input_ids, labels = lf._encode_data_example(
        deepcopy(messages[:-1]),
        deepcopy(messages[-1:]),
        system,
        json.dumps(tools) if tools is not None else None,
        [],
        [],
        [],
    )
    expected_messages = deepcopy(swift_messages)
    if system is not None:
        expected_messages.insert(0, {"role": "system", "content": system})
    sample = {
        "messages": expected_messages,
        "chat_template_kwargs": {"preserve_thinking": preserve},
    }
    if tools is not None:
        sample["tools"] = deepcopy(tools)
    expected = swift.encode(sample)

    assert input_ids == expected["input_ids"]
    assert labels == expected["labels"]
