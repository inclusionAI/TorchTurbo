"""Optional Swift template tests using checkpoint assets without model weights.

Set BAILING_VL_REFERENCE to the reference checkpoint directory. These tests
exercise the Ling SFT chat protocol, media expansion, and the Swift batch collator.
"""

import json
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from torchturbo.models.bailing_moe_v3_vl.model import (
    TurboBailingMoeV3VLForConditionalGeneration,
)

from .test_model import make_config


@pytest.fixture(scope="module")
def processor():
    reference = os.environ.get("BAILING_VL_REFERENCE")
    if reference is None:
        pytest.skip("Set BAILING_VL_REFERENCE to run native Swift integration tests.")
    pytest.importorskip("swift")
    from swift.model.model_meta import ModelInfo, ModelMeta
    from transformers import AutoProcessor

    from torchturbo.utils.transformers_utils import patch_dynamic_module_loading

    # Launchers install this through torchturbo.activate() before loading.
    patch_dynamic_module_loading()
    processor = AutoProcessor.from_pretrained(
        reference, trust_remote_code=True, local_files_only=True
    )
    processor.model_info = ModelInfo(
        model_type="turbo_bailing_moe_v3_vl",
        model_dir=reference,
        torch_dtype=torch.float32,
        max_model_len=32768,
        quant_method=None,
        quant_bits=None,
        task_type="causal_lm",
    )
    processor.model_meta = ModelMeta(
        model_type="turbo_bailing_moe_v3_vl", model_groups=[], is_multimodal=True
    )
    return processor


def _build_template(processor, template_type):
    from torchturbo.models.bailing_moe_v3_vl.swift_template import (
        BailingVLTemplate,
        BailingVLOrderedTemplate,
        _make_template_meta,
        _register_agent_template,
    )

    template_cls = (
        BailingVLTemplate if template_type == "bailing_moe_v3_vl" else BailingVLOrderedTemplate
    )
    _register_agent_template()
    template = template_cls(
        processor,
        _make_template_meta(template_type, template_cls),
        loss_scale="ignore_empty_think",
    )
    template.set_mode("train")
    return template


@pytest.fixture(
    params=["bailing_moe_v3_vl", "bailing_moe_v3_vl_agent"],
    ids=["pairing", "ordered"],
)
def template(processor, monkeypatch, request):
    monkeypatch.delenv("LING_SFT_AUTO_THINKING", raising=False)
    return _build_template(processor, request.param)


@pytest.fixture
def ordered_template(processor, monkeypatch):
    monkeypatch.delenv("LING_SFT_AUTO_THINKING", raising=False)
    return _build_template(processor, "bailing_moe_v3_vl_agent")


@pytest.fixture
def video(tmp_path):
    frames = []
    for index in range(4):
        frame = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (64, 64), (index * 30, 0, 0)).save(frame)
        frames.append(str(frame))
    return "frames_list://" + json.dumps(frames)


def test_template_dill_roundtrip(template):
    import dill

    restored = dill.loads(dill.dumps(template))

    assert type(restored) is type(template)
    assert type(restored.processor) is type(template.processor)
    assert type(restored.tokenizer) is type(template.tokenizer)
    assert type(restored.processor.image_processor) is type(
        template.processor.image_processor
    )
    assert restored.auto_sft_thinking == template.auto_sft_thinking


def test_auto_thinking_environment_is_resolved_once(template, monkeypatch):
    from torchturbo.models.bailing_moe_v3_vl.swift_template import _make_template_meta

    def fresh_template():
        meta = _make_template_meta(template.template_meta.template_type, type(template))
        return type(template)(template.processor, meta)

    monkeypatch.setenv("LING_SFT_AUTO_THINKING", "1")
    enabled = fresh_template()
    monkeypatch.setenv("LING_SFT_AUTO_THINKING", "0")

    assert enabled.auto_sft_thinking
    assert not template.auto_sft_thinking
    monkeypatch.setenv("LING_SFT_AUTO_THINKING", "invalid")
    with pytest.raises(ValueError, match="LING_SFT_AUTO_THINKING"):
        fresh_template()


@pytest.mark.parametrize(
    "answer,thinking,target",
    [
        ("answer", False, "answer<|role_end|>"),
        ("<think></think>answer", False, "answer<|role_end|>"),
        (
            "<think>reasoning</think>answer",
            True,
            "<think>reasoning</think>answer<|role_end|>",
        ),
    ],
)
def test_auto_thinking_matches_sample_protocol(template, answer, thinking, target):
    template.auto_sft_thinking = True
    template.enable_thinking = not thinking
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": answer},
    ]

    encoded = template.encode({"messages": messages})
    expected = template.tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=thinking, preserve_thinking=True
    )

    assert template.tokenizer.decode(encoded["input_ids"]) == expected
    targets = [token for token in encoded["labels"] if token != -100]
    assert template.tokenizer.decode(targets) == target
    assert template.enable_thinking is not thinking


@pytest.mark.parametrize(
    "answer,explicit,system,thinking",
    [
        ("answer", True, None, True),
        ("<think>reasoning</think>answer", False, None, False),
        ("answer", None, "detailed thinking on", True),
        ("<think>reasoning</think>answer", None, "detailed thinking off", False),
        ("answer", True, "detailed thinking on", True),
        ("answer", False, "detailed thinking off", False),
    ],
)
def test_auto_thinking_honors_explicit_mode(
    template, answer, explicit, system, thinking
):
    template.auto_sft_thinking = True
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.extend(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ]
    )
    kwargs = {} if explicit is None else {"enable_thinking": explicit}

    encoded = template.encode({"messages": messages, "chat_template_kwargs": kwargs})
    expected = template.tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=thinking, preserve_thinking=True
    )

    assert template.tokenizer.decode(encoded["input_ids"]) == expected


@pytest.mark.parametrize(
    "explicit,system,error",
    [
        ("false", None, "JSON boolean"),
        (True, "detailed thinking off", "Conflicting thinking modes"),
        (False, "detailed thinking on", "Conflicting thinking modes"),
        (None, "detailed thinking on; detailed thinking off", "Conflicting system"),
    ],
)
def test_auto_thinking_rejects_conflicting_modes(template, explicit, system, error):
    template.auto_sft_thinking = True
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.extend(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    kwargs = {} if explicit is None else {"enable_thinking": explicit}

    with pytest.raises(ValueError, match=error):
        template.encode({"messages": messages, "chat_template_kwargs": kwargs})


def test_auto_thinking_does_not_leak_between_samples(template):
    template.auto_sft_thinking = True
    template.enable_thinking = False
    for answer, mode in [
        ("<think>reasoning</think>answer", "on"),
        ("answer", "off"),
    ]:
        encoded = template.encode(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        assert f"detailed thinking {mode}" in template.tokenizer.decode(
            encoded["input_ids"]
        )
    assert not template.enable_thinking


def test_thinking_directive_reuses_trailing_system_newline(template):
    encoded = template.encode(
        {
            "messages": [
                {"role": "system", "content": "custom system\n"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )

    text = template.tokenizer.decode(encoded["input_ids"])
    assert "custom system\ndetailed thinking off" in text
    assert "custom system\n\ndetailed thinking off" not in text


def test_auto_thinking_considers_earlier_assistant_turns(template):
    template.auto_sft_thinking = True
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<think>reasoning</think>first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    encoded = template.encode({"messages": messages})
    expected = template.tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=True, preserve_thinking=True
    )

    assert template.tokenizer.decode(encoded["input_ids"]) == expected


def test_auto_thinking_does_not_infer_generation_mode(template):
    template.auto_sft_thinking = True
    template.enable_thinking = True
    template.set_mode("transformers")

    encoded = template.encode({"messages": [{"role": "user", "content": "question"}]})

    assert "detailed thinking on" in template.tokenizer.decode(encoded["input_ids"])


def test_reasoning_content_field_does_not_change_baseline_encoding(template):
    template.auto_sft_thinking = True
    sample = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    expected = template.encode(deepcopy(sample))
    sample["messages"][-1]["reasoning_content"] = "separate reasoning"

    actual = template.encode(sample)

    assert actual["input_ids"] == expected["input_ids"]
    assert actual["labels"] == expected["labels"]


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
@pytest.mark.parametrize("media_type", ["image", "video"])
def test_multiprocess_dataset_encoding(
    template, video, tmp_path, monkeypatch, media_type, start_method
):
    import multiprocess
    from datasets import Dataset, arrow_dataset
    from swift.dataset import EncodePreprocessor

    if start_method not in multiprocess.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable on this platform.")
    pool = multiprocess.get_context(start_method).Pool
    # datasets 4.0 imports Pool by name; newer releases call multiprocess.Pool.
    monkeypatch.setattr(arrow_dataset, "Pool", pool, raising=False)
    monkeypatch.setattr(multiprocess, "Pool", pool)
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64)).save(image)
    media = str(image) if media_type == "image" else video
    sample = {
        "messages": [
            {"role": "user", "content": f"<{media_type}>describe"},
            {"role": "assistant", "content": "answer"},
        ],
        f"{media_type}s": [media],
    }
    dataset = Dataset.from_list([sample, sample])
    preprocessor = EncodePreprocessor(template)
    kwargs = dict(
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={"strict": True, "ignore_max_length_error": False},
    )

    expected = dataset.map(preprocessor.batched_preprocess, **kwargs)
    actual = dataset.map(preprocessor.batched_preprocess, num_proc=2, **kwargs)

    assert len(actual) == len(dataset)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [None, False, True])
@pytest.mark.parametrize("training", [False, True])
def test_chat_template_matches_native(template, thinking, preserve, training):
    template.enable_thinking = thinking
    template.preserve_thinking = preserve
    template.set_mode("train" if training else "transformers")
    messages = [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "content": "<think>\nfirst reasoning\n</think>first answer",
        },
        {"role": "user", "content": "second question"},
        # Swift preserves history verbatim during inference with preserve=True.
        {"role": "assistant", "content": "<think></think>second answer"},
        {"role": "user", "content": "third question"},
    ]
    if training:
        messages.append(
            {"role": "assistant", "content": "<think>\nreasoning\n</think>\nanswer"}
        )
    encoded = template.encode({"messages": messages})
    expected = template.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=thinking,
        preserve_thinking=training if preserve is None else preserve,
        add_generation_prompt=not training,
    )
    assert template.tokenizer.decode(encoded["input_ids"]) == expected
    if training:
        targets = [token for token in encoded["labels"] if token != -100]
        assert template.tokenizer.decode(targets).endswith("answer<|role_end|>")


@pytest.mark.parametrize("num_calls", [1, 2])
@pytest.mark.parametrize("content", ["", "answer", "<think>reasoning</think>"])
def test_tool_round_matches_native(template, content, num_calls):
    calls = [
        {"name": "lookup", "arguments": {"key": "value", "count": 2}},
        {"name": "search", "arguments": {"query": "<think>literal</think>"}},
    ][:num_calls]
    messages = [{"role": "user", "content": "lookup"}]
    if content:
        messages.append({"role": "assistant", "content": content})
    messages.extend(
        {"role": "tool_call", "content": json.dumps(call)} for call in calls
    )
    messages.extend(
        {"role": "tool", "content": f"result {i}"} for i in range(num_calls)
    )
    messages.append({"role": "assistant", "content": "final answer"})
    native_messages = [
        {"role": "user", "content": "lookup"},
        {"role": "assistant", "content": content, "tool_calls": calls},
        *({"role": "tool", "content": f"result {i}"} for i in range(num_calls)),
        {"role": "assistant", "content": "final answer"},
    ]
    encoded = template.encode({"messages": messages})
    expected = template.tokenizer.apply_chat_template(
        native_messages, tokenize=False, enable_thinking=False
    )
    assert template.tokenizer.decode(encoded["input_ids"]) == expected


@pytest.mark.parametrize("num_observations", [1, 2])
def test_standalone_tool_observations_match_native(template, num_observations):
    messages = [
        {"role": "user", "content": "Read the observations."},
        *(
            {"role": "tool", "content": f"observation {index}"}
            for index in range(num_observations)
        ),
        {"role": "assistant", "content": "final answer"},
    ]

    encoded = template.encode({"messages": messages})
    expected_ids = template.tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, enable_thinking=False
    )

    assert encoded["input_ids"] == expected_ids
    targets = template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert targets == "final answer<|role_end|>"
    assert "observation" not in targets


def test_auto_thinking_keeps_reasoning_across_tool_steps(template):
    template.auto_sft_thinking = True
    template.preserve_thinking = False
    call = {"name": "lookup", "arguments": {"key": "value"}}
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "<think>old reasoning</think>old answer"},
        {"role": "user", "content": "lookup"},
        {"role": "assistant", "content": "<think>first step</think>"},
        {"role": "tool_call", "content": json.dumps(call)},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "<think>second step</think>answer"},
    ]
    native_messages = [
        *messages[:3],
        {
            "role": "assistant",
            "content": "<think>first step</think>",
            "tool_calls": [call],
        },
        *messages[5:],
    ]

    encoded = template.encode({"messages": messages})
    expected = template.tokenizer.apply_chat_template(
        native_messages,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=False,
    )

    assert template.tokenizer.decode(encoded["input_ids"]) == expected
    targets = template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert "old reasoning" not in targets
    assert "first step" in targets
    assert "second step" in targets
    assert "<tool_call>lookup\n<arg_key>key</arg_key>" in targets
    assert "<tool_response>" not in targets


def test_multi_round_agent_conversation_matches_native(ordered_template):
    ordered_template.auto_sft_thinking = True
    ordered_template.preserve_thinking = True
    first_call = {"name": "lookup", "arguments": {"key": "value"}}
    second_call = {"name": "search", "arguments": {"query": "first result"}}
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>first step</think>"},
        {"role": "tool_call", "content": json.dumps(first_call)},
        {"role": "tool", "content": "first result"},
        {"role": "assistant", "content": "<think>second step</think>"},
        {"role": "tool_call", "content": json.dumps(second_call)},
        {"role": "tool", "content": "second result"},
        {"role": "assistant", "content": "<think>final step</think>answer"},
    ]
    native_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "<think>first step</think>",
            "tool_calls": [first_call],
        },
        {"role": "tool", "content": "first result"},
        {
            "role": "assistant",
            "content": "<think>second step</think>",
            "tool_calls": [second_call],
        },
        {"role": "tool", "content": "second result"},
        {"role": "assistant", "content": "<think>final step</think>answer"},
    ]

    encoded = ordered_template.encode({"messages": messages})
    expected = ordered_template.tokenizer.apply_chat_template(
        native_messages, tokenize=False, enable_thinking=True, preserve_thinking=True
    )

    assert ordered_template.tokenizer.decode(encoded["input_ids"]) == expected
    targets = ordered_template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert "first step" in targets
    assert "second step" in targets
    assert "<tool_call>lookup\n<arg_key>key</arg_key>" in targets
    assert "<tool_call>search" in targets
    assert targets.endswith("final step</think>answer<|role_end|>")
    assert "<tool_response>" not in targets
    assert "<role>OBSERVATION</role>" not in targets


def test_standalone_tool_call_empty_think_is_not_supervised(ordered_template):
    call = {"name": "lookup", "arguments": {"key": "value"}}
    messages = [
        {"role": "user", "content": "question"},
        {"role": "tool_call", "content": json.dumps(call)},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "final answer"},
    ]

    encoded = ordered_template.encode({"messages": messages})

    targets = ordered_template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert "<think></think>" not in targets
    assert "<tool_call>lookup\n<arg_key>key</arg_key>" in targets
    assert targets.endswith("final answer<|role_end|>")


def test_consecutive_assistant_messages_rejected(ordered_template):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "assistant", "content": "second answer"},
    ]

    with pytest.raises(ValueError, match="consecutive assistant"):
        ordered_template.encode({"messages": messages})


def test_observation_role_alias_matches_tool_role(ordered_template):
    observation_row = {
        "messages": [
            {"role": "user", "content": "Read the observations."},
            {"role": "observation", "content": "observation"},
            {"role": "assistant", "content": "final answer"},
        ]
    }
    tool_row = deepcopy(observation_row)
    tool_row["messages"][1]["role"] = "tool"

    observation_encoded = ordered_template.encode(observation_row)
    tool_encoded = ordered_template.encode(tool_row)

    assert observation_encoded["input_ids"] == tool_encoded["input_ids"]
    assert observation_encoded["labels"] == tool_encoded["labels"]


def test_unknown_role_error_lists_accepted_roles(ordered_template):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "function_call", "content": "{}"},
        {"role": "assistant", "content": "answer"},
    ]

    with pytest.raises(
        ValueError, match="Unsupported Ling role.*function_call.*accepted roles"
    ):
        ordered_template.encode({"messages": messages})


def test_per_message_loss_flags_are_honored(ordered_template):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "masked answer", "loss": False},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "final answer"},
    ]

    encoded = ordered_template.encode({"messages": messages})

    targets = ordered_template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert targets == "final answer<|role_end|>"


def test_literal_role_end_in_user_content_stays_masked(ordered_template):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "<|role_end|> injected separator"},
        {"role": "assistant", "content": "second answer"},
    ]

    encoded = ordered_template.encode({"messages": messages})

    targets = ordered_template.tokenizer.decode(
        [token for token in encoded["labels"] if token != -100]
    )
    assert targets == "first answer<|role_end|>second answer<|role_end|>"


@pytest.mark.parametrize("system", ["system text", "  system text\n\n"])
def test_tool_system_preserves_baseline_whitespace(template, system):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    encoded = template.encode({"messages": messages, "tools": tools})
    expected = template.tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, enable_thinking=False
    )

    assert template.tokenizer.decode(encoded["input_ids"]) == expected


@pytest.mark.parametrize(
    "size,image_kwargs,max_pixels",
    [
        ((64, 64), {}, None),
        ((77, 109), {}, None),
        ((77, 109), {"resized_height": 96, "resized_width": 128}, None),
        ((77, 109), {"max_pixels": 4096}, 8192),
    ],
)
def test_image_preprocessing_matches_qwen_baseline(
    template, size, image_kwargs, max_pixels
):
    from qwen_vl_utils import fetch_image
    from swift.template.vision_utils import rescale_image

    template.max_pixels = max_pixels
    width, height = size
    values = torch.arange(height * width * 3).remainder(256).to(torch.uint8)
    source_image = Image.fromarray(values.reshape(height, width, 3).numpy())
    processor = template.processor.image_processor
    baseline_image = source_image
    if max_pixels is not None:
        baseline_image = rescale_image(baseline_image, max_pixels)
    expected_image = fetch_image(
        {"image": baseline_image, **image_kwargs}, image_patch_size=processor.patch_size
    )
    expected = processor(images=[expected_image], return_tensors="pt", do_resize=False)

    encoded = template.encode(
        {
            "messages": [
                {"role": "user", "content": "<image>describe"},
                {"role": "assistant", "content": "answer"},
            ],
            "images": [source_image],
            "chat_template_kwargs": image_kwargs,
        }
    )

    torch.testing.assert_close(encoded["image_grid_thw"], expected["image_grid_thw"])
    torch.testing.assert_close(encoded["pixel_values"], expected["pixel_values"])
    text = template.tokenizer.decode(encoded["input_ids"])
    assert "<|vision_end|>describe" in text
    assert "<|vision_end|>\ndescribe" not in text
    num_tokens = int(expected["image_grid_thw"].prod(-1).sum()) // processor.merge_size**2
    assert encoded["input_ids"].count(template.processor.image_token_id) == num_tokens


@pytest.mark.parametrize("packing", [False, True])
def test_mixed_media_collator(template, video, packing):
    samples = [
        {"content": "text"},
        {"content": "<image>describe", "images": [Image.new("RGB", (64, 64))]},
        {"content": "<video>describe", "videos": [video]},
        {
            "content": "<video>compare<image>",
            "images": [Image.new("RGB", (96, 64))],
            "videos": [video],
        },
    ]
    encoded = []
    for sample in samples:
        content = sample.pop("content")
        sample["messages"] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "answer"},
        ]
        encoded.append(template.encode(sample, return_length=True))
    assert template.support_padding_free
    template.padding_free = packing
    batch = template.data_collator(deepcopy(encoded))
    for media_type, pixel_key in [
        ("image", "pixel_values"),
        ("video", "pixel_values_videos"),
    ]:
        grid_key = f"{media_type}_grid_thw"
        expected_grids = torch.cat([row[grid_key] for row in encoded if grid_key in row])
        expected_pixels = torch.cat([row[pixel_key] for row in encoded if pixel_key in row])
        torch.testing.assert_close(batch[grid_key], expected_grids)
        torch.testing.assert_close(batch[pixel_key], expected_pixels)
        token_id = (
            template.processor.image_token_id
            if media_type == "image"
            else template.processor.video_token_id
        )
        merge_size = template.processor.image_processor.merge_size
        num_tokens = expected_grids.prod(-1).sum() // merge_size**2
        assert batch["input_ids"].eq(token_id).sum() == num_tokens
    if packing:
        expected_ids = sum((row["input_ids"] for row in encoded), [])
        expected_positions = sum((list(range(row["length"])) for row in encoded), [])
        assert batch["input_ids"].tolist() == [expected_ids]
        assert batch["position_ids"].tolist() == [expected_positions]
    else:
        assert batch["attention_mask"].sum(-1).tolist() == [
            row["length"] for row in encoded
        ]


def test_packed_samples_keep_boundaries_and_supervision(template):
    from torchturbo.adapters.msswift.inputs_bridging import SwiftInputCollator

    template.auto_sft_thinking = True
    template.padding_free = True
    rows = []
    for answer in ["answer", "<think>reasoning</think>answer"]:
        rows.append(
            template.encode(
                {
                    "messages": [
                        {"role": "user", "content": "<image>question"},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [Image.new("RGB", (64, 64))],
                },
                return_length=True,
            )
        )
    lengths = [len(row["input_ids"]) for row in rows]
    for row in rows:
        row["labels"][0] = row["input_ids"][0]
    collator = SwiftInputCollator(template.data_collator, template)

    batch = collator(deepcopy(rows))

    boundaries = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], boundaries)
    torch.testing.assert_close(batch["cu_seq_lens_k"], boundaries)
    assert batch["input_ids"].tolist() == [sum((row["input_ids"] for row in rows), [])]
    expected_labels = sum(([-100, *row["labels"][1:]] for row in rows), [])
    assert batch["labels"].tolist() == [expected_labels]
    assert batch["position_ids"].tolist() == [
        list(range(lengths[0])) + list(range(lengths[1]))
    ]
    text = template.tokenizer.decode(batch["input_ids"][0])
    assert text.count("detailed thinking off") == 1
    assert text.count("detailed thinking on") == 1
    assert "attention_mask" not in batch


def test_text_post_encode_adds_dummy_vision_dependency(template):
    config = make_config(mtp=1)
    config.vision_config.patch_size = template.processor.image_processor.patch_size
    model = TurboBailingMoeV3VLForConditionalGeneration(config).train()
    input_ids = torch.tensor([[2, 3, 4]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
    }

    with (
        patch.object(
            model.model.visual, "forward", wraps=model.model.visual.forward
        ) as visual,
        patch.object(
            model.linear_proj, "forward", wraps=model.linear_proj.forward
        ) as projector,
    ):
        args, model_inputs = template.pre_forward_hook(model, (), inputs)
        result = model(*args, **model_inputs)

    assert visual.call_count == 1
    assert projector.call_count == 1
    assert "input_ids" not in model_inputs
    torch.testing.assert_close(model_inputs["mtp_input_ids"], input_ids)
    assert model_inputs["inputs_embeds"].shape == (
        1,
        3,
        config.text_config.hidden_size,
    )
    result.loss.backward()
    assert model.model.visual.patch_embed.proj.weight.grad is not None
    assert model.linear_proj[0].weight.grad is not None
    assert torch.count_nonzero(model.model.visual.patch_embed.proj.weight.grad) == 0
    assert torch.count_nonzero(model.linear_proj[0].weight.grad) == 0
