"""Optional integration tests using checkpoint tokenizer and processor assets.

Set BAILING_VL_REFERENCE to the reference checkpoint directory. No weights are
needed; these tests exercise the real pinned LF collator and native processor.
"""

import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from torchturbo.models.bailing_moe_v3_vl.model import (
    TurboBailingMoeV3VLForConditionalGeneration,
)

from .test_model import make_config


@pytest.fixture(scope="module")
def components():
    reference = os.environ.get("BAILING_VL_REFERENCE")
    if reference is None:
        pytest.skip("Set BAILING_VL_REFERENCE to run native LF integration tests.")
    pytest.importorskip("llamafactory")
    from llamafactory.data.template import TEMPLATES
    from transformers import AutoProcessor

    from torchturbo.models.bailing_moe_v3_vl.lf_template import register
    from torchturbo.utils.transformers_utils import patch_dynamic_module_loading

    register()
    # Launchers install this through torchturbo.activate() before loading.
    patch_dynamic_module_loading()
    processor = AutoProcessor.from_pretrained(
        reference, trust_remote_code=True, local_files_only=True
    )
    processor.image_max_pixels = 3145728
    processor.image_min_pixels = 4096
    template = deepcopy(TEMPLATES["bailing_moe_v3_vl"])
    template.auto_sft_thinking = False
    return template, processor


@pytest.fixture
def video(tmp_path):
    frame_paths = []
    for index in range(4):
        frame = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (64, 64), (index * 30, 0, 0)).save(frame)
        frame_paths.append(str(frame))
    return "frames_list://" + json.dumps(frame_paths)


def test_auto_thinking_environment_and_dill(components, monkeypatch):
    import dill
    from datasets.fingerprint import Hasher

    template, _ = components
    options = vars(template).copy()
    options.pop("auto_sft_thinking")
    monkeypatch.setenv("LING_SFT_AUTO_THINKING", "1")
    enabled = type(template)(**options)
    monkeypatch.setenv("LING_SFT_AUTO_THINKING", "0")
    disabled = type(template)(**options)
    restored = dill.loads(dill.dumps(enabled))

    assert enabled.auto_sft_thinking
    assert not disabled.auto_sft_thinking
    assert type(restored) is type(template)
    assert restored.auto_sft_thinking
    assert Hasher.hash(enabled) != Hasher.hash(disabled)


def test_auto_thinking_does_not_mutate_samples(components):
    template, processor = components
    template = deepcopy(template)
    template.auto_sft_thinking = True
    template.enable_thinking = False
    for answer, mode in [
        ("answer", "off"),
        ("<think>reasoning</think>answer", "on"),
        ("answer", "off"),
    ]:
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ]
        original = deepcopy(messages)
        pairs = template.encode_multiturn(processor.tokenizer, messages)
        input_ids = [token for pair in pairs for part in pair for token in part]

        assert f"detailed thinking {mode}" in processor.tokenizer.decode(input_ids)
        assert messages == original
        assert template.enable_thinking is False


def test_oneturn_keeps_a_common_prompt_for_generation_and_preference_data(components):
    template, processor = components
    template = deepcopy(template)
    template.auto_sft_thinking = True
    template.enable_thinking = True
    prompts = []
    for answer in ["", "answer", "<think>reasoning</think>answer"]:
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ]
        prompt, _ = template.encode_oneturn(processor.tokenizer, messages)
        prompts.append(prompt)
        text = processor.tokenizer.decode(prompt)
        assert "detailed thinking on" in text
        assert text.endswith("<role>ASSISTANT</role>\n")
    assert prompts[0] == prompts[1] == prompts[2]


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
@pytest.mark.parametrize("media_type", ["image", "video"])
def test_multiprocess_preprocessing(
    components, video, tmp_path, monkeypatch, media_type, start_method
):
    import dill
    import multiprocess
    from datasets import Dataset, arrow_dataset
    from llamafactory.data import loader
    from llamafactory.hparams import DataArguments

    if start_method not in multiprocess.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable on this platform.")
    pool = multiprocess.get_context(start_method).Pool
    # datasets 4.0 imports Pool by name; newer releases call multiprocess.Pool.
    monkeypatch.setattr(arrow_dataset, "Pool", pool, raising=False)
    monkeypatch.setattr(multiprocess, "Pool", pool)
    template, processor = components

    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64)).save(image)
    media = str(image) if media_type == "image" else video
    sample = {
        "_prompt": [{"role": "user", "content": f"<{media_type}>describe"}],
        "_response": [{"role": "assistant", "content": "answer"}],
        "_system": "",
        "_tools": "",
        "_images": [media] if media_type == "image" else [],
        "_videos": [media] if media_type == "video" else [],
        "_audios": [],
    }
    dataset = Dataset.from_list([sample, sample])
    data_args = DataArguments(
        template="bailing_moe_v3_vl",
        cutoff_len=512,
        preprocessing_num_workers=1,
        preprocessing_batch_size=1,
        overwrite_cache=True,
    )
    training_args = SimpleNamespace(
        predict_with_generate=False, local_process_index=0, should_log=False
    )
    expected = loader._get_preprocessed_dataset(
        dataset, data_args, training_args, "sft", template,
        processor.tokenizer, processor,
    )
    # Fail in the parent before starting workers if classes are rebuilt.
    restored = dill.loads(dill.dumps(processor))
    assert type(restored) is type(processor)
    assert type(restored.image_processor) is type(processor.image_processor)

    data_args.preprocessing_num_workers = 2
    actual = loader._get_preprocessed_dataset(
        dataset, data_args, training_args, "sft", template,
        processor.tokenizer, processor,
    )
    assert len(actual) == len(dataset)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_chat_template_matches_native_rendering(components, thinking, preserve):
    template, processor = components
    template = deepcopy(template)
    template.enable_thinking = thinking
    template.preserve_thinking = preserve
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<think>reasoning</think>first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    parts = template._encode(processor.tokenizer, messages, "system text", None)
    expected = processor.tokenizer.apply_chat_template(
        [{"role": "system", "content": "system text"}, *messages],
        tokenize=False,
        enable_thinking=thinking,
        preserve_thinking=preserve,
    )
    assert (
        processor.tokenizer.decode(sum(parts, []), skip_special_tokens=False)
        == expected
    )
    thought = "reasoning" if preserve else ""
    assert processor.tokenizer.decode(parts[1]).startswith(f"<think>{thought}</think>")
    assert processor.tokenizer.decode(parts[3]).endswith("<|role_end|>")


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_lf_thinking_arguments_reach_template(
    components, monkeypatch, thinking, preserve
):
    from llamafactory.data.template import TEMPLATES, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    template, processor = components
    # LlamaFactory configures the registered template object in place.
    monkeypatch.setitem(TEMPLATES, "bailing_moe_v3_vl", deepcopy(template))
    data_args = DataArguments(
        template="bailing_moe_v3_vl",
        enable_thinking=thinking,
        preserve_thinking=preserve,
    )
    tokenizer = deepcopy(processor.tokenizer)
    configured = get_template_and_fix_tokenizer(tokenizer, data_args)
    assert configured.enable_thinking is thinking
    assert configured.preserve_thinking is preserve

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<think>reasoning</think>first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "<think>more</think>second answer"},
    ]
    pairs = configured.encode_multiturn(tokenizer, messages, "system text", None)
    expected = tokenizer.apply_chat_template(
        [{"role": "system", "content": "system text"}, *messages],
        tokenize=False,
        enable_thinking=thinking,
        preserve_thinking=preserve,
    )
    token_ids = [token for pair in pairs for part in pair for token in part]
    assert tokenizer.decode(token_ids) == expected

    prompt_ids, response_ids = configured.encode_oneturn(
        tokenizer, messages, "system text", None
    )
    assert tokenizer.decode(prompt_ids + response_ids) == expected


@pytest.mark.parametrize("reasoning", ["", "reasoning", "\nreasoning\n"])
@pytest.mark.parametrize("string_arguments", [False, True])
def test_tool_calls(components, reasoning, string_arguments):
    template, processor = components
    arguments = {"key": "value", "count": 2, "literal": "<think>literal</think>"}
    calls = [
        {
            "name": "lookup",
            "arguments": json.dumps(arguments) if string_arguments else arguments,
        }
    ]
    thought = f"<think>{reasoning}</think>\n" if reasoning else ""
    messages = [
        {"role": "user", "content": "lookup"},
        {"role": "function", "content": thought + json.dumps(calls)},
        {"role": "observation", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]
    parts = template._encode(processor.tokenizer, messages, None, None)
    rendered = processor.tokenizer.decode(sum(parts, []))
    native_messages = [
        messages[0],
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [{"name": "lookup", "arguments": arguments}],
        },
        {"role": "tool", "content": "result"},
        messages[-1],
    ]
    expected = processor.tokenizer.apply_chat_template(
        native_messages, tokenize=False, enable_thinking=False
    )
    assert rendered == expected
    assert "<tool_call>lookup" in rendered
    assert "<arg_value>value</arg_value>" in rendered
    assert "<role>OBSERVATION</role>" in rendered


@pytest.mark.parametrize("preserve", [False, True])
def test_function_thinking_history_matches_native(components, preserve):
    template, processor = components
    template = deepcopy(template)
    template.preserve_thinking = preserve
    call = {"name": "lookup", "arguments": {"key": "value"}}
    messages = [
        {"role": "user", "content": "lookup"},
        {"role": "function", "content": "<think>reasoning</think>" + json.dumps(call)},
        {"role": "observation", "content": "result"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "final answer"},
    ]
    parts = template._encode(processor.tokenizer, messages, None, None)
    native_messages = deepcopy(messages)
    native_messages[1] = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reasoning",
        "tool_calls": [call],
    }
    native_messages[2]["role"] = "tool"
    expected = processor.tokenizer.apply_chat_template(
        native_messages,
        tokenize=False,
        enable_thinking=False,
        preserve_thinking=preserve,
    )
    assert processor.tokenizer.decode(sum(parts, [])) == expected


@pytest.mark.parametrize("expand", [False, True])
def test_video_plugin_matches_native_rendering(components, video, expand):
    template, processor = components
    template = deepcopy(template)
    template.mm_plugin.expand_mm_tokens = expand
    messages = [
        {"role": "user", "content": "<video>describe"},
        {"role": "assistant", "content": "answer"},
    ]
    messages = template.mm_plugin.process_messages(
        messages, [], [video], [], processor
    )
    prompt, response = template.encode_oneturn(processor.tokenizer, messages)
    native_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": "describe"},
            ],
        },
        messages[-1],
    ]
    expected = processor.tokenizer.apply_chat_template(
        native_messages, tokenize=False, enable_thinking=False
    )
    if expand:
        _, videos, _ = processor.process_vision_info(native_messages)
        native_inputs = processor(text=expected, videos=videos, return_tensors="pt")
        expected = processor.tokenizer.decode(native_inputs["input_ids"][0])
    assert processor.tokenizer.decode(prompt + response) == expected


@pytest.mark.parametrize("media", ["none", "image", "video"])
def test_native_media_collator_and_model(components, media, video):
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq

    template, processor = components
    config = make_config()
    config.text_config.vocab_size = len(processor.tokenizer)
    config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    config.vision_config.patch_size = processor.image_processor.patch_size
    for name in ("image_token_id", "video_token_id"):
        setattr(config, name, getattr(processor, name))
    model = TurboBailingMoeV3VLForConditionalGeneration(config)
    images = [Image.new("RGB", (64, 64), "red")] if media == "image" else []
    videos = [video] if media == "video" else []
    messages = [
        {"role": "user", "content": "hello" if media == "none" else f"<{media}>describe"},
        {"role": "assistant", "content": "answer"},
    ]
    messages = template.mm_plugin.process_messages(messages, images, videos, [], processor)
    prompt, response = template.encode_oneturn(processor.tokenizer, messages)
    ids = prompt + response
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        template=template,
        processor=processor,
    )
    batch = collator(
        [
            dict(
                input_ids=ids,
                attention_mask=[1] * len(ids),
                labels=[-100] * len(prompt) + response,
                images=images,
                videos=videos,
            )
        ]
    )
    assert batch["position_ids"].shape[0] == 4
    torch.testing.assert_close(batch["position_ids"][0], batch["position_ids"][1])
    output = model.model(
        **{k: v for k, v in batch.items() if k != "labels"}, projector=model.linear_proj
    )
    without_positions = {
        k: v for k, v in batch.items() if k not in ("labels", "position_ids")
    }
    expected = model.model(**without_positions, projector=model.linear_proj)
    # LF assigns separate positions to the masked dummy-image padding.
    valid_tokens = batch["attention_mask"].bool()
    torch.testing.assert_close(
        output.last_hidden_state[valid_tokens], expected.last_hidden_state[valid_tokens]
    )
    assert torch.isfinite(output.last_hidden_state).all()
    output.last_hidden_state.sum().backward()
    assert model.linear_proj[0].weight.grad is not None


def test_native_video_expansion(components):
    _, processor = components
    # Exercise native frame-to-patch conversion without a filesystem video decoder.
    frames = torch.zeros(4, 3, 64, 64)
    native = processor.image_processor(
        images=None,
        videos=[frames],
        do_resize=False,
        videos_timestamps_seconds=[[0.0, 0.5, 1.0, 1.5]],
        return_tensors="pt",
    )
    expanded = processor._expand_video_tokens(
        [processor.video_token], deepcopy(native)
    )[0]
    grid = native["video_grid_thw"]
    assert expanded.count(processor.video_token) == int(grid.prod(-1).sum()) // 4
    assert "seconds>" in expanded


@pytest.mark.parametrize("media", ["none", "image", "video"])
def test_lf_packing_preserves_independent_multimodal_samples(
    components, media, video, monkeypatch
):
    import transformers.integrations.flash_attention as flash_integration
    import transformers.modeling_flash_attention_utils as flash_utils
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

    from torchturbo.adapters.llamafactory_v0.patches import PackedInputCollatorPatch
    from torchturbo.engine import TorchTurboEngine

    monkeypatch.setattr(flash_integration, "_flash_attention_forward", flash_integration._flash_attention_forward)
    monkeypatch.setattr(flash_utils, "_get_unpad_data", flash_utils._get_unpad_data)
    TorchTurboEngine.__new__(TorchTurboEngine)._patch_transformers()

    template, processor = components
    config = make_config(hybrid=True)
    config.text_config.vocab_size = len(processor.tokenizer)
    config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    config.vision_config.patch_size = processor.image_processor.patch_size
    config.image_token_id = processor.image_token_id
    config.video_token_id = processor.video_token_id
    model = TurboBailingMoeV3VLForConditionalGeneration(config).eval()
    examples = []
    images = (
        [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 96), "blue")]
        if media == "image"
        else []
    )
    videos = [video, video] if media == "video" else []
    for index in range(2):
        sample_images = images[index : index + 1]
        sample_videos = videos[index : index + 1]
        messages = [
            {"role": "user", "content": "hello" if media == "none" else f"<{media}>describe"},
            {"role": "assistant", "content": f"answer {index}"},
        ]
        messages = template.mm_plugin.process_messages(
            messages, sample_images, sample_videos, [], processor
        )
        prompt, response = template.encode_oneturn(processor.tokenizer, messages)
        ids = prompt + response
        examples.append(
            dict(
                input_ids=ids,
                attention_mask=[1] * len(ids),
                labels=[-100] * len(prompt) + response,
                images=sample_images,
                videos=sample_videos,
            )
        )
    collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=processor.tokenizer,
        model=model,
        template=template,
        processor=processor,
        neat_packing=True,
        attn_implementation="flash_attention_2",
    )
    patch = PackedInputCollatorPatch()
    patch.patch()
    try:
        separate = []
        for example in examples:
            batch = collator([deepcopy(example)])
            output = model.model(
                **{k: v for k, v in batch.items() if k != "labels"},
                projector=model.linear_proj,
            )
            separate.append(output.last_hidden_state[:, : len(example["input_ids"])])
        first_len = len(examples[0]["input_ids"])
        total_len = first_len + len(examples[1]["input_ids"])
        combined = dict(
            input_ids=examples[0]["input_ids"] + examples[1]["input_ids"],
            attention_mask=[1] * first_len + [2] * (total_len - first_len),
            labels=examples[0]["labels"] + examples[1]["labels"],
            images=images,
            videos=videos,
            packing_params=dict(
                sequence_boundaries=[0, first_len, total_len],
                image_subseq_ids=[0, 1] if images else [],
                video_subseq_ids=[0, 1] if videos else [],
                right_padding_length=0,
            ),
        )
        batch = collator([combined])
        output = model.model(
            **{k: v for k, v in batch.items() if k != "labels"},
            projector=model.linear_proj,
        )
        torch.testing.assert_close(
            output.last_hidden_state[:, :total_len],
            torch.cat(separate, 1),
            atol=1e-5,
            rtol=1e-5,
        )
    finally:
        patch.revert()


def test_video_plugin_preserves_temporal_grids(components, video):
    template, processor = components
    messages = template.mm_plugin.process_messages(
        [{'role': 'user', 'content': '<video>describe'}], [], [video], [], processor,
    )
    ids = processor.tokenizer.encode(messages[0]['content'], add_special_tokens=False)
    media = template.mm_plugin.get_mm_inputs([], [video], [], [0], [1], [0], [ids], processor)
    assert media['video_grid_thw'].shape == (1, 3)
    assert 'video_timestamps_seconds' not in media
    assert messages[0]['content'].count(processor.video_token) == int(media['video_grid_thw'].prod()) // 4
