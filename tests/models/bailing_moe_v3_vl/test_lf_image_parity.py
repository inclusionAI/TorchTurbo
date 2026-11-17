"""Compare LF image inputs with the Swift GPU baseline preprocessing path."""

import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="module")
def processor():
    reference = os.environ.get("BAILING_VL_REFERENCE")
    if reference is None:
        pytest.skip("Set BAILING_VL_REFERENCE to run native LF image tests.")
    pytest.importorskip("llamafactory")
    from transformers import AutoProcessor

    from torchturbo.utils.transformers_utils import patch_dynamic_module_loading

    patch_dynamic_module_loading()
    processor = AutoProcessor.from_pretrained(
        reference, trust_remote_code=True, local_files_only=True
    )
    # LlamaFactory sets these attributes from ModelArguments in patch_processor.
    processor.image_max_pixels = 3145728
    processor.image_min_pixels = 4096
    return processor


@pytest.fixture
def plugin():
    pytest.importorskip("llamafactory")
    from torchturbo.models.bailing_moe_v3_vl.mm_plugin import BailingVLPlugin

    return BailingVLPlugin(
        image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token=None
    )


def make_image(size, mode):
    width, height = size
    channels = len(mode)
    values = np.arange(width * height * channels, dtype=np.uint32)
    values = ((values * 37 + values // 13) % 256).astype(np.uint8)
    return Image.fromarray(values.reshape(height, width, channels))


def baseline_inputs(images, processor):
    pytest.importorskip("swift")
    from qwen_vl_utils import fetch_image
    from swift.template.vision_utils import load_image, rescale_image

    images = [
        fetch_image(
            {"image": rescale_image(load_image(image), processor.image_max_pixels)},
            image_patch_size=processor.image_processor.patch_size,
        )
        for image in images
    ]
    return processor.image_processor(images=images, return_tensors="pt", do_resize=False)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("size", [(83, 109), (1739, 1937)])
def test_lf_image_tensors_match_swift_baseline(plugin, processor, size, mode):
    image = make_image(size, mode)
    expected = baseline_inputs([image], processor)
    actual = plugin.get_mm_inputs([image], [], [], [1], [0], [0], [], processor)

    torch.testing.assert_close(
        actual["image_grid_thw"], expected["image_grid_thw"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["pixel_values"], expected["pixel_values"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "input_format", ["pil", "path", "bytes", "dict_path", "dict_bytes"]
)
def test_lf_image_input_formats_match_baseline(plugin, processor, tmp_path, input_format):
    image = make_image((77, 109), "RGBA")
    path = tmp_path / "image.png"
    image.save(path)
    image_bytes = path.read_bytes()
    inputs = {
        "pil": image,
        "path": str(path),
        "bytes": image_bytes,
        "dict_path": {"path": str(path), "bytes": None},
        "dict_bytes": {"path": None, "bytes": image_bytes},
    }
    expected = baseline_inputs([image], processor)
    actual = plugin.get_mm_inputs(
        [inputs[input_format]], [], [], [1], [0], [0], [], processor
    )

    torch.testing.assert_close(
        actual["image_grid_thw"], expected["image_grid_thw"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["pixel_values"], expected["pixel_values"], rtol=0, atol=0
    )


def test_lf_pixel_limits_control_images(plugin, processor):
    processor = deepcopy(processor)
    image = make_image((173, 227), "RGB")
    processor.image_max_pixels = 8192
    expected = baseline_inputs([image], processor)
    actual = plugin.get_mm_inputs([image], [], [], [1], [0], [0], [], processor)

    torch.testing.assert_close(
        actual["image_grid_thw"], expected["image_grid_thw"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["pixel_values"], expected["pixel_values"], rtol=0, atol=0
    )
    assert processor.image_processor.max_pixels != processor.image_max_pixels

    processor.image_max_pixels = 3145728
    processor.image_min_pixels = 16384
    actual = plugin.get_mm_inputs(
        [make_image((16, 16), "RGB")], [], [], [1], [0], [0], [], processor
    )
    grid = actual["image_grid_thw"][0]
    assert int(grid.prod()) * processor.image_processor.patch_size**2 == 16384


@pytest.mark.parametrize("merge_size", [1, 2])
@pytest.mark.parametrize("wrapped", [False, True])
def test_lf_image_tokens_use_grid_without_extra_newline(
    plugin, processor, merge_size, wrapped
):
    processor = deepcopy(processor)
    processor.image_processor.merge_size = merge_size
    images = [make_image((96, 64), "RGB"), make_image((64, 96), "RGBA")]
    placeholder = "<image>"
    if wrapped:
        placeholder = (
            processor.vision_start_token + placeholder + processor.vision_end_token
        )
    source = [
        {"role": "user", "content": placeholder + "first" + placeholder + "\nsecond"},
        {"role": "assistant", "content": "answer"},
    ]
    messages = plugin.process_messages(source, images, [], [], processor)
    actual = plugin.get_mm_inputs(images, [], [], [2], [0], [0], [], processor)
    grids = actual["image_grid_thw"]
    segments = [
        processor.vision_start_token
        + processor.image_token * (int(grid.prod()) // merge_size**2)
        + processor.vision_end_token
        for grid in grids
    ]
    assert messages[0]["content"] == segments[0] + "first" + segments[1] + "\nsecond"
    assert source[0]["content"] == placeholder + "first" + placeholder + "\nsecond"


@pytest.mark.parametrize("plugin_name", ["bailing", "qwen2_vl", "qwen3_vl"])
@pytest.mark.parametrize("size,min_pixels", [((16, 1000), 4096), ((16, 16), 16384)])
def test_pretrain_image_tokens_match_collator(
    plugin, processor, tmp_path, plugin_name, size, min_pixels
):
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from llamafactory.data.mm_plugin import Qwen2VLPlugin, Qwen3VLPlugin

    from torchturbo.adapters.llamafactory_v0.utils import preprocess_multimodal_pretrain

    if plugin_name != "bailing":
        plugin_cls = Qwen2VLPlugin if plugin_name == "qwen2_vl" else Qwen3VLPlugin
        plugin = plugin_cls(
            image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token=None
        )
    processor = deepcopy(processor)
    processor.image_min_pixels = min_pixels
    image_path = tmp_path / "image.png"
    make_image(size, "RGB").save(image_path)
    template = SimpleNamespace(mm_plugin=plugin)
    dataset_processor = SimpleNamespace(
        tokenizer=processor.tokenizer,
        processor=processor,
        template=template,
        data_args=SimpleNamespace(template="bailing_moe_v3_vl", cutoff_len=256),
    )
    packed = preprocess_multimodal_pretrain(
        dataset_processor,
        {"_prompt": [[{"role": "user", "content": "<image:0>describe"}]]},
        [[str(image_path)]],
    )
    features = []
    for index, image_paths in enumerate(packed["images"]):
        features.append(
            {
                "input_ids": packed["input_ids"][index].tolist(),
                "attention_mask": packed["attention_mask"][index].tolist(),
                "labels": packed["labels"][index].tolist(),
                "images": image_paths,
            }
        )
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, template=template, processor=processor
    )
    batch = collator(features)
    expected_count = int(batch["image_grid_thw"].prod(-1).sum())
    expected_count //= processor.image_processor.merge_size**2
    actual_count = int((batch["input_ids"] == processor.image_token_id).sum())
    assert actual_count == expected_count


def test_pretrain_does_not_substitute_one_token_for_invalid_image(
    plugin, processor, tmp_path
):
    from torchturbo.adapters.llamafactory_v0.utils import preprocess_multimodal_pretrain

    dataset_processor = SimpleNamespace(
        tokenizer=processor.tokenizer,
        processor=processor,
        template=SimpleNamespace(mm_plugin=plugin),
        data_args=SimpleNamespace(template="bailing_moe_v3_vl", cutoff_len=256),
    )
    with pytest.raises(FileNotFoundError):
        preprocess_multimodal_pretrain(
            dataset_processor,
            {"_prompt": [[{"role": "user", "content": "<image>describe"}]]},
            [[str(tmp_path / "missing.png")]],
        )


def test_pretrain_empty_image_batch_keeps_existing_empty_result(processor):
    from torchturbo.adapters.llamafactory_v0.utils import preprocess_multimodal_pretrain

    dataset_processor = SimpleNamespace(
        tokenizer=processor.tokenizer,
        processor=None,
        template=None,
        data_args=SimpleNamespace(template="bailing_moe_v3_vl", cutoff_len=256),
    )
    result = preprocess_multimodal_pretrain(
        dataset_processor,
        {"_prompt": [[{"role": "user", "content": "plain text"}]]},
        [None],
    )
    assert result == {"input_ids": [], "attention_mask": [], "labels": []}
