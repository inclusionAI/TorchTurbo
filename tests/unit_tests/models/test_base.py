from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.config import EngineConfig, MTPConfig
from torchturbo.models.base import TurboModel, TurboVLMModel
from torchturbo.models.kimi_k25.model import TurboKimiK25ForConditionalGeneration


@pytest.mark.parametrize("model_cls, media_keys", [
    (TurboModel, ()),
    (TurboVLMModel, (
        "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
    )),
    (TurboKimiK25ForConditionalGeneration, ("pixel_values", "grid_thws")),
])
def test_input_hook_selects_model_specific_media_fields(model_cls, media_keys):
    model = object.__new__(model_cls)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "cu_seq_lens_q": torch.tensor([0, 3], dtype=torch.int32),
    }
    source = {
        "input_ids": inputs["input_ids"].squeeze(0),
        "cu_seqlens": inputs["cu_seq_lens_q"],
        "ppo_loss_mask": torch.ones(3),
        "pixel_values": torch.ones(4, 6),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "pixel_values_videos": torch.ones(8, 6),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
        "grid_thws": torch.tensor([[1, 2, 2]]),
    }

    prepared = model.on_prepare_inputs(inputs.copy(), source_inputs=source)

    assert prepared.keys() == inputs.keys() | set(media_keys)
    assert prepared["input_ids"] is inputs["input_ids"]
    assert source["input_ids"].shape == (3,)
    for key in media_keys:
        assert prepared[key] is source[key]


def test_init_context_strips_mtp_without_restoring_model_structure_config():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        num_nextn_predict_layers=2,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=False))

    with TurboModel.init_context(model_config, engine_config):
        assert model_config.dtype == torch.float32
        assert model_config.num_nextn_predict_layers == 0

    assert model_config.dtype == torch.bfloat16
    assert model_config.num_nextn_predict_layers == 0


def test_init_context_retains_mtp_when_enabled():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        num_nextn_predict_layers=2,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=True))

    with TurboModel.init_context(model_config, engine_config):
        assert model_config.num_nextn_predict_layers == 2

    assert model_config.num_nextn_predict_layers == 2


def test_init_context_forces_flash_attention_2():
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    engine_config = EngineConfig()

    with TurboModel.init_context(model_config, engine_config):
        assert model_config._attn_implementation == "flash_attention_2"


def test_vlm_init_context_propagates_flash_attention_2_to_subconfigs():
    from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )

    model_config = Qwen3_5MoeConfig()
    model_config._attn_implementation = "sdpa"
    model_config.text_config._attn_implementation = "sdpa"
    model_config.vision_config._attn_implementation = "sdpa"

    with TurboVLMModel.init_context(model_config, EngineConfig()):
        assert model_config._attn_implementation == "flash_attention_2"
        assert model_config.text_config._attn_implementation == "flash_attention_2"
        assert model_config.vision_config._attn_implementation == "flash_attention_2"


def test_vlm_init_context_forwards_engine_config_and_restores_nested_dtype():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        text_config=SimpleNamespace(dtype=torch.float16),
        vision_config=SimpleNamespace(dtype=torch.float16),
        num_nextn_predict_layers=1,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=False))

    with TurboVLMModel.init_context(model_config, engine_config):
        assert model_config.dtype == torch.float32
        assert model_config.text_config.dtype == torch.float32
        assert model_config.vision_config.dtype == torch.float32
        assert model_config.num_nextn_predict_layers == 0

    assert model_config.dtype == torch.bfloat16
    assert model_config.text_config.dtype == torch.float16
    assert model_config.vision_config.dtype == torch.float16
    assert model_config.num_nextn_predict_layers == 0
