"""Tests for the FSDP2 HF export config."""

import json
import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch.distributed as dist
from transformers import PretrainedConfig

from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import (
    ShardedParamManager,
    _sync_mtp_layer_count,
)
from torchturbo.engine.config import EngineConfig
from torchturbo.models.bailing_moe_v3_vl.configuration_bailing_moe_v3_vl import (
    BailingMoeV3VLConfig,
)
from torchturbo.models.base import TurboModel
from torchturbo.models.kimi_k25.configuration_kimi_k25 import KimiK25Config
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)


def _disable_mtp(config):
    """Apply the config changes of model construction with MTP disabled."""
    with TurboModel.init_context(config, EngineConfig()):
        pass


@pytest.mark.parametrize(
    ("make_config", "mtp_paths"),
    [
        (
            lambda: PretrainedConfig(num_nextn_predict_layers=2),
            [("num_nextn_predict_layers",)],
        ),
        (
            lambda: Qwen3_5MoeTextConfig(mtp_num_hidden_layers=2),
            [("mtp_num_hidden_layers",)],
        ),
        (
            lambda: Qwen3_5MoeConfig(text_config={"mtp_num_hidden_layers": 2}),
            [("text_config", "mtp_num_hidden_layers")],
        ),
        (
            lambda: BailingMoeV3VLConfig(text_config={"num_nextn_predict_layers": 2}),
            [
                ("text_config", "num_nextn_predict_layers"),
                ("llm_config", "num_nextn_predict_layers"),
            ],
        ),
    ],
    ids=["top_level", "mtp_num_hidden_layers", "text_config_alias", "text_and_llm_config"],
)
def test_disabled_mtp_updates_only_the_mtp_layer_count(tmp_path, make_config, mtp_paths):
    config = make_config()
    # Reverse the key order so the test also checks that it is preserved.
    source = dict(reversed(list(config.to_dict().items())))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(source))
    _disable_mtp(config)

    _sync_mtp_layer_count(config_path, config.to_dict())

    expected = json.loads(json.dumps(source))
    for *parents, key in mtp_paths:
        section = expected
        for parent in parents:
            section = section[parent]
        assert section[key] == 2
        section[key] = 0
    assert json.dumps(json.loads(config_path.read_text())) == json.dumps(expected)


@pytest.mark.parametrize(
    "source",
    [
        {"use_cache": True, "num_nextn_predict_layers": 0, "model_type": "test"},
        {"use_cache": True, "model_type": "test"},
    ],
    ids=["matching_count", "no_mtp_field"],
)
def test_unchanged_mtp_layer_count_keeps_the_source_bytes(tmp_path, source):
    config_path = tmp_path / "config.json"
    # Non-default formatting, so any rewrite would change the bytes.
    config_path.write_text(json.dumps(source, separators=(",", ":")))
    original = config_path.read_bytes()
    runtime_config = {"model_type": "test", "num_nextn_predict_layers": 0, "use_cache": False}

    _sync_mtp_layer_count(config_path, runtime_config)

    assert config_path.read_bytes() == original


def test_mtp_layer_count_is_read_where_the_source_declares_it(tmp_path):
    # Kimi-K2.5 has no top-level MTP property, so training keeps the nested value.
    config = KimiK25Config(text_config={"num_nextn_predict_layers": 1})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict(), separators=(",", ":")))
    original = config_path.read_bytes()
    _disable_mtp(config)

    _sync_mtp_layer_count(config_path, config.to_dict())

    assert config_path.read_bytes() == original


def test_hf_export_copies_the_source_config(tmp_path):
    source = tmp_path / "source"
    PretrainedConfig(num_nextn_predict_layers=1, use_cache=True).save_pretrained(source)
    (source / "tokenizer_config.json").write_text("{}")
    config = PretrainedConfig.from_pretrained(source)
    # Activation checkpointing disables the KV cache during training.
    config.use_cache = False
    _disable_mtp(config)

    saver = ShardedParamManager.__new__(ShardedParamManager)
    saver.rank = 0
    saver.world_size = 1
    saver.sharded_param_groups = []
    saver.local_params = set()
    saver.src_path = source
    saver.processing_class = None
    saver.model = SimpleNamespace(config=config)
    saver._src_metadata = {}
    export = tmp_path / "export"
    export.mkdir()
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        saver.save_hf({}, export)
    finally:
        dist.destroy_process_group()

    expected = json.loads((source / "config.json").read_text())
    expected["num_nextn_predict_layers"] = 0
    assert json.loads((export / "config.json").read_text()) == expected
    assert (export / "tokenizer_config.json").read_text() == "{}"
    assert (export / "model.safetensors.index.json").is_file()
