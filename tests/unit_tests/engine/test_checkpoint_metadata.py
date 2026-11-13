"""Checkpoint metadata rebuilds models for export and checks resumed adapters."""

import json

import pytest
from peft import LoraConfig, get_peft_config, get_peft_model
from torch import nn
from transformers import PretrainedConfig

from torchturbo.engine.checkpoint.metadata import (
    build_metadata,
    load_metadata,
    save_metadata,
    validate_peft_configs,
)
from torchturbo.engine.config import EngineConfig


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.linear = nn.Linear(3, 2, bias=False)
        self.other = nn.Linear(3, 2, bias=False)
        self.extra = nn.Linear(3, 2, bias=False)


def _model(**overrides):
    settings = dict(r=8, lora_alpha=32, target_modules=["linear"], use_rslora=True)
    settings.update(overrides)
    return get_peft_model(LinearModel(), LoraConfig(**settings))


def _build(model):
    return build_metadata(model, EngineConfig.from_dict({}), "/models/base")


def test_metadata_round_trip_keeps_the_full_peft_config(tmp_path):
    model = _model(
        target_modules=["other", "linear"],
        # Several names, so their set iteration order is rarely sorted.
        exclude_modules=["unused", "extra", "absent", "frozen", "skipped"],
        alpha_pattern={"other": 16},
    )
    metadata = _build(model)
    save_metadata(metadata, str(tmp_path))
    restored = load_metadata(str(tmp_path))

    assert restored == metadata
    saved_config = restored["peft_configs"]["default"]
    assert saved_config["lora_alpha"] == 32
    assert saved_config["use_rslora"] is True
    assert saved_config["alpha_pattern"] == {"other": 16}
    assert saved_config["target_modules"] == ["linear", "other"]
    assert saved_config["exclude_modules"] == [
        "absent", "extra", "frozen", "skipped", "unused"
    ]
    live_config = model.peft_config["default"]
    rebuilt = get_peft_config(saved_config)
    assert rebuilt.target_modules == live_config.target_modules
    assert rebuilt.exclude_modules == live_config.exclude_modules
    assert rebuilt.to_dict() == live_config.to_dict()


def test_metadata_is_saved_as_one_json_file(tmp_path):
    metadata = _build(_model())
    save_metadata(metadata, str(tmp_path))

    files = [
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    ]
    assert files == ["torchturbo/checkpoint.json"]
    document = json.loads((tmp_path / "torchturbo/checkpoint.json").read_text())
    assert document == metadata
    assert list(document["peft_configs"]) == ["default"]


def test_resume_accepts_the_saved_adapter_config(tmp_path):
    settings = dict(
        target_modules=["other", "linear"],
        # Several names, so their set iteration order is rarely sorted.
        exclude_modules=["unused", "extra", "absent", "frozen", "skipped"],
    )
    save_metadata(_build(_model(**settings)), str(tmp_path))

    validate_peft_configs(_model(**settings), load_metadata(str(tmp_path)))


def test_resume_allows_changes_to_training_and_bookkeeping_fields():
    saved = _build(_model())
    saved_config = saved["peft_configs"]["default"]
    saved_config["peft_version"] = "0.0.1"
    saved_config["base_model_name_or_path"] = "/models/moved"

    current = _model(lora_dropout=0.25, init_lora_weights=False, bias="lora_only")
    validate_peft_configs(current, saved)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lora_alpha", 16),
        ("use_rslora", False),
        ("alpha_pattern", {"linear": 16}),
        ("r", 4),
        ("use_dora", True),
        ("target_modules", ["linear", "other"]),
    ],
)
def test_resume_rejects_a_changed_adapter_config(key, value):
    saved = _build(_model())

    with pytest.raises(ValueError, match=f"differs from the checkpoint: .*'{key}'"):
        validate_peft_configs(_model(**{key: value}), saved)


def test_resume_requires_the_adapters_of_this_run():
    saved = _build(LinearModel())

    with pytest.raises(ValueError, match=r"missing required PEFT adapters: \['default'\]"):
        validate_peft_configs(_model(), saved)


@pytest.mark.parametrize("key", ["use_qalora", "alora_invocation_tokens", "alpha_pattern"])
def test_resume_accepts_missing_fields_at_their_peft_defaults(key):
    current = _model()
    saved = _build(current)
    del saved["peft_configs"]["default"][key]

    validate_peft_configs(current, saved)

    assert key not in saved["peft_configs"]["default"]


@pytest.mark.parametrize(
    ("key", "default", "changed"),
    [("use_rslora", False, True), ("alpha_pattern", {}, {"linear": 16})],
)
def test_resume_rejects_missing_fields_with_nondefault_values(key, default, changed):
    saved = _build(_model(**{key: default}))
    del saved["peft_configs"]["default"][key]

    with pytest.raises(ValueError, match=f"differs from the checkpoint: .*'{key}'"):
        validate_peft_configs(_model(**{key: changed}), saved)


def test_resume_rejects_enabling_alora_after_a_peft_upgrade(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    def make_model(**settings):
        base = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=4, vocab_size=8, n_positions=8)
        )
        return get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, target_modules=["c_attn"],
            fan_in_fan_out=True, task_type="CAUSAL_LM", **settings,
        ))

    trained = make_model()
    metadata = _build(trained)
    # Older PEFT versions did not record aLoRA's invocation tokens.
    del metadata["peft_configs"]["default"]["alora_invocation_tokens"]
    save_metadata(metadata, str(tmp_path))

    resumed = make_model(alora_invocation_tokens=[7])
    # Tensor names and shapes alone cannot distinguish LoRA from aLoRA.
    resumed.load_state_dict(trained.state_dict(), strict=True)
    with pytest.raises(ValueError, match=r"'alora_invocation_tokens': \(None, \[7\]\)"):
        validate_peft_configs(resumed, load_metadata(str(tmp_path)))


def test_resume_skips_fields_unknown_to_the_current_peft():
    saved = _build(_model())
    saved_config = saved["peft_configs"]["default"]
    # A newer PEFT recorded a field that this PEFT does not know.
    saved_config["field_from_newer_peft"] = 1

    validate_peft_configs(_model(), saved)


def test_full_fine_tuning_records_no_peft_configs(tmp_path):
    save_metadata(_build(LinearModel()), str(tmp_path))

    assert load_metadata(str(tmp_path))["peft_configs"] == {}


def test_checkpoint_without_metadata_is_legacy(tmp_path):
    assert load_metadata(str(tmp_path)) is None


@pytest.mark.parametrize(
    "content",
    [None, '{"schema_version": 1', '{"schema_version": 2}'],
    ids=["missing", "corrupt", "unsupported_schema"],
)
def test_invalid_metadata_file_is_rejected(tmp_path, content):
    metadata_dir = tmp_path / "torchturbo"
    metadata_dir.mkdir()
    if content is not None:
        (metadata_dir / "checkpoint.json").write_text(content)

    with pytest.raises(RuntimeError, match="Invalid checkpoint metadata"):
        load_metadata(str(tmp_path))
