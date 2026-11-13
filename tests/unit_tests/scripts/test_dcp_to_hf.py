"""Tests for the HF export entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest

from torchturbo.engine.checkpoint import CheckpointFormat
import torchturbo.engine as engine_module
import torchturbo.engine.args as args_module


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "dcp_to_hf", REPO_ROOT / "scripts/checkpoint/dcp_to_hf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dcp_to_hf = _load_script()

_REQUIRED = [
    "--model_name_or_path", "/models/base",
    "--checkpoint_dir", "/out/checkpoint-10",
    "--output_dir", "/out/export",
]


def test_lora_config_matches_the_training_adapters():
    from peft import TaskType

    config = dcp_to_hf.build_lora_config(
        dcp_to_hf.parse_args(_REQUIRED + ["--lora_rank", "16", "--lora_alpha", "32", "--lora_target", "all-linear", "--no-use_rslora", "--no-use_dora"])
    )

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.target_modules == "all-linear"
    assert config.task_type == TaskType.CAUSAL_LM
    assert config.use_rslora is False
    assert config.use_dora is False


def test_lora_config_accepts_explicit_targets():
    args = dcp_to_hf.parse_args(
        _REQUIRED
        + ["--lora_rank", "8", "--lora_alpha", "4", "--lora_target", "q_proj, v_proj"]
        + ["--use_rslora", "--use_dora"]
    )

    config = dcp_to_hf.build_lora_config(args)

    assert config.lora_alpha == 4
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    assert config.use_rslora is True
    assert config.use_dora is True


def _grouped_mlp_model():
    from torch import nn
    from transformers import PretrainedConfig

    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_config import MoEConfig

    class MoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.experts = GroupedMLP(
                num_local_experts=2,
                use_te=False,
                config=MoEConfig(
                    hidden_size=4, moe_ffn_hidden_size=3, gated_linear_unit=True
                ),
            )

    return MoEModel()


@pytest.mark.parametrize("source", ["saved_metadata", "explicit_targets"])
def test_targets_without_all_linear_wrap_grouped_mlp(tmp_path, source):
    from peft import LoraConfig

    from torchturbo.engine.checkpoint.metadata import (
        build_metadata,
        load_metadata,
        save_metadata,
    )
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.peft import apply_peft
    from torchturbo.engine.peft.lora import GroupedMLPLoRA

    trained = apply_peft(
        _grouped_mlp_model(),
        LoraConfig(r=2, lora_alpha=4, target_modules="all-linear"),
    )
    if source == "saved_metadata":
        # Metadata keeps the expanded module names, not the "all-linear" marker.
        save_metadata(
            build_metadata(trained, EngineConfig.from_dict({}), "/models/base"),
            str(tmp_path),
        )
        metadata = load_metadata(str(tmp_path))
        assert "all-linear" not in metadata["peft_configs"]["default"]["target_modules"]
        config = dcp_to_hf.build_lora_config(dcp_to_hf.parse_args(_REQUIRED), metadata)
    else:
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "experts"])

    rebuilt = apply_peft(_grouped_mlp_model(), config)

    assert isinstance(rebuilt.base_model.model.experts, GroupedMLPLoRA)
    assert rebuilt.state_dict().keys() == trained.state_dict().keys()


def test_saved_lora_config_ignores_fields_from_a_newer_peft():
    saved = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj"],
        "field_from_newer_peft": 1,
    }
    metadata = {"peft_configs": {"default": saved}}

    with pytest.warns(UserWarning, match="field_from_newer_peft"):
        config = dcp_to_hf.build_lora_config(dcp_to_hf.parse_args(_REQUIRED), metadata)

    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.target_modules == {"q_proj"}


@pytest.mark.parametrize("metadata", [None, {"peft_configs": {}}])
def test_full_fine_tuning_exports_without_peft(monkeypatch, metadata):
    monkeypatch.setitem(sys.modules, "peft", None)

    assert dcp_to_hf.build_lora_config(dcp_to_hf.parse_args(_REQUIRED), metadata) is None


def test_main_restores_the_checkpoint_and_exports_hf(monkeypatch):
    calls = []

    class Engine:
        def __init__(self, config):
            calls.append(("engine", config.parallel_plan.ep_size))

        def prepare_model(self, **kwargs):
            calls.append(
                (
                    "prepare_model",
                    kwargs["model_name_or_path"],
                    kwargs["peft_config"].r,
                )
            )
            return "model"

        def load_from_checkpoint(self, model, checkpoint_dir):
            calls.append(("load_from_checkpoint", model, checkpoint_dir))

        def save_model(self, model, output_dir, **kwargs):
            calls.append(("save_model", model, output_dir, kwargs))

    monkeypatch.setattr(engine_module, "TorchTurboEngine", Engine)
    monkeypatch.setattr(
        engine_module,
        "init_process_group_if_needed",
        lambda enable_cpu_backend: calls.append(("init", enable_cpu_backend)),
    )
    # main() stashes the parsed --tt_* args; restore the global afterwards.
    monkeypatch.setattr(args_module, "_current_torchturbo_args", None)

    dcp_to_hf.main(_REQUIRED + ["--tt_ep_size", "4", "--lora_rank", "16", "--lora_alpha", "32", "--lora_target", "all-linear", "--no-use_rslora", "--no-use_dora"])

    assert calls == [
        ("init", True),
        ("engine", 4),
        ("prepare_model", "/models/base", 16),
        ("load_from_checkpoint", "model", "/out/checkpoint-10"),
        ("save_model", "model", "/out/export", {"format": CheckpointFormat.HF}),
    ]


def test_missing_required_arguments_fail_fast():
    with pytest.raises(SystemExit):
        dcp_to_hf.parse_args(["--model_name_or_path", "/models/base"])


@pytest.mark.parametrize("adapter_file", [False, True])
def test_export_uses_saved_swift_scaling_and_rejects_overrides(tmp_path, adapter_file):
    from peft import LoraConfig

    config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj"], use_rslora=True)
    metadata = {"peft_configs": {"default": config.to_dict()}}
    options = _REQUIRED
    if adapter_file:
        config.save_pretrained(tmp_path)
        options = _REQUIRED + ["--adapter_config", str(tmp_path / "adapter_config.json")]
        metadata = None
    args = dcp_to_hf.parse_args(options)
    actual = dcp_to_hf.build_lora_config(args, metadata)
    assert actual.lora_alpha == 32
    assert actual.use_rslora is True
    for option in (
        ["--lora_rank", "4"], ["--lora_alpha", "16"], ["--lora_target", "v_proj"],
        ["--no-use_rslora"], ["--use_dora"],
    ):
        with pytest.raises(ValueError, match="differs from saved"):
            dcp_to_hf.build_lora_config(dcp_to_hf.parse_args(options + option), metadata)


def test_legacy_lora_export_never_guesses_scaling():
    with pytest.raises(ValueError, match="explicit rank, alpha"):
        dcp_to_hf.build_lora_config(dcp_to_hf.parse_args(_REQUIRED + ["--lora_rank", "8"]))


@pytest.mark.parametrize("destination", ["checkpoint", "checkpoint/huggingface", "."])
def test_export_rejects_paths_that_modify_checkpoint_tree(tmp_path, destination):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="separate"):
        dcp_to_hf.validate_export_paths(str(checkpoint), str(tmp_path / destination))


@pytest.mark.parametrize(
    "name", ["model_distcp", "optimizer_distcp", "scheduler.pt", "trainer_state.json"]
)
def test_export_rejects_output_with_training_checkpoint_state(tmp_path, name):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    output = tmp_path / "checkpoint-900"
    output.mkdir()
    if name.endswith("_distcp"):
        (output / name).mkdir()
    else:
        (output / name).write_text("{}")

    with pytest.raises(ValueError, match=f"training checkpoint state \\({name}\\)"):
        dcp_to_hf.validate_export_paths(str(checkpoint), str(output))


def test_export_accepts_a_previous_export_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    output = tmp_path / "export"
    (output / "huggingface").mkdir(parents=True)
    (output / "torchturbo").mkdir()

    dcp_to_hf.validate_export_paths(str(checkpoint), str(output))


def test_main_rejects_an_output_dir_inside_the_source_model(monkeypatch, tmp_path):
    calls = []

    class Engine:
        def __init__(self, config):
            calls.append("engine")

    monkeypatch.setattr(engine_module, "TorchTurboEngine", Engine)
    monkeypatch.setattr(
        engine_module, "init_process_group_if_needed", lambda enable_cpu_backend: None
    )
    monkeypatch.setattr(args_module, "_current_torchturbo_args", None)
    source = tmp_path / "base"
    source.mkdir()

    with pytest.raises(ValueError, match="must not be the source model directory"):
        dcp_to_hf.main([
            "--model_name_or_path", str(source),
            "--checkpoint_dir", str(tmp_path / "checkpoint-10"),
            "--output_dir", str(source / "export"),
        ])
    assert calls == []


@pytest.mark.parametrize("use_dora", [False, True])
def test_export_round_trip_with_saved_configuration(tmp_path, monkeypatch, use_dora):
    import json

    import torch
    import torch.distributed as dist
    from peft import LoraConfig
    from safetensors.torch import load_file, save_file
    from torch import nn
    from transformers import GPT2Config, PreTrainedModel

    from torchturbo.engine import TorchTurboEngine
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.peft.weights import merge_lora_linear
    from torchturbo.models.base import TurboModel
    from torchturbo.runtime.cpu import CpuRuntime
    import torchturbo.engine.core as core_module
    import torchturbo.models.registry as registry

    class LinearTurboModel(TurboModel, PreTrainedModel):
        config_class = GPT2Config

        def __init__(self, config):
            config._attn_implementation = "eager"
            super().__init__(config)
            self.linear = nn.Linear(3, 2, bias=False)
            self.other = nn.Linear(3, 2, bias=False)
            self.post_init()

    source = tmp_path / "base"
    source.mkdir()
    GPT2Config().save_pretrained(source)
    save_file(
        {"linear.weight": torch.ones(2, 3), "other.weight": torch.ones(2, 3)},
        str(source / "model.safetensors"),
    )
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {
            "linear.weight": "model.safetensors",
            "other.weight": "model.safetensors",
        },
    }))
    monkeypatch.setattr(core_module, "get_device_runtime", lambda: CpuRuntime())
    monkeypatch.setattr(core_module, "get_backend", lambda name: FSDP2Backend)
    monkeypatch.setattr(registry, "get_model", lambda name: LinearTurboModel)
    monkeypatch.setattr(engine_module, "init_process_group_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(args_module, "_current_torchturbo_args", None)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1)
    try:
        engine = TorchTurboEngine({"checkpoint": {"hf_export": "final"}})
        model = engine.prepare_model(
            str(source), peft_config=LoraConfig(
                r=2, lora_alpha=32, target_modules=["linear", "other"],
                use_rslora=True, use_dora=use_dora,
            )
        )
        layer = model.base_model.model.linear
        with torch.no_grad():
            layer.base_layer.weight.fill_(3)
            layer.lora_A["default"].weight.fill_(0.5)
            layer.lora_B["default"].weight.fill_(0.25)
        model.config.n_positions = 4096
        checkpoint = tmp_path / "checkpoint-1"
        engine.save_model(model, str(checkpoint))
        expected = merge_lora_linear(layer.base_layer.weight.full_tensor(), layer, "weight")
        output = tmp_path / "export"
        import torchturbo.engine.backends.fsdp2.checkpoint.weight_saver as saver_module

        synced_inodes = set()
        original_fsync = os.fsync
        original_save_json = saver_module.save_json

        def record_fsync(fd):
            original_fsync(fd)
            synced_inodes.add(os.fstat(fd).st_ino)

        def publish_index(data, path):
            directory = Path(path).parent
            # The export keeps the source config; the runtime n_positions
            # change is not written back.
            assert (directory / "config.json").read_bytes() == (
                source / "config.json"
            ).read_bytes()
            for file in directory.rglob("*"):
                if file.is_file():
                    assert file.stat().st_ino in synced_inodes, file
            original_save_json(data, path)

        monkeypatch.setattr(os, "fsync", record_fsync)
        monkeypatch.setattr(saver_module, "save_json", publish_index)
        dcp_to_hf.main(["--checkpoint_dir", str(checkpoint), "--output_dir", str(output)])
        index = json.loads((output / "huggingface/model.safetensors.index.json").read_text())
        shard = load_file(str(output / "huggingface" / index["weight_map"]["linear.weight"]))
        torch.testing.assert_close(shard["linear.weight"], expected)
        assert (checkpoint / "model_distcp/.metadata").is_file()
        assert json.loads((source / "config.json").read_text())["n_positions"] == 1024
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_export_restores_structural_settings_and_rejects_ep_override():
    from torchturbo.engine import ParameterGroup, TorchTurboArgs
    from torchturbo.engine.args import parse_torchturbo_args

    metadata = {"engine_config": {
        "backend": "fsdp2", "mixed_precision": "fp32",
        "parallel_plan": {"ep_size": 2},
        "mtp": {"enabled": True, "loss_scaling_factor": 0.3, "log_loss": True},
        "freeze_policy": {"parameter_groups": ["all"]},
    }}
    config = dcp_to_hf.build_export_config(TorchTurboArgs(), metadata)
    assert config.parallel_plan.ep_size == 2
    assert config.mtp.enabled is True
    assert config.freeze_policy.enabled is False
    config = dcp_to_hf.build_export_config(
        TorchTurboArgs(
            freeze_language_model=True,
            mixed_precision="bf16",
            mtp_loss_scaling_factor=0.9,
        ),
        metadata,
    )
    assert config.freeze_policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)
    assert config.mixed_precision.value == "bf16"
    assert config.mtp.loss_scaling_factor == 0.9
    args, _ = parse_torchturbo_args(["--tt_ep_size", "1"])
    with pytest.raises(ValueError, match="EP size must match"):
        dcp_to_hf.build_export_config(args, metadata)
