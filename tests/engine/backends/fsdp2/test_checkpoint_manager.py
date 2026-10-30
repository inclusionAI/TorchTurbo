"""Tests for the FSDP2 checkpoint manager."""

import json
import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
from torch import nn

from torchturbo.engine.checkpoint import ComponentStatus
from torchturbo.engine.checkpoint import CheckpointFormat
from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator, CheckpointExportPluginConsumer
from torchturbo.engine.checkpoint import coordinator as coordinator_module
from torchturbo.engine.backends.fsdp2.checkpoint import layout as fsdp_layout
from torchturbo.engine.checkpoint import layout
from torchturbo.engine.backends.fsdp2.checkpoint.manager import FSDP2CheckpointManager
from torchturbo.engine.config import EngineConfig, MuonOptimizerConfig
from torchturbo.engine.optimizers import MuonPlan, MuonRecipe, OptimizersContainer
from torchturbo.engine.optimizers.lr_scheduler import create_lr_scheduler
from torchturbo.runtime.cpu import CpuRuntime


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning_rank0(self, message, *args, **kwargs):
        self.warnings.append(message % args)

    def info_rank0(self, *args, **kwargs):
        pass


class ModelConfig:
    model_type = "test"

    def to_dict(self):
        return {"model_type": self.model_type}


class ConvertibleModel(nn.Module):
    """Records the source paths of requested weight converters."""

    def __init__(self, lora: bool = False) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)
        if lora:
            self.layer.lora_A = nn.ModuleDict(
                {"default": nn.Linear(2, 1, bias=False)}
            )
        self.converter_paths = []
        self.config = ModelConfig()

    def get_weight_converter(self, parallel_context, model_path):
        self.converter_paths.append(model_path)
        return object()


@pytest.fixture
def export_plugin_consumer():
    return CheckpointExportPluginConsumer()


@pytest.fixture
def manager(export_plugin_consumer):
    parallel_context = {
        "ep": {"ep_rank": 0, "ep_size": 1},
        "sp": {"sp_rank": 0, "sp_size": 1},
    }

    checkpoint_manager = FSDP2CheckpointManager(
        rank=0,
        mesh={"edp": object()},
        world_mesh=None,
        parallel_context=parallel_context,
        device_runtime=CpuRuntime(),
        cpu_offload=False,
        require_optimizer_not_suspended=lambda: None,
    )
    config = EngineConfig.from_dict({"checkpoint": {"hf_export": "final"}})
    return CheckpointCoordinator(
        config, checkpoint_manager, parallel_context=parallel_context,
        export_plugin_consumer=export_plugin_consumer,
    )


@pytest.fixture
def recorder(monkeypatch):
    recording_logger = RecordingLogger()
    monkeypatch.setattr(coordinator_module, "logger", recording_logger)
    return recording_logger


def _make_training_state():
    """Return a model and a two-child optimizer with per-child state."""
    model = nn.Linear(2, 2)
    model.config = ModelConfig()
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=2.5e-5, adamw_lr=1e-5, weight_decay=0.0),
    )
    groups = plan.get_param_groups_by_optimizer_name()
    optimizers = {
        "Muon": torch.optim.SGD(
            [group.get_optimizer_param_group() for group in groups["Muon"]],
            lr=2.5e-5,
        ),
        "AdamW": torch.optim.AdamW(
            [group.get_optimizer_param_group() for group in groups["AdamW"]],
            lr=1e-5,
        ),
    }
    return model, OptimizersContainer(plan, optimizers)


def _make_scheduler(optimizer):
    return create_lr_scheduler(
        "cosine_with_min_lr",
        optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        scheduler_specific_kwargs={"min_lr": 1e-7},
    )


def _train_step(model, optimizer, scheduler, step):
    for index, param in enumerate(model.parameters()):
        param.grad = torch.full_like(param, (step + 1) * (index + 1) / 10)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def _write_hf_export(checkpoint_dir):
    hf_dir = checkpoint_dir / layout.HF_MODEL_DIR
    hf_dir.mkdir(parents=True)
    (hf_dir / "model.safetensors").write_bytes(b"shard")
    (hf_dir / layout.HF_INDEX_FILE).write_text(
        json.dumps({"weight_map": {"layer.weight": "model.safetensors"}})
    )
    return hf_dir


@pytest.mark.parametrize("accelerated", [False, True])
def test_resume_matches_uninterrupted_training(manager, tmp_path, accelerated):
    from accelerate import Accelerator

    def build():
        model, optimizer = _make_training_state()
        stepped_optimizer = optimizer
        if accelerated:
            stepped_optimizer = Accelerator(cpu=True).prepare_optimizer(
                optimizer, device_placement=False
            )
        return model, optimizer, stepped_optimizer, _make_scheduler(stepped_optimizer)

    torch.manual_seed(0)
    model, optimizer, stepped_optimizer, scheduler = build()
    for step in range(4):
        _train_step(model, stepped_optimizer, scheduler, step)
    checkpoint_dir = str(tmp_path / "checkpoint-4")
    manager.save_model(
        model,
        checkpoint_dir,
        source_path="unused",
    )
    manager.save_optimizer_and_scheduler(model, optimizer, scheduler, checkpoint_dir)

    torch.manual_seed(1)
    resumed = build()
    resumed_model, resumed_optimizer, resumed_stepped, resumed_scheduler = resumed
    stale_groups = list(resumed_optimizer.param_groups)
    manager.load_model(resumed_model, checkpoint_dir, source_path="unused")
    manager.load_optimizer_and_scheduler(
        resumed_model, resumed_optimizer, resumed_scheduler, checkpoint_dir
    )

    # Loading replaced the child group dicts; the container serves the new ones.
    for group, stale_group, child in zip(
        resumed_stepped.param_groups,
        stale_groups,
        resumed_optimizer.optimizers,
        strict=True,
    ):
        assert group is not stale_group
        assert group is child.param_groups[0]

    for step in range(4, 10):
        _train_step(model, stepped_optimizer, scheduler, step)
        _train_step(resumed_model, resumed_stepped, resumed_scheduler, step)
        assert resumed_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
    torch.testing.assert_close(
        resumed_model.state_dict(), model.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("saved_ep_size,current_ep_size", [(1, 2), (2, 1)])
def test_optimizer_restore_rejects_ep_changes_before_mutation(
    manager, tmp_path, saved_ep_size, current_ep_size
):
    manager._config.parallel_plan.ep_size = saved_ep_size
    model = ConvertibleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    model.layer(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    manager.save_model(model, str(tmp_path), source_path="unused")
    manager.save_optimizer_and_scheduler(model, optimizer, None, str(tmp_path))

    manager._parallel_context["ep"]["ep_size"] = current_ep_size
    resumed = ConvertibleModel()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.07)
    with pytest.raises(
        ValueError, match=f"saved EP size {saved_ep_size}.*{current_ep_size}"
    ):
        manager.load_optimizer_and_scheduler(
            resumed, resumed_optimizer, None, str(tmp_path)
        )
    assert not resumed_optimizer.state
    assert resumed_optimizer.param_groups[0]["lr"] == 0.07


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("frozen", [False, True])
def test_optimizer_restore_keeps_unused_parameters_without_state(
    manager, tmp_path, ep_size, frozen
):
    manager._config.parallel_plan.ep_size = ep_size
    manager._parallel_context["ep"]["ep_size"] = ep_size

    def build():
        model = ConvertibleModel()
        model.unused = nn.Linear(2, 2)
        model.unused.requires_grad_(not frozen)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.02,
        )
        return model, optimizer

    model, optimizer = build()
    model.layer(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    manager.save_model(model, str(tmp_path), source_path="unused")
    manager.save_optimizer_and_scheduler(model, optimizer, None, str(tmp_path))

    resumed, resumed_optimizer = build()
    manager.load_optimizer_and_scheduler(
        resumed, resumed_optimizer, None, str(tmp_path)
    )
    assert all(
        parameter not in resumed_optimizer.state
        for parameter in resumed.unused.parameters()
    )
    torch.testing.assert_close(
        resumed_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("lora", [False, True])
def test_load_model_restores_hf_exports_with_a_warning(
    manager, recorder, tmp_path, monkeypatch, lora
):
    hf_dir = _write_hf_export(tmp_path)
    loaded_paths = []

    class RecordingLoader:
        def __init__(self, path):
            self.path = path

        def load_weights(self, *, assign=False):
            loaded_paths.append(self.path)

    monkeypatch.setattr(
        manager._manager,
        "_hf_loader",
        lambda model, path, converter: RecordingLoader(path),
    )
    model = ConvertibleModel(lora=lora)

    manager.load_model(model, str(tmp_path), source_path="/source")

    assert loaded_paths == [str(hf_dir)]
    assert model.converter_paths == ["/source"]
    (warning,) = recorder.warnings
    assert "rounds away fp32 updates" in warning
    assert ("LoRA adapters" in warning) is lora


def test_coordinator_reuses_model_converter_for_hf_load_and_save(
    manager, tmp_path, monkeypatch
):
    loads = []
    saves = []

    class RecordingLoader:
        def __init__(self, model, path, converter):
            self.args = (model, path, converter)

        def load_weights(self, *, assign=False):
            loads.append((*self.args, assign))

    class RecordingSaver:
        def __init__(self, model, source_path, converter):
            self.args = (model, source_path, converter)

        def save_model(self, output_dir):
            saves.append((*self.args, output_dir))

    monkeypatch.setattr(manager._manager, "_hf_loader", RecordingLoader)
    monkeypatch.setattr(manager._manager, "_hf_saver", RecordingSaver)
    policy = ConvertibleModel()
    reference = ConvertibleModel()

    manager.load_pretrained(policy, "/source")
    manager.load_pretrained(reference, "/source")
    manager.load_pretrained(policy, "/source")
    manager.save_model(
        policy, str(tmp_path / "export"), source_path="/source", format=CheckpointFormat.HF
    )
    hf_dir = _write_hf_export(tmp_path / "resume")
    manager.load_model(policy, str(hf_dir.parent), source_path="/source")
    manager.load_pretrained(policy, "/new-source")

    assert policy.converter_paths == ["/source", "/new-source"]
    assert reference.converter_paths == ["/source"]
    assert loads[0][2] is loads[2][2] is loads[3][2] is saves[0][2]
    assert loads[1][2] is not loads[0][2]
    assert loads[4][2] is not loads[0][2]
    assert [load[3] for load in loads] == [True, True, True, False, True]


def test_load_model_rejects_checkpoints_without_complete_weights(manager, tmp_path):
    (tmp_path / fsdp_layout.MODEL_DCP_DIR).mkdir()
    with pytest.raises(RuntimeError, match="interrupted"):
        manager.load_model(nn.Linear(2, 2), str(tmp_path), source_path="unused")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No model weights"):
        manager.load_model(nn.Linear(2, 2), str(empty_dir), source_path="unused")


def test_missing_optimizer_state_is_not_resumable(manager, tmp_path):
    model, optimizer = _make_training_state()
    scheduler = _make_scheduler(optimizer)

    with pytest.raises(
        FileNotFoundError, match="cannot be used with resume_from_checkpoint"
    ) as error:
        manager.load_optimizer_and_scheduler(
            model, optimizer, scheduler, str(tmp_path)
        )
    message = str(error.value)
    assert "interrupted" in message
    assert "save_only_model" in message
    assert "new output_dir" in message
    assert f"model_name_or_path={tmp_path / layout.HF_MODEL_DIR}" in message


def test_starting_a_dcp_save_discards_stale_training_state(
    manager, tmp_path, monkeypatch
):
    model, optimizer = _make_training_state()
    scheduler = _make_scheduler(optimizer)
    manager.save_model(
        model,
        str(tmp_path),
        source_path="unused",
    )
    manager.save_optimizer_and_scheduler(model, optimizer, scheduler, str(tmp_path))
    _write_hf_export(tmp_path)

    def fail_save(model, dcp_dir):
        assert os.listdir(dcp_dir) == []
        assert not (tmp_path / fsdp_layout.OPTIMIZER_DCP_DIR).exists()
        assert not (tmp_path / layout.SCHEDULER_FILE).exists()
        assert not (tmp_path / layout.HF_MODEL_DIR).exists()
        raise RuntimeError("DCP save failed")

    monkeypatch.setattr(manager._manager._dcp, "save_model", fail_save)

    with pytest.raises(RuntimeError, match="DCP save failed"):
        manager.save_model(
            ConvertibleModel(),
            str(tmp_path),
            source_path="/models/source",
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        manager.load_model(model, str(tmp_path), source_path="unused")
    with pytest.raises(
        FileNotFoundError, match="cannot be used with resume_from_checkpoint"
    ):
        manager.load_optimizer_and_scheduler(model, optimizer, scheduler, str(tmp_path))


def test_model_only_dcp_loads_weights_but_cannot_resume_training(manager, tmp_path):
    model, optimizer = _make_training_state()
    manager.save_model(
        model,
        str(tmp_path),
        format=CheckpointFormat.NATIVE,
        source_path="/models/source",
    )

    manager.load_model(model, str(tmp_path), source_path="/models/source")
    with pytest.raises(
        FileNotFoundError, match="cannot be used with resume_from_checkpoint"
    ):
        manager.load_optimizer_and_scheduler(
            model, optimizer, None, str(tmp_path)
        )


def test_interrupted_optimizer_save_is_rejected(manager, tmp_path):
    (tmp_path / fsdp_layout.OPTIMIZER_DCP_DIR).mkdir()
    model, optimizer = _make_training_state()

    with pytest.raises(RuntimeError, match="interrupted"):
        manager.load_optimizer_and_scheduler(model, optimizer, None, str(tmp_path))


def test_starting_optimizer_save_discards_a_stale_scheduler(
    manager, tmp_path, monkeypatch
):
    model, optimizer = _make_training_state()
    scheduler = _make_scheduler(optimizer)
    layout.save_scheduler(scheduler.state_dict(), str(tmp_path))

    def fail_save(model, optimizer, dcp_dir):
        assert os.listdir(dcp_dir) == []
        assert not (tmp_path / layout.SCHEDULER_FILE).exists()
        raise RuntimeError("optimizer save failed")

    monkeypatch.setattr(manager._manager, "save_optimizer", fail_save)

    with pytest.raises(RuntimeError, match="optimizer save failed"):
        manager.save_optimizer_and_scheduler(model, optimizer, scheduler, str(tmp_path))


def test_missing_scheduler_state_is_rejected(manager, tmp_path):
    model, optimizer = _make_training_state()
    scheduler = _make_scheduler(optimizer)
    manager.save_model(
        model,
        str(tmp_path),
        source_path="unused",
    )
    manager.save_optimizer_and_scheduler(model, optimizer, scheduler, str(tmp_path))
    os.remove(tmp_path / layout.SCHEDULER_FILE)

    with pytest.raises(FileNotFoundError, match="missing"):
        manager.load_optimizer_and_scheduler(
            model, optimizer, scheduler, str(tmp_path)
        )


def test_save_without_scheduler_removes_stale_scheduler_state(manager, tmp_path):
    model, optimizer = _make_training_state()
    scheduler = _make_scheduler(optimizer)
    manager.save_model(
        model,
        str(tmp_path),
        source_path="unused",
    )
    layout.save_scheduler(scheduler.state_dict(), str(tmp_path))

    manager.save_optimizer_and_scheduler(model, optimizer, None, str(tmp_path))

    assert not (tmp_path / layout.SCHEDULER_FILE).exists()
    with pytest.raises(FileNotFoundError, match="missing"):
        manager.load_optimizer_and_scheduler(
            model, optimizer, scheduler, str(tmp_path)
        )


@pytest.mark.parametrize(
    "model_format", [CheckpointFormat.NATIVE, CheckpointFormat.HF]
)
def test_save_model_replaces_the_previous_save(
    manager, tmp_path, monkeypatch, model_format
):
    subdir = (
        fsdp_layout.MODEL_DCP_DIR
        if model_format is CheckpointFormat.NATIVE
        else layout.HF_MODEL_DIR
    )
    target_dir = tmp_path / subdir
    target_dir.mkdir()
    (target_dir / "stale").write_bytes(b"previous save")
    saved_paths = []

    class RecordingSaver:
        def save_model(self, path):
            saved_paths.append(path)

    monkeypatch.setattr(
        manager._manager,
        "_hf_saver",
        lambda model, source_path, converter: RecordingSaver(),
    )
    model = ConvertibleModel()

    manager.save_model(model, str(tmp_path), format=model_format, source_path="/source")

    assert not (target_dir / "stale").exists()
    if model_format is CheckpointFormat.HF:
        assert saved_paths == [str(target_dir)]
        assert model.converter_paths == ["/source"]
    else:
        assert fsdp_layout.dcp_status(str(target_dir)) is ComponentStatus.COMPLETE


@pytest.mark.parametrize("symlink", [False, True])
def test_export_rejects_source_overlap_before_removing_files(manager, tmp_path, symlink):
    source = tmp_path / "huggingface"
    source.mkdir()
    marker = source / "model.safetensors"
    marker.write_bytes(b"original model")
    source_path = source
    if symlink:
        source_path = tmp_path / "source_link"
        source_path.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps source"):
        manager.save_model(
            ConvertibleModel(), str(tmp_path), source_path=str(source_path),
            format=CheckpointFormat.HF,
        )
    assert marker.read_bytes() == b"original model"


def test_model_only_save_removes_previous_native_and_optimizer_state(manager, tmp_path, monkeypatch):
    model = ConvertibleModel()
    optimizer = torch.optim.AdamW(model.parameters())
    manager.save_model(model, str(tmp_path), source_path="/source")
    manager.save_optimizer_and_scheduler(model, optimizer, _make_scheduler(optimizer), str(tmp_path))
    monkeypatch.setattr(manager._manager, "save_hf", lambda *args, **kwargs: None)
    manager.save_model(model, str(tmp_path), source_path="/source", format=CheckpointFormat.HF)
    assert not (tmp_path / fsdp_layout.MODEL_DCP_DIR).exists()
    assert not (tmp_path / fsdp_layout.OPTIMIZER_DCP_DIR).exists()
    assert not (tmp_path / layout.SCHEDULER_FILE).exists()


def test_combined_save_keeps_just_written_native_state(manager, tmp_path, monkeypatch):
    model = ConvertibleModel()
    monkeypatch.setattr(manager._manager, "save_hf", lambda *args, **kwargs: None)
    manager.save_model(model, str(tmp_path), source_path="/source", final=True)
    assert (tmp_path / fsdp_layout.MODEL_DCP_DIR / ".metadata").is_file()


@pytest.mark.parametrize("extra_adapter", [False, True])
def test_native_load_ignores_checkpoint_only_parameters(
    manager, recorder, tmp_path, extra_adapter
):
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig

    saved = ConvertibleModel()
    resumed = ConvertibleModel()
    if extra_adapter:
        saved.config = PretrainedConfig()
        resumed.config = PretrainedConfig()
        config = LoraConfig(r=1, target_modules=["layer"])
        saved = get_peft_model(saved, config)
        resumed = get_peft_model(resumed, LoraConfig(r=1, target_modules=["layer"]))
        saved.add_adapter(
            "checkpoint_only",
            LoraConfig(r=2, lora_alpha=64, target_modules=["layer"], use_dora=True),
        )
    else:
        saved.checkpoint_only = nn.Linear(2, 2)

    with torch.no_grad():
        for parameter in saved.parameters():
            parameter.fill_(0.5)
    manager.save_model(saved, str(tmp_path), source_path="/source")
    manager.load_model(resumed, str(tmp_path), source_path="/source")

    expected = {name: saved.state_dict()[name] for name in resumed.state_dict()}
    torch.testing.assert_close(resumed.state_dict(), expected, rtol=0, atol=0)
    assert recorder.warnings == []


def test_native_load_keeps_new_training_settings_and_rejects_scaling_changes(
    manager, tmp_path
):
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig

    from torchturbo.engine.config import FreezePolicy

    def build(**settings):
        base = nn.Module()
        base.config = PretrainedConfig()
        base.linear = nn.Linear(2, 2, bias=False)
        base.other = nn.Linear(2, 2, bias=False)
        return get_peft_model(
            base, LoraConfig(r=2, target_modules=["linear", "other"], **settings)
        )

    trained = build(lora_alpha=32)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.fill_(0.5)
    manager.save_model(trained, str(tmp_path), source_path="/source")

    resumed = build(lora_alpha=32, lora_dropout=0.25, init_lora_weights=False)
    manager._config.freeze_policy = FreezePolicy.from_list(["all"])
    resumed.requires_grad_(False)
    manager.load_model(resumed, str(tmp_path), source_path="/source")
    torch.testing.assert_close(resumed.state_dict(), trained.state_dict())
    assert resumed.peft_config["default"].lora_dropout == 0.25
    assert resumed.peft_config["default"].init_lora_weights is False
    assert all(not parameter.requires_grad for parameter in resumed.parameters())
    assert manager._config.freeze_policy.enabled

    # Scaling lives outside the saved tensors, so only metadata can catch it.
    changed_scaling = build(lora_alpha=16)
    with pytest.raises(ValueError, match=r"'lora_alpha': \(32, 16\)"):
        manager.load_model(changed_scaling, str(tmp_path), source_path="/source")


@pytest.mark.parametrize("export_fails", [False, True])
def test_export_plugins_run_only_after_successful_hf_save(
    manager, export_plugin_consumer, tmp_path, monkeypatch, export_fails
):
    from torchturbo.engine.plugins import CheckpointExportPlugin, PluginManager

    events = []

    class ExportPlugin(CheckpointExportPlugin):
        def on_hf_export(self, export_dir: str) -> None:
            assert os.path.isfile(os.path.join(export_dir, "complete"))
            events.append("upload")

    plugins = PluginManager()
    plugins.plugins.append(ExportPlugin())
    plugins.try_consume(export_plugin_consumer)

    def save_hf(model, export_dir, **kwargs):
        events.append("save")
        if export_fails:
            raise RuntimeError("export failed")
        with open(os.path.join(export_dir, "complete"), "w") as marker:
            marker.write("saved")

    monkeypatch.setattr(manager._manager, "save_hf", save_hf)
    if export_fails:
        with pytest.raises(RuntimeError, match="export failed"):
            manager.save_model(
                ConvertibleModel(), str(tmp_path), source_path="/source",
                format=CheckpointFormat.HF,
            )
        assert events == ["save"]
    else:
        manager.save_model(
            ConvertibleModel(), str(tmp_path), source_path="/source",
            format=CheckpointFormat.HF,
        )
        assert events == ["save", "upload"]


def test_export_plugin_failure_does_not_fail_save(
    manager, export_plugin_consumer, tmp_path, monkeypatch, recorder
):
    from torchturbo.engine.plugins import CheckpointExportPlugin

    class FailingPlugin(CheckpointExportPlugin):
        def on_hf_export(self, export_dir: str) -> None:
            raise RuntimeError("upload failed")

    export_plugin_consumer.consume(FailingPlugin())
    monkeypatch.setattr(manager._manager, "save_hf", lambda *a, **kw: None)
    manager.save_model(
        ConvertibleModel(), str(tmp_path), source_path="/source",
        format=CheckpointFormat.HF,
    )
    assert any("FailingPlugin failed" in message for message in recorder.warnings)


def test_native_only_save_does_not_notify_export_plugin(
    manager, export_plugin_consumer, tmp_path, monkeypatch
):
    from torchturbo.engine.plugins import CheckpointExportPlugin

    class ExportPlugin(CheckpointExportPlugin):
        def on_hf_export(self, export_dir: str) -> None:
            pytest.fail("Native-only saves must not trigger export plugins.")

    export_plugin_consumer.consume(ExportPlugin())
    monkeypatch.setattr(manager._manager, "save_native", lambda *a, **kw: None)
    manager.save_model(ConvertibleModel(), str(tmp_path), source_path="/source")
