"""LoRA preference-training checkpoints retain unmerged adapters on resume."""

import copy
import json
import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from transformers import PretrainedConfig

from torchturbo.engine.backends.fsdp2.checkpoint.manager import FSDP2CheckpointManager
from torchturbo.engine.checkpoint import CheckpointFormat, layout
from torchturbo.engine.checkpoint.coordinator import (
    CheckpointCoordinator,
    CheckpointExportPluginConsumer,
)
from torchturbo.engine.config import EngineConfig
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.runtime.cpu import CpuRuntime


class PreferenceModel(nn.Module):
    """A token projection with two possible responses for each prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.all_tied_weights_keys = {}
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def get_weight_converter(
        self, parallel_context: dict, model_path: str
    ) -> BaseWeightConverter:
        return BaseWeightConverter(self, parallel_context, model_path)


@pytest.fixture
def checkpoint_manager(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("edp", "ep"))
        parallel_context = {
            "ep": {"ep_rank": 0, "ep_size": 1, "ep_group": mesh["ep"].get_group()},
        }
        manager = FSDP2CheckpointManager(
            rank=0,
            mesh=mesh,
            world_mesh=mesh,
            parallel_context=parallel_context,
            device_runtime=CpuRuntime(),
            cpu_offload=False,
            require_optimizer_not_suspended=lambda: None,
        )
        yield CheckpointCoordinator(
            EngineConfig.from_dict({"checkpoint": {"hf_export": "final"}}),
            manager,
            parallel_context=parallel_context,
            export_plugin_consumer=CheckpointExportPluginConsumer(),
        )
    finally:
        dist.destroy_process_group()


def _log_ratio(model, inputs):
    logps = model(inputs).log_softmax(-1)
    return logps[:, 0] - logps[:, 1]


def _dpo_step(model, optimizer, inputs, reference_log_ratio):
    logits = 0.2 * (_log_ratio(model, inputs) - reference_log_ratio)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def _load_hf_export(checkpoint_dir):
    output = checkpoint_dir / layout.HF_MODEL_DIR
    index = json.loads((output / layout.HF_INDEX_FILE).read_text())
    state = {}
    for filename in set(index["weight_map"].values()):
        state.update(load_file(str(output / filename)))
    assert set(state) == set(index["weight_map"])
    return state


@pytest.mark.parametrize("use_rslora", [False, True])
def test_dpo_resume_keeps_adapter_and_exports_one_merge(
    tmp_path, checkpoint_manager, use_rslora
):
    torch.manual_seed(17)
    base = PreferenceModel()
    original_base = copy.deepcopy(base.state_dict())
    source = tmp_path / "source"
    base.config.save_pretrained(source)
    save_file(original_base, str(source / "model.safetensors"))
    (source / layout.HF_INDEX_FILE).write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in original_base},
    }))

    def build():
        model = get_peft_model(
            copy.deepcopy(base),
            LoraConfig(
                r=2, lora_alpha=3, target_modules=["linear"], use_rslora=use_rslora
            ),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.02,
            weight_decay=0.0,
        )
        return model, optimizer

    model, optimizer = build()
    inputs = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.75]])
    with torch.no_grad():
        reference_log_ratio = _log_ratio(base, inputs)
    initial_adapter = copy.deepcopy(model.state_dict())
    for _ in range(3):
        _dpo_step(model, optimizer, inputs, reference_log_ratio)
    trained = copy.deepcopy(model.state_dict())
    adapter_names = [name for name in trained if "lora_" in name]
    assert all(not torch.equal(trained[name], initial_adapter[name]) for name in adapter_names)

    def save_and_check(current_model, checkpoint_dir):
        before_save = copy.deepcopy(current_model.state_dict())
        formats = checkpoint_manager.save_model(
            current_model, str(checkpoint_dir), source_path=str(source), final=True
        )
        assert formats == (CheckpointFormat.NATIVE, CheckpointFormat.HF)
        torch.testing.assert_close(current_model.state_dict(), before_save, rtol=0, atol=0)
        layer = current_model.base_model.model.linear
        assert not layer.base_layer.weight.requires_grad
        assert layer.base_layer.weight.grad is None
        assert not layer.merged_adapters
        torch.testing.assert_close(layer.base_layer.weight, original_base["linear.weight"])

        merged = copy.deepcopy(current_model).merge_and_unload()
        exported = _load_hf_export(checkpoint_dir)
        assert set(exported) == set(original_base)
        torch.testing.assert_close(exported, merged.state_dict(), rtol=0, atol=0)
        restored_base = copy.deepcopy(base)
        restored_base.load_state_dict(exported)
        torch.testing.assert_close(restored_base(inputs), current_model(inputs))

    first_checkpoint = tmp_path / "checkpoint-3"
    save_and_check(model, first_checkpoint)
    checkpoint_manager.save_optimizer_and_scheduler(
        model, optimizer, None, str(first_checkpoint)
    )

    resumed, resumed_optimizer = build()
    checkpoint_manager.load_model(resumed, str(first_checkpoint), source_path=str(source))
    checkpoint_manager.load_optimizer_and_scheduler(
        resumed, resumed_optimizer, None, str(first_checkpoint)
    )
    torch.testing.assert_close(resumed.state_dict(), trained, rtol=0, atol=0)
    torch.testing.assert_close(resumed_optimizer.state_dict(), optimizer.state_dict())
    torch.testing.assert_close(resumed(inputs), model(inputs), rtol=0, atol=0)
    for _ in range(2):
        expected_loss = _dpo_step(model, optimizer, inputs, reference_log_ratio)
        actual_loss = _dpo_step(resumed, resumed_optimizer, inputs, reference_log_ratio)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed_optimizer.state_dict(), optimizer.state_dict())
    save_and_check(resumed, tmp_path / "checkpoint-5")
