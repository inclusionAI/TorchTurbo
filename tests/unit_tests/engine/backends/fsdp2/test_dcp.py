"""Tests for FSDP2 DCP model and optimizer state."""

import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import PretrainedConfig, PreTrainedModel

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.runtime.cpu import CpuRuntime


def _checkpoint() -> DcpCheckpoint:
    # Plain tensors are never sharded on the edp mesh, so no key gets an EP
    # suffix. DCP supports single-process CPU save/load without a process group.
    return DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())


class FrozenBaseModel(nn.Module):
    """A frozen base layer next to a trainable layer and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3)
        self.register_buffer("bias_state", torch.zeros(3))
        self.base.requires_grad_(False)


class SmallerModel(nn.Module):
    """A model that consumes a subset of a larger checkpoint's state."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3)


class LargerModel(SmallerModel):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_only = nn.Linear(3, 3)


def _saved_model_keys(model_dir: str) -> set[str]:
    metadata = FileSystemReader(model_dir).read_metadata()
    return set(metadata.state_dict_metadata)


def test_load_adamw_checkpoint_without_parameter_names(tmp_path):
    torch.manual_seed(7)
    reference_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    reference_optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in reference_model.parameters() if p.ndim > 1],
                "weight_decay": 0.1,
            },
            {
                "params": [p for p in reference_model.parameters() if p.ndim == 1],
                "weight_decay": 0.0,
            },
        ],
        lr=0.002,
        weight_decay=0.1,
    )
    for step in range(3):
        for index, param in enumerate(reference_model.parameters()):
            param.grad = torch.full_like(param, (step + 1) * (index + 1) / 10)
        reference_optimizer.step()
        reference_optimizer.zero_grad()

    checkpoint = _checkpoint()
    model_dir = str(tmp_path / "model")
    optimizer_dir = str(tmp_path / "optimizer")
    checkpoint.save_model(reference_model, model_dir)
    checkpoint.save_optimizer(reference_optimizer, reference_model, optimizer_dir)

    resumed_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    plan = OptimizerClassPlan(
        resumed_model,
        torch.optim.AdamW,
        {"lr": 0.002, "weight_decay": 0.1},
    )
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._device_runtime = CpuRuntime()
    backend.fsdp_config = FSDPConfig()
    backend._mesh = {"edp": object()}
    resumed_optimizer = backend.prepare_optimizer(plan)

    checkpoint.load_model(resumed_model, model_dir)
    checkpoint.load_optimizer(resumed_optimizer, resumed_model, optimizer_dir)
    torch.testing.assert_close(
        get_optimizer_state_dict(resumed_model, resumed_optimizer)["state"],
        get_optimizer_state_dict(reference_model, reference_optimizer)["state"],
        rtol=0,
        atol=0,
    )

    for model, optimizer in (
        (reference_model, reference_optimizer),
        (resumed_model, resumed_optimizer),
    ):
        for index, param in enumerate(model.parameters()):
            param.grad = torch.full_like(param, -(index + 1) / 5)
        optimizer.step()

    torch.testing.assert_close(
        resumed_model.state_dict(), reference_model.state_dict(), rtol=0, atol=0
    )
    resumed_state = get_optimizer_state_dict(resumed_model, resumed_optimizer)
    reference_state = get_optimizer_state_dict(reference_model, reference_optimizer)
    assert resumed_state["param_groups"] == reference_state["param_groups"]
    torch.testing.assert_close(
        resumed_state["state"], reference_state["state"], rtol=0, atol=0
    )


def test_model_state_stores_frozen_parameters_and_buffers(tmp_path):
    torch.manual_seed(0)
    trained = FrozenBaseModel()
    trained.bias_state.fill_(3.0)
    model_dir = str(tmp_path / "model")
    _checkpoint().save_model(trained, model_dir)

    assert _saved_model_keys(model_dir) == {
        "model.base.weight",
        "model.base.bias",
        "model.adapter.weight",
        "model.adapter.bias",
        "model.bias_state",
    }

    torch.manual_seed(1)
    resumed = FrozenBaseModel()
    _checkpoint().load_model(resumed, model_dir)

    torch.testing.assert_close(
        resumed.state_dict(), trained.state_dict(), rtol=0, atol=0
    )


def test_load_restores_stored_entries_that_are_frozen_now(tmp_path):
    torch.manual_seed(0)
    trained = FrozenBaseModel()
    trained.base.requires_grad_(True)
    model_dir = str(tmp_path / "model")
    _checkpoint().save_model(trained, model_dir)

    torch.manual_seed(1)
    resumed = FrozenBaseModel()
    _checkpoint().load_model(resumed, model_dir)

    torch.testing.assert_close(
        resumed.state_dict(), trained.state_dict(), rtol=0, atol=0
    )


class TiedModel(PreTrainedModel):
    """An embedding shared with an output projection."""

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self.embed = nn.Embedding(4, 3)
        self.head = nn.Linear(3, 4, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.head.weight = self.embed.weight


def test_load_preserves_tied_parameter_identity(tmp_path):
    torch.manual_seed(0)
    trained = TiedModel()
    model_dir = str(tmp_path / "model")
    _checkpoint().save_model(trained, model_dir)

    resumed = TiedModel()
    original_parameter = resumed.embed.weight

    _checkpoint().load_model(resumed, model_dir)

    assert resumed.embed.weight is original_parameter
    assert resumed.head.weight is original_parameter
    torch.testing.assert_close(
        resumed.embed.weight, trained.embed.weight, rtol=0, atol=0
    )


class CloningStateDictModel(nn.Module):
    """A module whose state-dict hook detaches values from live parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.register_state_dict_post_hook(self._clone_state_dict)

    @staticmethod
    def _clone_state_dict(module, state_dict, prefix, local_metadata) -> None:
        for name, value in state_dict.items():
            state_dict[name] = value.clone()


def test_load_applies_populated_state_through_model_load_state_dict(tmp_path):
    saved = CloningStateDictModel()
    saved.weight.data.fill_(1.0)
    model_dir = str(tmp_path / "model")
    _checkpoint().save_model(saved, model_dir)

    resumed = CloningStateDictModel()
    resumed.weight.data.fill_(9.0)
    _checkpoint().load_model(resumed, model_dir)

    torch.testing.assert_close(resumed.weight, saved.weight, rtol=0, atol=0)


def test_load_rejects_missing_entries(tmp_path):
    model_dir = str(tmp_path / "model")
    _checkpoint().save_model(SmallerModel(), model_dir)
    with pytest.raises(RuntimeError, match=r"missing entries.*checkpoint_only"):
        _checkpoint().load_model(LargerModel(), model_dir)


def test_load_ignores_checkpoint_only_entries(tmp_path):
    model_dir = str(tmp_path / "model")
    saved = LargerModel()
    _checkpoint().save_model(saved, model_dir)
    resumed = SmallerModel()
    _checkpoint().load_model(resumed, model_dir)
    torch.testing.assert_close(resumed.shared.state_dict(), saved.shared.state_dict())


def test_ep_size_validation_rejects_a_different_size():
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    saved_keys = {
        "model.experts.weight::ep0",
        "model.experts.weight::ep1",
    }

    with pytest.raises(ValueError, match=r"saved experts\.weight on EP ranks \[0, 1\]"):
        checkpoint._validate_ep_size(
            saved_keys,
            frozenset({"experts.weight"}),
            "/checkpoint/model_distcp",
        )


def test_ep_size_validation_ignores_checkpoint_only_entries():
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    saved_keys = {
        "model.removed_experts.weight::ep0",
        "model.removed_experts.weight::ep1",
    }

    checkpoint._validate_ep_size(
        saved_keys,
        frozenset({"current_experts.weight"}),
        "/checkpoint/model_distcp",
    )


def test_ep_size_validation_rejects_a_missing_rank():
    checkpoint = DcpCheckpoint(ep_rank=1, ep_size=2, edp_mesh=object())
    saved_keys = {"model.experts.weight::ep1"}

    with pytest.raises(ValueError, match=r"saved experts\.weight on EP ranks \[1\]"):
        checkpoint._validate_ep_size(
            saved_keys,
            frozenset({"experts.weight"}),
            "/checkpoint/model_distcp",
        )


def test_trained_then_frozen_weights_survive_subsequent_checkpoints(tmp_path):
    trained = FrozenBaseModel()
    trained.base.weight.data.fill_(3)
    checkpoint = _checkpoint()
    checkpoint.save_model(trained, str(tmp_path / "first"))
    resumed = FrozenBaseModel()
    resumed.base.weight.data.fill_(1)
    checkpoint.load_model(resumed, str(tmp_path / "first"))
    checkpoint.save_model(resumed, str(tmp_path / "second"))
    restored = FrozenBaseModel()
    checkpoint.load_model(restored, str(tmp_path / "second"))
    torch.testing.assert_close(restored.state_dict(), trained.state_dict())


class ExtraStateModel(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.extra = {"version": version, "nested": {"value": torch.tensor([version])}}

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state


def test_nested_extra_state_round_trip(tmp_path):
    model = ExtraStateModel(7)
    _checkpoint().save_model(model, str(tmp_path))
    resumed = ExtraStateModel(0)
    _checkpoint().load_model(resumed, str(tmp_path))
    torch.testing.assert_close(resumed.state_dict(), model.state_dict())


def test_dcp_rejects_missing_base_weights_after_removing_lora_wrappers(tmp_path):
    from peft import LoraConfig, get_peft_model

    def build(targets):
        base = nn.Module()
        base.a = nn.Linear(2, 2, bias=False)
        base.b = nn.Linear(2, 2, bias=False)
        return get_peft_model(base, LoraConfig(r=1, target_modules=targets))

    model = build(["a", "b"])
    _checkpoint().save_model(model, str(tmp_path))
    with pytest.raises(RuntimeError, match=r"missing entries.*b.weight"):
        _checkpoint().load_model(build(["a"]), str(tmp_path))
