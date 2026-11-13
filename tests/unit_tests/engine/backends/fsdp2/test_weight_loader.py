"""Unit tests for FSDP2 weight loading behavior."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch
import torch.nn as nn

import torchturbo.engine.backends.fsdp2.checkpoint.weight_loader as weight_loader_mod
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig, CPUOffloadConfig, CPUOffloadMode
from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("buf", torch.ones(2))
        self.tie_called = False

    def tie_weights(self):
        self.tie_called = True


def test_load_weights_cpu_offload_assigns_and_restores_buffers(monkeypatch):
    model = TinyModel()

    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cuda:0")
    loader.is_cpu_offload = True
    loader.rank = 0
    loader._sync_log_rank0 = lambda *args, **kwargs: None
    loader._load_local_shards = lambda: {}
    loader._broadcast_buffers = lambda sharded_sd: sharded_sd.update(
        {"buf": torch.tensor([2.0, 2.0])}
    )
    loader._broadcast_params = lambda sharded_sd: sharded_sd.update(
        {"weight": torch.tensor([3.0, 3.0])}
    )
    loader._cleanup = lambda: None

    load_calls = {}

    def fake_load_state_dict(state_dict, assign=False, strict=True):
        load_calls["assign"] = assign
        load_calls["strict"] = strict
        load_calls["devices"] = {k: v.device.type for k, v in state_dict.items()}
        load_calls["keys"] = tuple(state_dict.keys())
        return None

    model.load_state_dict = fake_load_state_dict

    loader.load_weights(assign=True)

    assert load_calls["assign"] is True
    assert load_calls["strict"] is False
    assert load_calls["devices"] == {"buf": "cpu", "weight": "cpu"}
    assert load_calls["keys"] == ("buf", "weight")
    assert model.tie_called is True
    assert torch.equal(model.buf, torch.ones(2))


def test_materialize_places_model_on_cpu_when_cpu_offload():
    manager = FSDP2Backend.__new__(FSDP2Backend)
    manager._device_runtime = types.SimpleNamespace(device=torch.device("cuda"))
    manager.fsdp_config = FSDPConfig(
        cpu_offload=CPUOffloadConfig(enabled=True, mode=CPUOffloadMode.FULL)
    )

    model = TinyModel()
    to_empty_calls = []

    def fake_to_empty(device):
        to_empty_calls.append(torch.device(device))
        return model

    model.to_empty = fake_to_empty

    helper_calls = []
    manager._keep_non_offloaded_on_device = lambda m: helper_calls.append(
        (m, manager._device_runtime.device)
    )

    manager.materialize(model)

    assert to_empty_calls == [torch.device("cpu")]
    assert helper_calls == [(model, torch.device("cuda"))]


def test_broadcast_buffers_writes_directly_into_module(monkeypatch):
    model = TinyModel()
    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cpu")
    loader.is_cpu_offload = False
    loader.rank = 0
    loader.param_converter = types.SimpleNamespace(
        is_persist_buffer=lambda name: name == "buf",
        convert_persist_buffer=lambda name: "hf_buf",
        get_default_hf_dtype=lambda name: torch.float32,
    )
    loader.param_to_filename = {"hf_buf": "file.safetensors"}
    loader.param_to_rank = {"hf_buf": 0}
    loader.local_sd = {"hf_buf": torch.tensor([7.0, 8.0])}

    broadcast_calls = []

    def fake_broadcast(tensor, src, async_op=False, group=None):
        broadcast_calls.append((tensor.device.type, src, async_op))
        return None

    monkeypatch.setattr(weight_loader_mod.dist, "broadcast", fake_broadcast)

    sharded_sd = {}
    loader._broadcast_buffers(sharded_sd)

    assert broadcast_calls == [("cpu", 0, False)]
    assert torch.equal(sharded_sd["buf"], torch.tensor([7.0, 8.0]))
    assert torch.equal(model.buf, torch.ones(2))


def test_initialize_dora_magnitude_after_base_weights_are_loaded():
    from peft import LoraConfig, get_peft_model

    reference_base = nn.Module()
    reference_base.linear = nn.Linear(3, 2, bias=False)
    reference = get_peft_model(
        reference_base,
        LoraConfig(
            r=2,
            lora_alpha=3,
            target_modules=["linear"],
            init_lora_weights=False,
            use_dora=True,
        ),
    )
    expected = (
        reference.base_model.model.linear.lora_magnitude_vector["default"]
        .weight.detach()
        .clone()
    )
    state_without_magnitude = {
        name: value
        for name, value in reference.state_dict().items()
        if ".lora_magnitude_vector." not in name
    }

    with torch.device("meta"):
        target_base = nn.Module()
        target_base.linear = nn.Linear(3, 2, bias=False)
        model = get_peft_model(
            target_base,
            LoraConfig(
                r=2,
                lora_alpha=3,
                target_modules=["linear"],
                init_lora_weights=False,
                use_dora=True,
            ),
        )
    model.load_state_dict(state_without_magnitude, assign=True, strict=False)

    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cpu")
    loader.is_cpu_offload = False

    loader._initialize_dora_magnitudes(assign=True)

    actual = model.base_model.model.linear.lora_magnitude_vector["default"].weight
    assert not actual.is_meta
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("use_dora", [False, True])
def test_hf_resume_preserves_parameters_hooks_and_optimizer_references(use_dora):
    from peft import LoraConfig, get_peft_model

    base = nn.Module()
    base.linear = nn.Linear(3, 2, bias=False)
    model = get_peft_model(
        base, LoraConfig(r=2, target_modules=["linear"], use_dora=use_dora)
    )
    model.register_parameter("no_shard", nn.Parameter(torch.ones(2)))
    original = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = []
    for name, parameter in original.items():
        if parameter.requires_grad:
            parameter.register_post_accumulate_grad_hook(
                lambda parameter, name=name: hooks.append(name)
            )

    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cpu")
    loader.is_cpu_offload = False
    loader._sync_log_rank0 = lambda *args, **kwargs: None
    loader._load_local_shards = lambda: {}
    loader._broadcast_buffers = lambda state: None
    loader._broadcast_params = lambda state: state.update({
        name: torch.full_like(parameter, 2)
        for name, parameter in original.items()
        if ".lora_magnitude_vector." not in name
    })
    loader._cleanup = lambda: None
    loader.load_weights()

    for name, parameter in model.named_parameters():
        assert parameter is original[name]
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {
        id(p) for p in model.parameters()
    }
    sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
    assert set(hooks) == {name for name, p in original.items() if p.requires_grad}
    torch.testing.assert_close(model.no_shard, torch.full_like(model.no_shard, 2))
