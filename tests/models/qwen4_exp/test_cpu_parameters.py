from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode
from torchturbo.models.qwen4_exp.ple import NGramEmbedding


@pytest.mark.parametrize("full_offload", [False, True])
def test_backend_materializes_resident_parameters_on_cpu(
    tmp_path, node_group, full_offload
):
    with torch.device("meta"):
        model = nn.Module()
        model.table = NGramEmbedding(torch.empty(16, 4))
        model.proj = nn.Linear(4, 4)
    model.table.weight.directory = tmp_path
    backend = object.__new__(FSDP2Backend)
    backend.fsdp_config = SimpleNamespace(
        cpu_offload=SimpleNamespace(enabled=full_offload, mode=CPUOffloadMode.FULL)
    )
    backend._device_runtime = SimpleNamespace(
        device=torch.device("cpu" if full_offload else "meta")
    )
    backend.materialize(model)
    assert all(
        p.device.type == "cpu" and not p.is_meta for p in model.table.parameters()
    )
    assert model.table.weight.is_shared()
    assert model.proj.weight.device.type == ("cpu" if full_offload else "meta")
    assert list(tmp_path.iterdir()) == []


def test_backend_keeps_resident_descendants_on_cpu(monkeypatch):
    model = nn.Module()
    model.table = NGramEmbedding(torch.arange(64).float().view(16, 4))
    model.proj = nn.Linear(4, 4)
    original = model.table.weight.clone()
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("meta"))
    table_storage = model.table.weight.untyped_storage().data_ptr()
    moved = []

    def stage(tensor, device):
        moved.append(tensor.untyped_storage().data_ptr())
        return tensor.clone()

    monkeypatch.setattr(torch.Tensor, "to", stage)
    backend._keep_non_offloaded_on_device(model)
    assert table_storage not in moved
    assert len(moved) == 2
    assert all(p.device.type == "cpu" for p in model.table.parameters())
    torch.testing.assert_close(model.table.weight, original)


def test_deepcopy_keeps_registered_weight_connected_to_lookup(
    tmp_path, node_group
):
    import copy

    original = NGramEmbedding(torch.arange(28).float().view(7, 4))
    original.weight.directory = tmp_path
    original.weight.materialize_()
    copied = copy.deepcopy(original)
    with torch.no_grad():
        for parameter in copied.parameters():
            parameter.add_(10)
    torch.testing.assert_close(
        copied(torch.tensor([0, 3, 6])), original(torch.tensor([0, 3, 6])) + 10
    )
    copied.weight.materialize_()
    assert copied.weight.is_shared()
    torch.testing.assert_close(copied.weight, original.weight + 10)
    torch.testing.assert_close(original.weight, torch.arange(28).float().view(7, 4))
