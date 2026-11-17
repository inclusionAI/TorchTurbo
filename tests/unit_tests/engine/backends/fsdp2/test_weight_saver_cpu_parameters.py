"""Tests for exporting frozen registered CPU parameters with trained weights."""

import json
import os
import struct
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from transformers import PretrainedConfig

import torchturbo.engine.backends.fsdp2.checkpoint.weight_saver as saver_module
from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import (
    ShardedParamManager,
)
from torchturbo.models.modules.cpu_resident import NodeSharedCPUResidentParameter
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.runtime.cpu import CpuRuntime

_NGRAM_NAMES = (
    "ngram_embedding.shard_0.weight",
    "ngram_embedding.shard_1.weight",
    "ngram_embedding.shard_2.weight",
)


def _ngram_tensors():
    return {
        _NGRAM_NAMES[0]: torch.tensor(
            [[1.25, -2.5], [3.75, 4.5], [-5.25, 6.0]], dtype=torch.bfloat16
        ),
        _NGRAM_NAMES[1]: torch.tensor(
            [[1.0000000001, -2.0000000002], [3.0000000003, 4.0000000004]],
            dtype=torch.float64,
        ),
        _NGRAM_NAMES[2]: torch.tensor([1, 2**40, -2**40, 4], dtype=torch.int64),
    }


class _ExportModel(nn.Module):
    def __init__(self, include_frozen_parameters=True):
        super().__init__()
        self.config = PretrainedConfig()
        self.all_tied_weights_keys = {}
        self.linear = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]]))
        if include_frozen_parameters:
            self.ngram_embedding = nn.Module()
            for name, tensor in _ngram_tensors().items():
                shard = nn.Module()
                shard.register_parameter("weight", nn.Parameter(tensor, requires_grad=False))
                self.ngram_embedding.add_module(name.split(".")[1], shard)


def _update_frozen_parameters(model):
    with torch.no_grad():
        for parameter in model.ngram_embedding.parameters():
            parameter.add_(10)


class _SmallShardSaver(ShardedParamManager):
    def _assign_param_to_ranks(self, ep_params, edp_ranks, common_params):
        self.WEIGHT_SHARD_SIZE = 30
        super()._assign_param_to_ranks(ep_params, edp_ranks, common_params)


@pytest.fixture
def process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cpu", (1, 1), mesh_dim_names=("edp", "ep"))
    finally:
        dist.destroy_process_group()
        NodeSharedCPUResidentParameter._node_group = None


@pytest.fixture
def source_checkpoint(tmp_path):
    source = tmp_path / "source"
    PretrainedConfig().save_pretrained(source)
    tensors = {
        "linear.weight": torch.zeros(2, 3, dtype=torch.float16),
        **_ngram_tensors(),
        "mtp.weight": torch.ones(5),
    }
    first_names = ("linear.weight", _NGRAM_NAMES[0], "mtp.weight")
    save_file({name: tensors[name] for name in first_names}, str(source / "first.safetensors"))
    save_file(
        {name: tensors[name] for name in _NGRAM_NAMES[1:]},
        str(source / "second.safetensors"),
    )
    index = {
        "metadata": {"total_size": 108, "source": "original"},
        "weight_map": {
            name: "first.safetensors" if name in first_names else "second.safetensors"
            for name in tensors
        },
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    return source, tensors


def _remove_source_tensor_data(source):
    # Keep dtype/shape headers readable by CheckpointInfo, but make any attempt
    # to load tensor values from the source checkpoint fail.
    for path in source.glob("*.safetensors"):
        with path.open("r+b") as shard:
            header_size = struct.unpack("<Q", shard.read(8))[0]
            shard.truncate(8 + header_size)


def _make_converter(model, mesh, source):
    return BaseWeightConverter(
        model,
        {"ep": {"ep_group": mesh["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
        source,
    )


def _make_saver(model, mesh, source, converter):
    return ShardedParamManager(
        model, 0, mesh, converter, CpuRuntime(), path=source
    )


def _read_export(output: Path):
    index = json.loads((output / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in set(index["weight_map"].values()):
        shard = load_file(str(output / filename))
        assert not (tensors.keys() & shard.keys())
        tensors.update(shard)
    assert set(tensors) == set(index["weight_map"])
    return index, tensors


def test_frozen_cpu_parameters_export_current_values_without_source_tensor_data(
    tmp_path, process_group, source_checkpoint
):
    source, originals = source_checkpoint
    _remove_source_tensor_data(source)
    model = _ExportModel()
    parameters = dict(model.named_parameters())
    assert set(parameters) == {"linear.weight", *_NGRAM_NAMES}
    assert set(model.state_dict()) == set(parameters)
    for name in _NGRAM_NAMES:
        assert parameters[name].device.type == "cpu"
        assert not parameters[name].requires_grad

    converter = _make_converter(model, process_group, source)
    saver = _make_saver(model, process_group, source, converter)
    _update_frozen_parameters(model)
    output = tmp_path / "export"
    saver.save_model(output)

    index, saved = _read_export(output)
    assert set(saved) == set(parameters)
    assert index["metadata"] == {"total_size": 88, "source": "original"}
    torch.testing.assert_close(saved["linear.weight"], model.linear.weight.to(torch.float16))
    for name in _NGRAM_NAMES:
        assert saved[name].device.type == "cpu"
        assert saved[name].dtype == originals[name].dtype
        assert torch.equal(saved[name], originals[name] + 10)


def test_export_omits_source_tensors_absent_from_the_model(
    tmp_path, process_group, source_checkpoint
):
    source, _ = source_checkpoint
    model = _ExportModel(include_frozen_parameters=False)
    converter = _make_converter(model, process_group, source)
    saver = _make_saver(model, process_group, source, converter)
    output = tmp_path / "export"
    saver.save_model(output)

    index, saved = _read_export(output)
    assert set(saved) == {"linear.weight"}
    assert index["metadata"] == {"total_size": 12, "source": "original"}


def _distributed_export_worker(rank, source, output, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("edp", "ep"))
        model = _ExportModel()
        converter = _make_converter(model, mesh, source)
        saver = _SmallShardSaver(model, rank, mesh, converter, CpuRuntime(), path=source)
        _update_frozen_parameters(model)
        saver.save_model(output)
    finally:
        dist.destroy_process_group()


def test_distributed_export_saves_frozen_cpu_parameters_once(tmp_path, source_checkpoint):
    source, originals = source_checkpoint
    _remove_source_tensor_data(source)
    output = tmp_path / "export"
    mp.spawn(
        _distributed_export_worker,
        args=(str(source), str(output), str(tmp_path / "distributed-init")),
        nprocs=2,
        join=True,
    )

    index, saved = _read_export(output)
    assert set(saved) == {"linear.weight", *_NGRAM_NAMES}
    assert index["metadata"]["total_size"] == 88
    assert len(set(index["weight_map"].values())) == 2
    torch.testing.assert_close(saved["linear.weight"], _ExportModel().linear.weight.half())
    for name in _NGRAM_NAMES:
        assert saved[name].dtype == originals[name].dtype
        assert torch.equal(saved[name], originals[name] + 10)


def _ordered_export_worker(rank, root, created, release, finished, failure):
    root = Path(root)
    source, output = root / "source", root / "export"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'ordered-init'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("edp", "ep"))
        model = _ExportModel()
        converter = _make_converter(model, mesh, source)
        saver = _SmallShardSaver(
            model, rank, mesh, converter, CpuRuntime(), path=source
        )
        _update_frozen_parameters(model)
        original_save = saver_module.save_file
        original_walk = os.walk
        original_check = saver_module.collective_check
        original_copy = saver._copy_non_safetensor_files
        temporary_name = ".pending-rank1"
        metadata_started = False

        def coordinated_save(tensors, filename):
            if rank == 0:
                return original_save(tensors, filename)
            temporary = output / temporary_name
            original_save(tensors, str(temporary))
            created.set()
            assert release.wait(30), "rank zero did not finish tensor writes"
            if failure == "writer":
                raise OSError("injected tensor writer failure")
            os.replace(temporary, filename)
            finished.set()

        def coordinated_walk(path, *args, **kwargs):
            if rank == 0 and Path(path) == output:
                assert created.wait(30), "rank one did not create its temporary file"
            for directory, directories, files in original_walk(path, *args, **kwargs):
                if rank == 0 and Path(directory) == output and temporary_name in files:
                    # Capture the name before its atomic rename. Correct ordering
                    # completes the writer before this metadata enumeration.
                    release.set()
                    assert finished.wait(30), (
                        "rank one did not rename its temporary file"
                    )
                yield directory, directories, files

        def coordinated_check(action):
            def checked_action():
                try:
                    return action()
                finally:
                    if rank == 0 and action.__name__ == "write_files":
                        release.set()

            return original_check(checked_action)

        def copy_metadata(*args):
            nonlocal metadata_started
            metadata_started = True
            if failure == "metadata":
                raise OSError("injected metadata copy failure")
            original_copy(*args)

        with (
            patch.object(saver_module, "save_file", coordinated_save),
            patch.object(os, "walk", coordinated_walk),
            patch.object(saver_module, "collective_check", coordinated_check),
            patch.object(saver, "_copy_non_safetensor_files", copy_metadata),
        ):
            try:
                saver.save_model(output)
            except OSError as error:
                result = f"{type(error).__name__}: {error}"
            else:
                result = "success"
        (root / f"ordered-result-{rank}.json").write_text(
            json.dumps({"result": result, "metadata_started": metadata_started})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [None, "writer", "metadata"])
def test_distributed_export_finishes_writers_before_metadata(
    tmp_path, source_checkpoint, failure
):
    source, originals = source_checkpoint
    source_bytes = {path.name: path.read_bytes() for path in source.iterdir()}
    context = mp.get_context("spawn")
    mp.spawn(
        _ordered_export_worker,
        args=(
            str(tmp_path),
            context.Event(),
            context.Event(),
            context.Event(),
            failure,
        ),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"ordered-result-{rank}.json").read_text())
        for rank in range(2)
    ]
    output = tmp_path / "export"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == source_bytes
    assert not results[1]["metadata_started"]
    if failure is not None:
        message = (
            "injected tensor writer failure"
            if failure == "writer"
            else "injected metadata copy failure"
        )
        assert [result["result"] for result in results] == [f"OSError: {message}"] * 2
        assert results[0]["metadata_started"] == (failure == "metadata")
        assert not (output / "model.safetensors.index.json").exists()
        return

    assert [result["result"] for result in results] == ["success"] * 2
    assert results[0]["metadata_started"]
    assert not (output / ".pending-rank1").exists()
    assert (output / "config.json").read_bytes() == source_bytes["config.json"]
    index, saved = _read_export(output)
    assert set(saved) == {"linear.weight", *_NGRAM_NAMES}
    assert index["metadata"] == {"total_size": 88, "source": "original"}
    torch.testing.assert_close(
        saved["linear.weight"], _ExportModel().linear.weight.half()
    )
    for name in _NGRAM_NAMES:
        assert saved[name].dtype == originals[name].dtype
        assert torch.equal(saved[name], originals[name] + 10)



def test_buffer_beside_resident_parameter_uses_ordinary_converter_policy(
    tmp_path, process_group
):
    class Converter(BaseWeightConverter):
        def is_persist_buffer(self, name):
            return name == "linear.scale"

        def convert_persist_buffer(self, name):
            return name.replace("linear.scale", "linear.saved_scale")

    model = _ExportModel(include_frozen_parameters=False)
    model.linear.weight = NodeSharedCPUResidentParameter(
        model.linear.weight.detach(), requires_grad=False, directory=tmp_path
    )
    model.linear.weight.load_([2], lambda index, target: None)
    model.linear.register_buffer("scale", torch.tensor([1.25, 2.5], dtype=torch.float64))
    model.linear.register_buffer("cache", torch.tensor([7]), persistent=False)
    source = tmp_path / "source"
    PretrainedConfig().save_pretrained(source)
    tensors = {
        "linear.weight": model.linear.weight.detach().clone(),
        "linear.saved_scale": model.linear.scale.clone(),
    }
    save_file(tensors, str(source / "model.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    converter = Converter(
        model,
        {"ep": {"ep_group": process_group["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
        source,
    )
    saver = _make_saver(model, process_group, source, converter)
    model.linear.scale.add_(10)
    output = tmp_path / "export"
    saver.save_model(output)
    _, saved = _read_export(output)
    assert set(saved) == {"linear.weight", "linear.saved_scale"}
    torch.testing.assert_close(saved["linear.saved_scale"], model.linear.scale)


def test_cpu_export_uses_parameter_layout_instead_of_source_shard_rows(
    tmp_path, process_group
):
    from torchturbo.models.qwen4_exp.model import Qwen4ExpWeightConverter

    model = nn.Module()
    model.config = PretrainedConfig(text_config=PretrainedConfig(split_ngram_parts=3))
    model.all_tied_weights_keys = {}
    model.ngram_embedding = nn.Module()
    model.ngram_embedding.weight = NodeSharedCPUResidentParameter(
        torch.empty(10, 2), requires_grad=False, directory=tmp_path
    )
    expected = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    chunks = expected.split([4, 3, 3])
    model.ngram_embedding.weight.load_(
        [4, 3, 3], lambda index, target: target.copy_(chunks[index])
    )
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    source_tensors = {
        name: chunk.to(torch.bfloat16).clone()
        for name, chunk in zip(_NGRAM_NAMES, expected.split([2, 5, 3]), strict=True)
    }
    save_file(source_tensors, str(source / "weights.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(_NGRAM_NAMES, "weights.safetensors")})
    )
    converter = Qwen4ExpWeightConverter(
        model,
        {"ep": {"ep_group": process_group["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
        source,
    )
    saver = _make_saver(model, process_group, source, converter)
    output = tmp_path / "export"
    saver.save_model(output)
    index, saved = _read_export(output)
    assert index["metadata"]["total_size"] == 40
    for name, chunk in zip(_NGRAM_NAMES, chunks, strict=True):
        torch.testing.assert_close(saved[name], chunk.bfloat16())


def test_cpu_export_requires_parameter_shard_layout(
    tmp_path, process_group, source_checkpoint
):
    source, _ = source_checkpoint
    model = _ExportModel(include_frozen_parameters=False)
    model.linear.weight = NodeSharedCPUResidentParameter(
        model.linear.weight.detach(), requires_grad=False, directory=tmp_path
    )
    converter = _make_converter(model, process_group, source)
    with pytest.raises(RuntimeError, match="shard layout"):
        _make_saver(model, process_group, source, converter)
