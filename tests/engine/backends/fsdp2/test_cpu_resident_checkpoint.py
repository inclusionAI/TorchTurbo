"""CPU table loading and export through the production checkpoint classes."""

import io
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.checkpoint import (
    CheckpointException,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.device_mesh import init_device_mesh
from transformers import PretrainedConfig

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.checkpoint import weight_loader as loader_module
from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader
from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import ShardedParamManager
from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode
from torchturbo.models.modules.cpu_resident import NodeSharedCPUResidentParameter
from torchturbo.models.qwen4_exp.model import Qwen4ExpWeightConverter
from torchturbo.models.qwen4_exp.ple import NGramEmbedding
from torchturbo.runtime.cpu import CpuRuntime


class ResidentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(
            text_config=PretrainedConfig(split_ngram_parts=3)
        )
        self.all_tied_weights_keys = {}
        self.ngram_embedding = NGramEmbedding(torch.empty(10, 2, device="meta"))
        self.proj = nn.Linear(2, 2, bias=False, device="meta")

    def tie_weights(self):
        pass


def _source(path):
    path.mkdir()
    PretrainedConfig().save_pretrained(path)
    rows = (torch.arange(20).reshape(10, 2) / 7).bfloat16()
    weight_map = {}
    for i, chunk in enumerate(rows.split([2, 5, 3])):
        key = f"ngram_embedding.shard_{i}.weight"
        filename = f"table-{i}.safetensors"
        save_file({key: chunk.clone()}, str(path / filename))
        weight_map[key] = filename
    save_file({"proj.weight": torch.eye(2)}, str(path / "proj.safetensors"))
    weight_map["proj.weight"] = "proj.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return rows


def _worker(rank, world_size, root):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'init'}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size, 1), mesh_dim_names=("edp", "ep"))
        world_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        model = ResidentModel()
        parameter = model.ngram_embedding.weight
        parameter.directory = root
        converter = Qwen4ExpWeightConverter(
            model,
            {"ep": {"ep_group": mesh["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
            str(root / "source"),
        )
        loader = DistParamLoader(
            model,
            root / "source",
            rank,
            mesh,
            world_mesh,
            CpuRuntime(),
            False,
            converter,
        )
        reads = []
        native_open = loader_module.safe_open

        def record_open(filename, **kwargs):
            reads.append(Path(filename).name)
            return native_open(filename, **kwargs)

        native_prepare = loader._prepare_tensor

        def prepare(name, *args, **kwargs):
            assert not name.startswith("ngram_embedding."), (
                "CPU table reached device broadcast"
            )
            return native_prepare(name, *args, **kwargs)

        loader._prepare_tensor = prepare
        with patch.object(loader_module, "safe_open", record_open):
            loader.load_weights(assign=True)
        expected = (torch.arange(20).reshape(10, 2) / 7).bfloat16()
        assert reads == (
            ["table-0.safetensors", "table-2.safetensors"]
            if rank % 2 == 0
            else ["table-1.safetensors"]
        )
        assert model.ngram_embedding.weight is parameter
        assert isinstance(parameter, NodeSharedCPUResidentParameter)
        assert parameter.dtype == torch.bfloat16 and parameter.device.type == "cpu"
        assert parameter.is_shared()
        assert parameter.shard_sizes == (2, 5, 3)
        torch.testing.assert_close(parameter, expected)
        torch.testing.assert_close(model.proj.weight, torch.eye(2))
        torch.testing.assert_close(
            model.ngram_embedding(torch.tensor([0, 4, 9])), expected[[0, 4, 9]]
        )
        pointer = parameter.data_ptr()
        dist.barrier()
        # Prove that peers in a node see writes to the same physical allocation.
        if rank % 2 == 0:
            with torch.no_grad():
                parameter.add_(2)
        dist.barrier()
        torch.testing.assert_close(parameter, expected + 2)
        saver = ShardedParamManager(
            model, rank, mesh, converter, CpuRuntime(), path=root / "source"
        )
        state = saver.get_local_hf_state_dict()
        cpu_keys = sorted(key for key in state if key.startswith("ngram_embedding."))
        wanted = {
            0: ["ngram_embedding.shard_0.weight", "ngram_embedding.shard_2.weight"],
            1: ["ngram_embedding.shard_1.weight"],
        }
        assert cpu_keys == wanted.get(rank, [])
        saver.save_hf(state, root / "output")
        dist.barrier()
        if rank == 0:
            index = json.loads(
                (root / "output/model.safetensors.index.json").read_text()
            )
            restored = {}
            for filename in set(index["weight_map"].values()):
                restored.update(load_file(str(root / "output" / filename)))
            for i, chunk in enumerate((expected + 2).split([2, 5, 3])):
                torch.testing.assert_close(
                    restored[f"ngram_embedding.shard_{i}.weight"], chunk
                )
        loader.load_weights(assign=False)
        assert parameter.data_ptr() == pointer
        torch.testing.assert_close(parameter, expected)
        checkpoint = DcpCheckpoint(0, 1, mesh["edp"])
        model.alias = parameter
        native_write = FileSystemWriter.write_data
        cpu_writes = []

        def record_dcp_write(writer, plan, planner):
            cpu_writes.extend(
                (item.index.fqn, tuple(item.tensor_data.chunk.offsets))
                for item in plan.items
                if item.index.fqn
                in {"model.alias", "model.ngram_embedding.weight"}
            )
            return native_write(writer, plan, planner)

        with patch.object(FileSystemWriter, "write_data", record_dcp_write):
            checkpoint.save_model(model, str(root / "dcp"))
        wanted_offsets = {0: [(0, 0), (7, 0)], 1: [(2, 0)]}.get(rank, [])
        for key in ("model.alias", "model.ngram_embedding.weight"):
            assert sorted(offset for fqn, offset in cpu_writes if fqn == key) == (
                wanted_offsets
            )
        metadata = FileSystemReader(root / "dcp").read_metadata()
        for key in ("model.alias", "model.ngram_embedding.weight"):
            saved_tensor = metadata.state_dict_metadata[key]
            assert saved_tensor.size == parameter.shape
            chunks = sorted(saved_tensor.chunks, key=lambda chunk: chunk.offsets)
            assert [chunk.sizes[0] for chunk in chunks] == [2, 5, 3]
            for index, storage in metadata.storage_data.items():
                if index.fqn != key:
                    continue
                with (root / "dcp" / storage.relative_path).open("rb") as stream:
                    stream.seek(storage.offset)
                    tensor = torch.load(io.BytesIO(stream.read(storage.length)))
                assert tensor.untyped_storage().nbytes() == (
                    tensor.numel() * tensor.element_size()
                )
        if rank % 2 == 0:
            with torch.no_grad():
                parameter.zero_()
        dist.barrier()
        native_read = FileSystemReader.read_data
        cpu_reads = []

        def record_dcp_read(reader, plan, planner):
            cpu_reads.extend(
                (item.dest_index.fqn, tuple(item.dest_index.offset))
                for item in plan.items
                if item.dest_index.fqn
                in {"model.alias", "model.ngram_embedding.weight"}
            )
            return native_read(reader, plan, planner)

        with patch.object(FileSystemReader, "read_data", record_dcp_read):
            checkpoint.load_model(model, str(root / "dcp"))
        for key in ("model.alias", "model.ngram_embedding.weight"):
            assert sorted(offset for fqn, offset in cpu_reads if fqn == key) == (
                {0: [(0, 0), (7, 0)], 1: [(2, 0)]}[rank % 2]
            )
        assert model.alias is parameter
        del model.alias
        assert model.ngram_embedding.weight is parameter
        assert parameter.data_ptr() == pointer
        torch.testing.assert_close(parameter, expected)

        restored_model = ResidentModel()
        restored_parameter = restored_model.ngram_embedding.weight
        assert restored_parameter.is_meta and restored_parameter.shard_sizes is None
        restored_model.proj.to_empty(device="cpu")
        restored_model.alias = restored_parameter
        checkpoint.load_model(restored_model, str(root / "dcp"))
        assert restored_model.alias is restored_parameter
        assert restored_model.ngram_embedding.weight is restored_parameter
        assert restored_parameter.shard_sizes == (2, 5, 3)
        torch.testing.assert_close(restored_parameter, expected)
        torch.testing.assert_close(restored_model.proj.weight, torch.eye(2))
        del restored_model.alias
        restored_converter = Qwen4ExpWeightConverter(
            restored_model,
            {"ep": {"ep_group": mesh["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
            str(root / "source"),
        )
        restored_saver = ShardedParamManager(
            restored_model, rank, mesh, restored_converter, CpuRuntime(),
            path=root / "source",
        )
        restored_state = restored_saver.get_local_hf_state_dict()
        assert sorted(
            key for key in restored_state if key.startswith("ngram_embedding.")
        ) == wanted.get(rank, [])
        restored_saver.save_hf(restored_state, root / "cold-output")
        if rank == 0:
            index = json.loads(
                (root / "cold-output/model.safetensors.index.json").read_text()
            )
            saved = {}
            for filename in set(index["weight_map"].values()):
                tensors = load_file(str(root / "cold-output" / filename))
                assert not saved.keys() & tensors.keys()
                saved.update(tensors)
            assert set(saved) == {
                "proj.weight",
                *(f"ngram_embedding.shard_{i}.weight" for i in range(3)),
            }
            assert index["metadata"]["total_size"] == 56
            for i, chunk in enumerate(expected.split([2, 5, 3])):
                torch.testing.assert_close(saved[f"ngram_embedding.shard_{i}.weight"], chunk)
            torch.testing.assert_close(saved["proj.weight"], torch.eye(2))

        # Backend allocation must retain resident identity even when other params move.
        fresh = ResidentModel()
        fresh_parameter = fresh.ngram_embedding.weight
        backend = object.__new__(FSDP2Backend)
        backend.fsdp_config = SimpleNamespace(
            cpu_offload=SimpleNamespace(enabled=False, mode=CPUOffloadMode.FULL)
        )
        backend._device_runtime = SimpleNamespace(device=torch.device("meta"))
        backend.materialize(fresh)
        assert fresh.ngram_embedding.weight is fresh_parameter
        assert fresh_parameter.device.type == "cpu" and fresh_parameter.is_shared()
        assert fresh.proj.weight.is_meta
        dist.barrier()

        def fail_one_reader(filename, **kwargs):
            if rank == world_size - 1:
                raise OSError("injected node-local source failure")
            return native_open(filename, **kwargs)

        with (
            patch.object(loader_module, "safe_open", fail_one_reader),
            pytest.raises(RuntimeError, match="injected node-local source failure"),
        ):
            loader.load_weights()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_shared_cpu_table_hf_and_dcp_round_trip(tmp_path, world_size):
    _source(tmp_path / "source")
    (tmp_path / "output").mkdir()
    (tmp_path / "cold-output").mkdir()
    mp.spawn(_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)


@pytest.fixture
def checkpoint_group(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'single-init'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("edp",))
    finally:
        dist.destroy_process_group()
        NodeSharedCPUResidentParameter._node_group = None


def test_omitted_tied_cpu_parameter_hf_round_trip(tmp_path, checkpoint_group):
    from torchturbo.models.weight_converter import BaseWeightConverter

    model = nn.Module()
    model.config = PretrainedConfig()
    model.target = NodeSharedCPUResidentParameter(
        torch.empty(2, 2, device="meta"), requires_grad=False, directory=tmp_path
    )
    model.source = model.target
    model.all_tied_weights_keys = {"target": "source"}
    expected = torch.tensor([[1.25, 2.5], [3.75, 4.0]], dtype=torch.bfloat16)
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    save_file({"source": expected}, str(source / "weights.safetensors"))
    weight_map = {"source": "weights.safetensors"}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    context = {"ep": {"ep_group": dist.group.WORLD, "ep_rank": 0, "ep_size": 1}}
    converter = BaseWeightConverter(model, context, source)
    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("edp", "ep"))
    loader = DistParamLoader(
        model, source, 0, mesh, checkpoint_group, CpuRuntime(), False, converter
    )
    loader.load_weights()
    parameter = model.target
    pointer = parameter.data_ptr()
    assert parameter is model.source
    torch.testing.assert_close(parameter, expected)

    parameter.add_(2)
    saver = ShardedParamManager(model, 0, mesh, converter, CpuRuntime(), path=source)
    output = tmp_path / "output"
    saver.save_model(output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"source"}
    assert index["metadata"]["total_size"] == 8
    saved = load_file(str(output / index["weight_map"]["source"]))
    torch.testing.assert_close(saved["source"], expected + 2)

    parameter.zero_()
    converter = BaseWeightConverter(model, context, output)
    loader = DistParamLoader(
        model, output, 0, mesh, checkpoint_group, CpuRuntime(), False, converter
    )
    loader.load_weights()
    assert model.target is model.source is parameter
    assert parameter.data_ptr() == pointer
    torch.testing.assert_close(parameter, expected + 2)


def _cpu_export_metadata_failure_worker(rank, root):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'metadata-init'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("edp", "ep"))
        model = nn.Module()
        model.config = PretrainedConfig(
            text_config=PretrainedConfig(split_ngram_parts=3)
        )
        model.all_tied_weights_keys = {}
        model.ngram_embedding = nn.Module()
        model.ngram_embedding.weight = NodeSharedCPUResidentParameter(
            torch.ones(10, 2), requires_grad=False, directory=root
        )
        model.ngram_embedding.weight.load_([2, 5, 3], lambda index, target: None)
        source = root / f"source-{rank}"
        converter = Qwen4ExpWeightConverter(
            model,
            {"ep": {"ep_group": mesh["ep"].get_group(), "ep_rank": 0, "ep_size": 1}},
            source,
        )
        try:
            ShardedParamManager(model, rank, mesh, converter, CpuRuntime(), path=source)
        except Exception as error:  # noqa: BLE001 - Compare failures from every rank.
            result = {"type": type(error).__name__, "message": str(error)}
        else:
            result = {"type": None, "message": "Export accepted missing source metadata"}
        (root / f"metadata-result-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_cpu_export_metadata_failure_reaches_every_rank(tmp_path):
    for rank in range(2):
        _source(tmp_path / f"source-{rank}")
    save_file(
        {"unexpected.weight": torch.ones(3, 2)},
        str(tmp_path / "source-1" / "table-2.safetensors"),
    )
    mp.spawn(
        _cpu_export_metadata_failure_worker,
        args=(str(tmp_path),),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"metadata-result-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert results[0] == results[1]
    assert results[0]["type"] == "KeyError"
    assert "ngram_embedding.shard_2.weight" in results[0]["message"]


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="PyTorch DCP cannot pickle Python 3.14 traceback code objects",
)
def test_cpu_dcp_file_failure_uses_native_checkpoint_exception(
    tmp_path, checkpoint_group
):
    model = nn.Module()
    model.weight = NodeSharedCPUResidentParameter(
        torch.ones(3, 2), requires_grad=False, directory=tmp_path
    )
    model.weight.materialize_()
    checkpoint = DcpCheckpoint(0, 1, checkpoint_group)
    checkpoint.save_model(model, str(tmp_path / "dcp"))
    for shard in (tmp_path / "dcp").glob("*.distcp"):
        shard.unlink()
    with pytest.raises(CheckpointException, match="No such file"):
        checkpoint.load_model(model, str(tmp_path / "dcp"))
