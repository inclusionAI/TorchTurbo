"""Native DCP chunks for node-shared CPU storage."""

import io
import os
import sys
import weakref
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.models.modules.cpu_resident import (
    NodeSharedCPUResidentParameter,
)


class ResidentStateModel(nn.Module):
    _version = 7

    def __init__(self):
        super().__init__()
        self.extra = {"step": 12, "label": "resident checkpoint"}

    def get_extra_state(self):
        return self.extra.copy()

    def set_extra_state(self, state):
        self.extra = state.copy()


@pytest.fixture
def single_rank_group(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        NodeSharedCPUResidentParameter._node_group = None
        dist.destroy_process_group()


def _shared_model(root):
    model = ResidentStateModel()
    sources = {
        "table": (torch.arange(20).reshape(10, 2).bfloat16(), [2, 5, 3]),
        "scalar": (torch.tensor(7.5, dtype=torch.float64), [1]),
        "empty": (torch.empty(0, 2, dtype=torch.int64), [0]),
    }
    for name, (tensor, sizes) in sources.items():
        parameter = NodeSharedCPUResidentParameter(
            torch.empty_like(tensor, device="meta"),
            requires_grad=False,
            directory=root,
        )
        chunks = [tensor] if tensor.ndim == 0 else tensor.split(sizes)
        parameter.load_(sizes, lambda index, target: target.copy_(chunks[index]))
        model.register_parameter(name, parameter)
    model.alias = model.table
    model.local = nn.Parameter(torch.tensor([3.0, 4.0]))
    model.weight = nn.Parameter(torch.tensor([5.0, 6.0]))
    return model, {name: tensor for name, (tensor, _) in sources.items()}


def _clear_model(model, rank):
    with torch.no_grad():
        for parameter in model.parameters():
            if (
                not isinstance(parameter, NodeSharedCPUResidentParameter)
                or rank % 2 == 0
            ):
                parameter.zero_()
    dist.barrier()


def _assert_model(model, expected, pointers):
    assert model.alias is model.table
    for name, tensor in expected.items():
        parameter = model.get_parameter(name)
        assert parameter.data_ptr() == pointers[name]
        torch.testing.assert_close(parameter, tensor, rtol=0, atol=0)
    torch.testing.assert_close(model.local, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(model.weight, torch.tensor([5.0, 6.0]))


def _round_trip_worker(rank, world_size, root):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'init'}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("edp",))
        model, expected = _shared_model(root)
        pointers = {name: model.get_parameter(name).data_ptr() for name in expected}
        checkpoint = DcpCheckpoint(0, 1, mesh)
        native_write = FileSystemWriter.write_data
        writes = []
        write_source = [model.table.untyped_storage()]

        def record_write(writer, plan, planner):
            for item in plan.items:
                if item.index.fqn not in {"model.table", "model.alias"}:
                    continue
                source = planner.resolve_data(item)
                assert source.untyped_storage() is write_source[0]
                writes.append((item.index.fqn, tuple(item.index.offset)))
            return native_write(writer, plan, planner)

        with (
            patch.object(FileSystemWriter, "write_data", record_write),
            patch.object(dcp, "save", wraps=dcp.save) as save,
        ):
            checkpoint.save_model(model, str(root / "dcp"))
        assert save.call_count == 1
        assert not save.call_args.kwargs.get("no_dist", False)
        assert save.call_args.kwargs.get("planner") is None
        wanted = {0: [(0, 0), (7, 0)], 1: [(2, 0)]}.get(rank, [])
        for key in ("model.table", "model.alias"):
            assert sorted(offset for fqn, offset in writes if fqn == key) == wanted

        metadata = FileSystemReader(root / "dcp").read_metadata()
        for key in ("model.table", "model.alias"):
            saved = metadata.state_dict_metadata[key]
            assert saved.size == expected["table"].shape
            assert sorted((tuple(c.offsets), tuple(c.sizes)) for c in saved.chunks) == [
                ((0, 0), (2, 2)),
                ((2, 0), (5, 2)),
                ((7, 0), (3, 2)),
            ]
            payload_bytes = 0
            for index, storage in metadata.storage_data.items():
                if index.fqn != key:
                    continue
                with (root / "dcp" / storage.relative_path).open("rb") as stream:
                    stream.seek(storage.offset)
                    tensor = torch.load(io.BytesIO(stream.read(storage.length)))
                num_bytes = tensor.numel() * tensor.element_size()
                assert tensor.untyped_storage().nbytes() == num_bytes
                payload_bytes += num_bytes
            assert payload_bytes == model.table.numel() * model.table.element_size()

        _clear_model(model, rank)
        model.extra = {"step": -1, "label": "cleared"}
        native_read = FileSystemReader.read_data
        reads = []
        destinations = []
        state_keys = set(model.state_dict())
        hook_calls = []

        def before_load(
            module, state, prefix, local_metadata, strict, missing, unexpected, errors
        ):
            assert set(state) == state_keys
            assert local_metadata["version"] == ResidentStateModel._version
            for name in ("table", "alias", "scalar", "empty", "local"):
                parameter = module.get_parameter(name)
                assert state[name].untyped_storage() is parameter.untyped_storage()
            torch.testing.assert_close(state["weight"], torch.tensor([5.0, 6.0]))
            assert state["_extra_state"] == {
                "step": 12, "label": "resident checkpoint"
            }
            state["weight"] = state["weight"] + 10
            hook_calls.append("pre")

        hook = model.register_load_state_dict_pre_hook(before_load)

        def after_load(module, incompatible):
            assert not incompatible.missing_keys and not incompatible.unexpected_keys
            assert module.extra == {"step": 12, "label": "resident checkpoint"}
            hook_calls.append("post")

        post_hook = model.register_load_state_dict_post_hook(after_load)

        def record_read(reader, plan, planner):
            reads.extend(
                tuple(item.dest_index.offset)
                for item in plan.items
                if item.dest_index.fqn == "model.table"
            )
            for item in plan.items:
                name = item.dest_index.fqn.removeprefix("model.")
                if name not in {"table", "alias", "scalar", "empty", "local"}:
                    continue
                target = planner.resolve_tensor(item)
                parameter = model.get_parameter(name)
                assert target.untyped_storage() is parameter.untyped_storage()
                assert target.shape == item.lengths
                destinations.append(name)
            return native_read(reader, plan, planner)

        with (
            patch.object(FileSystemReader, "read_data", record_read),
            patch.object(dcp, "load", wraps=dcp.load) as load,
            patch.object(
                model, "load_state_dict", wraps=model.load_state_dict
            ) as apply,
        ):
            checkpoint.load_model(model, str(root / "dcp"))
        hook.remove()
        post_hook.remove()
        assert load.call_count == 1, load.call_args_list
        assert not load.call_args.kwargs.get("no_dist", False)
        planner = load.call_args.kwargs["planner"]
        assert type(planner) is dcp.DefaultLoadPlanner
        assert planner.flatten_sharded_tensors is False
        assert set(load.call_args.args[0]["model"]) == state_keys
        assert apply.call_count == 1 and apply.call_args.kwargs["strict"] is True
        assert apply.call_args.args[0]._metadata[""]["version"] == 7
        assert hook_calls == ["pre", "post"]
        assert {"table", "local"} <= set(destinations)
        torch.testing.assert_close(model.weight, torch.tensor([15.0, 16.0]))
        with torch.no_grad():
            model.weight.sub_(10)
        assert model.extra == {"step": 12, "label": "resident checkpoint"}
        assert sorted(reads) == {0: [(0, 0), (7, 0)], 1: [(2, 0)]}[rank % 2]
        assert model.table.shard_sizes == (2, 5, 3)
        _assert_model(model, expected, pointers)

        fresh = nn.Module()
        identities = {}
        for name, tensor in expected.items():
            parameter = NodeSharedCPUResidentParameter(
                torch.empty(tensor.shape, device="meta"),
                requires_grad=False,
                directory=root,
            )
            fresh.register_parameter(name, parameter)
            identities[name] = parameter
        fresh_reads = []

        def read_fresh(reader, plan, planner):
            for item in plan.items:
                if item.dest_index.fqn != "model.table":
                    continue
                target = planner.resolve_tensor(item)
                assert target.untyped_storage() is fresh.table.untyped_storage()
                fresh_reads.append(
                    (
                        tuple(item.dest_index.offset),
                        target.numel(),
                        target.storage_offset(),
                    )
                )
            return native_read(reader, plan, planner)

        with patch.object(FileSystemReader, "read_data", read_fresh):
            checkpoint.load_model(fresh, str(root / "dcp"))
        for name, tensor in expected.items():
            parameter = fresh.get_parameter(name)
            assert parameter is identities[name] and parameter.dtype == tensor.dtype
            torch.testing.assert_close(parameter, tensor, rtol=0, atol=0)
        assert fresh.table.shard_sizes == (2, 5, 3)
        assert fresh.scalar.shard_sizes == (1,)
        assert fresh.empty.shard_sizes == (0,)
        row_start = (rank % 2) * 5
        assert {offset for offset, _, _ in fresh_reads} == {(row_start, 0)}
        assert sum(numel for _, numel, _ in fresh_reads) == 10
        assert all(
            row_start * 2 <= start and start + numel <= (row_start + 5) * 2
            for _, numel, start in fresh_reads
        )

        # A full-tensor checkpoint from the default planner remains readable.
        dcp.save({"model": model.state_dict()}, checkpoint_id=root / "legacy")
        _clear_model(model, rank)
        checkpoint.load_model(model, str(root / "legacy"))
        _assert_model(model, expected, pointers)
        assert model.table.shard_sizes == (2, 5, 3)

        # Parameters initialized without source shards get balanced node slices.
        initialized = nn.Module()
        initialized.table = NodeSharedCPUResidentParameter(
            torch.arange(10).reshape(5, 2).float(),
            requires_grad=False,
            directory=root,
        ).materialize_()
        writes.clear()
        write_source[0] = initialized.table.untyped_storage()
        with patch.object(FileSystemWriter, "write_data", record_write):
            checkpoint.save_model(initialized, str(root / "initialized"))
        wanted = {0: [(0, 0)], 1: [(3, 0)]}.get(rank, [])
        assert [offset for fqn, offset in writes if fqn == "model.table"] == wanted
        _clear_model(initialized, rank)
        checkpoint.load_model(initialized, str(root / "initialized"))
        torch.testing.assert_close(
            initialized.table, torch.arange(10).reshape(5, 2).float(), rtol=0, atol=0
        )
        assert initialized.table.shard_sizes is None
    finally:
        NodeSharedCPUResidentParameter._node_group = None
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_node_shared_dcp_parallel_io(tmp_path, world_size):
    mp.spawn(_round_trip_worker, args=(world_size, str(tmp_path)), nprocs=world_size)


def _zero_shards_worker(rank, root):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'init'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("edp",))
        checkpoint = DcpCheckpoint(0, 1, mesh)
        for case, sizes in enumerate(([2, 0, 5, 3], [0, 2, 0, 5, 3, 0], [], [0, 0])):
            expected = torch.arange(sum(sizes) * 2).reshape(-1, 2).float()
            model = nn.Module()
            model.table = NodeSharedCPUResidentParameter(
                torch.empty_like(expected, device="meta"),
                requires_grad=False,
                directory=root,
            )
            chunks = expected.split(sizes)
            model.table.load_(sizes, lambda index, target: target.copy_(chunks[index]))
            native_write = FileSystemWriter.write_data
            writes = []

            def record_write(writer, plan, planner):
                writes.extend(
                    tuple(item.index.offset) for item in plan.items
                    if item.index.fqn == "model.table"
                )
                return native_write(writer, plan, planner)

            directory = root / str(case)
            with patch.object(FileSystemWriter, "write_data", record_write):
                checkpoint.save_model(model, str(directory))
            wanted = (
                [
                    (sum(sizes[:index]), 0)
                    for index, size in enumerate(sizes)
                    if size > 0 and index % 2 == rank
                ]
                if sum(sizes)
                else ([(0, 0)] if rank == 0 else [])
            )
            assert writes == wanted
            metadata = FileSystemReader(directory).read_metadata()
            saved = metadata.state_dict_metadata["model.table"]
            assert len(saved.chunks) == max(1, sum(size > 0 for size in sizes))
            _clear_model(model, rank)
            checkpoint.load_model(model, str(directory))
            torch.testing.assert_close(model.table, expected, rtol=0, atol=0)
            assert model.table.shard_sizes == tuple(sizes)
            fresh = nn.Module()
            fresh.table = NodeSharedCPUResidentParameter(
                torch.empty_like(expected, device="meta"),
                requires_grad=False,
                directory=root,
            )
            checkpoint.load_model(fresh, str(directory))
            torch.testing.assert_close(fresh.table, expected, rtol=0, atol=0)
            assert fresh.table.shard_sizes == tuple(sizes)

        columns = DTensor.from_local(
            torch.arange(16).reshape(4, 4).float()[:, rank * 2 : (rank + 1) * 2],
            mesh,
            [Shard(1)],
            shape=torch.Size((4, 4)),
            stride=(4, 1),
        )
        dcp.save({"model": {"table": columns}}, checkpoint_id=root / "columns")
        model = nn.Module()
        model.table = NodeSharedCPUResidentParameter(
            torch.empty(4, 4, device="meta"), requires_grad=False, directory=root
        )
        checkpoint.load_model(model, str(root / "columns"))
        torch.testing.assert_close(model.table, torch.arange(16).reshape(4, 4).float())
        assert model.table.shard_sizes is None
    finally:
        NodeSharedCPUResidentParameter._node_group = None
        dist.destroy_process_group()


def test_node_shared_dcp_zero_shards_and_chunk_geometry(tmp_path):
    mp.spawn(_zero_shards_worker, args=(str(tmp_path),), nprocs=2)


@pytest.mark.parametrize("mapping_state", ["unmaterialized", "converted"])
def test_node_shared_dcp_save_requires_current_mapping(
    tmp_path, single_rank_group, mapping_state
):
    other = NodeSharedCPUResidentParameter(
        torch.ones(2, 2), requires_grad=False, directory=tmp_path
    ).materialize_()
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.ones(4, 2), requires_grad=False, directory=tmp_path
    )
    if mapping_state == "converted":
        model.table.materialize_()
        model.double()
    assert model.table.group is other.group
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    with pytest.raises(
        RuntimeError, match="Call materialize_ before shared shard I/O"
    ):
        checkpoint.save_model(model, str(tmp_path / "invalid"))


def test_node_shared_dcp_load_rejects_stale_shard_coverage(
    tmp_path, single_rank_group
):
    source = nn.Module()
    source.table = NodeSharedCPUResidentParameter(
        torch.ones(4, 2), requires_grad=False, directory=tmp_path
    ).materialize_()
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_model(source, str(tmp_path / "dcp"))
    parameter = NodeSharedCPUResidentParameter(
        torch.ones(2, 2), requires_grad=False, directory=tmp_path
    )
    parameter.load_((2,), lambda index, target: target.fill_(1))
    parameter.data = torch.zeros(4, 2)
    parameter.materialize_()
    assert parameter.shape == (4, 2) and parameter.shard_sizes == (2,)
    model = nn.Module()
    model.table = parameter
    with pytest.raises(
        ValueError, match="Source shard sizes must cover the leading dimension"
    ):
        checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert model.table.shard_sizes == (2,)


def _native_layout_failure_worker(rank, root):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'init'}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        model = nn.Module()
        model.table = NodeSharedCPUResidentParameter(
            torch.empty(4, 2, device="meta"), requires_grad=False, directory=root
        )
        model.table.load_((2, 2), lambda index, target: target.fill_(index + 1))
        expected = model.table.detach().clone()
        pointer = model.table.data_ptr()
        checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
        checkpoint.save_model(model, str(root / "dcp"))
        if rank % 2 == 0:
            model.table.zero_()
        dist.barrier()

        # Each layout covers the tensor locally, but ranks 2 and 3 together
        # leave row 2 uncovered. Native validation fails only on that node.
        layout = (3, 1) if rank == 3 else (2, 2)
        model.table.shard_sizes = layout
        with pytest.raises(
            ValueError, match="do not cover the entire tensor"
        ) as failure:
            checkpoint.load_model(model, str(root / "dcp"))
        messages = [None] * 4
        dist.all_gather_object(messages, str(failure.value))
        assert len(set(messages)) == 1
        assert model.table.shard_sizes == layout

        model.table.shard_sizes = (2, 2)
        checkpoint.load_model(model, str(root / "dcp"))
        assert model.table.data_ptr() == pointer
        assert model.table.shard_sizes == (2, 2)
        torch.testing.assert_close(model.table, expected)
    finally:
        NodeSharedCPUResidentParameter._node_group = None
        dist.destroy_process_group()


def test_node_shared_dcp_native_layout_failure_reaches_every_node(tmp_path):
    mp.spawn(_native_layout_failure_worker, args=(str(tmp_path),), nprocs=4)


def _io_failure_worker(rank, root, operation):
    root = Path(root)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'init'}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("edp",))
        model, _ = _shared_model(root)
        checkpoint = DcpCheckpoint(0, 1, mesh)
        native_write = FileSystemWriter.write_data
        native_read = FileSystemReader.read_data
        message = f"injected shard {operation} failure"

        def fail_one_writer(writer, plan, planner):
            if rank == 1:
                raise OSError(message)
            return native_write(writer, plan, planner)

        def fail_one_reader(reader, plan, planner):
            if rank == 3 and any(
                item.dest_index.fqn == "model.table" for item in plan.items
            ):
                raise OSError(message)
            return native_read(reader, plan, planner)

        if operation == "read":
            checkpoint.save_model(model, str(root / "dcp"))
            model.table.shard_sizes = (5, 5)
        target, method, replacement = (
            (FileSystemWriter, "write_data", fail_one_writer)
            if operation == "write"
            else (FileSystemReader, "read_data", fail_one_reader)
        )
        with (
            patch.object(target, method, replacement),
            pytest.raises(dcp.CheckpointException, match=message),
        ):
            if operation == "write":
                checkpoint.save_model(model, str(root / "failed"))
            else:
                checkpoint.load_model(model, str(root / "dcp"))
        if operation == "read":
            assert model.table.shard_sizes == (5, 5)
        dist.barrier()
    finally:
        NodeSharedCPUResidentParameter._node_group = None
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="PyTorch DCP cannot pickle Python 3.14 traceback code objects",
)
@pytest.mark.parametrize("operation", ["read", "write"])
def test_node_shared_dcp_io_failure_reaches_every_node(tmp_path, operation):
    mp.spawn(_io_failure_worker, args=(str(tmp_path), operation), nprocs=4)


def test_node_shared_dcp_releases_replaced_storage_before_native_read(
    tmp_path, monkeypatch, single_rank_group
):
    source = nn.Module()
    source.first = NodeSharedCPUResidentParameter(
        torch.ones(4, 2, dtype=torch.bfloat16),
        requires_grad=False,
        directory=tmp_path,
    ).materialize_()
    source.second = NodeSharedCPUResidentParameter(
        torch.full((4, 2), 2, dtype=torch.bfloat16),
        requires_grad=False,
        directory=tmp_path,
    ).materialize_()
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_model(source, str(tmp_path / "dcp"))
    model = nn.Module()
    model.first = NodeSharedCPUResidentParameter(
        torch.zeros(4, 2, dtype=torch.float32),
        requires_grad=False,
        directory=tmp_path,
    ).materialize_()
    model.second = NodeSharedCPUResidentParameter(
        torch.zeros(4, 2, dtype=torch.float32),
        requires_grad=False,
        directory=tmp_path,
    ).materialize_()
    old_storage = weakref.ref(model.first.untyped_storage())
    assert old_storage() is not None
    native_read = FileSystemReader.read_data
    observed = []

    def read(reader, plan, planner):
        observed.append((model.first.dtype, model.second.dtype))
        assert old_storage() is None
        for item in plan.items:
            if not item.dest_index.fqn.startswith("model."):
                continue
            name = item.dest_index.fqn.removeprefix("model.")
            assert planner.resolve_tensor(item).untyped_storage() is (
                model.get_parameter(name).untyped_storage()
            )
        return native_read(reader, plan, planner)

    monkeypatch.setattr(FileSystemReader, "read_data", read)
    checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert observed == [(torch.bfloat16, torch.bfloat16)]
    torch.testing.assert_close(model.first, source.first)
    torch.testing.assert_close(model.second, source.second)


def test_node_shared_dcp_layout_changes_only_after_strict_model_load(
    tmp_path, single_rank_group
):
    source = nn.Module()
    source.table = NodeSharedCPUResidentParameter(
        torch.ones(10, 2), requires_grad=False, directory=tmp_path
    )
    source.table.load_((2, 0, 5, 3), lambda index, target: target.fill_(index))
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_model(source, str(tmp_path / "dcp"))
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.zeros(10, 2), requires_grad=False, directory=tmp_path
    )
    model.table.load_((5, 5), lambda index, target: target.zero_())

    def fail_load(module, incompatible):
        raise RuntimeError("injected strict model load failure")

    hook = model.register_load_state_dict_post_hook(fail_load)
    with pytest.raises(RuntimeError, match="injected strict model load failure"):
        checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert model.table.shard_sizes == (5, 5)
    hook.remove()
    checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert model.table.shard_sizes == (2, 0, 5, 3)
    torch.testing.assert_close(model.table, source.table)


def test_node_shared_dcp_rejects_conflicting_tied_layouts(tmp_path, single_rank_group):
    source = nn.Module()
    for name, sizes in (("table", (2, 2)), ("alias", (1, 3))):
        parameter = NodeSharedCPUResidentParameter(
            torch.ones(4, 2), requires_grad=False, directory=tmp_path
        )
        parameter.load_(sizes, lambda index, target: target.fill_(1))
        source.register_parameter(name, parameter)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_model(source, str(tmp_path / "dcp"))
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.zeros(4, 2), requires_grad=False, directory=tmp_path
    )
    model.table.load_((4,), lambda index, target: target.zero_())
    model.alias = model.table
    with pytest.raises(ValueError, match="Tied CPU checkpoint shard sizes differ"):
        checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert model.alias is model.table
    assert model.table.shard_sizes == (4,)


def test_node_shared_dcp_saves_state_dict_hook_values(tmp_path, single_rank_group):
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.ones(4, 2), requires_grad=False, directory=tmp_path
    ).materialize_()

    def export_hook(module, state_dict, prefix, local_metadata):
        key = f"{prefix}table"
        state_dict[key] = state_dict[key].bfloat16() * 2

    hook = model.register_state_dict_post_hook(export_hook)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_model(model, str(tmp_path / "dcp"))
    restored = {"model": {"table": torch.empty(4, 2, dtype=torch.bfloat16)}}
    dcp.load(restored, checkpoint_id=tmp_path / "dcp")
    torch.testing.assert_close(
        restored["model"]["table"], torch.full((4, 2), 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(model.table, torch.ones(4, 2))
    with torch.no_grad():
        model.table.fill_(7)
    checkpoint.load_model(model, str(tmp_path / "dcp"))
    assert model.table.dtype == torch.bfloat16
    torch.testing.assert_close(
        model.table, torch.full((4, 2), 2, dtype=torch.bfloat16)
    )
    hook.remove()


def test_node_shared_dcp_rejects_state_dict_hook_shape_change(
    tmp_path, single_rank_group
):
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.ones(4, 2), requires_grad=False, directory=tmp_path
    ).materialize_()
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())

    def reshape_hook(module, state_dict, prefix, local_metadata):
        key = f"{prefix}table"
        state_dict[key] = state_dict[key].flatten()

    model.register_state_dict_post_hook(reshape_hook)
    with pytest.raises(ValueError, match="must retain the parameter shape"):
        checkpoint.save_model(model, str(tmp_path / "invalid"))
