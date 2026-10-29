"""Node-shared parameter behavior, CPU backing, and per-rank file I/O."""

import gc
import importlib.util
import os
import sys
import weakref
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


def _package():
    name = "_torchturbo_cpu_resident_under_test"
    if name not in sys.modules:
        path = (
            Path(__file__).resolve().parents[3]
            / "src/torchturbo/models/modules/cpu_resident/__init__.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return sys.modules[name]


@pytest.fixture
def node_group(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    init_file = tmp_path_factory.mktemp("node-group") / "init"
    _initialize(0, 1, str(init_file))
    try:
        yield
    finally:
        dist.destroy_process_group()
        # Each test owns a separate distributed job in this pytest process.
        _package().NodeSharedCPUResidentParameter._node_group = None


def test_shared_materialization_preserves_optimizer_identity_and_trains(
    tmp_path, node_group
):
    parameter = _package().NodeSharedCPUResidentParameter(
        torch.empty(3, 2, device="meta"), directory=tmp_path
    )
    assert isinstance(parameter, _package().CPUResidentParameter)
    model = nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert parameter.materialize_() is parameter
    assert model.weight is optimizer.param_groups[0]["params"][0] is parameter
    assert parameter.device.type == "cpu"
    with torch.no_grad():
        parameter.fill_(2)
    parameter.sum().backward()
    optimizer.step()
    torch.testing.assert_close(parameter, torch.full((3, 2), 1.9))


def test_parameter_retains_loaded_layout_for_save_and_deepcopy(
    tmp_path, node_group
):
    cls = _package().NodeSharedCPUResidentParameter
    parameter = cls(torch.empty(3, 2, device="meta"), False, directory=tmp_path)
    assert parameter.shard_sizes is None
    with pytest.raises(RuntimeError, match="materialize"):
        parameter.save(lambda index, value: None)
    parameter.materialize_()
    sizes = [2, 0, 1]
    parameter.load_(sizes, lambda index, value: value.fill_(index + 1))
    sizes[0] = 9
    assert parameter.shard_sizes == (2, 0, 1)

    def fail(index, value):
        raise OSError("reader failed")

    with pytest.raises((OSError, RuntimeError), match="reader failed"):
        parameter.load_([1, 2], fail)
    assert parameter.shard_sizes == (2, 0, 1)
    copied = deepcopy(parameter)
    assert copied.shard_sizes == (2, 0, 1)
    assert copied.data_ptr() != parameter.data_ptr()
    assert copied.directory == tmp_path
    assert not copied.is_shared()
    copied.materialize_()
    copied.materialize_(dtype=torch.float64)
    assert copied.shard_sizes == (2, 0, 1)
    saved = []
    copied.save(lambda index, value: saved.append((index, value.clone())))
    assert [(index, len(value)) for index, value in saved] == [(0, 2), (1, 0), (2, 1)]
    torch.testing.assert_close(
        torch.cat([value for _, value in saved]), parameter.double()
    )


@pytest.mark.parametrize("shape", [(), (3, 2), (0, 2)])
def test_save_without_source_layout_preserves_storage_and_values(
    tmp_path, shape, node_group
):
    cls = _package().NodeSharedCPUResidentParameter
    expected = torch.full(shape, 7.0)
    parameter = cls(expected.clone(), False, directory=tmp_path)
    parameter.materialize_()
    pointer = parameter.data_ptr()
    saved = []
    parameter.save(lambda index, value: saved.append((index, value.clone())))
    assert len(saved) == 1 and saved[0][0] == 0
    torch.testing.assert_close(saved[0][1], expected)
    torch.testing.assert_close(parameter, expected)
    assert parameter.data_ptr() == pointer
    assert parameter.shard_sizes is None


def test_local_shards_returns_views_without_collectives_or_layout_changes(
    tmp_path, node_group, monkeypatch
):
    cls = _package().NodeSharedCPUResidentParameter
    parameter = cls(torch.arange(6.0).reshape(3, 2), directory=tmp_path)
    parameter.materialize_()
    pointer = parameter.data_ptr()

    def unexpected(*args, **kwargs):
        raise AssertionError("local_shards must not allocate or communicate")

    monkeypatch.setattr(cls, "materialize_", unexpected)
    monkeypatch.setattr(dist, "all_gather_object", unexpected)
    monkeypatch.setattr(dist, "broadcast_object_list", unexpected)
    monkeypatch.setattr(dist, "barrier", unexpected)
    monkeypatch.setattr(dist, "new_subgroups", unexpected)
    for for_save in (False, True):
        full = parameter.local_shards(for_save=for_save)
        assert [(index, offset, len(view)) for index, offset, view in full] == [
            (0, 0, 3)
        ]
        assert parameter.shard_sizes is None
        explicit = parameter.local_shards([2, 0, 1], for_save=for_save)
        assert [(index, offset, len(view)) for index, offset, view in explicit] == [
            (0, 0, 2),
            (1, 2, 0),
            (2, 2, 1),
        ]
        assert all(not view.requires_grad for _, _, view in explicit)
        assert explicit[0][2].data_ptr() == pointer
        assert explicit[2][2].data_ptr() == pointer + 4 * parameter.element_size()
        torch.testing.assert_close(
            torch.cat([view for _, _, view in explicit]), parameter
        )
        assert parameter.shard_sizes is None
    parameter.shard_sizes = (1, 2)
    remembered = parameter.local_shards()
    assert [(index, offset, len(view)) for index, offset, view in remembered] == [
        (0, 0, 1),
        (1, 1, 2),
    ]
    parameter.local_shards([2, 1])
    assert parameter.shard_sizes == (1, 2)
    assert parameter.data_ptr() == pointer
    torch.testing.assert_close(parameter, torch.arange(6.0).reshape(3, 2))


@pytest.mark.parametrize("shape,sizes", [((), [1]), ((0, 2), []), ((0, 2), [0, 0])])
def test_local_shards_preserves_scalar_and_empty_source_intervals(
    shape, sizes, tmp_path, node_group
):
    parameter = _package().NodeSharedCPUResidentParameter(
        torch.empty(shape), directory=tmp_path
    ).materialize_()
    actual = parameter.local_shards(sizes)
    assert [(index, offset) for index, offset, _ in actual] == [
        (index, 0) for index in range(len(sizes))
    ]
    assert all(view.shape == parameter.shape for _, _, view in actual)
    assert parameter.shard_sizes is None


def test_load_allocates_checkpoint_dtype_without_replacing_parameter(
    tmp_path, node_group
):
    cls = _package().NodeSharedCPUResidentParameter
    parameter = cls(
        torch.empty(3, 2, device="meta"), requires_grad=False, directory=tmp_path
    )
    shards = [torch.full((1, 2), 2.5), torch.full((2, 2), -3.0)]
    result = parameter.load_(
        [1, 2],
        lambda index, target: target.copy_(shards[index]),
        dtype=torch.bfloat16,
    )
    assert result is parameter
    assert parameter.dtype == torch.bfloat16
    assert parameter.device.type == "cpu"
    torch.testing.assert_close(
        parameter, torch.tensor([[2.5, 2.5], [-3, -3], [-3, -3]]).bfloat16()
    )
    pointer = parameter.data_ptr()
    parameter.load_(
        [1, 2],
        lambda index, target: target.copy_(shards[index]),
    )
    assert parameter.data_ptr() == pointer
    assert parameter.dtype == torch.bfloat16
    parameter.materialize_(dtype=torch.float64)
    assert parameter.dtype == torch.float64
    torch.testing.assert_close(
        parameter, torch.tensor([[2.5, 2.5], [-3, -3], [-3, -3]]).double()
    )


@pytest.mark.parametrize("device", ["meta", "cpu"])
def test_materialization_preserves_hooks_and_removable_handles(
    tmp_path, device, node_group
):
    cls = _package().NodeSharedCPUResidentParameter
    parameter = cls(torch.empty(2, device=device), directory=tmp_path)
    calls = []

    def scale_gradient(gradient):
        calls.append("pre")
        return gradient * 3

    def after_accumulation(value):
        assert value is parameter
        calls.append("post")

    pre_handle = parameter.register_hook(scale_gradient)
    post_handle = parameter.register_post_accumulate_grad_hook(after_accumulation)
    parameter.materialize_()
    parameter.materialize_()
    with torch.no_grad():
        parameter.fill_(1)
    parameter.sum().backward()
    assert calls == ["pre", "post"]
    torch.testing.assert_close(parameter.grad, torch.full((2,), 3.0))
    pre_handle.remove()
    parameter.grad = None
    calls.clear()
    parameter.materialize_()
    parameter.sum().backward()
    assert calls == ["post"]
    torch.testing.assert_close(parameter.grad, torch.ones(2))
    post_handle.remove()
    parameter.grad = None
    calls.clear()
    parameter.sum().backward()
    assert calls == []
    torch.testing.assert_close(parameter.grad, torch.ones(2))


def test_single_process_shared_training_reuse_deepcopy_and_dtype(tmp_path, node_group):
    cls = _package().NodeSharedCPUResidentParameter
    parameter = cls(torch.empty(3, 2, device="meta"), directory=tmp_path)
    with pytest.raises(RuntimeError, match="materialize"):
        parameter.save(lambda index, value: None)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    assert parameter.materialize_() is parameter
    assert parameter.is_shared()

    def initialize(index, target):
        assert not target.requires_grad
        target.fill_(2)

    parameter.load_([1, 2], initialize)
    parameter.sum().backward()
    optimizer.step()
    torch.testing.assert_close(parameter, torch.full((3, 2), 1.9))
    saved = []

    def writer(index, value):
        assert not value.requires_grad
        saved.append(value.clone())

    parameter.save(writer)
    torch.testing.assert_close(torch.cat(saved), torch.full((3, 2), 1.9))
    pointer = parameter.data_ptr()
    parameter.materialize_()
    assert parameter.data_ptr() == pointer
    copied = deepcopy(parameter)
    assert isinstance(copied, cls)
    assert not copied.is_shared()
    with pytest.raises(RuntimeError, match="materialize"):
        copied.save(lambda index, value: None)
    copied.materialize_()
    with torch.no_grad():
        copied.zero_()
    torch.testing.assert_close(parameter, torch.full((3, 2), 1.9))
    parameter.grad = None
    old_storage = weakref.ref(parameter.untyped_storage())
    model = nn.Module()
    model.weight = parameter
    model.double()
    gc.collect()
    assert old_storage() is None
    with pytest.raises(RuntimeError, match="materialize"):
        parameter.save(lambda index, value: None)
    parameter.materialize_()
    assert model.weight is parameter
    assert parameter.dtype == torch.float64
    assert parameter.is_shared()
    torch.testing.assert_close(parameter, torch.full((3, 2), 1.9).double())
    assert list(tmp_path.iterdir()) == []


def test_shared_mapping_copies_noncontiguous_values_and_saves_plain_state(
    tmp_path, node_group
):
    cls = _package().NodeSharedCPUResidentParameter
    model = nn.Module()
    model.weight = cls(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t(), False, directory=tmp_path
    )
    assert not model.weight.is_contiguous()
    model.weight.materialize_()
    assert model.weight.is_contiguous()
    torch.testing.assert_close(model.weight, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
    state = model.state_dict()
    assert type(state["weight"]) is torch.Tensor
    torch.save(state, tmp_path / "state.pt")
    model.weight.zero_()
    model.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    torch.testing.assert_close(model.weight, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def _initialize(rank, num_ranks, init_file):
    os.environ["LOCAL_WORLD_SIZE"] = "2" if num_ranks > 1 else "1"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=num_ranks,
        timeout=timedelta(seconds=30),
    )


def _io_worker(rank, init_file, directory):
    _initialize(rank, 2, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        parameter = cls(
            torch.full((7, 2), float(rank + 2)).share_memory_(), False, directory=root
        )
        parameter.materialize_()
        group = parameter.group
        assert parameter.is_shared() and parameter.is_contiguous()
        torch.testing.assert_close(parameter, torch.full((7, 2), 2.0))
        reads = []

        def reader(index, target):
            reads.append(index)
            assert not target.requires_grad and target.device.type == "cpu"
            with safe_open(
                root / f"source-{index}.safetensors", framework="pt", device="cpu"
            ) as source:
                target.copy_(source.get_tensor("weight"))

        assert parameter.load_([2, 1, 3, 1], reader) is parameter
        assert reads == ([0, 2] if rank == 0 else [1, 3])
        assert [parameter.shard_owner(index) for index in range(4)] == [0, 1, 0, 1]
        expected = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
            ]
        )
        torch.testing.assert_close(parameter, expected)
        writes = []

        def writer(index, value):
            writes.append(index)
            assert not value.requires_grad
            save_file({"weight": value}, str(root / f"saved-{index}.safetensors"))

        parameter.save(writer)
        assert writes == ([0, 2] if rank == 0 else [1, 3])
        pointer = parameter.data_ptr()
        parameter.materialize_()
        assert parameter.data_ptr() == pointer
        if rank == 1:
            parameter.requires_grad_(True)
        with pytest.raises(RuntimeError, match="optimizer ownership"):
            parameter.materialize_()
        parameter.requires_grad_(False)
        previous = weakref.ref(parameter.untyped_storage())
        if rank == 1:
            parameter.data = torch.zeros(7, 2)
        parameter.materialize_()
        gc.collect()
        assert previous() is None
        torch.testing.assert_close(parameter, expected)
        dist.barrier(group=group)
        if rank == 1:
            parameter[0, 0] = 17
        dist.barrier(group=group)
        assert parameter[0, 0].item() == 17
        copied = deepcopy(parameter)
        copied.materialize_()
        assert copied.group is group
        if rank == 0:
            copied.fill_(23)
        dist.barrier(group=group)
        torch.testing.assert_close(copied, torch.full((7, 2), 23.0))
        assert parameter[0, 0].item() == 17
        old_storage = weakref.ref(parameter.untyped_storage())
        model = nn.Module()
        model.weight = parameter
        model.double()
        gc.collect()
        assert old_storage() is None
        parameter.materialize_()
        if rank == 1:
            parameter[0, 0] = 29
        dist.barrier(group=group)
        assert parameter.dtype == torch.float64 and parameter[0, 0].item() == 29
        for shape, sizes, assigned in [
            ((), [1], [0] if rank == 0 else []),
            ((1, 2), [1], [0] if rank == 0 else []),
            ((0, 2), [0, 0], [0] if rank == 0 else [1]),
            ((0, 2), [], []),
            ((3, 0), [1, 2], [0] if rank == 0 else [1]),
            ((3, 2), [0, 3], [0] if rank == 0 else [1]),
        ]:
            value = cls(torch.empty(shape, device="meta"), False, directory=root)
            reads = []

            def fill(index, target, reads=reads):
                reads.append(index)
                target.fill_(41)

            value.load_(sizes, fill)
            assert reads == assigned
            torch.testing.assert_close(value, torch.full(shape, 41.0))
        assert not list(root.glob("torchturbo-cpu-resident-*"))
    finally:
        dist.destroy_process_group()


def test_source_shards_are_loaded_and_saved_by_assigned_node_ranks(tmp_path):
    shards = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        torch.tensor([[13.0, 14.0]]),
    ]
    for index, shard in enumerate(shards):
        save_file({"weight": shard}, str(tmp_path / f"source-{index}.safetensors"))
    mp.spawn(
        _io_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True
    )
    restored = torch.cat(
        [
            load_file(tmp_path / f"saved-{index}.safetensors")["weight"]
            for index in range(4)
        ]
    )
    torch.testing.assert_close(
        restored,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
            ]
        ),
    )


def _failure_worker(rank, init_file, directory):
    _initialize(rank, 2, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    anchor = cls(torch.empty(0), False, directory=root).materialize_()
    group = anchor.group
    try:
        for data in [
            torch.empty((2 + rank, 2), device="meta"),
            torch.empty(
                2, dtype=torch.float32 if rank == 0 else torch.float64, device="meta"
            ),
        ]:
            parameter = cls(data, False, directory=root)
            with pytest.raises(RuntimeError, match="metadata"):
                parameter.materialize_()
            assert parameter.is_meta
        parameter = cls(
            torch.empty(2, device="meta"), requires_grad=rank == 1, directory=root
        )
        with pytest.raises(RuntimeError, match="optimizer ownership"):
            parameter.materialize_()
        parameter = cls(
            torch.empty(2, device="meta"), False, directory=root / "missing"
        )
        with pytest.raises(RuntimeError, match="No such file|not found"):
            parameter.materialize_()
        assert parameter.is_meta
        # Sparse metadata has no file-mapping representation.
        data = torch.eye(2).to_sparse() if rank == 1 else torch.eye(2)
        parameter = cls(data, False, directory=root)
        with pytest.raises(RuntimeError, match="strided"):
            parameter.materialize_()
        parameter = cls(torch.empty(2, device="meta"), False, directory=root)

        def fail_resize(*args):
            raise OSError("injected file resize failure")

        with pytest.MonkeyPatch.context() as patch:
            if rank == 0:
                patch.setattr(os, "ftruncate", fail_resize)
            with pytest.raises(RuntimeError, match="file resize failure"):
                parameter.materialize_()
        assert parameter.is_meta
        dist.barrier(group=group)
        assert not list(root.glob("torchturbo-cpu-resident-*"))

        def fail_map(*args, **kwargs):
            raise OSError("injected peer mapping failure")

        with pytest.MonkeyPatch.context() as patch:
            if rank == 1:
                patch.setattr(torch, "from_file", fail_map)
            with pytest.raises(RuntimeError, match="peer mapping failure"):
                parameter.materialize_()
        assert parameter.is_meta
        dist.barrier(group=group)
        assert not list(root.glob("torchturbo-cpu-resident-*"))
        parameter.materialize_()
        if rank == 0:
            parameter.fill_(7)
        dist.barrier(group=group)
        torch.testing.assert_close(parameter, torch.full((2,), 7.0))
    finally:
        dist.destroy_process_group()


def test_group_metadata_and_allocation_errors_reach_every_rank(tmp_path):
    mp.spawn(
        _failure_worker,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _cleanup_worker(rank, init_file, directory):
    _initialize(rank, 2, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    anchor = cls(torch.empty(0), False, directory=root).materialize_()
    group = anchor.group
    unlink = Path.unlink

    def fail_unlink(path, *args, **kwargs):
        if path.name.startswith("torchturbo-cpu-resident-"):
            raise PermissionError("injected cleanup unlink failure")
        return unlink(path, *args, **kwargs)

    def fail_map(*args, **kwargs):
        raise OSError("injected primary mapping failure")

    try:
        for fail_mapping in (False, True):
            parameter = cls(torch.empty(2, device="meta"), False, directory=root)
            message = (
                "primary mapping failure" if fail_mapping else "cleanup unlink failure"
            )
            with pytest.MonkeyPatch.context() as patch:
                if rank == 0:
                    patch.setattr(Path, "unlink", fail_unlink)
                if rank == 1 and fail_mapping:
                    patch.setattr(torch, "from_file", fail_map)
                with pytest.raises(RuntimeError, match=message) as caught:
                    parameter.materialize_()
            assert parameter.is_meta
            errors = [None, None]
            dist.all_gather_object(errors, str(caught.value), group=group)
            assert errors[0] == errors[1]
            # A failed OS unlink cannot remove its file; restore fixture access.
            if rank == 0:
                remaining = list(root.glob("torchturbo-cpu-resident-*"))
                assert len(remaining) == 1
                remaining[0].unlink()
            dist.barrier(group=group)
            parameter.materialize_()
            if rank == 0:
                parameter.fill_(11)
            dist.barrier(group=group)
            torch.testing.assert_close(parameter, torch.full((2,), 11.0))
            assert not list(root.glob("torchturbo-cpu-resident-*"))
    finally:
        dist.destroy_process_group()


def test_cleanup_failures_preserve_group_agreement_and_allow_retry(tmp_path):
    mp.spawn(
        _cleanup_worker,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _world_materialize_worker(rank, init_file, directory, reuse_mapping, failure):
    _initialize(rank, 4, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        parameter = cls(torch.full((3,), 7.0), False, directory=root)
        if reuse_mapping:
            parameter.materialize_()
            if rank >= 2:
                parameter.data = parameter.detach().clone()
        original_pointer = parameter.data_ptr()
        if failure == "allocation" and rank >= 2:
            parameter.directory = root / "missing"

        def fail_bind(self, value):
            raise OSError("injected second-node binding failure")

        error = None
        with pytest.MonkeyPatch.context() as patch:
            if failure == "binding" and rank == 3:
                patch.setattr(cls, "_bind", fail_bind)
            try:
                parameter.materialize_()
            except RuntimeError as exc:
                error = str(exc)
        errors = [None] * 4
        dist.all_gather_object(errors, error)
        assert all(message is not None for message in errors), errors
        assert len(set(errors)) == 1
        if reuse_mapping and rank < 2:
            assert parameter.data_ptr() == original_pointer
        torch.testing.assert_close(parameter, torch.full((3,), 7.0))
        parameter.directory = root
        pointer = parameter.data_ptr()
        assert parameter.materialize_() is parameter
        assert parameter._mapping_is_current()
        assert parameter.shard_sizes is None
        if rank < 2:
            assert parameter.data_ptr() == pointer
        torch.testing.assert_close(parameter, torch.full((3,), 7.0))
        assert not list(root.glob("torchturbo-cpu-resident-*"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reuse_mapping", [False, True])
@pytest.mark.parametrize("failure", ["allocation", "binding"])
def test_materialize_errors_reach_every_node_including_cached_peers(
    tmp_path, reuse_mapping, failure
):
    mp.spawn(
        _world_materialize_worker,
        args=(str(tmp_path / "init"), str(tmp_path), reuse_mapping, failure),
        nprocs=4,
        join=True,
    )


def _groups_worker(rank, init_file, directory):
    _initialize(rank, 4, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        with torch.device("meta"):
            anchor = cls(torch.empty(0), False, directory=root)
        # Construction is local; no node group exists before materialization.
        assert cls._node_group is None and anchor.is_meta
        with pytest.raises(RuntimeError, match="materialize"):
            _ = anchor.group
        with pytest.raises(RuntimeError, match="materialize"):
            anchor.local_shards()
        copied = deepcopy(anchor)
        assert copied.is_meta and cls._node_group is None
        if rank == 0:
            extra = cls(torch.empty(2, device="meta"), False, directory=root)
            assert extra.is_meta and cls._node_group is None
        assert not list(root.glob("torchturbo-cpu-resident-*"))
        anchor.materialize_()
        group = anchor.group
        assert dist.get_rank(group) == rank % 2
        assert dist.get_world_size(group) == 2
        assert dist.get_process_group_ranks(group) == ([0, 1] if rank < 2 else [2, 3])
        assert copied.group is group
        dist.barrier(group=group)
        parameter = cls(torch.full((3,), float(rank + 10)), False, directory=root)
        assert parameter.group is group
        failed = cls(
            torch.empty(2, device="meta"),
            False,
            directory=root / "missing" if rank < 2 else root,
        )
        with pytest.raises(RuntimeError, match="No such file|not found"):
            failed.materialize_()
        parameter.materialize_()
        assert parameter.group is group
        balanced = parameter.local_shards()
        expected_shard = (0, 0, 2) if rank % 2 == 0 else (1, 2, 1)
        assert [(index, offset, len(view)) for index, offset, view in balanced] == [
            expected_shard
        ]
        assert parameter.shard_sizes is None
        for explicit in ([0, 1, 0, 2], [1, 2]):
            views = parameter.local_shards(explicit)
            assert [index for index, _, _ in views] == list(
                range(rank % 2, len(explicit), 2)
            )
            for index, offset, view in views:
                assert offset == sum(explicit[:index])
                torch.testing.assert_close(
                    view, parameter[offset : offset + explicit[index]]
                )
            exported = parameter.local_shards(explicit, for_save=True)
            assert [index for index, _, _ in exported] == (
                [index for index, _, _ in views] if rank < 2 else []
            )
        empty = anchor.local_shards([0, 0])
        assert [(index, offset, view.numel()) for index, offset, view in empty] == [
            (rank % 2, 0, 0)
        ]
        assert len(anchor.local_shards()) == (1 if rank % 2 == 0 else 0)
        expected = 10.0 if rank < 2 else 12.0
        torch.testing.assert_close(parameter, torch.full((3,), expected))
        default_writes = []
        pointer = parameter.data_ptr()
        parameter.save(
            lambda index, value: default_writes.append((index, len(value)))
        )
        assert default_writes == ([(rank, 2 - rank)] if rank < 2 else [])
        assert parameter.shard_sizes is None and parameter.data_ptr() == pointer
        torch.testing.assert_close(parameter, torch.full((3,), expected))
        dist.barrier(group=group)
        if rank % 2:
            parameter.fill_(expected + 10)
        dist.barrier(group=group)
        torch.testing.assert_close(parameter, torch.full((3,), expected + 10))
        parameter.load_([1, 2], lambda index, value: None)
        remembered = parameter.local_shards()
        expected_shard = (0, 0, 1) if rank % 2 == 0 else (1, 1, 2)
        assert [(index, offset, len(view)) for index, offset, view in remembered] == [
            expected_shard
        ]
        writes = []

        def writer(index, value):
            writes.append(index)
            save_file(
                {"weight": value}, str(root / f"selected-node-{index}.safetensors")
            )

        parameter.save(writer)
        assert writes == ([rank] if rank < 2 else [])
        assert (root / "selected-node-0.safetensors").is_file()
        assert (root / "selected-node-1.safetensors").is_file()

        def fail(index, value):
            if index == 1:
                raise OSError("selected-node writer failed")

        with pytest.raises(RuntimeError, match="selected-node writer failed"):
            parameter.save(fail)
        # Layout disagreement must fail before any rank enters a writer.
        if rank == 1:
            parameter.shard_sizes = None
        with pytest.raises(RuntimeError, match="shard sizes must agree"):
            parameter.save(writer)
        assert writes == ([rank] if rank < 2 else [])
        parameter.shard_sizes = (1, 2)
        parameter.save(lambda index, value: None)
    finally:
        dist.destroy_process_group()


def test_internal_node_groups_are_reused_and_only_selected_node_saves(tmp_path):
    mp.spawn(
        _groups_worker,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=4,
        join=True,
    )
    actual = torch.cat(
        [
            load_file(tmp_path / f"selected-node-{index}.safetensors")["weight"]
            for index in range(2)
        ]
    )
    torch.testing.assert_close(actual, torch.full((3,), 20.0))


def _world_load_worker(rank, init_file, directory, phase):
    _initialize(rank, 4, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        parameter = cls(torch.empty(3, device="meta"), False, directory=root)
        parameter.load_([1, 2], lambda index, value: value.fill_(7))
        if phase == "materialize" and rank >= 2:
            parameter.directory = root / "missing"

        def fail_on_second_node(index, value):
            if phase == "reader" and rank == 3:
                raise OSError("injected second-node read failure")
            value.fill_(9)

        error = None
        try:
            parameter.load_([2, 1], fail_on_second_node, dtype=torch.float64)
        except RuntimeError as exc:
            error = str(exc)
        outcomes = [None] * 4
        dist.all_gather_object(outcomes, (error, parameter.shard_sizes))
        assert all(message is not None for message, _ in outcomes), outcomes
        assert len({message for message, _ in outcomes}) == 1
        assert all(sizes == (1, 2) for _, sizes in outcomes)
        parameter.directory = root

        def recover(index, value):
            assert parameter.shard_sizes == (1, 2)
            value.fill_(11)

        assert parameter.load_([2, 1], recover, dtype=torch.float64) is parameter
        assert parameter.shard_sizes == (2, 1)
        torch.testing.assert_close(parameter, torch.full((3,), 11.0).double())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", ["materialize", "reader"])
def test_load_errors_reach_every_node_and_layout_commits_only_after_success(
    tmp_path, phase
):
    mp.spawn(
        _world_load_worker,
        args=(str(tmp_path / "init"), str(tmp_path), phase),
        nprocs=4,
        join=True,
    )


def _partition_worker(rank, init_file, directory):
    _initialize(rank, 2, init_file)
    root = Path(directory)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        parameter = cls(torch.full((3, 2), 7.0), False, directory=root)
        parameter.load_([1, 2], lambda index, value: None)
        calls = []

        def record(index, value):
            calls.append(index)
            value.zero_()

        target = parameter if rank == 0 else deepcopy(parameter)
        with pytest.raises(RuntimeError, match="group rank 1.*materialize"):
            target.save(record)
        assert calls == []
        for sizes in ([1, 1], [-1, 4], [1.5, 1.5], [1, 2] if rank == 0 else [2, 1]):
            with pytest.raises(RuntimeError, match="shard"):
                parameter.load_(sizes, record)
            parameter.shard_sizes = tuple(sizes)
            with pytest.raises(RuntimeError, match="shard"):
                parameter.save(record)
            assert calls == []
            torch.testing.assert_close(parameter, torch.full((3, 2), 7.0))
        scalar = cls(torch.tensor(5.0), False, directory=root).materialize_()
        scalar.shard_sizes = (0, 1)
        with pytest.raises(RuntimeError, match="scalar"):
            scalar.save(record)
        assert calls == []
        parameter.load_([1, 2], lambda index, value: None)

        def fail(index, value):
            if index == 1:
                raise OSError("source shard I/O failed")

        with pytest.raises(RuntimeError, match="source shard I/O failed"):
            parameter.load_([1, 2], fail)
        with pytest.raises(RuntimeError, match="source shard I/O failed"):
            parameter.save(fail)
        parameter.load_([1, 2], lambda index, value: value.fill_(19))
        torch.testing.assert_close(parameter, torch.full((3, 2), 19.0))
        writes = []
        parameter.save(lambda index, value: writes.append(index))
        assert writes == ([0] if rank == 0 else [1])
    finally:
        dist.destroy_process_group()


def test_shard_partition_and_callback_failures_are_collective_and_recoverable(tmp_path):
    mp.spawn(
        _partition_worker,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _configuration_worker(rank, init_file, directory):
    _initialize(rank, 2, init_file)
    cls = _package().NodeSharedCPUResidentParameter
    try:
        for setting in (None, "invalid", "0", "3", "1" if rank == 0 else "2"):
            if setting is None:
                os.environ.pop("LOCAL_WORLD_SIZE")
            else:
                os.environ["LOCAL_WORLD_SIZE"] = setting
            parameter = cls(torch.empty(2), False, directory=directory)
            with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
                parameter.materialize_()
            assert cls._node_group is None
        os.environ["LOCAL_WORLD_SIZE"] = "2"
        value = cls(
            torch.full((2,), float(rank)), False, directory=directory
        ).materialize_()
        torch.testing.assert_close(value, torch.zeros(2))
    finally:
        dist.destroy_process_group()


def test_node_configuration_errors_agree_before_subgroup_creation(tmp_path):
    mp.spawn(
        _configuration_worker,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def test_shared_construction_is_local_and_materialization_requires_distributed(
    tmp_path,
):
    assert not dist.is_initialized()
    cls = _package().NodeSharedCPUResidentParameter
    declared = cls(torch.empty(2, device="meta"), directory=tmp_path)
    assert declared.is_meta
    with pytest.raises(RuntimeError, match="materialize"):
        _ = declared.group
    with pytest.raises(RuntimeError, match="initialized"):
        declared.materialize_()
    with pytest.raises(RuntimeError, match="initialized"):
        declared.load_([2], lambda index, value: None)
    assert declared.is_meta and cls._node_group is None
    # Real values stay usable as private CPU data without any process group.
    local = cls(torch.tensor([1.0, 2.0]), False, directory=tmp_path)
    assert local.device.type == "cpu" and not local.is_shared()
    torch.testing.assert_close(deepcopy(local) + 1, torch.tensor([2.0, 3.0]))


def test_one_process_world_creates_cpu_node_group(tmp_path, node_group):
    parameter = _package().NodeSharedCPUResidentParameter(
        torch.empty(1), directory=tmp_path
    )
    parameter.load_([1], lambda index, value: value.fill_(3))
    assert dist.get_backend(parameter.group) == "gloo"
    assert dist.get_process_group_ranks(parameter.group) == [0]
    parameter.sum().backward()
    torch.testing.assert_close(parameter, torch.tensor([3.0]))
    torch.testing.assert_close(parameter.grad, torch.ones(1))
