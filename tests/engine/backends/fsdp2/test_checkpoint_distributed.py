"""Multi-process tests for EP-unique DCP keys and checkpoint failure handling."""

import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


_HAS_GLOO = dist.is_available() and dist.is_gloo_available()


class _Experts(nn.Module):
    """Expert weights plus a LoRA adapter, both local to one EP rank."""

    def __init__(self) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(4, 3))
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})


class _MoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(3, 4, bias=False)
        self.experts = _Experts()


def _build_model(world_mesh, edp_mesh, ep_rank: int, scale: float) -> _MoEModel:
    """Shard dense weights over all ranks and expert weights over edp.

    Expert values encode the EP rank, so state restored on the wrong EP rank
    is visible.
    """
    model = _MoEModel()
    dense = torch.arange(12, dtype=torch.float32).reshape(4, 3) * scale
    model.dense.weight = nn.Parameter(
        distribute_tensor(dense, world_mesh, [Shard(0)])
    )
    expert_params = (
        (model.experts, "weight1", (4, 3), 10.0),
        (model.experts.lora_A["default"], "weight", (2, 3), 20.0),
    )
    for module, name, shape, base in expert_params:
        value = torch.full(shape, (base + ep_rank) * scale)
        setattr(
            module,
            name,
            nn.Parameter(distribute_tensor(value, edp_mesh, [Shard(0)])),
        )
    return model


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _run_ep_checkpoint_test(
    rank: int, world_size: int, init_file: str, checkpoint_dir: str
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint

        world_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
        ep_rank = mesh["ep"].get_local_rank()
        checkpoint = DcpCheckpoint(
            ep_rank=ep_rank, ep_size=mesh["ep"].size(), edp_mesh=mesh["edp"]
        )

        trained = _build_model(world_mesh, mesh["edp"], ep_rank, scale=1.0)
        assert checkpoint._ep_partitioned_fqns(trained) == {
            "experts.weight1",
            "experts.lora_A.default.weight",
        }
        optimizer = torch.optim.AdamW(trained.parameters(), lr=0.1)
        for param in trained.parameters():
            param.grad = torch.ones_like(param) * (ep_rank + 1)
        optimizer.step()

        model_dir = os.path.join(checkpoint_dir, "model")
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        checkpoint.save_model(trained, model_dir)
        checkpoint.save_optimizer(optimizer, trained, optimizer_dir)

        resumed = _build_model(world_mesh, mesh["edp"], ep_rank, scale=0.0)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.1)
        checkpoint.load_model(resumed, model_dir)
        checkpoint.load_optimizer(resumed_optimizer, resumed, optimizer_dir)

        for name, expected in trained.state_dict().items():
            actual = resumed.state_dict()[name]
            torch.testing.assert_close(
                _local(actual), _local(expected), rtol=0, atol=0, msg=name
            )
        expected_state = get_optimizer_state_dict(trained, optimizer)["state"]
        actual_state = get_optimizer_state_dict(resumed, resumed_optimizer)["state"]
        assert actual_state.keys() == expected_state.keys()
        for name, expected_entry in expected_state.items():
            for key, expected in expected_entry.items():
                torch.testing.assert_close(
                    _local(actual_state[name][key]),
                    _local(expected),
                    rtol=0,
                    atol=0,
                    msg=f"{name}.{key}",
                )

        # A later stage can omit expert adapters while restoring the remaining
        # EP state from the same checkpoint.
        reduced = _build_model(world_mesh, mesh["edp"], ep_rank, scale=0.0)
        del reduced.experts.lora_A
        checkpoint.load_model(reduced, model_dir)
        for name, actual in reduced.state_dict().items():
            torch.testing.assert_close(
                _local(actual), _local(trained.state_dict()[name]), rtol=0, atol=0
            )

        from torchturbo.engine.checkpoint.distributed import collective_check, rank_zero_call

        assert rank_zero_call(lambda: rank) == 0

        def missing_on_rank_zero():
            raise FileNotFoundError("rank-zero preflight failure")

        with pytest.raises(FileNotFoundError, match="rank-zero preflight"):
            rank_zero_call(missing_on_rank_zero)

        def missing_on_one_rank():
            if rank == world_size - 1:
                raise FileNotFoundError("rank-local preflight failure")

        with pytest.raises(FileNotFoundError, match="rank-local preflight"):
            collective_check(missing_on_one_rank)

        # A rank with a different filesystem view must fail together with peers
        # before any of them starts the DCP load collective.
        local_model_dir = model_dir if rank != world_size - 1 else model_dir + "-missing"
        with pytest.raises(FileNotFoundError):
            checkpoint.load_model(resumed, local_model_dir)

        from torchturbo.engine.backends.fsdp2.checkpoint.manager import FSDP2CheckpointManager
        from torchturbo.engine.backends.fsdp2.optimizers.memory import OptimizerMemoryController
        from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator, CheckpointExportPluginConsumer
        from torchturbo.engine.config import EngineConfig
        from torchturbo.runtime.cpu import CpuRuntime

        controller = OptimizerMemoryController(optimizer, CpuRuntime())
        parallel_context = {"ep": {"ep_rank": ep_rank, "ep_size": mesh["ep"].size()}}
        manager = FSDP2CheckpointManager(
            rank=rank,
            mesh=mesh,
            world_mesh=world_mesh,
            parallel_context=parallel_context,
            device_runtime=CpuRuntime(),
            cpu_offload=False,
            require_optimizer_not_suspended=controller.require_not_suspended,
        )
        coordinator = CheckpointCoordinator(
            EngineConfig(), manager, parallel_context=parallel_context,
            export_plugin_consumer=CheckpointExportPluginConsumer(),
        )
        coordinator.save_optimizer_and_scheduler(
            trained, optimizer, None, checkpoint_dir
        )
        metadata = Path(manager.optimizer_dir(checkpoint_dir)) / ".metadata"
        saved_metadata = metadata.read_bytes()
        optimizer.zero_grad(set_to_none=True)
        # One suspended rank must stop every peer before any checkpoint I/O.
        if rank == world_size - 1:
            controller.offload()
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            coordinator.save_optimizer_and_scheduler(
                trained, optimizer, None, checkpoint_dir
            )
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            coordinator.load_optimizer_and_scheduler(
                trained, optimizer, None, checkpoint_dir
            )
        assert metadata.read_bytes() == saved_metadata
        if rank == world_size - 1:
            controller.restore()
        coordinator.load_optimizer_and_scheduler(
            trained, optimizer, None, checkpoint_dir
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO,
    reason="requires torch.distributed with Gloo support",
)
def test_expert_and_expert_lora_state_round_trip_per_ep_rank(tmp_path):
    mp.spawn(
        _run_ep_checkpoint_test,
        args=(4, str(tmp_path / "init"), str(tmp_path / "checkpoint")),
        nprocs=4,
        join=True,
    )


def _run_shard_read_failure_test(
    rank: int, world_size: int, init_file: str, weights_dir: str
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        from torchturbo.engine.backends.fsdp2.checkpoint import weight_loader
        from torchturbo.runtime.cpu import CpuRuntime

        def load_file(path, device):
            if rank == world_size - 1:
                raise OSError("rank-local shard read failure")
            return {"weight": torch.ones(2)}

        # Spawned workers import their own copy of the module.
        weight_loader.load_file = load_file
        loader = weight_loader.DistParamLoader.__new__(weight_loader.DistParamLoader)
        loader.model = nn.Module()
        loader.rank = rank
        loader.path = Path(weights_dir)
        loader.local_params = {"weight"}
        loader.param_to_filename = {"weight": "model.safetensors"}
        loader.param_converter = SimpleNamespace(is_virtual_key=lambda name: False)
        loader.device_runtime = CpuRuntime()
        loader.start_time = loader.prev_time = time.perf_counter()

        # Ranks that read their shards must not wait in a collective that the
        # failed rank never joins.
        with pytest.raises(OSError, match="rank-local shard read failure"):
            loader.load_weights()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO,
    reason="requires torch.distributed with Gloo support",
)
def test_rank_local_shard_read_failure_reaches_every_rank(tmp_path):
    mp.spawn(
        _run_shard_read_failure_test,
        args=(2, str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
