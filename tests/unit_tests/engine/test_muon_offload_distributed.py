"""Real FSDP CPU offload and optimizer migration on CUDA/NPU workers."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig, FSDPConfig
from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon
from torchturbo.engine.config import MuonOptimizerConfig
from torchturbo.engine.optimizers import MuonPlan, MuonRecipe, OptimizersContainer
from torchturbo.runtime import get_device_runtime
from torchturbo.utils.torch_utils import torch_version


def _device_count():
    runtime = get_device_runtime()
    if runtime.device.type == "cuda":
        return torch.cuda.device_count()
    if runtime.device.type == "npu":
        return torch.npu.device_count()
    return 0


def _run(rank, init_file, mode, fraction, overlap, dual_backend):
    runtime = get_device_runtime()
    runtime.set_device(rank)
    backend_name = runtime.dist_backend
    if dual_backend:
        backend_name = f"{runtime.device.type}:{backend_name},cpu:gloo"
    dist.init_process_group(
        backend_name,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(12)
        model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 3)).to(runtime.device)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh(runtime.device.type, (2,), mesh_dim_names=("dp",))
        kwargs = (
            {"offload_policy": CPUOffloadPolicy(pin_memory=True)}
            if mode == "full"
            else {}
        )
        fully_shard(model, mesh=mesh, **kwargs)
        backend = object.__new__(FSDP2Backend)
        backend._device_runtime = runtime
        backend._mesh = {"edp": object()}
        backend.fsdp_config = FSDPConfig(
            cpu_offload=CPUOffloadConfig(
                enabled=mode is not None,
                mode=mode or "full",
                optimizer_memory_fraction=fraction,
            )
        )
        plan = MuonPlan(
            model,
            recipe=MuonRecipe({}),
            config=MuonOptimizerConfig(
                lr=0.01,
                weight_decay=0.01,
                ns_steps=2,
                overlap_comm_compute=overlap,
                adamw_fused=False,
            ),
        )
        optimizer = backend.prepare_optimizer(plan)
        reference_muon = torch.optim.Muon(
            [param for param in reference.parameters() if param.ndim == 2],
            **plan.muon_kwargs,
        )
        reference_adamw = torch.optim.AdamW(
            [param for param in reference.parameters() if param.ndim == 1],
            **{**plan.adamw_kwargs, "weight_decay": 0.0},
        )
        parameters = list(model.parameters())
        identities = [
            (id(param), param.to_local().untyped_storage().data_ptr())
            for param in parameters
        ]
        generator = torch.Generator().manual_seed(38)
        children = (
            tuple(optimizer)
            if isinstance(optimizer, OptimizersContainer)
            else (optimizer,)
        )
        for _ in range(3):
            inputs = torch.randn(4, 7, generator=generator).to(runtime.device)
            model(inputs).square().mean().backward()
            reference(inputs).square().mean().backward()
            optimizer.step()
            reference_muon.step()
            reference_adamw.step()
            runtime.synchronize()
            for param, reference_param in zip(
                model.parameters(), reference.parameters()
            ):
                full = param.to(runtime.device).full_tensor()
                torch.testing.assert_close(full, reference_param, rtol=2e-3, atol=3e-4)
            assert identities == [
                (id(param), param.to_local().untyped_storage().data_ptr())
                for param in model.parameters()
            ]
            optimizer.zero_grad(set_to_none=True)
            reference_muon.zero_grad(set_to_none=True)
            reference_adamw.zero_grad(set_to_none=True)
            backend.offload_optimizer()
            for child in children:
                assert all(
                    value.device.type == "cpu"
                    for state in child.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
                if isinstance(child, FSDPDistributedMuon):
                    assert not child._workspace.slots
                    assert not child._workspace.local_buffers.compute_scratch
            with pytest.raises(RuntimeError, match="optimizer is suspended"):
                optimizer.step()
            inference_allocation = torch.empty(1024 * 1024, device=runtime.device)
            del inference_allocation
            backend.restore_optimizer()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    _device_count() < 2 or torch_version() < (2, 10, 0),
    reason="requires two CUDA/NPU devices and PyTorch 2.10",
)
@pytest.mark.parametrize(
    "mode,fraction,overlap,dual_backend",
    [
        (None, 0.3, True, True),
        ("full", 0.3, False, True),
        ("full", 0.3, True, True),
        ("optimizer", 0.3, False, True),
        ("optimizer", 0.3, True, True),
        ("full", 0, False, True),
        ("optimizer", 0, False, True),
        ("optimizer", 0, False, False),
    ],
)
def test_fsdp_muon_offload_training_and_inference(
    tmp_path, mode, fraction, overlap, dual_backend
):
    mp.spawn(
        _run,
        args=(str(tmp_path / "init"), mode, fraction, overlap, dual_backend),
        nprocs=2,
        join=True,
    )
