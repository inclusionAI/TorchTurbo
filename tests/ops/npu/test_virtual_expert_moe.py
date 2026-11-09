"""Reuse MoE/optimizer/checkpoint regressions with a selected NPU runtime."""

import importlib

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from tests.models.test_moon_ep_distributed import _check_moon, _check_shared_moon_layers
from tests.ops.npu.test_virtual_expert_weight_exchange import weight_env
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import is_npu_available
import torchturbo.runtime.runtime as runtime_module
from torchturbo.models.modules.moe.virtual_expert_load_balancer import (
    _VirtualExpertConfig, _VirtualExpertStorage,
)

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture
def moe_env(weight_env, monkeypatch):
    local_dispatcher, group = weight_env
    runtime = local_dispatcher.device_runtime
    monkeypatch.setattr(runtime_module, "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(
        device_runtime=runtime, default_policy=OpPolicy.NATIVE,
        overrides={
            "plan_virtual_expert_routes": "npu_triton_plan_virtual_expert_routes",
            "virtual_expert_weight_exchange": "npu_shmem_virtual_expert_weight_exchange",
            "virtual_expert_grad_reduce": "npu_shmem_virtual_expert_grad_reduce",
        },
    )
    def reject_weight_all_to_all(*args, **kwargs):
        raise AssertionError("Virtual-expert weights must use direct SHMEM communication")

    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.providers.native.virtual_expert_weight_exchange"),
        "ep_all_to_all", reject_weight_all_to_all,
    )
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.providers.native.virtual_expert_grad_reduce"),
        "ep_all_to_all", reject_weight_all_to_all,
    )
    yield runtime, group


@pytest.fixture(scope="module")
def fsdp_mesh(weight_env):
    dispatcher, world = weight_env
    if dist.get_world_size(world) != 4:
        pytest.skip("Run with torchrun EDP2 x EP2")
    mesh = init_device_mesh(dispatcher.device_runtime.device.type, (2, 2), mesh_dim_names=("edp", "ep"))
    yield mesh
    dist.destroy_process_group(mesh["ep"].get_group())
    dist.destroy_process_group(mesh["edp"].get_group())


def _check_shmem_peer_access(runtime, group, dtype):
    config = _VirtualExpertConfig(
        group, torch.device("npu", runtime.current_device()), 2,
        (torch.Size((8, 12)), torch.Size((12, 8))), dtype, torch.float32,
    )
    storage = _VirtualExpertStorage(config)
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    for arena, handle in (
        (storage.weight_arena, storage.weight_handle),
        (storage.grad_arena, storage.grad_handle),
    ):
        arena_dtype = arena.dtype
        assert handle is not None
        assert handle.world_size == size
        arena.fill_(dist.get_rank() + 1)
        runtime.synchronize()
        dist.barrier()
        peer = (rank + 1) % size
        view = handle.get_buffer(peer, arena.shape, arena_dtype)
        wanted = dist.get_global_rank(group, peer) + 1
        torch.testing.assert_close(view.cpu(), torch.full(arena.shape, wanted, dtype=arena_dtype))
        runtime.synchronize()
        dist.barrier()
        view.fill_(dist.get_rank() + 11)
        runtime.synchronize()
        dist.barrier()
        sender = dist.get_global_rank(group, (rank - 1) % size)
        torch.testing.assert_close(arena.cpu(), torch.full(arena.shape, sender + 11, dtype=arena_dtype))
        runtime.synchronize()
        dist.barrier()
        # The same mapped allocation must be usable directly by native HCCL.
        original_ptr = arena.data_ptr()
        dist.all_reduce(arena, group=group)
        runtime.synchronize()
        dist.barrier()
        total = sum(dist.get_global_rank(group, peer) + 11 for peer in range(size))
        torch.testing.assert_close(arena.cpu(), torch.full(arena.shape, total, dtype=arena_dtype))
        torch.testing.assert_close(view.cpu(), torch.full(arena.shape, total, dtype=arena_dtype))
        assert arena.data_ptr() == original_ptr
        runtime.synchronize()
        dist.barrier()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_storage_exposes_weight_and_gradient_peers(moe_env, dtype):
    runtime, group = moe_env
    _check_shmem_peer_access(runtime, group, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_fsdp_storage_maps_ep_peers_after_world_initialization(moe_env, fsdp_mesh, dtype):
    runtime, _ = moe_env
    _check_shmem_peer_access(runtime, fsdp_mesh["ep"].get_group(), dtype)


@pytest.mark.parametrize("mismatch", ["size", "dtype"])
def test_npu_storage_rejects_inconsistent_world_layout(moe_env, mismatch):
    runtime, group = moe_env
    if dist.get_world_size() < 2:
        pytest.skip("Requires multiple WORLD ranks")
    rank = dist.get_rank()
    config = _VirtualExpertConfig(
        group, torch.device("npu", runtime.current_device()),
        2 + rank if mismatch == "size" else 2,
        (torch.Size((8, 12)), torch.Size((12, 8))),
        torch.bfloat16 if mismatch == "dtype" and rank else torch.float32,
        torch.float32,
    )
    with pytest.raises(RuntimeError, match="identical buffer sizes and dtypes"):
        _VirtualExpertStorage(config)


@pytest.mark.parametrize("boundary,reentrant", [
    ("none", False), ("whole", False), ("whole", True),
    ("experts", False), ("experts", True),
])
@pytest.mark.parametrize("options", [
    {}, {"empty": "one"}, {"empty": "all"}, {"freeze_experts": True},
    {"freeze_inputs": True}, {"route_mode": "toggle"}, {"route_mode": "toggle_reverse"},
    {"backward_together": True},
    {"inference_warmup": True},
])
def test_npu_moe_outputs_gradients_optimizer_and_checkpoints(moe_env, boundary, reentrant, options):
    runtime, group = moe_env
    if dist.get_world_size(group) != 2:
        pytest.skip("Run with torchrun EP2")
    _check_moon(group, runtime=runtime, boundary=boundary, reentrant=reentrant, **options)


@pytest.mark.parametrize("boundary,reentrant", [
    ("none", False), ("whole", False), ("whole", True),
    ("experts", False), ("experts", True),
])
@pytest.mark.parametrize("options", [
    {}, {"empty": "one"}, {"empty": "all"}, {"freeze_inputs": True},
    {"freeze_experts": True}, {"route_mode": "edp_mixed"},
    {"route_mode": "toggle", "mixed_precision": True, "cast_inputs": False},
    {"route_mode": "toggle_reverse", "mixed_precision": True, "cast_inputs": True},
    {"backward_together": True},
])
def test_npu_fsdp_edp2_ep2_checkpoints_and_bf16(moe_env, fsdp_mesh, boundary, reentrant, options):
    runtime, world = moe_env
    if dist.get_world_size(world) != 4:
        pytest.skip("Run with torchrun EDP2 x EP2")
    _check_moon(fsdp_mesh["ep"].get_group(), runtime=runtime, fsdp_mesh=fsdp_mesh,
                boundary=boundary, reentrant=reentrant, **options)


@pytest.mark.parametrize("boundary,reentrant", [
    ("none", False), ("whole", False), ("whole", True),
    ("experts", False), ("experts", True),
])
def test_npu_layers_share_storage_and_repush_before_backward(moe_env, boundary, reentrant):
    runtime, group = moe_env
    _check_shared_moon_layers(group, runtime=runtime, boundary=boundary, reentrant=reentrant)


@pytest.mark.parametrize("boundary,reentrant", [
    ("none", False), ("whole", False), ("whole", True),
    ("experts", False), ("experts", True),
])
def test_npu_fsdp_layers_share_storage_and_repush_before_backward(
    moe_env, fsdp_mesh, boundary, reentrant,
):
    runtime, _ = moe_env
    _check_shared_moon_layers(
        fsdp_mesh["ep"].get_group(), runtime=runtime, fsdp_mesh=fsdp_mesh,
        boundary=boundary, reentrant=reentrant,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-s"]))
