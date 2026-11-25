"""Reuse MoE/optimizer/checkpoint regressions with a selected NPU runtime."""

import importlib

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from tests.models.test_moon_ep_distributed import _check_moon, _check_shared_moon_layers
from tests.models.test_virtual_expert_weights import _check_finalize_rebuild
from tests.ops.npu.test_virtual_expert_weight_exchange import weight_env
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import is_npu_available
from torchturbo.models.modules.moe.virtual_expert_load_balancer import (
    _VirtualExpertConfig, _VirtualExpertStorage, VirtualExpertLoadBalancer,
)
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.grouped_linear import grouped_linear
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture
def moe_env(weight_env, monkeypatch):
    local_dispatcher, group = weight_env
    runtime = local_dispatcher.device_runtime
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(
        device_runtime=runtime, default_policy=OpPolicy.NATIVE,
        overrides={
            "grouped_matmul_add": "torch_npu_grouped_matmul_add",
            "plan_virtual_expert_routes": "npu_triton_plan_virtual_expert_routes",
            "hybrid_ep_dispatch": "npu_hybrid_ep_dispatch",
            "hybrid_ep_combine": "npu_hybrid_ep_combine",
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
    VirtualExpertLoadBalancer.finalize()


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
        handle.barrier()
        runtime.synchronize()
        dist.barrier(group=group)
        peer = (rank + 1) % size
        view = handle.get_buffer(peer, arena.shape, arena_dtype)
        wanted = dist.get_global_rank(group, peer) + 1
        torch.testing.assert_close(view.cpu(), torch.full(arena.shape, wanted, dtype=arena_dtype))
        runtime.synchronize()
        dist.barrier(group=group)
        view.fill_(dist.get_rank() + 11)
        runtime.synchronize()
        dist.barrier(group=group)
        sender = dist.get_global_rank(group, (rank - 1) % size)
        torch.testing.assert_close(arena.cpu(), torch.full(arena.shape, sender + 11, dtype=arena_dtype))
        runtime.synchronize()
        dist.barrier(group=group)
        # The same mapped allocation must be usable directly by native HCCL.
        original_ptr = arena.data_ptr()
        dist.all_reduce(arena, group=group)
        runtime.synchronize()
        dist.barrier(group=group)
        total = sum(dist.get_global_rank(group, peer) + 11 for peer in range(size))
        torch.testing.assert_close(arena.cpu(), torch.full(arena.shape, total, dtype=arena_dtype))
        torch.testing.assert_close(view.cpu(), torch.full(arena.shape, total, dtype=arena_dtype))
        assert arena.data_ptr() == original_ptr
        runtime.synchronize()
        dist.barrier(group=group)


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
    with pytest.raises(RuntimeError, match="buffer sizes, dtypes and heap capacities must match"):
        _VirtualExpertStorage(config)


@pytest.mark.parametrize("inference_init", [False, True])
def test_npu_runtime_initializes_shared_communication_before_routing(
    moe_env, monkeypatch, inference_init,
):
    runtime, group = moe_env
    config = MoEConfig(
        hidden_size=12, moe_ffn_hidden_size=10, num_moe_experts=2 * dist.get_world_size(group),
        expert_model_parallel_size=dist.get_world_size(group), moe_router_topk=2,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32", params_dtype=torch.bfloat16,
        moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True,
    )
    layers = [
        MoELayer(config, group, submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
        .to(runtime.device)
        for _ in range(2)
    ]
    with torch.no_grad():
        for layer in layers:
            for parameter in layer.parameters():
                parameter.normal_(std=0.1)
    inputs = torch.randn((1, 5, 12), device=runtime.device, dtype=torch.bfloat16, requires_grad=True)
    managers = [layer.token_dispatcher._comm_manager for layer in layers]
    with torch.inference_mode(inference_init):
        for manager in managers:
            manager.wrap_layer_input(inputs)
            assert manager.planner.histogram_handle is not None
            assert manager.storage.weight_handle is not None
            assert manager.storage.grad_handle is not None
    assert managers[0].planner is managers[1].planner
    assert managers[0].storage is managers[1].storage

    def reject_late_allocation(*args, **kwargs):
        pytest.fail("Routing and backward must reuse the communication buffers initialized at input")

    monkeypatch.setattr(type(runtime), "allocate_communication_buffer", reject_late_allocation)
    outputs = [layer(inputs)[0] for layer in layers]
    sum(output.sum() for output in outputs).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for layer in layers:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_npu_finalize_rebuilds_layers_with_default_world_group(moe_env):
    runtime, group = moe_env
    config = MoEConfig(
        hidden_size=12, moe_ffn_hidden_size=10, num_moe_experts=2 * dist.get_world_size(group),
        expert_model_parallel_size=dist.get_world_size(group), moe_router_topk=2,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        params_dtype=torch.bfloat16, moe_token_dispatcher_type="flex",
        moe_virtual_expert_load_balance=True,
    )
    _check_finalize_rebuild(config, None, runtime.device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("impl", ["torch_grouped_matmul", "torch_npu_grouped_matmul"])
@pytest.mark.parametrize("use_store", [False, True])
def test_npu_empty_linear_partition_preserves_autocast_dtype(moe_env, monkeypatch, dtype, impl, use_store):
    runtime, _ = moe_env
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(
        device_runtime=runtime, default_policy=OpPolicy.NATIVE,
        overrides={"grouped_matmul": impl},
    )
    inputs = torch.ones((2, 4), device=runtime.device, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weights = torch.ones((1, 4, 3), device=runtime.device, requires_grad=use_store)
    wgrads = []
    store = wgrads.append if use_store else None
    with torch.autocast("npu", dtype=dtype):
        active = grouped_linear(inputs, torch.tensor([2]), weight=weights, wgrad_store=store)
        empty = grouped_linear(inputs[:0], torch.tensor([0]), weight=weights, wgrad_store=store)
        assert active.dtype == empty.dtype == dtype
        output = torch.cat((active, empty))
        expected = ref_inputs @ weights[0]
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)
    if use_store:
        assert len(wgrads) == 2
        torch.testing.assert_close(wgrads[0] + wgrads[1], weights.grad)


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
