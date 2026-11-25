"""MoonEP provider selection and NPU allocation preflight without NPU hardware."""

import ctypes
import gc
import importlib
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer, WeightDirection
from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.providers.npu.plan_virtual_expert_routes import NpuPlanVirtualExpertRoutes
from torchturbo.ops.providers.npu.virtual_expert_grad_reduce import NpuShmemVirtualExpertGradReduce
from torchturbo.ops.providers.npu.virtual_expert_weight_exchange import NpuShmemVirtualExpertWeightExchange
from torchturbo.ops.registry import registry
from torchturbo.runtime.npu import NpuRuntime


PROVIDERS = (
    NpuPlanVirtualExpertRoutes, NpuShmemVirtualExpertWeightExchange, NpuShmemVirtualExpertGradReduce,
)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_missing_post7_shmem_symbols_disable_provider(monkeypatch, provider):
    monkeypatch.setattr("torchturbo.runtime.runtime._SELECTED_RUNTIME", NpuRuntime())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post7"))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "triton.language.extra.cann.extension", SimpleNamespace(sort=sorted))
    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(shmem_team_split_strided=object()))
    assert not provider().available()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_multinode_torchrun_keeps_shmem_provider_available(monkeypatch, provider):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("torchturbo.runtime.runtime._SELECTED_RUNTIME", NpuRuntime())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post7"))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "triton.language.extra.cann.extension", SimpleNamespace(sort=sorted))
    monkeypatch.setattr(ctypes, "CDLL", Mock())
    assert provider().available()


def test_missing_triton_falls_back_with_available_shmem(monkeypatch):
    runtime = NpuRuntime()
    monkeypatch.setattr("torchturbo.runtime.runtime._SELECTED_RUNTIME", runtime)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post7"))
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(ctypes, "CDLL", Mock())

    assert runtime.shmem.available()
    dispatcher = OpDispatcher(device_runtime=runtime, registry=registry)
    for provider in PROVIDERS:
        assert not dispatcher.requires_peer_memory(provider.op_spec().op_name)


def test_configured_peer_memory_query_does_not_require_a_process_group(monkeypatch):
    for provider in PROVIDERS:
        monkeypatch.setattr(provider, "available", lambda self: True)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)

    def reject_topology_check():
        pytest.fail("Selecting a provider must not inspect a process group")

    monkeypatch.setattr(dist, "is_initialized", reject_topology_check)
    for provider in PROVIDERS:
        assert dispatcher.requires_peer_memory(provider.op_spec().op_name)


@pytest.mark.parametrize("policy,available,overrides,peer_access", [
    (OpPolicy.NATIVE, True, (), (False, False, False)),
    (OpPolicy.AUTO, False, (), (False, False, False)),
    (OpPolicy.NATIVE, True, (0, 2), (True, False, True)),
    (OpPolicy.NATIVE, True, (1,), (False, True, False)),
    (OpPolicy.AUTO, True, (), (True, True, True)),
])
def test_model_allocates_only_selected_peer_buffers(monkeypatch, policy, available, overrides, peer_access):
    runtime = NpuRuntime()
    monkeypatch.setattr(VirtualExpertLoadBalancer, "_peer_resources", [])
    monkeypatch.setattr(VirtualExpertLoadBalancer, "runtimes", weakref.WeakValueDictionary())
    monkeypatch.setattr(VirtualExpertLoadBalancer, "storages", weakref.WeakValueDictionary())
    for provider in PROVIDERS:
        monkeypatch.setattr(provider, "available", lambda self: available)
    dispatcher = OpDispatcher(
        device_runtime=runtime, registry=registry, default_policy=policy,
        overrides={PROVIDERS[i].op_spec().op_name: PROVIDERS[i].op_spec().impl_name for i in overrides},
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", dispatcher)
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(NpuRuntime, "supports_async_streams", False)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    group = object()
    calls = []
    prepared = []

    def allocate(numel, *, dtype, device, group, peer_access=True):
        calls.append((numel * dtype.itemsize, peer_access))
        return torch.empty(numel, dtype=dtype, device=device), object() if peer_access else None

    def prepare(sizes):
        assert not calls, "Preflight must size the heap before allocating the first arena"
        prepared.extend(sizes)

    monkeypatch.setattr(runtime, "allocate_communication_buffer", allocate)
    monkeypatch.setattr(runtime.shmem, "prepare_heap", prepare)
    config = MoEConfig(
        hidden_size=2, moe_ffn_hidden_size=3, num_moe_experts=4,
        expert_model_parallel_size=2, moe_router_topk=1,
        moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True,
    )
    manager = VirtualExpertLoadBalancer()
    manager.initialize_virtual_expert_load_balancer(
        group=group, num_local_experts=2, num_experts=4, config=config, router_topk=1,
    )
    manager.bind_experts(GroupedMLP(2, False, config))
    manager._runtime_init(torch.empty(1, 2))
    assert [mapped for _, mapped in calls] == list(peer_access)
    assert prepared == [size if mapped else None for size, mapped in calls]

    planner = weakref.ref(manager._shared_runtime)
    storage = weakref.ref(manager.storage)
    del manager
    gc.collect()
    assert (planner() is not None) == peer_access[0]
    assert (storage() is not None) == any(peer_access[1:])

    release_unused = Mock()
    monkeypatch.setattr(runtime.shmem, "release_unused", release_unused)
    VirtualExpertLoadBalancer.finalize()
    release_unused.assert_called_once_with()


def test_expert_dtype_change_rebinds_weight_and_gradient_arenas(monkeypatch):
    runtime = NpuRuntime()
    group = object()
    dtypes = []

    def empty(numel, *, dtype, device, group):
        dtypes.append(dtype)
        return torch.empty(numel, dtype=dtype, device=device)

    monkeypatch.setattr("torchturbo.runtime.runtime._SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(NpuRuntime, "supports_async_streams", False)
    monkeypatch.setattr(runtime.shmem, "available", lambda: True)
    monkeypatch.setattr(runtime.shmem, "prepare_heap", lambda sizes: None)
    monkeypatch.setattr(runtime.shmem, "empty", empty)
    monkeypatch.setattr(runtime.shmem, "rendezvous", lambda arena, group: object())
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", SimpleNamespace(
        requires_peer_memory=lambda name: True,
    ))
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(VirtualExpertLoadBalancer, "_peer_resources", [])
    config = MoEConfig(
        hidden_size=2, moe_ffn_hidden_size=3, num_moe_experts=4,
        expert_model_parallel_size=2, moe_router_topk=1, params_dtype=torch.float32,
        moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True,
    )
    manager = VirtualExpertLoadBalancer()
    manager.initialize_virtual_expert_load_balancer(
        group=group, num_local_experts=2, num_experts=4, config=config, router_topk=1,
    )
    experts = GroupedMLP(2, False, config)
    manager.bind_experts(experts)
    manager.wrap_layer_input(torch.empty(1, 2))
    experts.to(torch.bfloat16)
    manager._plan = SimpleNamespace(routing=SimpleNamespace(has_global_replicas=False))
    manager._start_weight_push(WeightDirection.FORWARD)
    assert dtypes == [torch.int64, torch.float32, torch.float32, torch.bfloat16, torch.float32]


def test_explicit_native_buffers_need_no_ep_collectives(monkeypatch):
    for provider in PROVIDERS:
        monkeypatch.setattr(provider, "available", lambda self: True)
    runtime = NpuRuntime()
    overrides = {
        "plan_virtual_expert_routes": "torch_plan_virtual_expert_routes",
        "virtual_expert_weight_exchange": "torch_virtual_expert_weight_exchange",
        "virtual_expert_grad_reduce": "torch_virtual_expert_grad_reduce",
    }
    dispatcher = OpDispatcher(device_runtime=runtime, registry=registry, overrides=overrides)

    def reject(*args, **kwargs):
        pytest.fail("A native buffer invoked an EP collective")

    monkeypatch.setattr(dist, "all_gather_object", reject)
    for op_name in overrides:
        arena, handle = runtime.allocate_communication_buffer(
            7, dtype=torch.int64, device=torch.device("cpu"), group=None,
            peer_access=dispatcher.requires_peer_memory(op_name),
        )
        assert handle is None and arena.shape == (7,)
