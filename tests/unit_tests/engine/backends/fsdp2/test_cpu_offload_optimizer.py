"""Tests for AdamW optimizer-state CPU offload with a per-step device workspace."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch
import torch.distributed as dist

from torchturbo.engine.backends.fsdp2.config import (
    CPUOffloadConfig,
    FSDPConfig,
)
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.optimizers.adamw import (
    AdamWSwap,
    DeviceWorkspace,
)
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


class SyncRuntime(CpuRuntime):
    """Plain CPU runtime for exercising host-side control flow."""


# torch mis-dispatches CPU work to the MPS backend on MPS builds without
# CUDA (Apple silicon laptops): any op on a pin_memory=True CPU tensor and
# CPU fused AdamW both hit "missing kernel for mps" or crash the process.
# CUDA/NPU hosts are unaffected.
CPU_FUSED_ADAMW_BROKEN = (
    torch.backends.mps.is_built() and not torch.cuda.is_available()
)
requires_cpu_fused_adamw = pytest.mark.skipif(
    CPU_FUSED_ADAMW_BROKEN,
    reason="pin_memory/fused CPU ops mis-dispatch to the MPS backend on this torch build",
)


def _offload_adamw(param_groups, optimizer_memory_fraction=0.8, **kwargs):
    return AdamWSwap(
        param_groups,
        device_runtime=SyncRuntime(),
        optimizer_memory_fraction=optimizer_memory_fraction,
        **kwargs,
    )


def _clone_groups(params):
    return [
        {"params": [params[0], params[1]], "lr": 2e-3, "weight_decay": 0.1},
        {"params": [params[2]], "lr": 5e-4, "weight_decay": 0.0},
    ]


@requires_cpu_fused_adamw
def test_cpu_offload_adamw_matches_adamw(monkeypatch):
    def _workspace_init(self, _runtime, _fraction):
        self._storage = torch.empty(1024, dtype=torch.uint8)

    monkeypatch.setattr(DeviceWorkspace, "__init__", _workspace_init)
    torch.manual_seed(7)
    reference_params = [
        torch.nn.Parameter(torch.randn(3, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    offloaded_params = [
        torch.nn.Parameter(param.detach().clone()) for param in reference_params
    ]
    reference = torch.optim.AdamW(_clone_groups(reference_params), fused=True)
    offload = _offload_adamw(_clone_groups(offloaded_params))
    offload.load_state_dict(offload.state_dict())
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference, step_size=1, gamma=0.8)
    offload_scheduler = torch.optim.lr_scheduler.StepLR(offload, step_size=1, gamma=0.8)

    for step in range(4):
        for index, (reference_param, offload_param) in enumerate(
            zip(reference_params, offloaded_params)
        ):
            if step == 0 and index == 1:
                continue
            grad = torch.randn_like(reference_param)
            reference_param.grad = grad
            offload_param.grad = grad.clone()

        reference.step()
        offload.step()
        assert offload.workspace is None
        reference_scheduler.step()
        offload_scheduler.step()

        for reference_param, offload_param in zip(reference_params, offloaded_params):
            torch.testing.assert_close(offload_param, reference_param, rtol=0, atol=0)
        reference.zero_grad()
        offload.zero_grad()

    for reference_param, offload_param in zip(reference_params, offloaded_params):
        for key in offload.state_keys:
            cpu_state = offload.param_to_cpu_states_map[offload_param][key]
            torch.testing.assert_close(
                cpu_state, reference.state[reference_param][key], rtol=0, atol=0
            )
            assert cpu_state.device.type == "cpu"
        torch.testing.assert_close(
            offload.param_to_step_map[offload_param],
            reference.state[reference_param]["step"],
        )
    assert set(offload.state) == set(offloaded_params)
    for offload_param in offloaded_params:
        assert offload.state[offload_param]["step"] is offload.param_to_step_map[offload_param]
        assert (
            offload.state[offload_param]["exp_avg"]
            is offload.param_to_cpu_states_map[offload_param]["exp_avg"]
        )
    assert [group["lr"] for group in offload.param_groups] == [
        group["lr"] for group in reference.param_groups
    ]


@requires_cpu_fused_adamw
def test_host_update_adamw_matches_adamw():
    torch.manual_seed(11)
    reference_params = [
        torch.nn.Parameter(torch.randn(3, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    cpu_params = [
        torch.nn.Parameter(param.detach().clone()) for param in reference_params
    ]
    reference = torch.optim.AdamW(_clone_groups(reference_params), fused=True)
    cpu_update = _offload_adamw(
        _clone_groups(cpu_params), optimizer_memory_fraction=0
    )
    assert not hasattr(cpu_update, "param_to_cpu_param_map")

    for _ in range(4):
        for reference_param, cpu_param in zip(reference_params, cpu_params):
            grad = torch.randn_like(reference_param)
            reference_param.grad = grad
            cpu_param.grad = grad.clone()
        reference.step()
        cpu_update.step()
        reference.zero_grad()
        cpu_update.zero_grad()

    for reference_param, cpu_param in zip(reference_params, cpu_params):
        torch.testing.assert_close(cpu_param, reference_param, rtol=0, atol=0)
        for key in cpu_update.state_keys:
            torch.testing.assert_close(
                cpu_update.param_to_cpu_states_map[cpu_param][key],
                reference.state[reference_param][key],
                rtol=0,
                atol=0,
            )
    assert set(cpu_update.state) == set(cpu_params)


@requires_cpu_fused_adamw
def test_step_on_device_requires_workspace():
    params = [torch.nn.Parameter(torch.ones(4))]
    optimizer = _offload_adamw(params)
    params[0].grad = torch.ones_like(params[0])

    with pytest.raises(RuntimeError, match="workspace is unavailable"):
        optimizer.step()


def test_device_workspace_views_share_temporary_storage():
    workspace = DeviceWorkspace.__new__(DeviceWorkspace)
    workspace._storage = torch.empty(256, dtype=torch.uint8)
    tensor = torch.zeros(4)

    view = workspace.view_like(0, tensor)

    assert view.untyped_storage().data_ptr() == workspace._storage.data_ptr()


def test_device_workspace_skips_cpu_runtime():
    assert not DeviceWorkspace(SyncRuntime(), memory_fraction=0.5).is_available


def test_device_workspace_raises_when_capacity_insufficient():
    runtime = types.SimpleNamespace(
        turbo_device=TurboDevice.NPU,
        device=torch.device("meta"),
        memory_info=lambda: (10, 1000),
    )
    with pytest.raises(RuntimeError, match="too small"):
        DeviceWorkspace(runtime, memory_fraction=0.5)


def test_device_workspace_synchronizes_capacity_across_ranks(monkeypatch):
    runtime = types.SimpleNamespace(
        turbo_device=TurboDevice.NPU,
        device=torch.device("cpu"),
        memory_info=lambda: (100, 1000),
    )

    def all_reduce(tensor, op, group):
        assert op is dist.ReduceOp.MIN
        tensor.fill_(32)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    workspace = DeviceWorkspace(runtime, memory_fraction=0.8)

    assert workspace.capacity_bytes == 32


@requires_cpu_fused_adamw
def test_workspace_capacity_error_includes_param_name():
    params = [torch.nn.Parameter(torch.ones(8))]
    optimizer = _offload_adamw(
        params, param_names={params[0]: "model.layers.3.mlp.up_proj"}
    )
    workspace = DeviceWorkspace.__new__(DeviceWorkspace)
    workspace._storage = torch.empty(16, dtype=torch.uint8)
    optimizer.workspace = workspace

    with pytest.raises(RuntimeError, match="model.layers.3.mlp.up_proj"):
        optimizer._ensure_workspace_capacity([(params[0], {})])


@requires_cpu_fused_adamw
def test_workspace_capacity_failure_is_synchronized(monkeypatch):
    params = [torch.nn.Parameter(torch.ones(4))]
    optimizer = _offload_adamw(
        params, param_names={params[0]: "model.layers.3.mlp.up_proj"}
    )
    workspace = DeviceWorkspace.__new__(DeviceWorkspace)
    workspace._storage = torch.empty(128, dtype=torch.uint8)
    optimizer.workspace = workspace

    def all_reduce(tensor, op, group):
        assert op is dist.ReduceOp.MIN
        tensor.fill_(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    with pytest.raises(RuntimeError, match="failed on another rank"):
        optimizer._ensure_workspace_capacity([(params[0], {})])


@requires_cpu_fused_adamw
def test_workspace_stages_multiple_batches_when_each_state_pair_fits():
    params = [torch.nn.Parameter(torch.ones(4)) for _ in range(5)]
    optimizer = _offload_adamw(params)
    workspace = DeviceWorkspace.__new__(DeviceWorkspace)
    workspace._storage = torch.empty(128, dtype=torch.uint8)
    optimizer.workspace = workspace
    params_with_groups = [(param, {}) for param in params]

    optimizer._ensure_workspace_capacity(params_with_groups)
    entries, next_index = optimizer._stage_workspace_batch(params_with_groups, 0)
    assert len(entries) == 4
    assert next_index == 4

    entries, next_index = optimizer._stage_workspace_batch(
        params_with_groups, next_index
    )
    assert len(entries) == 1
    assert next_index == 5


@requires_cpu_fused_adamw
def test_workspace_adamw_matches_adamw(monkeypatch):
    def _workspace_init(self, _runtime, _fraction):
        self._storage = torch.empty(512, dtype=torch.uint8)

    monkeypatch.setattr(DeviceWorkspace, "__init__", _workspace_init)
    torch.manual_seed(13)
    reference_param = torch.nn.Parameter(torch.randn(4, 3))
    workspace_param = torch.nn.Parameter(reference_param.detach().clone())
    reference = torch.optim.AdamW([reference_param], fused=True)
    workspace_update = _offload_adamw([workspace_param])

    assert workspace_update.workspace is None

    for _ in range(3):
        grad = torch.randn_like(reference_param)
        reference_param.grad = grad
        workspace_param.grad = grad.clone()
        reference.step()
        assert workspace_update._acquire_workspace() is not None
        entries, next_index = workspace_update._stage_workspace_batch(
            [(workspace_param, workspace_update.param_groups[0])], 0
        )
        assert next_index == 1
        assert (
            entries[0][2]["exp_avg"].untyped_storage().data_ptr()
            == workspace_update.workspace._storage.data_ptr()
        )
        workspace_update.step()
        assert workspace_update.workspace is None
        reference.zero_grad()
        workspace_update.zero_grad()

    torch.testing.assert_close(workspace_param, reference_param, rtol=0, atol=0)
    for key in workspace_update.state_keys:
        torch.testing.assert_close(
            workspace_update.param_to_cpu_states_map[workspace_param][key],
            reference.state[reference_param][key],
            rtol=0,
            atol=0,
        )


def test_prepare_optimizer_rejects_unsupported_optimizer():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig(
        cpu_offload=CPUOffloadConfig(
            enabled=True,
            mode="optimizer",
        )
    )
    backend._device_runtime = CpuRuntime()
    backend._mesh = {"edp": object()}
    plan = OptimizerClassPlan(
        torch.nn.Linear(1, 1),
        torch.optim.SGD,
        {"lr": 1e-3, "weight_decay": 0.0},
    )

    with pytest.raises(ValueError, match="only supports AdamW"):
        backend.prepare_optimizer(plan)


@requires_cpu_fused_adamw
def test_prepare_optimizer_builds_adamw_swap():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig(
        cpu_offload=CPUOffloadConfig(
            enabled=True,
            mode="optimizer",
        )
    )
    backend._device_runtime = SyncRuntime()
    backend._mesh = {"edp": object()}
    model = torch.nn.Linear(1, 1)
    plan = OptimizerClassPlan(
        model,
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 0.0, "fused": True},
    )

    optimizer = backend.prepare_optimizer(plan)

    assert isinstance(optimizer, AdamWSwap)
    assert optimizer.defaults["fused"] is True
    assert optimizer.param_to_name_map[model.weight] == "weight"
    assert optimizer.param_to_name_map[model.bias] == "bias"


def test_prepare_optimizer_builds_host_update_adamw():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig(
        cpu_offload=CPUOffloadConfig(
            enabled=True,
            mode="optimizer",
            optimizer_memory_fraction=0,
        )
    )
    backend._device_runtime = SyncRuntime()
    backend._mesh = {"edp": object()}
    plan = OptimizerClassPlan(
        torch.nn.Linear(1, 1),
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 0.0, "fused": True},
    )

    optimizer = backend.prepare_optimizer(plan)

    assert isinstance(optimizer, AdamWSwap)
    assert optimizer.optimizer_memory_fraction == 0
    assert optimizer.defaults["fused"] is True


def test_prepare_optimizer_keeps_original_class_without_optimizer_offload():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig(
        cpu_offload=CPUOffloadConfig(enabled=True, mode="full")
    )
    backend._device_runtime = SyncRuntime()
    backend._mesh = {"edp": object()}
    plan = OptimizerClassPlan(
        torch.nn.Linear(1, 1),
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 0.0, "fused": True},
    )

    optimizer = backend.prepare_optimizer(plan)

    assert type(optimizer) is torch.optim.AdamW
