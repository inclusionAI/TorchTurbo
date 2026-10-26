"""Optimizer lifecycle, aliases, and continuation across inference windows."""

import copy
import weakref

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig, FSDPConfig
from torchturbo.engine.backends.fsdp2.optimizers.adamw import AdamWSwap
from torchturbo.engine.backends.fsdp2.optimizers.memory import (
    OptimizerMemoryController,
    _TensorSpec,
)
from torchturbo.engine.backends.fsdp2.qk_clip import install_qk_clip
from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator, CheckpointExportPluginConsumer
from torchturbo.engine.config import EngineConfig, MuonOptimizerConfig, QKClipConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.optimizers import (
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonPlan,
    MuonRecipe,
    OptimizerClassPlan,
    OptimizersContainer,
)
from torchturbo.models.modules.attention.qk_clip import get_qk_clip_state
from torchturbo.runtime import get_device_runtime
from torchturbo.runtime.cpu import CpuRuntime


class _MpsMemoryRuntime(CpuRuntime):
    """Exercise actual host/device copies on developer Macs, without FSDP/MPS."""

    @property
    def device(self):
        return torch.device("mps")

    def synchronize(self):
        torch.mps.synchronize()

    def empty_cache(self):
        torch.mps.empty_cache()


@pytest.fixture(params=["cpu", "device"])
def runtime(request):
    if request.param == "cpu":
        return CpuRuntime()
    selected = get_device_runtime()
    if selected.device.type != "cpu":
        return selected
    if torch.backends.mps.is_available():
        return _MpsMemoryRuntime()
    pytest.skip("requires an accelerator for actual state migration")


def _backend(runtime, offload=None):
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = runtime
    backend._mesh = {"edp": object()}
    backend.fsdp_config = FSDPConfig(cpu_offload=offload)
    return backend


def _engine(runtime, model, offload=None):
    engine = object.__new__(TorchTurboEngine)
    engine.model = model
    engine.backend = _backend(runtime, offload)
    engine.backend.rank = 0
    engine.backend._world_mesh = object()
    engine.backend._optimizer_memory = None
    parallel_context = {"ep": {"ep_rank": 0, "ep_size": 1}}
    engine.backend.get_parallel_context = lambda: parallel_context
    # Match training: checkpoint I/O is created before the optimizer exists.
    engine._checkpoint = CheckpointCoordinator(
        EngineConfig(), engine.backend.create_checkpoint_manager(),
        parallel_context=parallel_context,
        export_plugin_consumer=CheckpointExportPluginConsumer(),
    )
    engine.optimizer = engine.backend.prepare_optimizer(
        OptimizerClassPlan(
            model, torch.optim.AdamW, {"lr": 0.03, "weight_decay": 0.1, "fused": False}
        )
    )
    return engine


def test_engine_optimizer_resume_preserves_training_and_scheduler(runtime):
    torch.manual_seed(10)
    model = nn.Linear(4, 3).to(runtime.device)
    reference = copy.deepcopy(model)
    engine = _engine(runtime, model)
    ref_engine = _engine(runtime, reference)
    optimizer = engine.optimizer
    reference_optimizer = ref_engine.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.7)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(
        reference_optimizer, 1, gamma=0.7
    )
    parameter_ids = [id(param) for param in model.parameters()]
    groups = list(optimizer.param_groups)

    for step in range(4):
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            gradient = torch.full_like(param, (step + 1) / 7)
            param.grad = gradient.clone()
            ref_param.grad = gradient.clone()
        optimizer.step()
        reference_optimizer.step()
        scheduler.step()
        reference_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        before = {
            (id(param), key): value.detach().cpu().clone()
            for param, state in optimizer.state.items()
            for key, value in state.items()
        }
        value = None
        device_states = [
            weakref.ref(value)
            for state in optimizer.state.values()
            for value in state.values()
            if value.device.type != "cpu"
        ]
        engine.offload_optimizer()
        engine.offload_optimizer()
        assert all(reference() is None for reference in device_states)
        assert all(
            value.device.type == "cpu"
            for state in optimizer.state.values()
            for value in state.values()
        )
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            optimizer.step()
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            optimizer.load_state_dict(optimizer.state_dict())
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            engine.save_optimizer_and_scheduler(model, optimizer, scheduler, "unused")
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            engine.load_optimizer_and_scheduler(model, optimizer, scheduler, "unused")
        engine.restore_optimizer()
        engine.restore_optimizer()
        for param, state in optimizer.state.items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value.cpu(), before[id(param), key], rtol=0, atol=0
                )
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=0, atol=0)
    assert [id(param) for param in model.parameters()] == parameter_ids
    assert all(
        current is original for current, original in zip(optimizer.param_groups, groups)
    )
    assert scheduler.state_dict() == reference_scheduler.state_dict()
    assert engine.optimizer is optimizer


@pytest.mark.parametrize("mode", ["full", "optimizer"])
def test_training_cpu_offload_allows_checkpoint_io(tmp_path, mode):
    model = nn.Linear(3, 2)
    engine = _engine(
        CpuRuntime(),
        model,
        CPUOffloadConfig(enabled=True, mode=mode, optimizer_memory_fraction=0),
    )
    optimizer = engine.optimizer
    checkpoint_dir = str(tmp_path / "checkpoint")
    engine.save_optimizer_and_scheduler(model, optimizer, None, checkpoint_dir)
    saved_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer.state.values():
        for tensor in state.values():
            tensor.fill_(7.0)
    engine.load_optimizer_and_scheduler(model, optimizer, None, checkpoint_dir)
    torch.testing.assert_close(optimizer.state_dict(), saved_state, rtol=0, atol=0)


@pytest.mark.parametrize("direct_checkpoint", [False, True])
def test_suspended_checkpoint_preserves_files_and_state(tmp_path, direct_checkpoint):
    model = nn.Linear(3, 2)
    engine = _engine(CpuRuntime(), model)
    optimizer = engine.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.7)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint = engine.checkpoint if direct_checkpoint else engine
    checkpoint_dir = str(tmp_path / "checkpoint")
    checkpoint.save_optimizer_and_scheduler(
        model, optimizer, scheduler, checkpoint_dir
    )
    saved_files = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())

    engine.offload_optimizer()
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        checkpoint.save_optimizer_and_scheduler(
            model, optimizer, scheduler, checkpoint_dir
        )
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        checkpoint.load_optimizer_and_scheduler(
            model, optimizer, scheduler, checkpoint_dir
        )

    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == saved_files
    torch.testing.assert_close(optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    assert scheduler.state_dict() == saved_scheduler

    engine.restore_optimizer()
    for param in model.parameters():
        param.grad = torch.full_like(param, 2.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint.load_optimizer_and_scheduler(
        model, optimizer, scheduler, checkpoint_dir
    )
    torch.testing.assert_close(optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    assert scheduler.state_dict() == saved_scheduler
    checkpoint.save_optimizer_and_scheduler(
        model, optimizer, scheduler, str(tmp_path / "resaved")
    )


def test_offload_rejects_pending_gradients_and_unsupported_optimizer():
    param = nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param])
    controller = OptimizerMemoryController(optimizer, CpuRuntime())
    param.grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="zero_grad"):
        controller.offload()
    assert not controller.is_suspended
    assert param.grad is not None
    param.grad = None
    sgd = OptimizerMemoryController(torch.optim.SGD([param], lr=0.1), CpuRuntime())
    with pytest.raises(NotImplementedError, match="SGD"):
        sgd.offload()
    assert not sgd.is_suspended


def test_fresh_optimizer_suspend_does_not_initialize_or_step():
    model = nn.Linear(3, 2)
    engine = _engine(CpuRuntime(), model)
    engine.offload_optimizer()
    engine.restore_optimizer()
    assert not engine.optimizer.state


def test_restore_allocation_failure_keeps_cpu_state(runtime, monkeypatch):
    if runtime.device.type == "cpu":
        pytest.skip("requires actual device state")
    model = nn.Linear(3, 2).to(runtime.device)
    engine = _engine(runtime, model)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    engine.optimizer.step()
    engine.optimizer.zero_grad(set_to_none=True)
    engine.offload_optimizer()
    original = _TensorSpec.restore
    count = 0

    def fail_second(spec, tensor):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected allocation failure")
        return original(spec, tensor)

    monkeypatch.setattr(_TensorSpec, "restore", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        engine.restore_optimizer()
    assert engine.backend._optimizer_memory.is_suspended
    assert all(
        value.device.type == "cpu"
        for state in engine.optimizer.state.values()
        for value in state.values()
    )
    monkeypatch.setattr(_TensorSpec, "restore", original)
    engine.restore_optimizer()
    assert not engine.backend._optimizer_memory.is_suspended


def test_swap_counter_alias_is_rebound(runtime):
    if runtime.device.type == "cpu":
        pytest.skip("requires actual device state")
    param = nn.Parameter(torch.ones(3, device=runtime.device))
    optimizer = AdamWSwap([param], device_runtime=runtime, optimizer_memory_fraction=0)
    # Isolate counter migration from the fused AdamW kernel availability.
    step = torch.tensor(7.0, device=runtime.device)
    optimizer.state[param]["step"] = step
    optimizer.param_to_step_map[param] = step
    del step
    controller = OptimizerMemoryController(optimizer, runtime)
    controller.offload()
    assert optimizer.state[param]["step"].device.type == "cpu"
    assert optimizer.param_to_step_map[param] is optimizer.state[param]["step"]
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        optimizer.load_state_dict(optimizer.state_dict())
    controller.restore()
    assert optimizer.param_to_step_map[param] is optimizer.state[param]["step"]
    assert optimizer.param_to_step_map[param].device.type == runtime.device.type
    assert optimizer.param_to_step_map[param].item() == 7


def test_qk_clip_pauses_collection_and_keeps_statistics(runtime):
    model = nn.Module()
    model.attention = nn.Module()
    model.attention.q_proj = nn.Linear(4, 6, bias=False)
    model.attention.kv_b_proj = nn.Linear(4, 8, bias=False)
    recipe = MuonRecipe(
        {},
        qk_clip_specs={
            "attention": MLAQKClipSpec("q_proj", "kv_b_proj", 2, 2, 1, 2),
        },
    )
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(
            lr=0.01,
            weight_decay=0.0,
            qk_clip=QKClipConfig(threshold=100),
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    qk = install_qk_clip(plan.qk_clip_plan, optimizer, device=runtime.device)
    state = get_qk_clip_state(model.attention)
    observations = torch.tensor([400.0, 50.0], device=runtime.device)
    state.record(observations)
    statistics = state.max_logits
    controller = OptimizerMemoryController(optimizer, runtime, qk)
    controller.offload()
    # Attention stops collecting, while the small statistics stay in place.
    assert get_qk_clip_state(model.attention) is None
    assert state.max_logits is statistics
    with pytest.raises(ValueError, match="already installed"):
        install_qk_clip(plan.qk_clip_plan, optimizer, device=runtime.device)
    controller.restore()
    assert get_qk_clip_state(model.attention) is state
    assert state.observed
    torch.testing.assert_close(state.max_logits, observations)
    # Device statistics still scale the CPU weights after the window.
    before = model.attention.q_proj.weight.detach().clone()
    qk.clip()
    torch.testing.assert_close(model.attention.q_proj.weight[:2], before[:2] * 0.5)
    assert not state.observed


@pytest.mark.parametrize("mode", ["optimizer", "full"])
def test_muon_empty_selection_uses_shared_adamw_offload_policy(mode):
    model = nn.Linear(3, 2)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"weight": ExcludeFromMuon()}),
        config=MuonOptimizerConfig(lr=0.01, weight_decay=0.1, adamw_fused=False),
    )
    backend = _backend(
        CpuRuntime(),
        CPUOffloadConfig(
            enabled=True,
            mode=mode,
            optimizer_memory_fraction=0,
        ),
    )
    optimizer = backend.prepare_optimizer(plan)
    expected = AdamWSwap if mode == "optimizer" else torch.optim.AdamW
    assert type(optimizer) is expected
    assert all(group["fused"] is False for group in optimizer.param_groups)
    backend.offload_optimizer()
    backend.restore_optimizer()


def test_container_guard_prevents_any_child_step(runtime):
    model = nn.Linear(3, 2).to(runtime.device)
    plan = OptimizerClassPlan(
        model, torch.optim.AdamW, {"lr": 0.01, "weight_decay": 0.01}
    )
    child = torch.optim.AdamW(
        [group.get_optimizer_param_group() for group in plan.get_param_groups()],
        lr=0.01,
    )
    container = OptimizersContainer(plan, {plan.optimizer_name: child})
    controller = OptimizerMemoryController(container, runtime)
    hooked = (container, child)
    # The guard exists only while offloaded and leaves no hooks behind.
    assert not any(optimizer._optimizer_step_pre_hooks for optimizer in hooked)
    controller.offload()
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        container.step()
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        child.step()
    groups = container.param_groups
    with pytest.raises(RuntimeError, match="optimizer is suspended"):
        container.load_state_dict(container.state_dict())
    assert container.param_groups is groups
    assert not child.state
    controller.restore()
    assert not any(optimizer._optimizer_step_pre_hooks for optimizer in hooked)
    assert not any(
        optimizer._optimizer_load_state_dict_pre_hooks for optimizer in hooked
    )
