from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from torchturbo.runtime import (
    DeviceRuntime,
    TurboDevice,
    get_device_name,
    get_device_runtime,
    is_cuda_available,
    is_device_available,
    is_npu_available,
)
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.npu import _NPU_DETERMINISTIC_ENV, NpuRuntime
import torchturbo.runtime.runtime as runtime_impl


class FakeRuntime(DeviceRuntime):
    def __init__(self, turbo_device: TurboDevice, available: bool = True) -> None:
        self.turbo_device = turbo_device
        self.dist_backend = {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}[
            turbo_device.value
        ]
        self._available = available
        self.available_calls = 0
        self.initialized = 0
        self.deterministic_calls = 0
        self.profiler_calls = []

    @property
    def device(self):
        return SimpleNamespace(type=self.turbo_device.value)

    def available(self) -> bool:
        self.available_calls += 1
        return self._available

    def initialize(self) -> None:
        self.initialized += 1

    def use_deterministic_algorithms(self) -> None:
        self.deterministic_calls += 1

    def current_device(self):
        return 0

    def set_device(self, device: int) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def memory_info(self) -> tuple[int, int]:
        return (1, 1)

    def stream(self, stream=None):
        return nullcontext()

    def current_stream(self):
        return None

    def new_stream(self, priority: int = 0):
        return None

    def manual_seed_all(self, seed: int) -> None:
        return None

    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        record_shapes: bool,
        aic_metrics: bool,
        with_stack: bool,
        profiling_step: int | None,
        memory_snapshot_steps: int | None,
    ):
        self.profiler_calls.append((output_dir, worker_name))
        return FakeProfiler()


class FakeProfiler:
    def start(self) -> None:
        return None

    def step(self) -> None:
        return None

    def stop(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_runtime_cache(monkeypatch):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    runtime_impl.is_device_available.cache_clear()
    yield
    runtime_impl.is_device_available.cache_clear()


def test_cpu_runtime_contract():
    runtime = CpuRuntime()

    assert runtime.turbo_device is TurboDevice.CPU
    assert runtime.device == torch.device("cpu")
    assert runtime.dist_backend == "gloo"
    assert runtime.supports_async_streams is False
    assert runtime.new_stream() is None
    assert runtime.current_stream() is None
    free, total = runtime.memory_info()
    assert 0 <= free <= total
    assert total > 0
    with runtime.stream(None):
        pass

    with pytest.raises(NotImplementedError, match="device events"):
        runtime.new_event()

    with pytest.raises(NotImplementedError, match="profiler"):
        runtime.profiler(
            output_dir="/tmp",
            worker_name="worker",
            profiling_enabled=True,
            record_shapes=False,
            aic_metrics=False,
            with_stack=False,
            profiling_step=None,
            memory_snapshot_steps=None,
        )


def test_use_deterministic_algorithms_sets_process_state(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    runtime = NpuRuntime()
    runtime.use_deterministic_algorithms()

    assert calls == [True]
    assert {name: os.environ[name] for name in _NPU_DETERMINISTIC_ENV} == _NPU_DETERMINISTIC_ENV


def test_device_runtime_synchronize_stream():
    runtime = CpuRuntime()

    class FakeStream:
        def __init__(self) -> None:
            self.synchronized = 0

        def synchronize(self) -> None:
            self.synchronized += 1

    stream = FakeStream()

    runtime.synchronize_stream(None)
    runtime.synchronize_stream(stream)

    assert stream.synchronized == 1


def test_runtime_availability_helpers_cache_and_probe(monkeypatch):
    cuda_runtime = FakeRuntime(TurboDevice.CUDA, available=False)
    npu_runtime = FakeRuntime(TurboDevice.NPU, available=True)
    cpu_runtime = FakeRuntime(TurboDevice.CPU, available=True)

    monkeypatch.setattr(
        runtime_impl,
        "_build_runtime",
        lambda device: {
            TurboDevice.CUDA: cuda_runtime,
            TurboDevice.NPU: npu_runtime,
            TurboDevice.CPU: cpu_runtime,
        }[device],
    )

    assert is_cuda_available() is False
    assert is_npu_available() is True
    assert is_device_available(TurboDevice.NPU) is True
    assert is_device_available(TurboDevice.NPU) is True
    assert cuda_runtime.available_calls == 1
    assert npu_runtime.available_calls == 1


def test_get_device_name_prefers_selected_runtime(monkeypatch):
    selected = FakeRuntime(TurboDevice.NPU, available=True)
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", selected)
    monkeypatch.setattr(
        runtime_impl,
        "_build_runtime",
        lambda device: pytest.fail("probe should not run when runtime is selected"),
    )

    assert get_device_name() == "npu"


def test_get_device_name_falls_back_to_probe_order(monkeypatch):
    cuda_runtime = FakeRuntime(TurboDevice.CUDA, available=False)
    npu_runtime = FakeRuntime(TurboDevice.NPU, available=True)
    cpu_runtime = FakeRuntime(TurboDevice.CPU, available=True)

    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    monkeypatch.setattr(
        runtime_impl,
        "_build_runtime",
        lambda device: {
            TurboDevice.CUDA: cuda_runtime,
            TurboDevice.NPU: npu_runtime,
            TurboDevice.CPU: cpu_runtime,
        }[device],
    )

    assert get_device_name() == "npu"
    assert cuda_runtime.available_calls == 1
    assert npu_runtime.available_calls == 1


def test_device_runtime_base_is_abstract():
    with pytest.raises(TypeError):
        DeviceRuntime()


def test_device_runtime_auto_detects_first_available_runtime(monkeypatch):
    cuda_runtime = FakeRuntime(TurboDevice.CUDA, available=False)
    npu_runtime = FakeRuntime(TurboDevice.NPU, available=False)
    cpu_runtime = FakeRuntime(TurboDevice.CPU, available=True)

    def fake_build_runtime(device):
        return {
            TurboDevice.CUDA: cuda_runtime,
            TurboDevice.NPU: npu_runtime,
            TurboDevice.CPU: cpu_runtime,
        }[device]

    monkeypatch.setattr(runtime_impl, "_build_runtime", fake_build_runtime)

    runtime = get_device_runtime()

    assert runtime is cpu_runtime
    assert runtime.turbo_device is TurboDevice.CPU
    assert cpu_runtime.initialized == 1
    assert cuda_runtime.initialized == 0
    assert npu_runtime.initialized == 0


def test_device_runtime_auto_detection_errors_when_none_available(monkeypatch):
    cuda_runtime = FakeRuntime(TurboDevice.CUDA, available=False)
    npu_runtime = FakeRuntime(TurboDevice.NPU, available=False)
    cpu_runtime = FakeRuntime(TurboDevice.CPU, available=False)

    def fake_build_runtime(device):
        return {
            TurboDevice.CUDA: cuda_runtime,
            TurboDevice.NPU: npu_runtime,
            TurboDevice.CPU: cpu_runtime,
        }[device]

    monkeypatch.setattr(runtime_impl, "_build_runtime", fake_build_runtime)

    with pytest.raises(RuntimeError, match="No available device runtime"):
        get_device_runtime()


def test_device_runtime_get_returns_selected_runtime(monkeypatch):
    selected = FakeRuntime(TurboDevice.CPU)
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", selected)

    current = get_device_runtime()

    assert current is selected
    assert current.turbo_device is TurboDevice.CPU


def _fake_torch_npu_module():
    """Minimal torch_npu stand-in covering what _NpuProfiler reads."""
    experimental_config_calls = []
    profile_calls = []
    schedule_calls = []

    class FakeExperimentalConfig:
        def __init__(self, **kwargs):
            experimental_config_calls.append(kwargs)

    class FakeProfile:
        def __init__(self, **kwargs):
            profile_calls.append(kwargs)

        def start(self):
            return None

        def step(self):
            return None

        def stop(self):
            return None

    def fake_schedule(**kwargs):
        schedule_calls.append(kwargs)
        return None

    profiler = SimpleNamespace(
        _ExperimentalConfig=FakeExperimentalConfig,
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level0="Level0", Level1="Level1"),
        AiCMetrics=SimpleNamespace(AiCoreNone="none", PipeUtilization="pipe"),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        schedule=fake_schedule,
        tensorboard_trace_handler=lambda *args, **kwargs: None,
        profile=FakeProfile,
    )
    module = SimpleNamespace(profiler=profiler)
    return module, experimental_config_calls, profile_calls, schedule_calls


@pytest.mark.parametrize(
    ("record_shapes", "with_stack", "expected_level", "expected_aic"),
    [
        # detailed_kernel_info raises the level and turns on utilization.
        (True, False, "Level1", "pipe"),
        # with_stack must not raise the level or enable metrics.
        (False, True, "Level0", "none"),
    ],
)
def test_npu_profiler_level_and_aic_metrics(
    monkeypatch, record_shapes, with_stack, expected_level, expected_aic
):
    fake_module, experimental_config_calls, profile_calls, _ = (
        _fake_torch_npu_module()
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_module)

    from torchturbo.runtime.npu import _NpuProfiler

    profiler = _NpuProfiler(
        "/tmp/profiling",
        "rank0",
        profiling_enabled=True,
        record_shapes=record_shapes,
        aic_metrics=record_shapes,
        with_stack=with_stack,
        profiling_step=None,
        memory_snapshot_steps=None,
    )
    # Constructing records torch_npu calls; start() is never called so this
    # test works on machines without torch_npu installed.
    assert profiler._profiler is not None
    assert experimental_config_calls[0]["profiler_level"] == expected_level
    assert experimental_config_calls[0]["aic_metrics"] == expected_aic
    assert experimental_config_calls[0]["data_simplification"] is True
    assert profile_calls[0]["record_shapes"] is record_shapes
    assert profile_calls[0]["with_stack"] is with_stack


def test_npu_profiler_schedule_from_profiling_step(monkeypatch):
    fake_module, _, _, schedule_calls = _fake_torch_npu_module()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_module)

    from torchturbo.runtime.npu import _NpuProfiler

    # Training step N (1-based) runs in the ProfilerStep#(N-1) window;
    # no profiling_step defaults to recording training step 1.
    for profiling_step, expected_skip_first in ((None, 0), (1, 0), (20, 19)):
        _NpuProfiler(
            "/tmp/profiling",
            "rank0",
            profiling_enabled=True,
            record_shapes=False,
            aic_metrics=False,
            with_stack=False,
            profiling_step=profiling_step,
            memory_snapshot_steps=None,
        )
        last = schedule_calls[-1]
        assert last["skip_first"] == expected_skip_first
        assert last["warmup"] == 0
