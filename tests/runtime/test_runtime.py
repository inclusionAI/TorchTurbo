from contextlib import nullcontext
import os
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
from torchturbo.runtime.base import Shmem, TorchShmem
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
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
    buffer, handle = runtime.allocate_communication_buffer(3, dtype=torch.float32, device=runtime.device, group=None)
    assert buffer.device == runtime.device and buffer.shape == (3,)
    assert handle is None
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
            memory_snapshot_steps=None,
        )


def test_selected_npu_runtime_owns_the_shmem_service(monkeypatch):
    monkeypatch.setattr(runtime_impl, "_auto_detect_runtime", NpuRuntime)
    runtime = get_device_runtime()

    assert isinstance(runtime.shmem, Shmem)
    assert get_device_runtime().shmem is runtime.shmem


@pytest.mark.parametrize("runtime_cls", [CpuRuntime, CudaRuntime])
def test_runtime_without_shmem_support_skips_heap_management(runtime_cls):
    shmem = runtime_cls().shmem
    assert not shmem.available()
    assert shmem.prepare_heap([None, None, None]) == 0
    shmem.release_unused()


@pytest.mark.parametrize("use_world", [False, True], ids=["subgroup", "world"])
def test_communication_buffer_delegates_peer_allocations_to_pytorch(monkeypatch, use_world):
    import torch.distributed._symmetric_memory as symm_mem

    class AvailableShmem(TorchShmem):
        def available(self) -> bool:
            return True

    runtime = CpuRuntime()
    runtime.shmem = AvailableShmem()
    device = torch.device("cpu")
    world = object()
    group = None if use_world else object()
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=world))
    calls = []
    arena = torch.empty(3)
    handle = object()

    def empty(numel, *, dtype, device):
        calls.append((numel, dtype, device))
        return arena

    def rendezvous(tensor, group):
        assert tensor is arena
        calls.append(group)
        return handle

    monkeypatch.setattr(symm_mem, "empty", empty)
    monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)
    args = {"dtype": torch.float32, "device": device, "group": group}

    assert runtime.shmem.prepare_heap([arena.nbytes, None]) == 0
    tensor, peers = runtime.allocate_communication_buffer(3, **args, peer_access=True)
    assert tensor is arena and peers is handle
    local, peers = runtime.allocate_communication_buffer(3, **args, peer_access=False)
    assert (local.shape, local.dtype, local.device, peers) == ((3,), torch.float32, device, None)
    runtime.shmem.release_unused()
    assert calls == [(3, torch.float32, device), world if use_world else group]


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


@pytest.mark.parametrize("base", [DeviceRuntime, Shmem])
def test_runtime_base_is_abstract(base):
    with pytest.raises(TypeError, match="abstract"):
        base()


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
