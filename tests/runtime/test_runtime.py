from contextlib import nullcontext
import os
import sys
import types
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


def _patch_init_device_mesh(monkeypatch):
    calls = []

    def fake_init_device_mesh(
        device_type, mesh_shape, *, mesh_dim_names=None, backend_override=None
    ):
        calls.append(
            {
                "device_type": device_type,
                "mesh_shape": mesh_shape,
                "mesh_dim_names": mesh_dim_names,
                "backend_override": backend_override,
            }
        )
        return "mesh"

    import torch.distributed.device_mesh as device_mesh

    monkeypatch.setattr(device_mesh, "init_device_mesh", fake_init_device_mesh)
    return calls


def _patch_torch_npu_hccl_options(monkeypatch):
    created = []

    class FakeOptions:
        def __init__(self) -> None:
            self.hccl_config = None
            created.append(self)

    distributed_c10d = types.ModuleType("torch_npu._C._distributed_c10d")
    distributed_c10d.ProcessGroupHCCL = types.SimpleNamespace(Options=FakeOptions)
    torch_npu_c = types.ModuleType("torch_npu._C")
    torch_npu_c._distributed_c10d = distributed_c10d
    torch_npu = types.ModuleType("torch_npu")
    torch_npu._C = torch_npu_c

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "torch_npu._C", torch_npu_c)
    monkeypatch.setitem(sys.modules, "torch_npu._C._distributed_c10d", distributed_c10d)
    return created


def test_init_device_mesh_forwards_without_override(monkeypatch):
    calls = _patch_init_device_mesh(monkeypatch)

    mesh = CpuRuntime().init_device_mesh(
        (4, 2), mesh_dim_names=("edp", "ep")
    )

    assert mesh == "mesh"
    assert calls == [
        {
            "device_type": "cpu",
            "mesh_shape": (4, 2),
            "mesh_dim_names": ("edp", "ep"),
            "backend_override": None,
        }
    ]


def test_init_device_mesh_uses_aiv_for_expert_parallel_dimension_on_npu(monkeypatch):
    calls = _patch_init_device_mesh(monkeypatch)
    created = _patch_torch_npu_hccl_options(monkeypatch)
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU")

    NpuRuntime().init_device_mesh(
        (2, 4),
        mesh_dim_names=("edp", "ep"),
        expert_parallel_dim="ep",
    )

    override = calls[0]["backend_override"]
    assert set(override) == {"ep"}
    assert override["ep"] is created[0]
    assert created[0].hccl_config == {"hccl_op_expansion_mode": 3}


def test_init_device_mesh_treats_unset_expansion_mode_as_ai_cpu(monkeypatch):
    calls = _patch_init_device_mesh(monkeypatch)
    _patch_torch_npu_hccl_options(monkeypatch)
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    NpuRuntime().init_device_mesh(
        (2, 4),
        mesh_dim_names=("edp", "ep"),
        expert_parallel_dim="ep",
    )

    assert set(calls[0]["backend_override"]) == {"ep"}


def test_init_device_mesh_skips_override_without_expert_parallel_dim(monkeypatch):
    calls = _patch_init_device_mesh(monkeypatch)
    created = _patch_torch_npu_hccl_options(monkeypatch)
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU")

    NpuRuntime().init_device_mesh((4,), mesh_dim_names=("dp",))

    assert calls[0]["backend_override"] is None
    assert created == []


def test_init_device_mesh_rejects_unknown_expert_parallel_dim(monkeypatch):
    calls = _patch_init_device_mesh(monkeypatch)
    _patch_torch_npu_hccl_options(monkeypatch)
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU")

    with pytest.raises(ValueError, match="expert_parallel_dim"):
        NpuRuntime().init_device_mesh(
            (2, 4),
            mesh_dim_names=("edp", "ep"),
            expert_parallel_dim="tp",
        )

    assert calls == []


@pytest.mark.parametrize("expansion_mode", ["AIV", "aiv", "HOST"])
def test_init_device_mesh_keeps_global_expansion_mode(monkeypatch, expansion_mode):
    calls = _patch_init_device_mesh(monkeypatch)
    created = _patch_torch_npu_hccl_options(monkeypatch)
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", expansion_mode)

    NpuRuntime().init_device_mesh(
        (2, 4),
        mesh_dim_names=("edp", "ep"),
        expert_parallel_dim="ep",
    )

    assert calls[0]["backend_override"] is None
    assert created == []
