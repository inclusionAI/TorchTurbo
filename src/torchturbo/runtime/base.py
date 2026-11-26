from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class BaseProfiler(ABC):
    @abstractmethod
    def __init__(
        self,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> None:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


class TurboDevice(StrEnum):
    """Supported TorchTurbo runtime devices."""

    CUDA = "cuda"
    NPU = "npu"
    CPU = "cpu"


class DeviceRuntime(ABC):
    # Concrete runtimes must set these class attributes.
    turbo_device: TurboDevice
    dist_backend: str

    @property
    def device(self) -> torch.device:
        """Device passed to Torch APIs."""
        return torch.device(self.turbo_device.value)

    @property
    def supports_async_streams(self) -> bool:
        """Whether this runtime provides asynchronous stream/event primitives."""
        return False

    def init_device_mesh(
        self,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...] | None = None,
        expert_parallel_dim: str | None = None,
    ) -> DeviceMesh:
        """Build a DeviceMesh on this runtime's device.

        Runtimes may specialize the process groups that back individual mesh
        dimensions; callers should build meshes through this method rather than
        calling :func:`torch.distributed.device_mesh.init_device_mesh` directly.

        Args:
            mesh_shape: Size of each mesh dimension.
            mesh_dim_names: Name of each mesh dimension.
            expert_parallel_dim: Name of the expert-parallel mesh dimension,
                which carries the MoE dispatch/combine communication. Runtimes
                may give its communicator a dedicated configuration; runtimes
                without such a notion ignore it.
        """
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh(
            self.device.type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def initialize(self) -> None:
        ...

    def use_deterministic_algorithms(self) -> None:
        """Enable PyTorch's deterministic algorithm mode."""
        torch.use_deterministic_algorithms(True)

    @abstractmethod
    def current_device(self):
        ...

    @abstractmethod
    def set_device(self, device: int) -> None:
        ...

    @abstractmethod
    def synchronize(self) -> None:
        ...

    @abstractmethod
    def empty_cache(self) -> None:
        ...

    @abstractmethod
    def memory_info(self) -> tuple[int, int]:
        ...

    def memory_reserved(self) -> int:
        """Return bytes the caching allocator holds on the current device."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement memory_reserved()."
        )

    def memory_allocated(self) -> int:
        """Return bytes occupied by live tensors on the current device."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement memory_allocated()."
        )

    @abstractmethod
    def stream(self, stream=None):
        ...

    @abstractmethod
    def current_stream(self):
        ...

    @abstractmethod
    def new_stream(self, priority: int = 0):
        ...

    def new_event(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement device events."
        )

    def record_stream(self, tensor: torch.Tensor, stream) -> None:
        if stream is not None:
            tensor.record_stream(stream)

    def synchronize_stream(self, stream) -> None:
        if stream is not None:
            stream.synchronize()

    @abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        ...

    @abstractmethod
    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        ...
