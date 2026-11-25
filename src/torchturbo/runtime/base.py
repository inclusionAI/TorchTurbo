from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol

import torch


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


class ShmemHandle(Protocol):
    """A window onto peer buffers, indexed by rank in the rendezvous group."""

    @property
    def buffer_ptrs(self) -> list[int]: ...

    @property
    def buffer_ptrs_dev(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def get_buffer(self, rank: int, sizes: tuple[int, ...], dtype: torch.dtype, /) -> torch.Tensor: ...

    def barrier(self) -> None: ...


class Shmem(ABC):
    """Symmetric memory operations provided by a device backend."""

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prepare_heap(self, buffer_sizes: list[int | None]) -> int:
        raise NotImplementedError

    @abstractmethod
    def empty(
        self, numel: int, *, dtype: torch.dtype, device: torch.device,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def rendezvous(
        self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup | None,
    ) -> ShmemHandle:
        raise NotImplementedError

    @abstractmethod
    def release_unused(self) -> None:
        raise NotImplementedError


class TorchShmem(Shmem):
    """PyTorch owns buffer lifetimes. Device backends opt in to peer access."""

    def available(self) -> bool:
        return False

    def prepare_heap(self, buffer_sizes: list[int | None]) -> int:
        """PyTorch handles heap sizing when it allocates a tensor."""
        return 0

    def empty(
        self, numel: int, *, dtype: torch.dtype, device: torch.device,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        import torch.distributed._symmetric_memory as symm_mem

        return symm_mem.empty(numel, dtype=dtype, device=device)

    def rendezvous(
        self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup | None,
    ) -> ShmemHandle:
        import torch.distributed._symmetric_memory as symm_mem

        group = torch.distributed.group.WORLD if group is None else group
        return symm_mem.rendezvous(tensor, group)

    def release_unused(self) -> None:
        """PyTorch releases buffers when their last tensor reference is dropped."""


class DeviceRuntime(ABC):
    # Concrete runtimes must set these class attributes.
    turbo_device: TurboDevice
    dist_backend: str
    shmem: Shmem = TorchShmem()

    @property
    def device(self) -> torch.device:
        """Device passed to Torch APIs."""
        return torch.device(self.turbo_device.value)

    @property
    def supports_async_streams(self) -> bool:
        """Whether this runtime provides asynchronous stream/event primitives."""
        return False

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

    def allocate_communication_buffer(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: torch.distributed.ProcessGroup | None,
        peer_access: bool = True,
    ) -> tuple[torch.Tensor, ShmemHandle | None]:
        """Allocate local storage or a peer window for callers supporting both."""
        if peer_access and self.shmem.available():
            buffer = self.shmem.empty(numel, dtype=dtype, device=device, group=group)
            return buffer, self.shmem.rendezvous(buffer, group)
        return torch.empty(numel, dtype=dtype, device=device), None

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
