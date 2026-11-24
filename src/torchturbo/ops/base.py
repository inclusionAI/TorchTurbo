from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from torchturbo.runtime import TurboDevice


class OpPolicy(StrEnum):
    AUTO = "auto"
    NATIVE = "native"
    CUSTOM = "custom"


class OpImplKind(StrEnum):
    NATIVE = "native"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class OpSpec:
    op_name: str
    impl_name: str
    kind: OpImplKind = OpImplKind.NATIVE
    priority: int = 0
    devices: tuple[TurboDevice, ...] = ()

    def __post_init__(self) -> None:
        if not self.op_name:
            raise ValueError("OpSpec.op_name must be set.")
        if not self.impl_name:
            raise ValueError("OpSpec.impl_name must be set.")
        if self.impl_name in {item.value for item in OpPolicy}:
            raise ValueError(
                f"OpSpec.impl_name cannot be a reserved policy name: {self.impl_name!r}."
            )
        if not isinstance(self.kind, OpImplKind):
            raise TypeError("OpSpec.kind must be an OpImplKind.")
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "devices", tuple(self.devices))


class OpImpl(ABC):
    requires_peer_memory: ClassVar[bool] = False
    __op_spec__: ClassVar[OpSpec | None] = None

    @classmethod
    def op_spec(cls) -> OpSpec:
        spec = cls.__op_spec__
        if spec is None:
            raise RuntimeError(f"{cls.__name__} is missing op spec metadata.")
        return spec

    def available(self) -> bool:
        return True

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


__all__ = [
    "OpImpl",
    "OpImplKind",
    "OpPolicy",
    "OpSpec",
]
