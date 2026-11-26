from __future__ import annotations

from collections.abc import Callable

from torchturbo.runtime import TurboDevice

from .base import OpImpl, OpImplKind, OpSpec


class OpRegistry:
    def __init__(self) -> None:
        self._specs: list[OpSpec] = []
        self._specs_by_impl: dict[str, OpSpec] = {}

    def copy(self) -> OpRegistry:
        """Return an independent snapshot of the registered metadata."""
        snapshot = OpRegistry()
        for spec in self._specs:
            snapshot.register(spec)
        return snapshot

    def register(self, spec: OpSpec) -> OpSpec:
        existing = self._specs_by_impl.get(spec.impl_name)
        if existing is not None and existing.op_name != spec.op_name:
            raise ValueError(
                f"OpSpec.impl_name {spec.impl_name!r} is already registered for "
                f"op {existing.op_name!r}."
            )

        replaced = False
        for idx, current in enumerate(self._specs):
            if current.impl_name == spec.impl_name:
                self._specs[idx] = spec
                replaced = True
                break
        if not replaced:
            self._specs.append(spec)

        self._specs_by_impl[spec.impl_name] = spec
        return spec

    def list(
        self,
        spec_filter: Callable[[OpSpec], bool] | None = None,
    ) -> list[OpSpec]:
        specs = list(self._specs)
        if spec_filter is not None:
            specs = [spec for spec in specs if spec_filter(spec)]
        # Rank by priority first; at equal priority prefer device-specific
        # implementations (non-empty ``devices``) over device-agnostic fallback,
        # then prefer custom over native so auto selection stays deterministic.
        specs.sort(key=lambda item: item.impl_name)
        specs.sort(
            key=lambda item: (
                item.priority,
                bool(item.devices),
                item.kind == OpImplKind.CUSTOM,
            ),
            reverse=True,
        )
        return specs

    def get_by_impl_name(self, impl_name: str) -> OpSpec:
        spec = self._specs_by_impl.get(impl_name)
        if spec is None:
            raise RuntimeError(f"No OpSpec registered for impl_name {impl_name!r}.")
        return spec


registry = OpRegistry()


def register_op(
    *,
    op_name: str,
    impl_name: str,
    kind: OpImplKind,
    priority: int,
    devices: tuple[TurboDevice, ...],
):
    def decorator(cls: type[OpImpl]) -> type[OpImpl]:
        spec = OpSpec(
            op_name=op_name,
            impl_name=impl_name,
            kind=kind,
            priority=priority,
            devices=devices,
        )
        cls.__op_spec__ = spec
        registry.register(spec)
        return cls

    return decorator


__all__ = [
    "OpRegistry",
    "register_op",
    "registry",
]
