from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from torchturbo.logging import get_logger
from torchturbo.runtime import DeviceRuntime, TurboDevice

from .base import OpImpl, OpImplKind, OpPolicy, OpSpec
from .providers import iter_provider_modules
from .registry import OpRegistry


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Candidate:
    impl: OpImpl
    spec: OpSpec
    fallback: bool


class OpDispatcher:
    def __init__(
        self,
        *,
        device_runtime: DeviceRuntime,
        registry: OpRegistry,
        default_policy: OpPolicy = OpPolicy.AUTO,
    ) -> None:
        if not isinstance(default_policy, OpPolicy):
            raise TypeError("default_policy must be an OpPolicy.")
        self.registry = registry
        self._device_runtime = device_runtime
        self._default_policy = default_policy
        self._impls: dict[str, OpImpl] = {}
        self._impl_classes: dict[str, type[OpImpl]] = {}
        self._candidate_impl_cache: dict[
            tuple[str, str, str | None], _Candidate
        ] = {}
        self._unresolved_ops: dict[str, str] = {}
        self._configure()

    @property
    def device_runtime(self) -> DeviceRuntime:
        return self._device_runtime

    @property
    def default_policy(self) -> OpPolicy:
        return self._default_policy

    def _index_provider_module(
        self,
        module: ModuleType,
        impl_classes: dict[str, type[OpImpl]],
    ) -> None:
        for value in module.__dict__.values():
            if not isinstance(value, type):
                continue
            if value.__module__ != module.__name__:
                continue
            if not issubclass(value, OpImpl) or value is OpImpl:
                continue

            try:
                spec = value.op_spec()
            except RuntimeError:
                continue

            existing = impl_classes.get(spec.impl_name)
            if existing is not None and existing is not value:
                raise ValueError(
                    f"Op implementation {spec.impl_name!r} is defined by both "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{value.__module__}.{value.__qualname__}."
                )
            impl_classes[spec.impl_name] = value

    def _load_provider_impl_classes(
        self, device: TurboDevice
    ) -> dict[str, type[OpImpl]]:
        impl_classes: dict[str, type[OpImpl]] = {}
        for module_name in iter_provider_modules(device):
            module = importlib.import_module(module_name)
            self._index_provider_module(module, impl_classes)
        return impl_classes

    def _configure(self) -> None:
        device = self._device_runtime.turbo_device

        self._impl_classes = self._load_provider_impl_classes(device)
        specs = self.registry.list(
            spec_filter=lambda spec: not spec.devices or device in spec.devices
        )
        op_names = sorted({spec.op_name for spec in specs})
        for op_name in op_names:
            try:
                self._candidate_impl(op_name, device)
            except RuntimeError as exc:
                self._unresolved_ops[op_name] = str(exc)

    def _candidate_key(
        self,
        op_name: str,
        device: TurboDevice,
        impl: str | None,
    ) -> tuple[str, str, str | None]:
        return (op_name, device.value, impl)

    def _candidate_impl(
        self,
        op_name: str,
        device: TurboDevice,
        impl: str | None = None,
    ) -> OpImpl:
        key = self._candidate_key(op_name, device, impl)
        cached = self._candidate_impl_cache.get(key)
        if cached is not None:
            return cached.impl

        if impl is not None:
            spec = self.registry.get_by_impl_name(impl)
            if spec.op_name != op_name:
                raise RuntimeError(
                    f"Implementation {impl!r} is registered for op "
                    f"{spec.op_name!r}, not {op_name!r}."
                )
            if spec.devices and device not in spec.devices:
                raise RuntimeError(
                    f"Implementation {impl!r} for op {op_name!r} does not support "
                    f"device={device.value}."
                )
            specs = [spec]
        else:
            def matches(spec: OpSpec) -> bool:
                if spec.op_name != op_name:
                    return False
                if spec.devices and device not in spec.devices:
                    return False
                if self._default_policy == OpPolicy.NATIVE:
                    return spec.kind == OpImplKind.NATIVE
                if self._default_policy == OpPolicy.CUSTOM:
                    return spec.kind == OpImplKind.CUSTOM
                return True

            specs = self.registry.list(spec_filter=matches)

        unavailable: list[str] = []
        for spec in specs:
            impl_obj = self._load_impl(spec)
            if not impl_obj.available():
                unavailable.append(spec.impl_name)
                continue
            self._candidate_impl_cache[key] = _Candidate(
                impl=impl_obj,
                spec=spec,
                fallback=bool(unavailable),
            )
            return impl_obj

        detail_msg = f" (unavailable={unavailable})" if unavailable else ""
        selected_policy = (
            f"policy={self._default_policy.value}"
            if impl is None
            else f"impl={impl}"
        )
        raise RuntimeError(
            f"No implementation for op={op_name!r}, device={device.value}, "
            f"{selected_policy}{detail_msg}."
        )

    def _load_impl(self, spec: OpSpec) -> OpImpl:
        impl = self._impls.get(spec.impl_name)
        if impl is not None:
            return impl

        impl_cls = self._impl_classes.get(spec.impl_name)
        if impl_cls is None:
            raise RuntimeError(
                f"No OpImpl class registered for impl_name {spec.impl_name!r}."
            )
        impl = impl_cls()
        self._impls[spec.impl_name] = impl
        return impl

    def log_ops_selection(self) -> None:
        """Log the implementations selected for this dispatcher."""
        default_candidates = {
            op_name: candidate
            for (op_name, _, impl_name), candidate in self._candidate_impl_cache.items()
            if impl_name is None
        }
        op_names = sorted(set(default_candidates) | set(self._unresolved_ops))
        rows: list[tuple[str, str, str, str, str]] = []
        for op_name in op_names:
            candidate = default_candidates.get(op_name)
            if candidate is None:
                rows.append((op_name, "-", "-", "-", "unavailable"))
                continue

            rows.append(
                (
                    op_name,
                    candidate.spec.impl_name,
                    candidate.spec.kind.value,
                    str(candidate.spec.priority),
                    "fallback" if candidate.fallback else "selected",
                )
            )

        if not rows:
            rows.append(("<no registered ops>", "-", "-", "-", "-"))

        headers = ("Op", "Implementation", "Kind", "Priority", "Status")
        widths = [
            max(len(header), *(len(row[index]) for row in rows)) + 2
            for index, header in enumerate(headers)
        ]
        header = " ".join(
            f"{column:<{width}}" for column, width in zip(headers, widths)
        ).rstrip()
        separator = "-" * len(header)
        lines = [separator, header, separator]
        lines.extend(
            " ".join(
                f"{value:<{width}}" for value, width in zip(row, widths)
            ).rstrip()
            for row in rows
        )
        lines.append(separator)

        unavailable = [
            f"  {op_name}: {self._unresolved_ops[op_name]}"
            for op_name in op_names
            if op_name in self._unresolved_ops
        ]
        if unavailable:
            lines.extend(["", "Unavailable details:", *unavailable])

        logger.info_rank0(
            "TorchTurbo ops selection (device=%s, policy=%s):\n%s",
            self._device_runtime.device.type,
            self._default_policy.value,
            "\n".join(lines),
        )

    def _dispatch(
        self,
        op_name: str,
        *args: Any,
        impl: str | None = None,
        **kwargs: Any,
    ):
        device = self._device_runtime.turbo_device
        impl_obj = self._candidate_impl(op_name, device, impl=impl)
        return impl_obj(*args, **kwargs)

    def rms_norm(
        self,
        hidden_states,
        weight,
        eps,
        *,
        impl: str | None = None,
    ):
        return self._dispatch(
            "rms_norm",
            hidden_states,
            weight,
            eps,
            impl=impl,
        )

    def grouped_matmul(
        self,
        x,
        weight,
        group_list,
        *,
        trans_b: bool = False,
        impl: str | None = None,
    ):
        return self._dispatch(
            "grouped_matmul",
            x,
            weight,
            group_list,
            trans_b=trans_b,
            impl=impl,
        )

    def gated_activation(
        self,
        x,
        *,
        activation_func,
        limit: float = 0.0,
        impl: str | None = None,
    ):
        return self._dispatch(
            "gated_activation",
            x,
            activation_func=activation_func,
            limit=limit,
            impl=impl,
        )

    def swiglu(
        self,
        x,
        *,
        impl: str | None = None,
    ):
        return self._dispatch(
            "swiglu",
            x,
            impl=impl,
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if not self.registry.list(spec_filter=lambda spec: spec.op_name == name):
            raise AttributeError(name)

        def facade(*args: Any, impl: str | None = None, **kwargs: Any):
            return self._dispatch(name, *args, impl=impl, **kwargs)

        return facade


__all__ = ["OpDispatcher"]
