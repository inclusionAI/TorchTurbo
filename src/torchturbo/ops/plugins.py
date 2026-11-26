"""Explicit descriptions and adapters for external Python operators."""

from __future__ import annotations

import importlib
import inspect
import keyword
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_leaves

from torchturbo.engine.plugins import OpPlugin, Plugin, PluginConsumer
from torchturbo.runtime import TurboDevice

from .base import OpImpl, OpImplKind, OpSpec


def _reference_parts(reference: str) -> tuple[str, str]:
    if not isinstance(reference, str):
        raise TypeError("Plugin references must be module:attribute strings.")
    parts = reference.split(":")
    if (
        len(parts) != 2
        or not all(part.isidentifier() and not keyword.iskeyword(part)
                   for part in parts[0].split("."))
        or not parts[1].isidentifier()
        or keyword.iskeyword(parts[1])
    ):
        raise ValueError(f"Expected module:attribute reference, got {reference!r}.")
    return parts[0], parts[1]


def _resolve(reference: str) -> Any:
    module_name, attribute = _reference_parts(reference)
    module = importlib.import_module(module_name)
    try:
        return vars(module)[attribute]
    except KeyError as exc:
        raise ValueError(f"Reference {reference!r} has no attribute {attribute!r}.") from exc


@dataclass(frozen=True, slots=True)
class OpBinding:
    """Bind a logical operator to an external callable and optional backward."""

    op_name: str
    impl_name: str
    target: str
    devices: tuple[TurboDevice, ...]
    kind: OpImplKind = OpImplKind.CUSTOM
    priority: int = 0
    availability: str | None = None
    backward: str | None = None
    setup_context: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.op_name, str)
            or not self.op_name.isidentifier()
            or keyword.iskeyword(self.op_name)
            or self.op_name.startswith("_")
        ):
            raise ValueError("OpBinding.op_name must be a public Python identifier.")
        if not isinstance(self.impl_name, str) or not self.impl_name:
            raise ValueError("OpBinding.impl_name must be a non-empty string.")
        _reference_parts(self.target)
        for reference in (self.availability, self.backward, self.setup_context):
            if reference is not None:
                _reference_parts(reference)
        if self.setup_context is not None and self.backward is None:
            raise ValueError("setup_context requires an explicit backward binding.")
        devices = tuple(self.devices)
        if any(not isinstance(device, TurboDevice) for device in devices):
            raise TypeError("OpBinding.devices must contain TurboDevice values.")
        object.__setattr__(self, "devices", devices)
        spec = self.to_spec()
        object.__setattr__(self, "priority", spec.priority)

    def to_spec(self) -> OpSpec:
        """Return the existing dispatcher metadata for this binding."""
        return OpSpec(
            op_name=self.op_name, impl_name=self.impl_name,
            kind=self.kind, priority=self.priority, devices=self.devices,
        )


@dataclass(frozen=True, slots=True)
class OpPluginSpec:
    """Immutable operator configuration collected from an Engine plugin."""

    name: str
    bindings: tuple[OpBinding, ...]


class OpPluginConsumer(PluginConsumer):
    """Collect operator bindings for dispatcher construction."""

    def __init__(self) -> None:
        self._specs: dict[str, OpPluginSpec] = {}

    @property
    def specs(self) -> tuple[OpPluginSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def matches(self, plugin: Plugin) -> bool:
        return isinstance(plugin, OpPlugin)

    def consume(self, plugin: Plugin) -> None:
        if not isinstance(plugin, OpPlugin):
            raise TypeError("Expected an OpPlugin.")
        name = type(plugin).plugin_name
        if not isinstance(name, str) or not name or "::" in name:
            raise ValueError("Operator plugin name must be non-empty and cannot contain '::'.")
        if name in self._specs:
            raise ValueError(f"Duplicate plugin ID {name!r}.")
        bindings = tuple(plugin.get_bindings())
        for binding in bindings:
            if not isinstance(binding, OpBinding):
                raise TypeError("get_bindings() must return OpBinding values.")
            prefix = f"{name}::"
            if not binding.impl_name.startswith(prefix) or binding.impl_name == prefix:
                raise ValueError(f"impl_name must use the {prefix!r} namespace.")
        self._specs[name] = OpPluginSpec(name, bindings)


def normalize_plugin_specs(specs: Sequence[OpPluginSpec]) -> tuple[OpPluginSpec, ...]:
    """Freeze the collected operator configuration for singleton comparison."""
    if any(not isinstance(spec, OpPluginSpec) for spec in specs):
        raise TypeError("plugin_specs must contain OpPluginSpec values from the consumer.")
    result = tuple(sorted(specs, key=lambda spec: spec.name))
    if len({spec.name for spec in result}) != len(result):
        raise ValueError("Duplicate operator plugin IDs.")
    return result


class CallableOpImpl(OpImpl):
    """Adapt a plugin binding while retaining its autograd execution path."""

    def __init__(self, plugin: OpPluginSpec, binding: OpBinding) -> None:
        self.binding = binding
        self.source = (
            f"plugin={plugin.name!r}, op_name={binding.op_name!r}, "
            f"impl_name={binding.impl_name!r}, target={binding.target!r}"
        )
        self.unavailable_reason: str | None = None
        self._checked = False
        self._target: Callable[..., Any] | None = None
        self._function: type[torch.autograd.Function] | None = None
        self._signature: inspect.Signature | None = None
        self.missing_dependency: ModuleNotFoundError | None = None

    def _callable(self, reference: str) -> Callable[..., Any]:
        target = _resolve(reference)
        if isinstance(target, type) or not callable(target):
            raise TypeError(f"{self.source}: {reference!r} must reference a callable, not a class.")
        return target

    def available(self) -> bool:
        if self._checked:
            return self.unavailable_reason is None
        try:
            if self.binding.availability is not None:
                available = self._callable(self.binding.availability)()
                if type(available) is not bool:
                    raise TypeError("Plugin availability must return bool.")
                if not available:
                    self.unavailable_reason = f"{self.source}: availability returned False"
                    self._checked = True
                    return False
            self._target = self._callable(self.binding.target)
            if self.binding.backward is not None:
                backward = self._callable(self.binding.backward)
                setup = (
                    self._callable(self.binding.setup_context)
                    if self.binding.setup_context is not None else None
                )
                self._function = self._wrap_backward(self._target, backward, setup)
        except ModuleNotFoundError as exc:
            self.missing_dependency = exc
            self.unavailable_reason = f"{self.source}: missing dependency {exc.name!r}: {exc}"
        except TypeError as exc:
            raise TypeError(f"Failed loading {self.source}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed loading {self.source}: {exc}") from exc
        self._checked = True
        return self.unavailable_reason is None

    def _wrap_backward(
        self,
        target: Callable[..., Any],
        backward_fn: Callable[..., Any],
        setup: Callable[..., Any] | None,
    ) -> type[torch.autograd.Function]:
        source = self.source
        signature = inspect.signature(target)
        if any(
            parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in signature.parameters.values()
        ):
            raise TypeError("Explicit backward targets need a fixed signature; use a wrapper for *args/**kwargs.")
        self._signature = signature

        class PluginFunction(torch.autograd.Function):
            @staticmethod
            def forward(*args: Any) -> Any:
                arguments = inspect.BoundArguments(signature, dict(zip(signature.parameters, args)))
                output = target(*arguments.args, **arguments.kwargs)
                if not (
                    isinstance(output, torch.Tensor)
                    or (isinstance(output, tuple) and all(
                        value is None or isinstance(value, torch.Tensor) for value in output
                    ))
                ):
                    raise TypeError("Explicit backward targets must return a Tensor or a flat tuple of Tensors/None.")
                return output

            @staticmethod
            def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
                if setup is not None:
                    setup(ctx, inputs, output)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                try:
                    return backward_fn(ctx, *grad_outputs)
                except Exception as exc:
                    exc.add_note(f"Backward for {source}.")
                    raise

        return PluginFunction

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            if self._target is None:
                raise RuntimeError(f"{self.source}: target has not been resolved.")
            if self._function is not None:
                assert self._signature is not None
                arguments = self._signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                values = tuple(arguments.arguments.values())
                for value in values:
                    if not isinstance(value, torch.Tensor) and any(
                        isinstance(leaf, torch.Tensor) for leaf in tree_leaves(value)
                    ):
                        raise TypeError("Explicit backward does not accept nested tensor inputs; flatten them in a wrapper.")
                return self._function.apply(*values)
            result = self._target(*args, **kwargs)
            if torch.is_grad_enabled():
                inputs = tree_leaves((args, kwargs))
                outputs = [item for item in tree_leaves(result)
                           if isinstance(item, torch.Tensor) and (item.is_floating_point() or item.is_complex())]
                if (
                    any(isinstance(item, torch.Tensor) and item.requires_grad for item in inputs)
                    and outputs
                    and not any(item.requires_grad for item in outputs)
                ):
                    raise RuntimeError(f"{self.source}: output is disconnected from autograd; provide backward.")
            return result
        except Exception as exc:
            exc.add_note(f"Calling {self.source}.")
            raise


__all__ = ["OpBinding", "OpPluginConsumer"]
