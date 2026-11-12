from __future__ import annotations

from collections.abc import Mapping

from torchturbo.runtime import DeviceRuntime

from .base import OpPolicy
from .dispatcher import OpDispatcher
from .registry import registry


_OPS: OpDispatcher | None = None


def get_or_create_ops(
    *,
    device_runtime: DeviceRuntime,
    default_policy: OpPolicy = OpPolicy.AUTO,
    overrides: Mapping[str, str] | None = None,
) -> OpDispatcher:
    """Return or create the process-wide dispatcher.

    Reuse requires matching device, policy, and per-op overrides.
    """
    global _OPS

    if not isinstance(default_policy, OpPolicy):
        raise TypeError("default_policy must be an OpPolicy.")
    requested_overrides = dict(overrides or {})

    if _OPS is None:
        _OPS = OpDispatcher(
            device_runtime=device_runtime,
            registry=registry,
            default_policy=default_policy,
            overrides=requested_overrides,
        )
        return _OPS

    conflicts: list[str] = []
    if _OPS.device_runtime.turbo_device != device_runtime.turbo_device:
        conflicts.append(
            f"device={_OPS.device_runtime.turbo_device.value!r} vs "
            f"{device_runtime.turbo_device.value!r}"
        )
    if _OPS.default_policy != default_policy:
        conflicts.append(
            f"policy={_OPS.default_policy.value!r} vs {default_policy.value!r}"
        )
    if _OPS.overrides != requested_overrides:
        conflicts.append(
            f"overrides={_OPS.overrides!r} vs {requested_overrides!r}"
        )
    if conflicts:
        raise RuntimeError(
            "Ops dispatcher is already configured for this process with an "
            "incompatible configuration (" + ", ".join(conflicts) + ")."
        )

    return _OPS


def get_ops() -> OpDispatcher:
    if _OPS is None:
        raise RuntimeError(
            "Ops dispatcher is not configured. Initialize TorchTurboEngine or "
            "call get_or_create_ops()."
        )
    return _OPS


class _OpsProxy:
    def __getattr__(self, name: str):
        return getattr(get_ops(), name)

    def __repr__(self) -> str:
        return "<ops proxy>"


ops = _OpsProxy()


__all__ = ["get_or_create_ops", "get_ops", "ops"]
