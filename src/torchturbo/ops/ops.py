from __future__ import annotations

from torchturbo.runtime import DeviceRuntime

from .base import OpPolicy
from .dispatcher import OpDispatcher
from .registry import registry


_OPS: OpDispatcher | None = None


def get_or_create_ops(
    *,
    device_runtime: DeviceRuntime,
    default_policy: OpPolicy = OpPolicy.AUTO,
) -> OpDispatcher:
    """Return the compatible process-wide dispatcher or create it once.

    A process-wide dispatcher is safe to share between engines only when its
    device and policy match. Conflicting engine configurations are rejected
    instead of silently changing the ops used by existing models.
    """
    global _OPS

    if not isinstance(default_policy, OpPolicy):
        raise TypeError("default_policy must be an OpPolicy.")

    if _OPS is None:
        _OPS = OpDispatcher(
            device_runtime=device_runtime,
            registry=registry,
            default_policy=default_policy,
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
