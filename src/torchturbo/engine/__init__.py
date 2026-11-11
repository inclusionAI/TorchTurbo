"""TorchTurbo engine configuration, backends, and public facade."""

# Import backends first so they auto-register
from . import backends  # noqa: F401
from .args import (
    TorchTurboArgs,
    get_torchturbo_args,
    parse_torchturbo_args,
    set_torchturbo_args,
    string_to_bool,
)
from .backend_registry import get_available_backends, get_backend, register_backend
from .backends.fsdp2 import FSDP2Backend, FSDPConfig
from .backends.base import BaseBackend
from .config import (
    ActivationCheckpointingConfig,
    CheckpointConfig,
    CommConfig,
    DistributedBackend,
    EngineConfig,
    FreezePolicy,
    MixedPrecisionPolicy,
    MfuConfig,
    MTPConfig,
    OpsConfig,
    ParameterGroup,
    build_from_args,
)
from .core import TorchTurboEngine
from .distributed import init_process_group_if_needed


__all__ = [
    # Shared EngineConfig builder (lf + msswift + ...)
    "build_from_args",
    # CLI args
    "TorchTurboArgs",
    "get_torchturbo_args",
    "parse_torchturbo_args",
    "set_torchturbo_args",
    "string_to_bool",
    # Common config
    "ActivationCheckpointingConfig",
    "CheckpointConfig",
    "CommConfig",
    "DistributedBackend",
    "EngineConfig",
    "FreezePolicy",
    "MixedPrecisionPolicy",
    "MfuConfig",
    "MTPConfig",
    "OpsConfig",
    "ParameterGroup",
    # Engine
    "BaseBackend",
    "FSDP2Backend",
    "FSDPConfig",
    # Registry
    "get_available_backends",
    "get_backend",
    "register_backend",
    # Distributed
    "init_process_group_if_needed",
]
