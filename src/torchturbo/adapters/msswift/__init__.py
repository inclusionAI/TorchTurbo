"""ms-swift adapter"""

from __future__ import annotations

from ...logging import get_logger
from ...registry import BaseAdapter, register_adapter
from .compat import get_swift_version, is_swift_supported

logger = get_logger(__name__)

__all__ = ["MSSwiftAdapter"]


@register_adapter("msswift", priority=20)
class MSSwiftAdapter(BaseAdapter):
    """Wires torchturbo into ms-swift through swift's native extension points."""

    framework_name = "msswift"

    def __init__(self) -> None:
        super().__init__()
        self._active = False

    def get_framework_version(self) -> str:
        return get_swift_version()

    def detect(self) -> bool:
        return is_swift_supported()

    def activate(self) -> None:
        """Register turbo models/trainers + swap stage pipeline globals (idempotent)."""
        if self._active:
            return
        import importlib

        from .model_registry import register_turbo_models_to_swift, register_turbo_trainers_to_swift
        from .stages import iter_stages, stage_wiring

        register_turbo_models_to_swift()
        register_turbo_trainers_to_swift()

        # Swap each stage's pipeline module global to the turbo class so native <stage>_main
        # builds it; a missing swift submodule (e.g. an absent RLHF stage) is skipped.
        for _spec in iter_stages():
            try:
                _mod = importlib.import_module(_spec.pipeline_module)
            except Exception:
                continue
            setattr(_mod, _spec.pipeline_attr, stage_wiring(_spec.name)[0])

        self._active = True
        logger.info_rank0("ms-swift adapter activated (turbo models/trainers registered)")

    def deactivate(self) -> None:
        from . import engine_registry

        engine_registry.clear()
        self._active = False
        logger.info_rank0("ms-swift adapter deactivated")
