"""torchturbo — optimized distributed training engine.

Importing torchturbo discovers the built-in adapters and backends (importing the
package side-effects nothing beyond registration). Adapter **activation is
explicit**: callers opt into a framework by naming it::

    import torchturbo
    torchturbo.activate("llamafactory_v0")   # or "msswift"

This avoids the wrong adapter activating in a dev image that has both
LlamaFactory and ms-swift installed — availability-based detection can't tell
which one a process is about to run, because torchturbo is imported *before*
the framework is loaded. Each launcher opts in unambiguously:
``scripts/runs/run_lf.py`` calls ``activate("llamafactory_v0")``; the swift
launchers (``run_swift.py`` / the offline tokenizers) call
``activate("msswift")``.

Extra engine config via env vars::

    TORCH_TURBO_PREFETCH_STRATEGY=auto TORCH_TURBO_FORWARD_PREFETCH_DEPTH=2 \\
        llamafactory-cli train config.yaml
"""

from __future__ import annotations

import os

from .logging import get_logger
from .detector import FrameworkDetector
from .engine import (
    BaseBackend,
    EngineConfig,
    FSDP2Backend,
    FSDPConfig,
    FreezePolicy,
    LossChunkConfig,
    MoEConfig,
    MTPConfig,
    MixedPrecisionPolicy,
    OpsConfig,
    ParameterGroup,
    TorchTurboEngine,
    get_available_backends,
    register_backend,
)
from .registry import (
    BaseAdapter,
    BasePatch,
    get_adapter,
    get_all_adapters,
    register_adapter,
)
from .utils.transformers_utils import patch_dynamic_module_loading


__version__ = "0.1.0"
__all__ = [
    "BaseAdapter",
    "BasePatch",
    "BaseBackend",
    "EngineConfig",
    "FSDP2Backend",
    "FrameworkDetector",
    "FSDPConfig",
    "FreezePolicy",
    "LossChunkConfig",
    "MoEConfig",
    "MTPConfig",
    "MixedPrecisionPolicy",
    "OpsConfig",
    "ParameterGroup",
    "TorchTurboEngine",
    "activate",
    "detect",
    "detect_all",
    "get_available_backends",
    "get_logger",
    "register_adapter",
    "register_backend",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Auto-discovery and activation
# ---------------------------------------------------------------------------


def _discover_builtin_adapters() -> None:
    """Import built-in adapter packages so they auto-register.

    Only the immediate children of :mod:`torchturbo.adapters` are imported —
    each adapter registers itself via ``@register_adapter`` in its package
    ``__init__``. Framework-coupled submodules (e.g. the swift-base subclasses
    in ``msswift.pipeline`` / ``loader`` / ``args``, which must live at module
    top level for dill pickling) are imported lazily by the adapter's
    ``activate()``. This keeps the swift and llamafactory adapters isolated:
    ``import torchturbo`` (and this discovery sweep) never drags in a framework
    that isn't being used, so running one framework emits no import warnings
    about — and requires — neither the other framework nor its adapter modules.

    Depth-1 ``iter_modules`` is used instead of recursive ``walk_packages`` so
    those framework-coupled submodules are not eagerly imported here.
    """
    import importlib
    import pkgutil

    from . import adapters as adapters_pkg

    for _importer, module_name, _is_pkg in pkgutil.iter_modules(
        adapters_pkg.__path__,
        prefix=adapters_pkg.__name__ + ".",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logger.warning_rank0("Failed to import adapter %s", module_name, exc_info=True)


def detect() -> str | None:
    """Detect which training framework is active."""
    return FrameworkDetector.detect()


def detect_all() -> list[str]:
    """Detect all active training frameworks."""
    return FrameworkDetector.detect_all()


def activate(name: str | None = None) -> BaseAdapter | None:
    """Activate one adapter.

    Adapter activation is **explicit** — ``import torchturbo`` no longer
    activates anything. Callers opt in by naming the framework (see module
    docstring). Idempotent: re-activating an already-active adapter is a no-op.

    * ``name`` given (the supported path): activate that adapter. Raises
      ``ValueError`` if no adapter is registered under *name*. This is what
      launchers use — ``run_lf.py`` calls ``activate("llamafactory_v0")``; the
      swift launchers call ``activate("msswift")``.
    * ``name`` ``None`` (convenience, not the recommended path): activate only
      when detection is unambiguous — exactly one adapter reports its framework
      present. In a multi-framework install this refuses to guess and returns
      ``None``; name the adapter explicitly instead.

    Activating an adapter also patches Transformers remote-code loading, so
    launchers must activate before loading models or processors.
    """
    if name is not None:
        adapter = get_adapter(name)
        if adapter is None:
            raise ValueError(f"No adapter registered for {name!r}")
        logger.info_rank0("Activating adapter for %r", name)
        patch_dynamic_module_loading()
        adapter.activate()
        return adapter

    if os.environ.get("TORCH_TURBO_DISABLE", "").lower() in ("1", "true", "yes"):
        logger.info_rank0("torchturbo disabled via TORCH_TURBO_DISABLE env var.")
        return None

    detected = FrameworkDetector.detect_all()
    if len(detected) == 1:
        framework = detected[0]
        adapter = get_adapter(framework)
        if adapter is None:
            return None
        logger.info_rank0("Auto-activating adapter for %r", framework)
        patch_dynamic_module_loading()
        adapter.activate()
        return adapter

    if len(detected) > 1:
        # Both frameworks installed: availability can't tell which one this
        # process will run, and torchturbo is imported before the framework is
        # loaded, so there is no reliable automatic signal. Stay inactive and
        # let the launcher activate its own adapter explicitly.
        logger.info_rank0(
            "Multiple frameworks detected (%s); skipping auto-activation — "
            "activate the intended adapter explicitly (e.g. via the launcher).",
            ", ".join(detected),
        )
    return None


# Discovery at import time, activation is explicit (see ``activate``).
_discover_builtin_adapters()

# Register built-in models
from . import models as _models_pkg  # noqa: F401 - trggers _register_builtins()
