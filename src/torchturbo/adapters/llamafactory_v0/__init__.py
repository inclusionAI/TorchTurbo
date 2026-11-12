"""LlamaFactory v0 adapter for torchturbo.

Two usage paths:

**One-line import** (zero changes to LlamaFactory code)::

    import torchturbo  # auto-detects LF v0, patches run_sft

**Explicit API** (full control)::

    from torchturbo.adapters.llamafactory_v0 import (
        build_config_from_lf_args,
        compute_sft_loss,
    )
    from torchturbo.engine import TorchTurboEngine

    The adapter creates the engine and patches LlamaFactory's native trainer;
    call that trainer's ``train()`` method to start the run.
"""

from __future__ import annotations

import sys
from typing import Any

from ...engine import TorchTurboEngine
from ...logging import get_logger
from ...registry import BaseAdapter, register_adapter
from .compat import is_llamafactory_v0
from .patches import (
    AcceleratorPatch,
    CreateRefModelPatch,
    DictFieldPatch,
    LoadModelPatch,
    LogCallbackPatch,
    PackedInputCollatorPatch,
    PretrainProcessorPatch,
    RegisterAutoclassPatch,
    RunExpPatch,
    TrainerClassPatch,
)
from .utils import build_config_from_lf_args, build_lora_config_from_lf_args, compute_sft_loss


logger = get_logger(__name__)

__all__ = [
    "LlamaFactoryV0Adapter",
    "build_config_from_lf_args",
    "build_lora_config_from_lf_args",
    "compute_sft_loss",
]


@register_adapter("llamafactory_v0", priority=10)
class LlamaFactoryV0Adapter(BaseAdapter):
    """Adapter for one-line import: auto-patches LlamaFactory v0.

    Patches are registered as ``BasePatch`` subclasses and managed
    by the ``BaseAdapter`` lifecycle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._active = False
        self.engine: TorchTurboEngine | None = None
        self.stage: str | None = None
        self.peft_config: Any | None = None

    def get_framework_version(self) -> str:
        """Return the LlamaFactory version string."""
        from .compat import get_llamafactory_version

        return get_llamafactory_version()

    def detect(self) -> bool:
        """Detect if LlamaFactory v0 is the framework being driven this process.
        """
        lf_loaded = any(
            key.startswith("llamafactory") and not key.startswith("llamafactory.v1")
            for key in sys.modules
        )
        if not lf_loaded:
            import importlib.util

            if importlib.util.find_spec("llamafactory") is None:
                return False
        return is_llamafactory_v0()

    def activate(self) -> None:
        """Register patches (engine is built later inside RunExpPatch)."""
        if self._active:
            return

        # Auto register LF templates/plugins for all torchturbo models
        self._register_lf_templates()

        # Register patches — inner patches applied by RunExpPatch
        self.add_patch(PackedInputCollatorPatch())
        self.add_patch(LoadModelPatch(adapter=self))
        self.add_patch(CreateRefModelPatch(adapter=self))
        self.add_patch(AcceleratorPatch())
        self.add_patch(TrainerClassPatch(adapter=self))
        self.add_patch(RegisterAutoclassPatch())
        self.add_patch(PretrainProcessorPatch())
        self.add_patch(LogCallbackPatch())

        # Fix HfArgumentParser: dict-typed CLI args like --lr_scheduler_kwargs
        # fail because argparse calls dict(cli_string).  Patch in __init__,
        # before get_train_args is ever called.
        self.add_patch(DictFieldPatch())

        # RunExpPatch is special: it intercepts run_sft, builds the
        # engine, then calls self.apply_patches(engine) for the rest
        run_exp_patch = RunExpPatch(adapter=self)
        run_exp_patch.patch()  # apply immediately (no engine needed yet)

        self._active = True
        logger.info_rank0("LlamaFactory v0 adapter activated (patches registered)")

    def deactivate(self) -> None:
        """Revert all patches."""
        if not self._active:
            return

        self.revert_patches()

        self.engine = None
        self._active = False
        logger.info_rank0("LlamaFactory v0 adapter deactivated.")

    def _register_lf_templates(self) -> None:
        """Auto-register LF templates/plugins and patches for all models.

        Scans model directories for ``lf_template.py`` modules with a
        ``register()`` function.  Each model that wants LF integration
        provides its own ``lf_template.py`` — no hardcoding here.
        """
        import importlib
        from ...models.registry import get_all_models

        for model_name in get_all_models():
            module_path = f"torchturbo.models.{model_name}.lf_template"
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, "get_patches"):
                    for patch in mod.get_patches():
                        self.add_patch(patch)
                if hasattr(mod, "register"):
                    mod.register()
                    logger.debug("Registered LF template for model %r", model_name)
            except ImportError:
                pass  # model doesn't provide LF integration
            except Exception:
                logger.debug(
                    "Failed to register LF template for %r", model_name, exc_info=True
                )
