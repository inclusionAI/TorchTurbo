"""Compatibility patches for Transformers remote-code loading."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

_patched_get_class_in_module: Callable[..., type] | None = None


def patch_dynamic_module_loading() -> None:
    """Bind remote-code modules to their parent packages."""

    global _patched_get_class_in_module
    from transformers import dynamic_module_utils

    original = dynamic_module_utils.get_class_in_module
    if original is _patched_get_class_in_module:
        return

    @wraps(original)
    def _get_class_in_module(*args: Any, **kwargs: Any) -> type:
        loaded_class = original(*args, **kwargs)
        prefix = f"{dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME}."
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith(prefix):
                continue
            parent_name, _, name = module_name.rpartition(".")
            parent = importlib.import_module(parent_name)
            setattr(parent, name, module)
        return loaded_class

    _patched_get_class_in_module = _get_class_in_module
    dynamic_module_utils.get_class_in_module = _get_class_in_module
