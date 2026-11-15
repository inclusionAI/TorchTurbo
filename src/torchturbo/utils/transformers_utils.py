"""Compatibility patches for Transformers remote-code loading."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

_patched_get_class_in_module: Callable[..., type] | None = None
_patched_compute_local_source_files_hash: Callable[..., str] | None = None


def patch_dynamic_module_loading() -> None:
    """Bind remote-code modules to parent packages and mirror cache files atomically."""
    global _patched_compute_local_source_files_hash, _patched_get_class_in_module
    from transformers import dynamic_module_utils

    original = dynamic_module_utils.get_class_in_module
    if original is not _patched_get_class_in_module:
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

    original_hash = dynamic_module_utils._compute_local_source_files_hash
    if original_hash is _patched_compute_local_source_files_hash:
        return

    @wraps(original_hash)
    def _compute_local_source_files_hash(*args: Any, **kwargs: Any) -> str:
        local_hash = original_hash(*args, **kwargs)
        module_file = kwargs.get("resolved_module_file") or kwargs.get("module_file")
        if module_file is None:
            for value in reversed(args):
                if isinstance(value, (str, os.PathLike)) and str(value).endswith(".py"):
                    module_file = value
                    break
        if module_file is None:
            return local_hash

        module_path = Path(module_file)
        source_files = sorted(module_path.parent.glob("*.py"))
        if source_files:
            module_root = dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME
            model_key = dynamic_module_utils._sanitize_module_name(
                module_path.parent.name
            )
            module_dir = Path(module_root) / model_key / local_hash
            dynamic_module_utils.create_dynamic_module(module_dir)
            target_dir = Path(dynamic_module_utils.HF_MODULES_CACHE) / module_dir
            for source_file in source_files:
                target_file = target_dir / source_file.name
                if target_file.is_file():
                    continue
                with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as handle:
                    tmp_path = Path(handle.name)
                try:
                    shutil.copyfile(source_file, tmp_path)
                    os.replace(tmp_path, target_file)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
        return local_hash

    _patched_compute_local_source_files_hash = _compute_local_source_files_hash
    dynamic_module_utils._compute_local_source_files_hash = _compute_local_source_files_hash
