"""Compatibility patches for Transformers remote-code loading."""

from __future__ import annotations

import filecmp
import hashlib
import importlib
import os
import shutil
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from packaging.version import Version

_patched_get_class_in_module: Callable[..., type] | None = None
_patched_get_cached_module_file: Callable[..., str] | None = None


def patch_dynamic_module_loading() -> None:
    """Bind remote-code modules to their parent packages as Transformers loads them.

    Transformers inserts remote-code modules into ``sys.modules`` without
    binding them to their parent packages. Dill needs that binding to locate
    classes; otherwise it copies them and can fail rebuilding TypedDict bases
    on Python >= 3.11 when multiprocess dataset workers load their tasks.
    Call before loading remote code. Repeated calls do not wrap an unchanged
    loader again. Each wrapper retains its original loader so other wrappers
    can be layered safely.

    Backport Transformers #46022 for versions < 5.10.1 to copy and hash transitive relative imports.
    """
    global _patched_get_class_in_module, _patched_get_cached_module_file
    from transformers import __version__, dynamic_module_utils

    if Version(__version__) < Version("5.10.1"):
        original_cached_loader = dynamic_module_utils.get_cached_module_file
        if original_cached_loader is not _patched_get_cached_module_file:
            # Backport local dependency copying and hashing from Transformers #46022.
            @wraps(original_cached_loader)
            def _get_cached_module_file(
                pretrained_model_name_or_path: str | os.PathLike,
                module_file: str,
                *args: Any,
                **kwargs: Any,
            ) -> str:
                model_path = Path(pretrained_model_name_or_path)
                if not model_path.is_dir():
                    return original_cached_loader(
                        pretrained_model_name_or_path, module_file, *args, **kwargs
                    )

                resolved_module_file = model_path / module_file
                dynamic_module_utils.check_imports(resolved_module_file)
                source_files = [resolved_module_file] + list(
                    map(Path, dynamic_module_utils.get_relative_import_files(resolved_module_file))
                )
                relative_files = sorted(
                    (source.relative_to(model_path).as_posix(), source) for source in source_files
                )
                source_hash = hashlib.sha256()
                for relative_path, source in relative_files:
                    source_hash.update(relative_path.encode("utf-8"))
                    source_hash.update(source.read_bytes())

                local_name = dynamic_module_utils._sanitize_module_name(model_path.name)
                submodule = (
                    Path(dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME)
                    / local_name
                    / source_hash.hexdigest()[:16]
                )
                dynamic_module_utils.create_dynamic_module(submodule)
                cache_path = Path(dynamic_module_utils.HF_MODULES_CACHE) / submodule
                for relative_path, source in relative_files:
                    target = cache_path / relative_path
                    if not target.exists() or not filecmp.cmp(source, target, shallow=False):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        importlib.invalidate_caches()
                return str(submodule / module_file)

            _patched_get_cached_module_file = _get_cached_module_file
            dynamic_module_utils.get_cached_module_file = _get_cached_module_file

    original = dynamic_module_utils.get_class_in_module
    if original is _patched_get_class_in_module:
        return

    @wraps(original)
    def _get_class_in_module(*args: Any, **kwargs: Any) -> type:
        loaded_class = original(*args, **kwargs)
        prefix = dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME + "."
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith(prefix):
                continue
            parent_name, _, name = module_name.rpartition(".")
            parent = importlib.import_module(parent_name)
            setattr(parent, name, module)
        return loaded_class

    _patched_get_class_in_module = _get_class_in_module
    dynamic_module_utils.get_class_in_module = _get_class_in_module
