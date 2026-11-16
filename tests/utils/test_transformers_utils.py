"""Regression tests for dynamic Transformers module imports."""

import importlib
import multiprocessing
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

dill = pytest.importorskip("dill")
pytest.importorskip("transformers")
from transformers import dynamic_module_utils

from torchturbo.utils import transformers_utils
from torchturbo.utils.transformers_utils import patch_dynamic_module_loading


@pytest.fixture(autouse=True)
def patched_loading(monkeypatch):
    # Restore the loader and installation state after each test.
    monkeypatch.setattr(
        dynamic_module_utils,
        "get_class_in_module",
        dynamic_module_utils.get_class_in_module,
    )
    monkeypatch.setattr(
        transformers_utils,
        "_patched_get_class_in_module",
        transformers_utils._patched_get_class_in_module,
    )
    monkeypatch.setattr(
        transformers_utils,
        "_patched_compute_local_source_files_hash",
        transformers_utils._patched_compute_local_source_files_hash,
    )
    patch_dynamic_module_loading()


@pytest.fixture
def checkpoint(tmp_path):
    checkpoint = tmp_path / f"remote_code_{uuid4().hex}"
    checkpoint.mkdir()
    (checkpoint / "processing.py").write_text(
        "from typing import TypedDict\n\n"
        "class ProcessorKwargs(TypedDict):\n    size: int\n\n"
        "class Processor:\n    valid_kwargs = ProcessorKwargs\n"
    )
    yield checkpoint
    package = dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME
    prefix = f"{package}.{checkpoint.name}"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]
    parent = sys.modules.get(package)
    if parent is not None:
        vars(parent).pop(checkpoint.name, None)
    shutil.rmtree(
        Path(dynamic_module_utils.HF_MODULES_CACHE) / package / checkpoint.name,
        ignore_errors=True,
    )


def _load_processor(checkpoint):
    return dynamic_module_utils.get_class_from_dynamic_module(
        "processing.Processor", checkpoint, local_files_only=True
    )


def _cached_module_dir(checkpoint: Path, local_hash: str) -> Path:
    return (
        Path(dynamic_module_utils.HF_MODULES_CACHE)
        / dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME
        / dynamic_module_utils._sanitize_module_name(checkpoint.name)
        / local_hash
    )


def _unpickle_processor(payload, module_name):
    was_loaded = module_name in sys.modules
    processor = dill.loads(payload)
    module = importlib.import_module(module_name)
    return (
        was_loaded,
        type(processor) is module.Processor,
        processor.valid_kwargs is module.ProcessorKwargs,
    )


def test_dynamic_module_preserves_dill_identity(checkpoint):
    processor_class = _load_processor(checkpoint)
    assert dill.loads(dill.dumps(processor_class)) is processor_class
    assert type(dill.loads(dill.dumps(processor_class()))) is processor_class

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        result = pool.apply_async(
            _unpickle_processor,
            (dill.dumps(processor_class()), processor_class.__module__),
        )
        assert result.get(timeout=30) == (False, True, True)


def test_cached_and_forced_loads_update_parent_binding(checkpoint):
    processor_class = _load_processor(checkpoint)
    assert _load_processor(checkpoint) is processor_class
    module_name = processor_class.__module__
    parent_name, _, name = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    assert vars(parent)[name] is sys.modules[module_name]

    reloaded = dynamic_module_utils.get_class_in_module(
        "Processor", Path(*module_name.split(".")).with_suffix(".py"),
        force_reload=True,
    )
    assert reloaded is not processor_class
    assert vars(parent)[name] is sys.modules[module_name]
    assert dill.loads(dill.dumps(reloaded)) is reloaded


@pytest.mark.parametrize("preload_component", [False, True])
def test_reexported_class_binds_both_modules(checkpoint, preload_component):
    source = checkpoint / "processing.py"
    (checkpoint / "component.py").write_text(source.read_text())
    source.write_text("from .component import Processor\n")
    if preload_component:
        dynamic_module_utils.get_class_from_dynamic_module(
            "component.Processor", checkpoint, local_files_only=True
        )
    processor_class = _load_processor(checkpoint)
    parent_name, _, name = processor_class.__module__.rpartition(".")
    assert name == "component"
    parent = importlib.import_module(parent_name)
    assert vars(parent)["processing"] is sys.modules[f"{parent_name}.processing"]
    assert vars(parent)["component"] is sys.modules[f"{parent_name}.component"]
    assert dill.loads(dill.dumps(processor_class)) is processor_class


def test_repeated_patching_wraps_once(checkpoint):
    installed = dynamic_module_utils.get_class_in_module
    patch_dynamic_module_loading()
    patch_dynamic_module_loading()
    assert dynamic_module_utils.get_class_in_module is installed
    processor_class = _load_processor(checkpoint)
    assert dill.loads(dill.dumps(processor_class)) is processor_class


def test_hash_patch_mirrors_local_remote_code(checkpoint, monkeypatch):
    (checkpoint / "component.py").write_text("from .video import HELPER\n")
    (checkpoint / "video.py").write_text("HELPER = 1\n")

    def fake_hash(*args, **kwargs):
        return "0123456789abcdef"

    monkeypatch.setattr(dynamic_module_utils, "_compute_local_source_files_hash", fake_hash)
    monkeypatch.setattr(transformers_utils, "_patched_compute_local_source_files_hash", None)
    patch_dynamic_module_loading()

    local_hash = dynamic_module_utils._compute_local_source_files_hash(
        resolved_module_file=checkpoint / "processing.py",
        modules_needed=[checkpoint / "component.py"],
    )

    target_dir = _cached_module_dir(checkpoint, local_hash)
    assert (target_dir / "processing.py").is_file()
    assert (target_dir / "component.py").is_file()
    assert (target_dir / "video.py").is_file()


def test_hash_patch_keeps_existing_cached_files(checkpoint, monkeypatch):
    cached = checkpoint / "processing.py"

    def fake_hash(*args, **kwargs):
        return "fedcba9876543210"

    monkeypatch.setattr(dynamic_module_utils, "_compute_local_source_files_hash", fake_hash)
    monkeypatch.setattr(transformers_utils, "_patched_compute_local_source_files_hash", None)
    patch_dynamic_module_loading()

    local_hash = dynamic_module_utils._compute_local_source_files_hash(
        resolved_module_file=cached,
        modules_needed=[],
    )
    target_dir = _cached_module_dir(checkpoint, local_hash)
    target_file = target_dir / "processing.py"
    sentinel = "# cached copy must stay untouched\n"
    target_file.write_text(sentinel)

    dynamic_module_utils._compute_local_source_files_hash(
        resolved_module_file=cached,
        modules_needed=[],
    )

    assert target_file.read_text() == sentinel


def test_repatching_preserves_outer_and_cached_wrappers(checkpoint, monkeypatch):
    installed = dynamic_module_utils.get_class_in_module
    outer = Mock(wraps=installed)

    with monkeypatch.context() as context:
        context.setattr(dynamic_module_utils, "get_class_in_module", outer)
        patch_dynamic_module_loading()
        repatched = dynamic_module_utils.get_class_in_module
        patch_dynamic_module_loading()
        assert dynamic_module_utils.get_class_in_module is repatched

        processor_class = _load_processor(checkpoint)
        outer.assert_called_once()
        assert dill.loads(dill.dumps(processor_class)) is processor_class

    assert dynamic_module_utils.get_class_in_module is installed
    assert _load_processor(checkpoint) is processor_class
    patch_dynamic_module_loading()
    assert _load_processor(checkpoint) is processor_class
    outer.assert_called_once()
