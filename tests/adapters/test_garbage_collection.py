"""Check trainer lifecycle integration using a device-free callback stub."""

import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def freeze_training_gc(monkeypatch):
    root = Path(__file__).parents[2] / "src/torchturbo"
    for name, path in [
        ("torchturbo.runtime.garbage_collection", root / "runtime/garbage_collection.py"),
        ("torchturbo.adapters.garbage_collection", root / "adapters/garbage_collection.py"),
    ]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return module.freeze_training_gc


class Trainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        self.callbacks.remove(callback)

    def begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(None, SimpleNamespace(global_step=0), None)


def test_freeze_waits_for_training_initialization(freeze_training_gc):
    trainer = Trainer()
    with freeze_training_gc(trainer):
        assert gc.get_freeze_count() == 0
        trainer.begin()
        assert gc.get_freeze_count() > 0
    assert not trainer.callbacks
    assert gc.get_freeze_count() == 0


@pytest.mark.parametrize("failure_after_begin", [False, True])
def test_cleanup_on_failure(freeze_training_gc, failure_after_begin):
    trainer = Trainer()
    with pytest.raises(RuntimeError, match="training failed"):
        with freeze_training_gc(trainer):
            if failure_after_begin:
                trainer.begin()
            raise RuntimeError("training failed")
    assert not trainer.callbacks
    assert gc.get_freeze_count() == 0


def test_repeated_training_leaves_no_callbacks(freeze_training_gc):
    trainer = Trainer()
    for _ in range(2):
        with freeze_training_gc(trainer):
            assert len(trainer.callbacks) == 1
            trainer.begin()
        assert not trainer.callbacks
        assert gc.get_freeze_count() == 0
