"""Verify native trainer GC defaults and cleanup without model execution."""

import gc
from types import SimpleNamespace

import pytest

from torchturbo.engine.config import EngineConfig
from torchturbo.adapters.llamafactory_v0.trainer import TorchTurboTrainerBase
from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import TurboTrainerMixin


@pytest.mark.parametrize("framework", ["swift", "llamafactory"])
@pytest.mark.parametrize("mode", [None, "auto", "freeze"])
@pytest.mark.parametrize("fail", [False, True])
def test_train_gc_default_override_and_cleanup(monkeypatch, tmp_path, framework, mode, fail):
    monkeypatch.setattr("torchturbo.engine.args._current_torchturbo_args", None)
    config = EngineConfig() if mode is None else EngineConfig(garbage_collection={"mode": mode})
    engine = SimpleNamespace(config=config, model_name_or_path=str(tmp_path / "model"))
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    expected_freeze = mode == "freeze"
    thresholds = gc.get_threshold()

    class BaseTrainer:
        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

        def train(self, *args, **kwargs):
            assert gc.get_freeze_count() == 0
            assert len(self.callbacks) == int(expected_freeze)
            for callback in self.callbacks:
                callback.on_train_begin(self.args, None, None)
            assert (gc.get_freeze_count() > 0) == expected_freeze
            assert gc.get_threshold() == thresholds
            if fail:
                raise RuntimeError("training failed")
            return "trained"

    mixin = TurboTrainerMixin if framework == "swift" else TorchTurboTrainerBase
    trainer = object.__new__(type("Trainer", (mixin, BaseTrainer), {}))
    trainer.engine = engine
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / "output"), num_train_epochs=1, max_steps=1)
    trainer.callbacks = []
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="training failed"):
                trainer.train()
        else:
            assert trainer.train() == "trained"
        assert trainer.callbacks == []
        assert gc.get_freeze_count() == 0
        assert gc.get_threshold() == thresholds
