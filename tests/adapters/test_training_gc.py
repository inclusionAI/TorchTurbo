"""Verify native trainer GC defaults and cleanup without model execution."""

import gc
from types import SimpleNamespace

import pytest

from torchturbo.engine.args import TorchTurboArgs
from torchturbo.engine.config import EngineConfig
from torchturbo.engine import garbage_collection as training_gc
from torchturbo.adapters.llamafactory_v0.trainer import TorchTurboTrainerBase
from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import TurboTrainerMixin


@pytest.fixture(autouse=True)
def restore_gc(monkeypatch):
    enabled = gc.isenabled()
    gc.enable()
    monkeypatch.setattr(training_gc, "_under_memory_pressure", lambda: False)
    yield
    if enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.mark.parametrize("framework", ["swift", "llamafactory"])
@pytest.mark.parametrize("mode", [None, "auto", "step"])
@pytest.mark.parametrize("fail", [False, True])
def test_train_gc_default_override_and_cleanup(monkeypatch, tmp_path, framework, mode, fail):
    monkeypatch.setattr("torchturbo.engine.args._current_torchturbo_args", None)
    config = EngineConfig() if mode is None else EngineConfig(garbage_collection={"mode": mode})
    engine = SimpleNamespace(config=config, model_name_or_path=str(tmp_path / "model"))
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    step_gc = mode == "step"
    thresholds = gc.get_threshold()

    class BaseTrainer:
        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

        def train(self, *args, **kwargs):
            assert gc.get_freeze_count() == 0
            assert len(self.callbacks) == int(step_gc)
            for callback in self.callbacks:
                callback.on_step_begin(self.args, None, None)
            assert gc.isenabled() == (not step_gc)
            assert gc.get_threshold() == thresholds
            # Accumulation substeps must keep cyclic GC paused.
            for _ in range(3):
                for callback in self.callbacks:
                    callback.on_substep_end(self.args, None, None)
                assert gc.isenabled() == (not step_gc)
            if fail:
                raise RuntimeError("training failed")
            for callback in self.callbacks:
                callback.on_step_end(self.args, None, None)
            assert gc.isenabled()
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
        assert gc.isenabled()


def test_swift_without_engine_keeps_native_training(monkeypatch):
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: None)
    monkeypatch.setattr(
        "torchturbo.engine.args._current_torchturbo_args",
        TorchTurboArgs(profiling=True, memory_snapshot_steps=2),
    )

    class BaseTrainer:
        def add_callback(self, callback):
            raise AssertionError("No TorchTurbo callbacks without an engine")

        def train(self, *args, **kwargs):
            return "native"

    trainer = object.__new__(type("Trainer", (TurboTrainerMixin, BaseTrainer), {}))
    assert trainer.train() == "native"
    assert gc.get_freeze_count() == 0


@pytest.mark.parametrize("framework", ["swift", "llamafactory"])
@pytest.mark.parametrize("step_gc", [False, True])
@pytest.mark.parametrize(
    ("profiling", "snapshot"), [(True, None), (False, 2), (True, 2)]
)
def test_gc_and_profiling_share_one_callback(
    monkeypatch, tmp_path, framework, step_gc, profiling, snapshot
):
    tt_args = TorchTurboArgs(profiling=profiling, memory_snapshot_steps=snapshot)
    monkeypatch.setattr("torchturbo.engine.args._current_torchturbo_args", tt_args)
    events = []

    def start(*args):
        events.append(("start", args))

    def step():
        assert gc.isenabled()
        events.append(("step",))

    def stop():
        assert gc.get_freeze_count() == 0
        assert trainer.callbacks == []
        events.append(("stop",))

    engine = SimpleNamespace(
        config=EngineConfig(garbage_collection={"mode": "step" if step_gc else "auto"}),
        model_name_or_path=str(tmp_path / "model"),
        start_profiling=start, step_profiling=step, stop_profiling=stop,
    )
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)

    class BaseTrainer:
        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

        def train(self, *args, **kwargs):
            assert gc.get_freeze_count() == 0
            assert len(self.callbacks) == 1
            callback = self.callbacks[0]
            for _ in range(2):
                callback.on_step_begin(self.args, None, None)
                assert gc.isenabled() == (not step_gc)
                callback.on_substep_end(self.args, None, None)
                assert gc.isenabled() == (not step_gc)
                callback.on_step_end(self.args, None, None)
                assert gc.isenabled()
            return "trained"

    mixin = TurboTrainerMixin if framework == "swift" else TorchTurboTrainerBase
    trainer = object.__new__(type("Trainer", (mixin, BaseTrainer), {}))
    trainer.engine = engine
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path / "output"), num_train_epochs=1, max_steps=2
    )
    trainer.callbacks = []
    for _ in range(2):
        assert trainer.train() == "trained"
        assert trainer.callbacks == []
        assert gc.get_freeze_count() == 0
    expected = [
        ("start", (trainer.args.output_dir, profiling, snapshot)),
        ("step",), ("step",), ("stop",),
    ]
    assert events == expected * 2


@pytest.mark.parametrize("framework", ["swift", "llamafactory"])
@pytest.mark.parametrize("failure", ["start", "before_begin", "training", "stop"])
def test_callback_and_gc_cleanup_on_lifecycle_failures(
    monkeypatch, tmp_path, framework, failure
):
    monkeypatch.setattr(
        "torchturbo.engine.args._current_torchturbo_args", TorchTurboArgs(profiling=True)
    )
    events = []

    def start(*args):
        events.append("start")
        if failure == "start":
            raise RuntimeError("start failed")

    def stop():
        assert gc.get_freeze_count() == 0
        assert trainer.callbacks == []
        events.append("stop")
        if failure == "stop":
            raise RuntimeError("stop failed")

    engine = SimpleNamespace(
        config=EngineConfig(garbage_collection={"mode": "step"}),
        model_name_or_path=str(tmp_path / "model"),
        start_profiling=start, stop_profiling=stop,
    )
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)

    class BaseTrainer:
        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

        def train(self, *args, **kwargs):
            if failure == "before_begin":
                raise RuntimeError("before_begin failed")
            self.callbacks[0].on_step_begin(self.args, None, None)
            assert not gc.isenabled()
            if failure == "training":
                raise RuntimeError("training failed")

    mixin = TurboTrainerMixin if framework == "swift" else TorchTurboTrainerBase
    trainer = object.__new__(type("Trainer", (mixin, BaseTrainer), {}))
    trainer.engine = engine
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path / "output"), num_train_epochs=1, max_steps=1
    )
    trainer.callbacks = []
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        trainer.train()
    assert trainer.callbacks == []
    assert gc.get_freeze_count() == 0
    assert gc.isenabled()
    assert events == (["start"] if failure == "start" else ["start", "stop"])


@pytest.mark.parametrize("framework", ["swift", "llamafactory"])
@pytest.mark.parametrize("num_micro_batches", [5, 6])
def test_gc_callback_in_real_trainer_accumulation_loop(
    monkeypatch, tmp_path, framework, num_micro_batches
):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    engine = SimpleNamespace()
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    mixin = TurboTrainerMixin if framework == "swift" else TorchTurboTrainerBase
    adapter = object.__new__(mixin)
    adapter.engine = engine
    events = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids):
            assert not gc.isenabled()
            events.append("forward")
            return {"loss": (input_ids * self.weight).square().mean()}

    class Observer(TrainerCallback):
        def on_substep_end(self, args, state, control, **kwargs):
            assert not gc.isenabled()
            events.append("substep")

        def on_optimizer_step(self, args, state, control, **kwargs):
            assert not gc.isenabled()
            events.append("optimizer")

        def on_step_end(self, args, state, control, **kwargs):
            assert gc.isenabled()
            events.append("end")

    args = TrainingArguments(
        output_dir=str(tmp_path / "output"),
        use_cpu=True,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        optim="sgd",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    dataset = [{"input_ids": torch.ones(1)} for _ in range(num_micro_batches)]
    with training_gc.GarbageCollector() as collector:
        callback = adapter._make_torchturbo_training_callback(collector, False)
        trainer = Trainer(
            model=Model(), args=args, train_dataset=dataset,
            callbacks=[callback, Observer()],
        )
        result = trainer.train()
    assert result.global_step == 2
    assert events.count("forward") == num_micro_batches
    assert events.count("substep") == num_micro_batches - 2
    assert events.count("optimizer") == events.count("end") == 2
    assert gc.isenabled()
