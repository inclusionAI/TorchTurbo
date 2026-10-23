from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from torchturbo.engine.checkpoint import CheckpointFormat
from torchturbo.adapters.llamafactory_v0.trainer import (
    LlamaFactoryInputCollator,
    TorchTurboDPOTrainer,
    TorchTurboTrainerBase,
    make_torchturbo_trainer,
)


@pytest.mark.parametrize("sp_rank", [0, 1])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_dpo_leaves_padded_inputs_global_for_model_slicing(
    monkeypatch, sp_rank, explicit_positions
):
    from torchturbo.models.base import TurboModel
    from torchturbo.models.modules.attention.ulysses import patch_forward_for_sp_input_slicing

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed.nn.functional, "all_reduce", lambda value, **kwargs: value
    )
    sp_info = {"sp_size": 2, "sp_group": object(), "sp_rank": sp_rank}

    class Model(TurboModel, torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return input_ids

    model = Model()
    ref_model = Model()
    patch_forward_for_sp_input_slicing(
        model.get_sequence_parallel_input_slicing_spec(), **sp_info
    )
    patch_forward_for_sp_input_slicing(
        ref_model.get_sequence_parallel_input_slicing_spec(), **sp_info
    )
    trainer = object.__new__(TorchTurboDPOTrainer)
    trainer.engine = SimpleNamespace(get_sp_info=lambda: sp_info)
    trainer.label_pad_token_id = -100
    trainer.ftx_gamma = 0.0
    trainer.loss_type = "sigmoid"
    trainer.finetuning_args = SimpleNamespace(use_ref_model=True)
    trainer.ref_model = ref_model
    seen_models = []

    def forward(model, batch, is_ref_model=False):
        seen_models.append(model)
        assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 0]] * 2
        assert batch["labels"].tolist() == [[1, 2, 3, 4, 5, -100]] * 2
        num_position_rows = 2 if explicit_positions else 1
        assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0]] * num_position_rows
        expected = [[1, 2, 3]] if sp_rank == 0 else [[4, 5, 0]]
        assert model(**batch).tolist() == expected * 2
        return (torch.ones(1),) * 5

    trainer.concatenated_forward = forward
    trainer.compute_preference_loss = lambda *args: (torch.ones(1),) * 3
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]] * 2),
        "labels": torch.tensor([[1, 2, 3, 4, 5]] * 2),
    }
    if explicit_positions:
        batch["position_ids"] = torch.arange(5).expand(2, -1)
    trainer.get_batch_loss_metrics(model, batch)

    assert seen_models == [model, ref_model]
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 5)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_pretraining_wraps_native_language_modeling_collation(padding_side):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

    from torchturbo.engine.utils import assert_packed_inputs

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
        padding_side=padding_side,
    )
    native_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class BaseTrainer:
        def __init__(self, *, data_collator):
            self.optimizer = None
            self.train_dataset = None
            self.data_collator = data_collator

    trainer_cls = make_torchturbo_trainer(BaseTrainer, engine=object(), stage="pt")
    trainer = trainer_cls(data_collator=native_collator)
    assert isinstance(trainer.data_collator, LlamaFactoryInputCollator)
    result = trainer.data_collator([
        {"input_ids": [10, 11]},
        {"input_ids": [20, 21, 22]},
    ])

    assert_packed_inputs(result)
    assert result["input_ids"].tolist() == [[10, 11, 20, 21, 22]]
    assert result["labels"].tolist() == [[-100, 11, -100, 21, 22]]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]


class _DiffusionEngine:
    def __init__(self) -> None:
        self.call = None

    def compute_diffusion_loss(
        self,
        model,
        inputs,
        *,
        ga,
        return_outputs,
        **kwargs,
    ):
        self.call = {
            "model": model,
            "inputs": inputs,
            "ga": ga,
            "return_outputs": return_outputs,
            "kwargs": kwargs,
        }
        return torch.tensor(1.5), SimpleNamespace()

    def step_profiling(self):
        raise AssertionError("diffusion microbatches must not advance profiling")


def test_diffusion_loss_delegates_without_advancing_profiler():
    engine = _DiffusionEngine()
    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = engine
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    model = object()
    inputs = {"input_ids": torch.tensor([[1, 2]])}

    loss, outputs = trainer._compute_diffusion_loss(
        model,
        inputs,
        return_outputs=True,
        num_items_in_batch=2,
    )

    torch.testing.assert_close(loss, torch.tensor(1.5))
    assert isinstance(outputs, SimpleNamespace)
    assert engine.call == {
        "model": model,
        "inputs": inputs,
        "ga": 4,
        "return_outputs": True,
        "kwargs": {"num_items_in_batch": 2},
    }


@pytest.mark.parametrize("loss_offset", [0.0, float("nan"), float("inf")])
def test_training_step_checks_loss_before_metrics_and_optimizer_step(loss_offset):
    model = torch.nn.Linear(1, 1, bias=False)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model.weight.sum() * 2 + loss_offset
    returned_loss = loss.detach()
    collected_models = []

    class BaseTrainer:
        def training_step(self, model, inputs):
            loss.backward()
            return returned_loss

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=SimpleNamespace(collect_metrics=collected_models.append),
        stage="sft",
    )
    trainer = object.__new__(trainer_cls)

    if loss_offset == 0.0:
        assert trainer.training_step(model, {}) is returned_loss
        assert collected_models == [model]
        optimizer.step()
        torch.testing.assert_close(model.weight, initial_weight - 0.2)
    else:
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.training_step(model, {})
            optimizer.step()
        assert collected_models == []
        torch.testing.assert_close(model.weight, initial_weight)

    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 2.0))


def test_muon_optimizer_bypasses_framework_optimizer_resolution():
    optimizer = object()

    class Engine:
        config = SimpleNamespace(
            optimizer=SimpleNamespace(name="muon", muon=SimpleNamespace(qk_clip=None)),
        )

        def prepare_optimizer(self, *args, **kwargs):
            assert args == ()
            assert kwargs == {}
            return optimizer

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.optimizer = None

    assert trainer.create_optimizer() is optimizer


@pytest.mark.parametrize("pass_optimizer", [False, True])
def test_muon_scheduler_is_created_by_engine_after_llamafactory_defaults(
    monkeypatch, pass_optimizer
):
    trainer_optimizer = object()
    passed_optimizer = object()
    scheduler = object()
    calls = []

    def create_custom_scheduler(training_args, num_training_steps):
        # LlamaFactory fills the WSD phase lengths before a scheduler is built.
        calls.append(("defaults", num_training_steps))
        training_args.lr_scheduler_kwargs = {
            "num_stable_steps": 2,
            "num_decay_steps": 6,
        }

    class Engine:
        config = SimpleNamespace(optimizer=SimpleNamespace(name="muon"))

        def prepare_lr_scheduler(self, *args, **kwargs):
            calls.append(("engine", args, kwargs))
            return scheduler

    trainer_utils = ModuleType("llamafactory.train.trainer_utils")
    trainer_utils.create_custom_scheduler = create_custom_scheduler
    monkeypatch.setitem(sys.modules, "llamafactory.train.trainer_utils", trainer_utils)
    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.optimizer = trainer_optimizer
    trainer.lr_scheduler = None
    trainer.args = SimpleNamespace(
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={},
        get_warmup_steps=lambda num_training_steps: 2,
    )
    if pass_optimizer:
        create_kwargs = {"optimizer": passed_optimizer}
        scheduler_optimizer = passed_optimizer
    else:
        create_kwargs = {}
        scheduler_optimizer = trainer_optimizer

    assert trainer.create_scheduler(10, **create_kwargs) is scheduler
    # A second call keeps the scheduler that the training loop already steps.
    assert trainer.create_scheduler(10, **create_kwargs) is scheduler
    assert trainer._created_lr_scheduler is True
    assert calls == [
        ("defaults", 10),
        (
            "engine",
            ("warmup_stable_decay",),
            {
                "optimizer": scheduler_optimizer,
                "num_warmup_steps": 2,
                "num_training_steps": 10,
                "scheduler_specific_kwargs": {
                    "num_stable_steps": 2,
                    "num_decay_steps": 6,
                },
            },
        ),
    ]


def test_framework_scheduler_is_kept_without_engine_optimizer():
    scheduler = object()
    calls = []

    class BaseTrainer:
        def create_scheduler(self, num_training_steps, optimizer=None):
            calls.append((num_training_steps, optimizer))
            return scheduler

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=SimpleNamespace(
            config=SimpleNamespace(optimizer=SimpleNamespace(name=None)),
        ),
        stage="sft",
    )
    trainer = object.__new__(trainer_cls)

    assert trainer.create_scheduler(10) is scheduler
    assert calls == [(10, None)]


def test_external_optimizer_is_rejected():
    class BaseTrainer:
        def __init__(self, *, optimizers):
            self.optimizer, self.lr_scheduler = optimizers

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=object(),
        stage="sft",
    )

    with pytest.raises(
        ValueError,
        match="does not support externally supplied optimizers",
    ):
        trainer_cls(optimizers=(object(), None))


def test_checkpoint_hooks_use_engine_optimizer():
    raw_optimizer = object()
    wrapped_optimizer = object()
    calls = []

    class Engine:
        def __init__(self):
            self.optimizer = raw_optimizer

        def save_optimizer_and_scheduler(self, *args):
            calls.append(("save", args))

        def load_optimizer_and_scheduler(self, *args):
            calls.append(("load", args))

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.model = object()
    trainer.optimizer = wrapped_optimizer
    trainer.lr_scheduler = object()

    trainer._save_optimizer_and_scheduler("output")
    trainer._load_optimizer_and_scheduler("checkpoint")

    assert calls[0][1][1] is raw_optimizer
    assert calls[1][1][1] is raw_optimizer


@pytest.mark.parametrize(("output", "rejected"), [("source/out", True), ("out", False)])
def test_train_checks_the_output_dir_before_training(
    monkeypatch, tmp_path, output, rejected
):
    # No stashed --tt_* args, so train() does not start profiling.
    monkeypatch.setattr("torchturbo.engine.args._current_torchturbo_args", None)
    source = tmp_path / "source"
    source.mkdir()
    calls = []

    class BaseTrainer:
        def train(self, *args, **kwargs):
            calls.append("train")

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=SimpleNamespace(model_name_or_path=str(source)),
        stage="sft",
    )
    trainer = object.__new__(trainer_cls)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / output))

    if rejected:
        with pytest.raises(ValueError, match="must not be the source model directory"):
            trainer.train()
        assert calls == []
    else:
        trainer.train()
        assert calls == ["train"]


@pytest.mark.parametrize(
    ("profiling", "profiling_config", "expected_start", "expected_stop"),
    [
        # Profiling on: the config dict is forwarded as the trailing argument.
        (
            True,
            {"with_shape": True},
            (True, None, {"with_shape": True}),
            True,
        ),
        # Config without --tt_profiling stays inert.
        (False, {"with_shape": True}, None, False),
    ],
)
def test_train_forwards_profiling_config(
    monkeypatch,
    tmp_path,
    profiling,
    profiling_config,
    expected_start,
    expected_stop,
):
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setattr(
        "torchturbo.engine.args._current_torchturbo_args",
        TorchTurboArgs(profiling=profiling, profiling_config=profiling_config),
    )
    source = tmp_path / "source"
    source.mkdir()
    output_dir = str(tmp_path / "out")
    calls = {"start": [], "stop": []}

    class Engine:
        model_name_or_path = str(source)

        def start_profiling(self, output_dir, profiling_enabled, steps, config):
            calls["start"].append((output_dir, profiling_enabled, steps, config))

        def stop_profiling(self):
            calls["stop"].append(None)

    class BaseTrainer:
        def train(self, *args, **kwargs):
            return None

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer, engine=Engine(), stage="sft"
    )
    trainer = object.__new__(trainer_cls)
    trainer.args = SimpleNamespace(output_dir=output_dir)

    # add_callback comes from the HF base trainer; provide a no-op.
    trainer.add_callback = lambda callback: None

    trainer.train()

    if expected_start is None:
        assert calls["start"] == []
        assert calls["stop"] == []
    else:
        expected_enabled, expected_steps, expected_config = expected_start
        assert calls["start"] == [
            (output_dir, expected_enabled, expected_steps, expected_config)
        ]
        assert calls["stop"] == [None]


@pytest.mark.parametrize(
    ("internal_call", "save_only_model", "global_step", "expected_kwargs"),
    [
        # Checkpoints from the HF machinery store native model state.
        (True, False, 5, {}),
        (True, False, 10, {}),
        # LF's final explicit save and model-only checkpoints are exports.
        (False, False, 5, {"format": CheckpointFormat.HF}),
        (True, True, 10, {"format": CheckpointFormat.HF}),
    ],
)
def test_save_model_reports_checkpoint_intent_to_the_engine(
    internal_call, save_only_model, global_step, expected_kwargs
):
    calls = []

    class Engine:
        def save_model(self, model, output_dir, **kwargs):
            calls.append((model, output_dir, kwargs))

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.model = object()
    trainer.args = SimpleNamespace(
        output_dir="output",
        save_only_model=save_only_model,
        push_to_hub=False,
    )
    trainer.state = SimpleNamespace(max_steps=10, global_step=global_step)

    trainer.save_model("output/checkpoint-1", _internal_call=internal_call)

    assert calls == [(trainer.model, "output/checkpoint-1", expected_kwargs)]
