"""Unit tests for ``build_config_from_swift_args`` and LoRA config mapping.

Most cases use a lightweight args namespace without Swift or model loading.
Optional native cases verify Swift's derived evaluation settings.

turbo ``--tt_*`` flags now live on the shared ``TorchTurboArgs``
(``set_torchturbo_args`` stashes them); swift-native fields (bf16/fp16, model
path, output_dir, tuner/rlhf type, LoRA) stay on the swift args namespace.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.checkpoint import CheckpointFormat

# Disable auto-activation so importing torchturbo never touches swift during
# these unit tests (matches the llamafactory adapter test setup).
os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

# Some import chains inside torchturbo reference the optional `grouped_gemm`
# package; stub it so the adapter modules import cleanly without it installed.
if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg


@pytest.fixture(autouse=True)
def _reset_cli_turchturbo(monkeypatch):
    # Clean sys.argv AND the stashed TorchTurboArgs so each test starts from a
    # known state. Tests set turbo flags via _set_tt(...).
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from torchturbo.engine.args import set_torchturbo_args
    set_torchturbo_args(None)
    yield
    set_torchturbo_args(None)


from torchturbo.adapters.msswift.utils import (  # noqa: E402
    build_config_from_swift_args,
    build_lora_config_from_swift_args,
)
from torchturbo.engine.args import TorchTurboArgs  # noqa: E402
from torchturbo.engine.backends.fsdp2.config import (  # noqa: E402
    CPUOffloadConfig,
)
from torchturbo.engine.config import (  # noqa: E402
    DistributedBackend,
    FreezePolicy,
    MixedPrecisionPolicy,
    ParameterGroup,
)


def _set_tt(**overrides):
    """Stash a ``TorchTurboArgs`` for the config builder to read back."""
    from torchturbo.engine.args import set_torchturbo_args
    tt = TorchTurboArgs(**overrides)
    set_torchturbo_args(tt)
    return tt


def _base_args(**overrides):
    """A minimal TurboSftArguments-like namespace (swift-native fields only)."""
    defaults = dict(
        bf16=True,
        fp16=False,
        model="/path/to/qwen3.5-moe",
        output_dir="/tmp/out",
        tuner_type="full",
        rlhf_type=None,
        router_aux_loss_coef=0.0,
        # Swift's own arguments carry the generation option; DPOConfig, used
        # as training_args for DPO, does not.
        predict_with_generate=False,
        training_args=SimpleNamespace(
            do_eval=False,
            do_predict=False,
            eval_strategy="no",
            eval_on_start=False,
        ),
        # lora fields (unused for full)
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        use_rslora=False,
        use_dora=False,
        target_modules=["all-linear"],
        learning_rate=0.002,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    "option,value",
    [
        ("do_eval", True),
        ("do_predict", True),
        ("predict_with_generate", True),
        ("eval_strategy", "steps"),
        ("eval_on_start", True),
    ],
)
def test_rejects_eval_and_generation_during_config_build(option, value):
    args = _base_args()
    target = args if option == "predict_with_generate" else args.training_args
    setattr(target, option, value)
    tt = _set_tt()
    with pytest.raises(NotImplementedError, match="does not support evaluation or generation"):
        build_config_from_swift_args(args)
    assert tt.hf_export is None
    assert tt.mixed_precision is None


def _save_tiny_llama(tmp_path):
    from transformers import LlamaConfig

    model_dir = tmp_path / "model"
    LlamaConfig(
        architectures=["LlamaForCausalLM"], vocab_size=16, hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.mark.parametrize("validation_dataset", [False, True])
def test_native_derived_eval_settings_are_checked_during_config_build(tmp_path, validation_dataset):
    pytest.importorskip("swift", minversion="4.0")
    from torchturbo.adapters.msswift.args import TurboSftArguments

    model_dir = _save_tiny_llama(tmp_path)
    args = TurboSftArguments(
        model=str(model_dir), model_type="llama", template="llama", dataset=["training-data"],
        val_dataset=["validation-data"] if validation_dataset else [],
        output_dir=str(tmp_path / "output"), add_version=False,
        tuner_type="full", report_to=[], use_cpu=True, check_model=False,
    )
    _set_tt()
    expected_strategy = "steps" if validation_dataset else "no"
    assert args.training_args.eval_strategy == expected_strategy
    assert args.training_args.do_eval is validation_dataset
    if validation_dataset:
        with pytest.raises(NotImplementedError, match="does not support evaluation or generation"):
            build_config_from_swift_args(args)
    else:
        config = build_config_from_swift_args(args)
        assert config.checkpoint.hf_export == "every"


@pytest.mark.parametrize("padding_free", [False, True])
def test_native_dpo_args_build_config(tmp_path, padding_free):
    pytest.importorskip("swift", minversion="4.0")
    from torchturbo.adapters.msswift.args import TurboRLHFArguments

    model_dir = _save_tiny_llama(tmp_path)
    args = TurboRLHFArguments(
        model=str(model_dir), model_type="llama", template="llama",
        dataset=["training-data"], rlhf_type="dpo", padding_free=padding_free,
        router_aux_loss_coef=0.02,
        attn_impl="flash_attn", output_dir=str(tmp_path / "output"),
        add_version=False, tuner_type="full", report_to=[], use_cpu=True,
        check_model=False,
    )
    # The layout stays the user's choice; the collator flattens either one.
    assert args.padding_free is padding_free
    # The engine-owned settings are forced off.
    assert args.gradient_checkpointing is False
    _set_tt()
    config = build_config_from_swift_args(args)
    assert config.checkpoint.hf_export == "every"
    assert config.moe.aux_loss_coeff == pytest.approx(0.02)


@pytest.mark.parametrize("turbo_coefficient", [0.0, 0.03])
def test_dpo_router_aux_loss_uses_engine_config(turbo_coefficient):
    _set_tt(moe_aux_loss_coeff=turbo_coefficient)
    args = _base_args(rlhf_type="dpo", router_aux_loss_coef=0.02)

    config = build_config_from_swift_args(args)

    assert config.moe.aux_loss_coeff == pytest.approx(turbo_coefficient or 0.02)
    assert args.router_aux_loss_coef == 0.02


@pytest.mark.parametrize("cli_coefficient", [None, 0.0, 0.03])
def test_dpo_router_aux_loss_preserves_config_and_cli_precedence(tmp_path, cli_coefficient):
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    config_file = tmp_path / "turbo.yaml"
    config_file.write_text("engine_config:\n  moe:\n    aux_loss_coeff: 0.04\n")
    argv = ["--tt_config", str(config_file)]
    if cli_coefficient is not None:
        argv += ["--tt_moe_aux_loss_coeff", str(cli_coefficient)]
    tt, _ = parse_torchturbo_args(argv)
    set_torchturbo_args(tt)

    config = build_config_from_swift_args(
        _base_args(rlhf_type="dpo", router_aux_loss_coef=0.02)
    )

    expected = 0.04 if cli_coefficient is None else cli_coefficient
    assert config.moe.aux_loss_coeff == pytest.approx(expected)


def test_sft_router_aux_loss_config_is_unchanged():
    _set_tt()

    config = build_config_from_swift_args(_base_args(router_aux_loss_coef=0.02))

    assert config.moe.aux_loss_coeff == 0.0


def test_native_dpo_args_reject_generation(tmp_path):
    pytest.importorskip("swift", minversion="4.0")
    from dataclasses import fields

    from torchturbo.adapters.msswift.args import TurboRLHFArguments

    model_dir = _save_tiny_llama(tmp_path)
    args = TurboRLHFArguments(
        model=str(model_dir), model_type="llama", template="llama",
        dataset=["training-data"], rlhf_type="dpo", predict_with_generate=True,
        # Otherwise Swift selects the "nlg" metric, which requires jieba.
        eval_metric="acc",
        output_dir=str(tmp_path / "output"), add_version=False,
        tuner_type="full", report_to=[], use_cpu=True, check_model=False,
    )
    # DPOConfig drops the option; only Swift's own arguments carry it.
    assert "predict_with_generate" not in {field.name for field in fields(args.training_args)}
    assert args.predict_with_generate is True
    _set_tt()
    with pytest.raises(NotImplementedError, match="does not support evaluation or generation"):
        build_config_from_swift_args(args)


def test_dpo_argument_mixin_preserves_implicit_logits_to_keep():
    pytest.importorskip("swift", minversion="4.0")
    from torchturbo.adapters.msswift.args import TurboSwiftArgsMixin

    class _Base:
        def __post_init__(self):
            pass

    class _Args(TurboSwiftArgsMixin, _Base):
        rlhf_type = "dpo"
        use_logits_to_keep = None

    args = _Args()
    args.__post_init__()
    assert args.use_logits_to_keep is None


def test_mixed_precision_from_bf16():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args(bf16=True))
    assert cfg.mixed_precision == MixedPrecisionPolicy.BF16


def test_mixed_precision_fp16_is_rejected():
    _set_tt()
    with pytest.raises(ValueError, match="does not support mixed_precision='fp16'"):
        build_config_from_swift_args(_base_args(bf16=False, fp16=True))


def test_mixed_precision_override_by_tt():
    _set_tt(mixed_precision="fp32")
    cfg = build_config_from_swift_args(_base_args(bf16=False, fp16=True))
    assert cfg.mixed_precision == MixedPrecisionPolicy.FP32


def test_backend_default_fsdp2_and_override():
    _set_tt()
    assert build_config_from_swift_args(_base_args()).backend == DistributedBackend.FSDP2
    _set_tt(backend="fsdp2")
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend == DistributedBackend.FSDP2


def test_parallel_plan_sp_ep():
    _set_tt(sp_size=4, ep_size=8)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.parallel_plan.sp_size == 4
    assert cfg.parallel_plan.ep_size == 8


def test_default_sp_ep_is_one():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.parallel_plan.sp_size == 1
    assert cfg.parallel_plan.ep_size == 1


def test_freeze_policy_mapping():
    _set_tt(
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=True,
    )
    cfg = build_config_from_swift_args(_base_args())
    groups = set(cfg.freeze_policy.parameter_groups)
    assert groups == {
        ParameterGroup.VISION_TOWER,
        ParameterGroup.MULTI_MODAL_PROJECTOR,
        ParameterGroup.LANGUAGE_MODEL,
    }
    assert cfg.freeze_policy.enabled is True


def test_no_freeze_default():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.freeze_policy.enabled is False


def test_mtp_config_mapping():
    _set_tt(enable_mtp=True, mtp_loss_scaling_factor=0.3, mtp_log_loss=True)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == pytest.approx(0.3)
    assert cfg.mtp.log_loss is True


def test_mtp_off_default():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.mtp.enabled is False
    assert cfg.mtp.log_loss is False


def test_chunked_loss_config_mapping():
    _set_tt(enable_chunk_loss=True, loss_chunk_size=384)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.loss_chunk.enabled is True
    assert cfg.loss_chunk.chunk_size == 384


def test_activation_checkpointing_owned_by_engine():
    _set_tt(gradient_checkpointing=True)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.activation_checkpointing.enabled is True
    _set_tt(gradient_checkpointing=False)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.activation_checkpointing.enabled is False


def test_cpu_offload_to_fsdp_config():
    _set_tt(cpu_offload=CPUOffloadConfig(enabled=True))
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "full"
    _set_tt(
        cpu_offload=CPUOffloadConfig(
            enabled=True,
            mode="optimizer",
        )
    )
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "optimizer"


def test_checkpoint_export_uses_the_engine_default():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.checkpoint.hf_export == "every"


def test_swift_optimizer_hyperparameters_do_not_configure_muon():
    _set_tt(optimizer_name="muon")

    cfg = build_config_from_swift_args(_base_args())

    assert cfg.optimizer.muon.lr is None
    assert cfg.optimizer.muon.weight_decay is None


@pytest.mark.parametrize("loss_offset", [0.0, float("nan"), float("inf")])
def test_training_step_checks_loss_before_metrics_and_optimizer_step(
    monkeypatch, loss_offset,
):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    model = torch.nn.Linear(1, 1, bias=False)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model.weight.sum() * 2 + loss_offset
    returned_loss = loss.detach()
    collected_models = []
    engine = SimpleNamespace(collect_metrics=collected_models.append)
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)

    class BaseTrainer:
        def training_step(self, model, inputs):
            loss.backward()
            return returned_loss

    class Trainer(TurboTrainerMixin, BaseTrainer):
        pass

    trainer = object.__new__(Trainer)

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


def test_muon_optimizer_bypasses_swift_optimizer_resolution(monkeypatch):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    optimizer = object()

    class Engine:
        config = SimpleNamespace(
            optimizer=SimpleNamespace(name="muon", muon=SimpleNamespace(qk_clip=None)),
        )

        def prepare_optimizer(self, *args, **kwargs):
            assert args == ()
            assert kwargs == {}
            return optimizer

    trainer = object.__new__(TurboTrainerMixin)
    trainer.optimizer = None
    trainer.get_optimizer_cls_and_kwargs = lambda *_args: (_ for _ in ()).throw(
        AssertionError("swift optimizer resolver must not be called")
    )
    monkeypatch.setattr(engine_registry, "get_engine", lambda: Engine())

    assert trainer.create_optimizer() is optimizer


def _install_swift_mixin(monkeypatch):
    """Install the ``SwiftMixin`` whose scheduler hook the adapter bypasses."""

    class SwiftMixin:
        def create_scheduler(self, *args, **kwargs):
            raise AssertionError("swift's OptimizerCallback scheduler must be bypassed")

    swift_mixin_module = types.ModuleType("swift.trainers.mixin")
    swift_mixin_module.SwiftMixin = SwiftMixin
    monkeypatch.setitem(sys.modules, "swift.trainers.mixin", swift_mixin_module)
    return SwiftMixin


@pytest.mark.parametrize("pass_optimizer", [False, True])
def test_muon_scheduler_is_created_by_engine(monkeypatch, pass_optimizer):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    _install_swift_mixin(monkeypatch)
    trainer_optimizer = object()
    passed_optimizer = object()
    scheduler = object()
    calls = []

    class Engine:
        config = SimpleNamespace(optimizer=SimpleNamespace(name="muon"))

        def prepare_lr_scheduler(self, *args, **kwargs):
            calls.append((args, kwargs))
            return scheduler

    monkeypatch.setattr(engine_registry, "get_engine", lambda: Engine())
    trainer = object.__new__(TurboTrainerMixin)
    trainer.optimizer = trainer_optimizer
    trainer.lr_scheduler = None
    trainer.args = SimpleNamespace(
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-7},
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
        (
            ("cosine_with_min_lr",),
            {
                "optimizer": scheduler_optimizer,
                "num_warmup_steps": 2,
                "num_training_steps": 10,
                "scheduler_specific_kwargs": {"min_lr": 1e-7},
            },
        )
    ]


def test_framework_scheduler_is_kept_without_engine_optimizer(monkeypatch):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    swift_mixin = _install_swift_mixin(monkeypatch)
    optimizer = object()
    scheduler = object()
    calls = []

    class HfTrainer:
        def create_scheduler(self, num_training_steps, optimizer=None):
            calls.append((num_training_steps, optimizer))
            return scheduler

    class SwiftTrainer(TurboTrainerMixin, swift_mixin, HfTrainer):
        pass

    engine = SimpleNamespace(
        config=SimpleNamespace(optimizer=SimpleNamespace(name=None)),
    )
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer = object.__new__(SwiftTrainer)
    trainer.optimizer = optimizer

    assert trainer.create_scheduler(10) is scheduler
    assert trainer.lr_scheduler is scheduler
    assert calls == [(10, optimizer)]


def test_accelerator_keeps_engine_optimizer_state_placement():
    from torchturbo.adapters.msswift.trainer import install_accelerator_overrides

    raw_optimizer = object()
    calls = []

    class Accelerator:
        def __init__(self):
            self._models = []

        def prepare_optimizer(self, optimizer, device_placement=None):
            calls.append((optimizer, device_placement))
            return optimizer

    trainer = SimpleNamespace(accelerator=Accelerator())
    engine = SimpleNamespace(optimizer=raw_optimizer)
    install_accelerator_overrides(trainer, engine)

    assert trainer.accelerator.prepare_optimizer(raw_optimizer) is raw_optimizer
    assert calls == [(raw_optimizer, False)]


def test_checkpoint_hooks_use_engine_optimizer(monkeypatch):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    raw_optimizer = object()
    calls = []

    class Engine:
        def __init__(self):
            self.optimizer = raw_optimizer

        def save_optimizer_and_scheduler(self, *args):
            calls.append(("save", args))

        def load_optimizer_and_scheduler(self, *args):
            calls.append(("load", args))

    engine = Engine()
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer = object.__new__(TurboTrainerMixin)
    trainer.model = object()
    trainer.optimizer = object()
    trainer.lr_scheduler = object()

    trainer._save_optimizer_and_scheduler("output")
    trainer._load_optimizer_and_scheduler(None)
    trainer._load_optimizer_and_scheduler("checkpoint")

    assert [kind for kind, _args in calls] == ["save", "load"]
    assert calls[0][1][1] is raw_optimizer
    assert calls[1][1][1] is raw_optimizer


@pytest.mark.parametrize(
    ("internal_call", "save_only_model", "global_step", "stop", "expected_kwargs"),
    [
        # Checkpoints from the HF machinery store native model state, and the
        # last checkpoint of the run also exports HF weights.
        (True, False, 5, False, {"final": False}),
        (True, False, 10, False, {"final": True}),
        # Training stopped early, for example by EarlyStoppingCallback.
        (True, False, 5, True, {"final": True}),
        # Explicit saves and model-only checkpoints are exports.
        (False, False, 5, False, {"format": CheckpointFormat.HF}),
        (True, True, 10, False, {"format": CheckpointFormat.HF}),
    ],
)
def test_save_model_reports_checkpoint_intent_to_the_engine(
    monkeypatch, tmp_path, internal_call, save_only_model, global_step, stop,
    expected_kwargs,
):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    calls = []

    class Engine:
        def save_model(self, model, output_dir, **kwargs):
            calls.append(("engine", output_dir, kwargs))

    class SwiftTrainer:
        def save_model(self, output_dir=None, _internal_call=False):
            calls.append(("swift", output_dir, _internal_call))

    monkeypatch.setattr(engine_registry, "get_engine", lambda: Engine())
    trainer = object.__new__(type("Trainer", (TurboTrainerMixin, SwiftTrainer), {}))
    trainer.model = object()
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), save_only_model=save_only_model)
    trainer.state = SimpleNamespace(max_steps=10, global_step=global_step)
    trainer.control = SimpleNamespace(should_training_stop=stop)
    output_dir = str(tmp_path / "checkpoint")

    trainer.save_model(output_dir, _internal_call=internal_call)

    assert calls == [
        ("engine", output_dir, expected_kwargs),
        ("swift", output_dir, internal_call),
    ]


@pytest.mark.parametrize(("output", "rejected"), [("source/out", True), ("out", False)])
def test_train_checks_the_output_dir_before_training(
    monkeypatch, tmp_path, output, rejected
):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboTrainerMixin

    source = tmp_path / "source"
    source.mkdir()
    calls = []

    class SwiftTrainer:
        def train(self, *args, **kwargs):
            calls.append("train")

    engine = SimpleNamespace(model_name_or_path=str(source))
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    trainer = object.__new__(type("Trainer", (TurboTrainerMixin, SwiftTrainer), {}))
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / output))

    if rejected:
        with pytest.raises(ValueError, match="must not be the source model directory"):
            trainer.train()
        assert calls == []
    else:
        trainer.train()
        assert calls == ["train"]


def test_unsupported_tuner_type_raises():
    _set_tt()
    with pytest.raises(NotImplementedError):
        build_config_from_swift_args(_base_args(tuner_type="adalora"))


def test_dpo_with_multimodal_architecture_is_accepted_for_text_batches():
    _set_tt()
    cfg = build_config_from_swift_args(
        _base_args(
            rlhf_type="dpo",
            model_meta=SimpleNamespace(is_multimodal=True),
        )
    )
    assert cfg.parallel_plan.sp_size == 1


def test_dpo_with_implicit_logits_to_keep_is_accepted():
    _set_tt()
    cfg = build_config_from_swift_args(
        _base_args(rlhf_type="dpo", use_logits_to_keep=None)
    )
    assert cfg.parallel_plan.sp_size == 1


def test_dpo_with_sp_is_accepted():
    _set_tt(sp_size=2)
    cfg = build_config_from_swift_args(_base_args(rlhf_type="dpo", tuner_type="full"))
    assert cfg.parallel_plan.sp_size == 2


def test_dpo_rejects_lora():
    _set_tt()
    with pytest.raises(NotImplementedError, match="DPO with tuner_type!=full"):
        build_config_from_swift_args(_base_args(rlhf_type="dpo", tuner_type="lora"))


def test_dpo_rejects_mtp():
    _set_tt(enable_mtp=True)
    with pytest.raises(NotImplementedError, match="DPO with MTP"):
        build_config_from_swift_args(_base_args(rlhf_type="dpo", tuner_type="full"))


def test_dpo_with_yaml_sp_is_accepted(tmp_path):
    """A resolved YAML sequence-parallel plan is accepted by DPO."""
    cfg = tmp_path / "tt.yaml"
    cfg.write_text("engine_config:\n  parallel_plan:\n    sp_size: 2\n")
    _set_tt(config_path=str(cfg))
    resolved = build_config_from_swift_args(_base_args(rlhf_type="dpo", tuner_type="full"))
    assert resolved.parallel_plan.sp_size == 2


def test_dpo_v1_baseline_matrix_accepted():
    """Test the supported full-tune DPO baseline matrix."""
    _set_tt()
    build_config_from_swift_args(
        _base_args(
            rlhf_type="dpo",
            tuner_type="full",
            packing=False,
            padding_free=False,
            loss_scale="last_round+ignore_empty_think",
            ref_adapters=[],
            adapters=[],
        )
    )


def test_lora_config_all_linear():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", lora_rank=16, lora_alpha=32, target_modules=["all-linear"])
    )
    assert lora is not None
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.target_modules == "all-linear"


def test_lora_config_explicit_targets():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", target_modules="q_proj,k_proj")
    )
    assert lora.target_modules == {"k_proj", "q_proj"}


def test_lora_config_dora_rslora_flags():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", use_rslora=True, use_dora=True)
    )
    assert lora.use_rslora is True
    assert lora.use_dora is True
