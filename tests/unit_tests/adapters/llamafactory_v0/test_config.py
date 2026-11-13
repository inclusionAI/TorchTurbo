"""Tests for LlamaFactory adapter config mapping."""

import os
import sys
import types

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg


@pytest.fixture(autouse=True)
def _reset_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])


def _training_args(**options):
    defaults = dict(
        do_eval=False,
        do_predict=False,
        predict_with_generate=False,
        eval_strategy="no",
        eval_on_start=False,
    )
    defaults.update(options)
    return types.SimpleNamespace(**defaults)


@pytest.mark.parametrize("option,value", [
    ("do_eval", True),
    ("do_predict", True),
    ("predict_with_generate", True),
    ("eval_strategy", "steps"),
    ("eval_on_start", True),
])
def test_build_config_rejects_evaluation_and_generation(option, value):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args

    with pytest.raises(NotImplementedError, match="does not support evaluation or generation"):
        build_config_from_lf_args(training_args=_training_args(**{option: value}))


def test_build_config_cli_prefetch_takes_effect():
    """--tt_prefetch_strategy / --tt_prefetch_direction must reach backend_config."""
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import (
        FSDPPrefetchDirection,
        FSDPPrefetchStrategy,
    )

    training_args = _training_args(
        fsdp=[],
        fsdp_config={
            "prefetch_strategy": "expert_lookahead",
            "prefetch_direction": "forward_backward",
        },
        gradient_checkpointing=True,
    )
    tt = TorchTurboArgs(
        prefetch_strategy="expert_with_layer", prefetch_direction="forward"
    )
    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.backend_config.prefetch_strategy == FSDPPrefetchStrategy.EXPERT_WITH_LAYER
    assert cfg.backend_config.prefetch_direction == FSDPPrefetchDirection.FORWARD


def test_build_config_maps_freeze_args_to_engine_policy():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import ParameterGroup

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(
        freeze_vision_tower=True,
        freeze_language_model=True,
    )

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.freeze_policy.parameter_groups == (
        ParameterGroup.VISION_TOWER,
        ParameterGroup.LANGUAGE_MODEL,
    )


def test_build_config_maps_mtp_args_to_engine_config():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(
        enable_mtp=True,
        mtp_loss_scaling_factor=0.3,
        mtp_log_loss=True,
    )

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.3
    assert cfg.mtp.log_loss is True


def test_build_config_maps_op_policy_to_engine_config():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.ops.base import OpPolicy

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(op_policy="native")

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.ops.policy is OpPolicy.NATIVE


@pytest.mark.parametrize("policy", ["fp16", "fp8"])
def test_build_config_rejects_unsupported_mixed_precision(policy):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args

    training_args = _training_args(**{policy: True})

    with pytest.raises(
        ValueError, match=f"does not support mixed_precision='{policy}'"
    ):
        build_config_from_lf_args(training_args=training_args)


@pytest.mark.parametrize("policy", ["fp16", "fp8"])
def test_build_config_mixed_precision_override_wins(policy):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import MixedPrecisionPolicy

    training_args = _training_args(
        fsdp=[],
        fsdp_config={},
        gradient_checkpointing=True,
        **{policy: True},
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(mixed_precision="fp32"),
    )

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32


def test_lf_optimizer_hyperparameters_do_not_configure_muon():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[],
        fsdp_config={},
        gradient_checkpointing=True,
        learning_rate=0.002,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(optimizer_name="muon"),
    )

    assert cfg.optimizer.muon.lr is None
    assert cfg.optimizer.muon.weight_decay is None


def test_muon_config_comes_from_torchturbo_yaml(tmp_path):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    config_path = tmp_path / "muon.yaml"
    config_path.write_text(
        """engine_config:
  optimizer:
    name: muon
    muon:
      lr: 0.004
      weight_decay: 0.2
      adamw_betas: [0.7, 0.9]
      adamw_eps: 1e-5
""",
        encoding="utf-8",
    )
    training_args = _training_args(
        fsdp=[],
        fsdp_config={},
        gradient_checkpointing=True,
        learning_rate=0.002,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(config_path=str(config_path)),
    )

    assert cfg.optimizer.muon.lr == 0.004
    assert cfg.optimizer.muon.weight_decay == 0.2
    assert cfg.optimizer.muon.adamw_betas == (0.7, 0.9)
    assert cfg.optimizer.muon.adamw_eps == 1e-5


def test_build_config_cli_prefetch_ignores_lf_yaml_prefetch():
    """TorchTurbo prefetch config does not read LF fsdp_config prefetch fields."""
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import (
        FSDPPrefetchDirection,
        FSDPPrefetchStrategy,
    )

    training_args = _training_args(
        fsdp=[],
        fsdp_config={
            "prefetch_strategy": "expert_lookahead",
            "prefetch_direction": "forward",
        },
        gradient_checkpointing=True,
    )
    tt = TorchTurboArgs()  # defaults: auto / forward_backward
    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.backend_config.prefetch_strategy == FSDPPrefetchStrategy.AUTO
    assert cfg.backend_config.prefetch_direction == FSDPPrefetchDirection.FORWARD_BACKWARD


def test_build_config_maps_lf_fsdp_without_offload_to_disabled_cpu_offload():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=["full_shard"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args, torchturbo_args=TorchTurboArgs()
    )

    assert cfg.backend_config.cpu_offload.enabled is False


def test_build_config_maps_lf_fsdp_offload_to_enabled_cpu_offload():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=["full_shard", "offload"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args, torchturbo_args=TorchTurboArgs()
    )

    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "full"


def test_build_config_cpu_offload_cannot_disable_lf_offload():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig

    training_args = _training_args(
        fsdp=["full_shard", "offload"], fsdp_config={}, gradient_checkpointing=True
    )
    with pytest.raises(ValueError, match="must use mode='full'"):
        build_config_from_lf_args(
            training_args=training_args,
            torchturbo_args=TorchTurboArgs(
                cpu_offload=CPUOffloadConfig(enabled=False)
            ),
        )


def test_resolve_cpu_offload_non_strict_preserves_non_full_mode():
    from torchturbo.adapters.llamafactory_v0.utils import resolve_cpu_offload
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig

    policy = resolve_cpu_offload(
        ["full_shard", "offload"],
        TorchTurboArgs(
            cpu_offload=CPUOffloadConfig(
                enabled=True,
                mode="optimizer",
            )
        ),
        strict=False,
    )

    assert policy.enabled is True
    assert policy.mode.value == "optimizer"


def test_build_config_cpu_offload_overrides_lf():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig

    training_args = _training_args(
        fsdp=["full_shard"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(
            cpu_offload=CPUOffloadConfig(enabled=True)
        ),
    )

    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "full"


def test_engine_activation_checkpointing_defaults_to_enabled():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.activation_checkpointing.enabled is True


def test_lf_gradient_checkpointing_is_taken_over_by_engine():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    model_args = types.SimpleNamespace(
        disable_gradient_checkpointing=False, use_reentrant_gc=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(gradient_checkpointing=True),
    )

    assert cfg.activation_checkpointing.enabled is True
    assert training_args.gradient_checkpointing is False
    assert model_args.disable_gradient_checkpointing is True
    assert model_args.use_reentrant_gc is True


def test_lf_gradient_checkpointing_uses_lf_value_when_tt_arg_is_default():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False


def test_lf_disable_gradient_checkpointing_does_not_enable_engine_ac():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=False)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False
    assert model_args.disable_gradient_checkpointing is True


@pytest.mark.parametrize("adapter_gc, tt_gc", [(True, False), (False, True)])
def test_lf_gradient_checkpointing_torchturbo_cli_mismatch_raises(
    adapter_gc, tt_gc, monkeypatch
):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=adapter_gc
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lf.py", "--tt_gradient_checkpointing", str(tt_gc).lower()],
    )

    with pytest.raises(ValueError, match="tt_gradient_checkpointing must match"):
        build_config_from_lf_args(
            training_args=training_args,
            model_args=model_args,
            torchturbo_args=TorchTurboArgs(gradient_checkpointing=tt_gc),
        )

    assert training_args.gradient_checkpointing is adapter_gc
    assert model_args.disable_gradient_checkpointing is True


@pytest.mark.parametrize("adapter_gc, tt_gc", [(True, True), (False, False)])
def test_lf_gradient_checkpointing_torchturbo_cli_match_takes_effect(
    adapter_gc, tt_gc, monkeypatch
):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=adapter_gc
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lf.py", f"--tt_gradient_checkpointing={str(tt_gc).lower()}"],
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(gradient_checkpointing=tt_gc),
    )

    assert cfg.activation_checkpointing.enabled is adapter_gc


def test_lf_gradient_checkpointing_env_mismatch_raises(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "false")
    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    with pytest.raises(ValueError, match="tt_gradient_checkpointing must match"):
        build_config_from_lf_args(
            training_args=training_args,
            model_args=model_args,
            torchturbo_args=TorchTurboArgs(),
        )

    assert training_args.gradient_checkpointing is True
    assert model_args.disable_gradient_checkpointing is True


def test_lf_gradient_checkpointing_env_match_takes_effect(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "false")
    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False


def test_lf_gradient_checkpointing_raw_args_are_disabled():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = _training_args(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, dict_args = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
        raw_args={
            "gradient_checkpointing": True,
            "disable_gradient_checkpointing": False,
        },
    )
    assert cfg.activation_checkpointing.enabled is False
    assert dict_args["gradient_checkpointing"] is False
    assert dict_args["disable_gradient_checkpointing"] is True

    _, list_args = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
        raw_args=[
            "--gradient_checkpointing=true",
            "--use_reentrant_gc",
            "true",
            "--foo",
            "bar",
        ],
    )
    assert "--gradient_checkpointing=true" not in list_args
    assert list_args[-1] == "--disable_gradient_checkpointing"
    assert "--use_reentrant_gc" in list_args
