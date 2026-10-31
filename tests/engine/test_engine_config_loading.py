"""Tests for unified EngineConfig loading (``--tt_config`` YAML/JSON + layering).

The config-loader feature assembles :class:`EngineConfig` from four layers:

1. ``TorchTurboArgs`` defaults / adapter-seeded values
2. ``TORCH_TURBO_*`` environment variables
3. the YAML/JSON config file passed via ``--tt_config``
4. explicitly supplied ``--tt_*`` CLI flags

Precedence is: CLI > file > env > TorchTurboArgs.
"""

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from torchturbo.engine.args import (
    TorchTurboArgs,
    get_arg_specs,
    parse_torchturbo_args,
)
from torchturbo.engine.config import (
    DSparkTrainingConfig,
    EngineConfig,
    MixedPrecisionPolicy,
    ParameterGroup,
    build_from_args,
)
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.base import TurboModel
from torchturbo.models.bailing_moe_v3_vl.parallel_plan import (
    get_parallel_plan as get_bailing_vl_plan,
)
from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
from torchturbo.models.qwen3_5_moe.parallel_plan import (
    get_parallel_plan as get_qwen3_5_moe_plan,
)
from torchturbo.ops.base import OpPolicy
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlanSpec,
    VisionBalanceMode,
    VisionBalanceSpec,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep every test hermetic against real ``TORCH_TURBO_*`` env vars."""
    for arg_spec in get_arg_specs():
        if arg_spec.env_name is not None:
            monkeypatch.delenv(arg_spec.env_name, raising=False)
    monkeypatch.delenv("TT_MODEL_NAME_OR_PATH", raising=False)
    monkeypatch.delenv("TT_OUTPUT_DIR", raising=False)


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _engine_config(content: str) -> str:
    return "engine_config:\n" + textwrap.indent(content, "  ")


def _build_from_argv(argv: list[str]) -> EngineConfig:
    tt, _ = parse_torchturbo_args(argv)
    return build_from_args(tt)


class _FakeQwen35Moe(TurboModel):
    """Model whose default plan mirrors ``qwen3_5_moe.parallel_plan``."""

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        return get_qwen3_5_moe_plan()


class _FakeBailingVL(TurboModel):
    """Model whose default plan mirrors ``bailing_moe_v3_vl.parallel_plan``."""

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        return get_bailing_vl_plan()


def _resolve_engine_plan(
    cfg: EngineConfig, model_cls: type[TurboModel] = _FakeQwen35Moe
):
    """Run the engine's model-default + engine-config plan merge."""
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.config = cfg
    return engine._resolve_parallel_plan(model_cls())


# ---------------------------------------------------------------------------
# Explicit CLI override extraction
# ---------------------------------------------------------------------------


def test_parse_overrides_only_explicit_flags():
    tt, remaining = parse_torchturbo_args(
        ["--tt_sp_size", "2", "--model_name_or_path", "Qwen/Qwen2.5-7B"]
    )

    assert remaining == ["--model_name_or_path", "Qwen/Qwen2.5-7B"]
    assert tt.config_overrides == {"tt_sp_size": 2}
    # Default-valued flags must not leak into overrides.
    assert "tt_cpu_offload" not in tt.config_overrides
    assert "tt_gradient_checkpointing" not in tt.config_overrides
    assert "tt_prefetch_strategy" not in tt.config_overrides


def test_parse_explicit_false_is_an_override():
    tt, _ = parse_torchturbo_args(["--tt_gradient_checkpointing", "false"])

    assert tt.config_overrides == {"tt_gradient_checkpointing": False}


def test_parse_bool_flag_without_value_is_true_override():
    tt, _ = parse_torchturbo_args(["--tt_gradient_checkpointing"])

    assert tt.config_overrides == {"tt_gradient_checkpointing": True}


def test_parse_runtime_flags_excluded_from_overrides():
    tt, _ = parse_torchturbo_args(
        [
            "--tt_config",
            "cfg.yaml",
            "--tt_profiling",
            "--tt_profiling_config",
            '{"detailed_kernel_info": true, "with_stack": true}',
            "--tt_log_level",
            "DEBUG",
            "--tt_memory_snapshot_steps",
            "2",
            "--tt_ep_size",
            "4",
        ]
    )

    assert tt.config_overrides == {"tt_ep_size": 4}
    assert tt.profiling_config == {"detailed_kernel_info": True, "with_stack": True}


def test_parse_explicit_default_value_still_an_override():
    """Passing the default explicitly must still beat a config-file value."""
    tt, _ = parse_torchturbo_args(["--tt_prefetch_strategy", "auto"])

    assert tt.config_overrides == {"tt_prefetch_strategy": "auto"}


# ---------------------------------------------------------------------------
# YAML / JSON file loading
# ---------------------------------------------------------------------------

YAML_CFG = """\
engine_config:
  mixed_precision: fp32
  parallel_plan:
    sp_size: 2
    ep_size: 4
  activation_checkpointing:
    enabled: false
"""


def test_build_from_yaml_with_engineconfig_wrapper(tmp_path):
    cfg = _build_from_argv(["--tt_config", _write(tmp_path, "cfg.yaml", YAML_CFG)])

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32
    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 4
    assert cfg.activation_checkpointing.enabled is False


def test_build_from_json_with_engineconfig_wrapper(tmp_path):
    data = {
        "engine_config": {
            "mixed_precision": "fp32",
            "parallel_plan": {"sp_size": 2, "ep_size": 4},
            "activation_checkpointing": {"enabled": False},
        }
    }
    cfg = _build_from_argv(
        ["--tt_config", _write(tmp_path, "cfg.json", json.dumps(data))]
    )

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32
    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 4
    assert cfg.activation_checkpointing.enabled is False


def test_vision_balance_defaults_to_off(tmp_path):
    path = _write(tmp_path, "vb.yaml", _engine_config("mixed_precision: fp32\n"))

    cfg = _build_from_argv(["--tt_config", path])

    plan = _resolve_engine_plan(cfg, _FakeBailingVL)

    vision_spec = plan.plan["model.visual"].vision_balance
    assert vision_spec.mode is None
    assert isinstance(vision_spec.io, BailingVisionBalanceIO)


@pytest.mark.parametrize("mode", ["pixel", True, False])
def test_invalid_vision_balance_mode_in_file_raises(tmp_path, mode):
    plan = {"model.visual": {"vision_balance": {"mode": mode}}}
    path = _write(
        tmp_path,
        "vb.yaml",
        _engine_config(
            yaml.safe_dump({"parallel_plan": {"plan": plan}})
        ),
    )

    with pytest.raises(ValueError, match="VisionBalanceMode"):
        _build_from_argv(["--tt_config", path])


@pytest.mark.parametrize("value", ["patch", True, False, []])
def test_vision_balance_config_requires_mapping(tmp_path, value):
    plan = {"model.visual": {"vision_balance": value}}
    path = _write(
        tmp_path,
        "vb.yaml",
        _engine_config(yaml.safe_dump({"parallel_plan": {"plan": plan}})),
    )

    with pytest.raises(ValueError, match="vision_balance must be a mapping"):
        _build_from_argv(["--tt_config", path])


@pytest.mark.parametrize("key", ["io", "enabled", "mdoe"])
def test_vision_balance_config_rejects_unknown_keys(tmp_path, key):
    plan = {"model.visual": {"vision_balance": {key: "patch"}}}
    path = _write(
        tmp_path,
        "vb.yaml",
        _engine_config(yaml.safe_dump({"parallel_plan": {"plan": plan}})),
    )

    with pytest.raises(ValueError, match=rf"Unknown key: .*vision_balance\.{key}"):
        _build_from_argv(["--tt_config", path])


def test_build_muon_runtime_policy_from_yaml(tmp_path):
    path = _write(
        tmp_path,
        "muon.yaml",
        _engine_config(
            """\
optimizer:
  name: muon
  muon:
    ns_steps: 3
    overlap_comm_compute: false
    bucket:
      strategy: size
      bucket_cap_mb: 32
"""
        ),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.optimizer.name == "muon"
    assert cfg.optimizer.muon.ns_steps == 3
    assert cfg.optimizer.muon.overlap_comm_compute is False
    assert cfg.optimizer.muon.bucket.strategy == "size"
    assert cfg.optimizer.muon.bucket.bucket_cap_mb == 32


def test_optimizer_config_file_paths_follow_dataclass_schema():
    from dataclasses import fields

    from torchturbo.engine.config import (
        MuonOptimizerConfig,
        OptimizerBucketPolicy,
        _validate_config_file,
    )

    muon = {
        field_info.name: None
        for field_info in fields(MuonOptimizerConfig)
        if field_info.name != "bucket"
    }
    muon["bucket"] = {
        field_info.name: None for field_info in fields(OptimizerBucketPolicy)
    }

    _validate_config_file({"optimizer": {"name": None, "muon": muon}})


def test_build_muon_per_step_coefficients_and_adamw_lr_from_yaml(tmp_path):
    from torch import nn

    from torchturbo.engine.optimizers import MuonPlan, MuonRecipe

    coefficients = [
        [4.0848, -6.8946, 2.9270],
        [3.9505, -6.3029, 2.6377],
        [3.7418, -5.5913, 2.3037],
        [2.8769, -3.1427, 1.2046],
        [2.8366, -3.0525, 1.2012],
    ]
    content = {
        "engine_config": {
            "optimizer": {
                "name": "muon",
                "muon": {
                    "lr": 2.5e-5,
                    "weight_decay": 0.01,
                    "ns_steps": 5,
                    "ns_coefficients": coefficients,
                    "adamw_lr": 1.0e-5,
                },
            }
        }
    }
    path = _write(tmp_path, "muon.yaml", yaml.safe_dump(content))

    cfg = _build_from_argv(["--tt_config", path])
    plan = MuonPlan(
        nn.Linear(2, 2), recipe=MuonRecipe({}), config=cfg.optimizer.muon
    )

    assert plan.muon_kwargs["ns_coefficients"] == tuple(map(tuple, coefficients))
    assert plan.muon_kwargs["ns_steps"] == 5
    assert plan.muon_kwargs["lr"] == 2.5e-5
    assert plan.adamw_kwargs["lr"] == 1.0e-5
    assert plan.muon_fqns == ("weight",)
    assert plan.adamw_no_decay_fqns == ("bias",)


def test_unknown_optimizer_config_file_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "unknown_optimizer.yaml",
        _engine_config("optimizer:\n  name: muon\n  muon:\n    typo_field: 1\n"),
    )

    with pytest.raises(ValueError, match="Unknown key: optimizer.muon.typo_field"):
        _build_from_argv(["--tt_config", path])


def test_config_without_engineconfig_wrapper_raises(tmp_path):
    data = {
        "mixed_precision": "fp32",
        "parallel_plan": {"sp_size": 3},
    }
    path = _write(tmp_path, "cfg.json", json.dumps(data))

    with pytest.raises(
        ValueError,
        match="Unknown key outside engine_config: mixed_precision",
    ):
        _build_from_argv(["--tt_config", path])


@pytest.mark.parametrize(
    "content",
    ["engine_config: {}\n", "engine_config: null\n", "engine_config:\n  # sp_size: 2\n"],
)
def test_empty_config_file_is_noop(tmp_path, content):
    cfg = _build_from_argv(
        ["--tt_config", _write(tmp_path, "empty.yaml", content)]
    )

    assert cfg.parallel_plan.sp_size == 1
    assert cfg.activation_checkpointing.enabled is True
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32


@pytest.mark.parametrize("value", ["false", "0", '""', "[]"])
def test_non_mapping_engine_config_raises(tmp_path, value):
    path = _write(tmp_path, "invalid.yaml", f"engine_config: {value}\n")

    with pytest.raises(ValueError, match="engine_config must be a mapping"):
        _build_from_argv(["--tt_config", path])


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_from_args(TorchTurboArgs(config_path=str(tmp_path / "nope.yaml")))


def test_malformed_yaml_raises(tmp_path):
    with pytest.raises(yaml.parser.ParserError):
        _build_from_argv(
            ["--tt_config", _write(tmp_path, "bad.yaml", "a: [\n")]
        )


def test_unknown_top_level_field_raises(tmp_path):
    path = _write(tmp_path, "unknown.yaml", "bogus: 1\n")

    with pytest.raises(ValueError, match="Unknown key outside engine_config: bogus"):
        _build_from_argv(["--tt_config", path])


def test_unknown_root_sibling_with_engineconfig_wrapper_raises(tmp_path):
    path = _write(
        tmp_path,
        "unknown_root_sibling.yaml",
        "engine_config:\n  backend: fsdp2\nbogus: 1\n",
    )

    with pytest.raises(
        ValueError,
        match="Unknown key outside engine_config: bogus",
    ):
        _build_from_argv(["--tt_config", path])


def test_unknown_nested_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("backend_config:\n  cpu_offloda: true\n"),
    )

    with pytest.raises(ValueError, match="Unknown key: backend_config.cpu_offloda"):
        _build_from_argv(["--tt_config", path])


def test_oc_env_interpolation_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_MODEL_NAME_OR_PATH", "/models/qwen")
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "mfu:\n  card: ${oc.env:TT_MODEL_NAME_OR_PATH}\n"
        ),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.mfu.card == "/models/qwen"


def test_oc_env_interpolation_without_env_raises(tmp_path):
    from omegaconf.errors import InterpolationResolutionError

    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "mfu:\n  card: ${oc.env:TT_MODEL_NAME_OR_PATH}\n"
        ),
    )

    with pytest.raises(InterpolationResolutionError):
        _build_from_argv(["--tt_config", path])


# ---------------------------------------------------------------------------
# Four-layer precedence: TT args < env < file < CLI
# ---------------------------------------------------------------------------


def test_env_overrides_tt_defaults(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")

    cfg = build_from_args()

    assert cfg.parallel_plan.sp_size == 2
    assert cfg.activation_checkpointing.enabled is False


def test_file_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_MIXED_PRECISION", "bf16")
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("parallel_plan:\n  sp_size: 4\nmixed_precision: fp32\n"),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32


@pytest.mark.parametrize("policy", ["fp16", "fp8"])
def test_file_rejects_unsupported_mixed_precision(tmp_path, policy):
    path = _write(
        tmp_path, "cfg.yaml", _engine_config(f"mixed_precision: {policy}\n")
    )

    with pytest.raises(
        ValueError, match=f"does not support mixed_precision='{policy}'"
    ):
        _build_from_argv(["--tt_config", path])


def test_cli_overrides_file(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "parallel_plan:\n  sp_size: 4\nactivation_checkpointing:\n  enabled: true\n"
        ),
    )

    cfg = _build_from_argv(
        [
            "--tt_config",
            path,
            "--tt_sp_size",
            "8",
            "--tt_gradient_checkpointing",
            "false",
        ]
    )

    assert cfg.parallel_plan.sp_size == 8
    assert cfg.activation_checkpointing.enabled is False


def test_full_four_layer_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")
    monkeypatch.setenv("TORCH_TURBO_MIXED_PRECISION", "bf16")
    monkeypatch.setenv("TORCH_TURBO_OP_POLICY", "native")
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  sp_size: 4
activation_checkpointing:
  enabled: true
mixed_precision: bf16
ops:
  policy: custom
"""
        ),
    )

    cfg = _build_from_argv(
        [
            "--tt_config",
            path,
            "--tt_sp_size",
            "8",
            "--tt_gradient_checkpointing",
            "false",
            "--tt_mixed_precision",
            "fp32",
            "--tt_op_policy",
            "auto",
        ]
    )

    assert cfg.parallel_plan.sp_size == 8
    assert cfg.activation_checkpointing.enabled is False
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32
    assert cfg.ops.policy is OpPolicy.AUTO


# ---------------------------------------------------------------------------
# Partial nested overrides keep unset values from lower layers
# ---------------------------------------------------------------------------


def test_partial_nested_overrides_keep_unset_base_values(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
backend_config:
  cpu_offload:
    enabled: true
diffusion:
  cap_lambda: 2.0
checkpoint:
  hf_export: final
mtp:
  enabled: true
"""
        ),
    )

    cfg = build_from_args(TorchTurboArgs(config_path=path))

    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "full"
    assert cfg.backend_config.prefetch_strategy.value == "auto"
    assert cfg.diffusion.cap_lambda == 2.0
    assert cfg.diffusion.bd_mask_low == 0.3
    assert cfg.checkpoint.hf_export == "final"
    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.1


def test_yaml_cpu_offload_nested_optimizer_path(tmp_path):
    """A nested cpu_offload struct is read hierarchically from the YAML file.

    Covers the optimizer mode with a custom device-memory budget.
    """
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
engine_config:
  backend_config:
    cpu_offload:
      enabled: true
      mode: optimizer
      optimizer_memory_fraction: 0.5
""",
    )

    cfg = build_from_args(
        TorchTurboArgs(config_path=path)
    )

    offload = cfg.backend_config.cpu_offload
    assert offload.enabled is True
    assert offload.mode.value == "optimizer"
    assert offload.optimizer_memory_fraction == 0.5
    assert offload.mode.value == "optimizer"


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_yaml_cpu_offload_parses_string_enabled(tmp_path, enabled):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            f'backend_config:\n  cpu_offload:\n    enabled: "{enabled}"\n'
            "    mode: optimizer\n    optimizer_memory_fraction: 0.5\n"
        ),
    )

    cfg = _build_from_argv(["--tt_config", path])

    offload = cfg.backend_config.cpu_offload
    assert offload.enabled is (enabled == "true")
    assert offload.mode.value == ("optimizer" if enabled == "true" else "full")
    assert offload.optimizer_memory_fraction == (0.5 if enabled == "true" else 0.8)


@pytest.mark.parametrize("enabled", [None, "true", "false"])
def test_yaml_cpu_offload_resolves_env_enabled(tmp_path, monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("TT_TEST_CPU_OFFLOAD", raising=False)
    else:
        monkeypatch.setenv("TT_TEST_CPU_OFFLOAD", enabled)
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "backend_config:\n  cpu_offload:\n"
            "    enabled: ${oc.env:TT_TEST_CPU_OFFLOAD,false}\n"
        ),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.backend_config.cpu_offload.enabled is (enabled == "true")


def test_yaml_cpu_offload_rejects_legacy_enable_key(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("backend_config:\n  cpu_offload:\n    enable: true\n"),
    )

    with pytest.raises(
        ValueError, match="Unknown key: backend_config.cpu_offload.enable"
    ):
        _build_from_argv(["--tt_config", path])


def test_yaml_cpu_offload_rejects_removed_swap_options(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "backend_config:\n  cpu_offload:\n    enabled: true\n"
            "    swap_optimizer: {enabled: true}\n"
        ),
    )

    with pytest.raises(
        ValueError, match="Unknown key: backend_config.cpu_offload.swap_optimizer"
    ):
        _build_from_argv(["--tt_config", path])


@pytest.mark.parametrize("cli_value", ["true", "false"])
def test_cli_cpu_offload_overrides_file_enabled(tmp_path, cli_value):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            "backend_config:\n  cpu_offload:\n    enabled: false\n"
            "    mode: optimizer\n    optimizer_memory_fraction: 0.5\n"
        ),
    )

    cfg = _build_from_argv(
        ["--tt_config", path, "--tt_cpu_offload", cli_value]
    )

    assert cfg.backend_config.cpu_offload.enabled is (cli_value == "true")
    assert cfg.backend_config.cpu_offload.mode.value == (
        "optimizer" if cli_value == "true" else "full"
    )


def test_file_checkpoint_export_overrides_the_default(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("checkpoint:\n  hf_export: final\n"),
    )

    cfg = build_from_args(TorchTurboArgs(config_path=path))

    assert cfg.checkpoint.hf_export == "final"


def test_hf_export_comes_from_file_and_cli(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("checkpoint:\n  hf_export: every\n"))

    assert _build_from_argv(["--tt_config", path]).checkpoint.hf_export == "every"
    assert (
        _build_from_argv(["--tt_config", path, "--tt_hf_export", "final"]).checkpoint.hf_export
        == "final"
    )


def test_hf_export_can_be_seeded_on_torchturbo_args():
    checkpoint = build_from_args(TorchTurboArgs(hf_export="every")).checkpoint

    assert checkpoint.hf_export == "every"


def test_default_save_format_is_not_supported(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("checkpoint:\n  default_save_format: hf\n"),
    )

    with pytest.raises(ValueError, match="Unknown key: checkpoint.default_save_format"):
        _build_from_argv(["--tt_config", path])


def test_removed_default_save_format_cli_is_not_consumed():
    _, remaining = parse_torchturbo_args(["--tt_default_save_format", "hf"])

    assert remaining == ["--tt_default_save_format", "hf"]


def test_invalid_hf_export_fails_when_the_config_is_built(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("checkpoint:\n  hf_export: always\n"),
    )

    with pytest.raises(ValueError, match="hf_export"):
        _build_from_argv(["--tt_config", path])


def test_tt_mtp_values_kept_when_file_only_touches_enabled(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("mtp:\n  enabled: false\n"))

    cfg = build_from_args(
        TorchTurboArgs(
            enable_mtp=True,
            mtp_loss_scaling_factor=0.5,
            config_path=path,
        )
    )

    assert cfg.mtp.enabled is False
    assert cfg.mtp.loss_scaling_factor == 0.5


def test_cli_mtp_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("mtp:\n  enabled: true\n"))

    cfg = _build_from_argv(["--tt_config", path, "--tt_enable_mtp", "false"])

    assert cfg.mtp.enabled is False


def test_cli_dspark_overrides_file_and_keeps_unset_defaults(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("dspark:\n  enabled: false\n  num_anchors: 256\n"),
    )

    cfg = _build_from_argv(["--tt_config", path, "--tt_dspark_num_anchors", "128"])

    assert cfg.dspark == DSparkTrainingConfig(enabled=False, num_anchors=128)


def test_cli_dspark_flags_map_to_config_fields():
    cfg = _build_from_argv(
        [
            "--tt_enable_dspark", "false",
            "--tt_dspark_loss_scaling_factor", "0.3",
            "--tt_dspark_num_anchors", "64",
            "--tt_dspark_loss_decay_gamma", "2.0",
            "--tt_dspark_ce_loss_alpha", "0.2",
            "--tt_dspark_l1_loss_alpha", "0.7",
            "--tt_dspark_confidence_loss_alpha", "0.5",
        ]
    )

    assert cfg.dspark == DSparkTrainingConfig(
        enabled=False,
        loss_scaling_factor=0.3,
        num_anchors=64,
        loss_decay_gamma=2.0,
        ce_loss_alpha=0.2,
        l1_loss_alpha=0.7,
        confidence_loss_alpha=0.5,
    )


def test_cli_op_policy_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("ops:\n  policy: custom\n"))

    cfg = _build_from_argv(["--tt_config", path, "--tt_op_policy", "auto"])

    assert cfg.ops.policy is OpPolicy.AUTO


def test_file_op_overrides_are_applied(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("ops:\n  overrides:\n    rms_norm: torch_rms_norm\n"),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.ops.overrides == {"rms_norm": "torch_rms_norm"}


def test_config_example_loads():
    path = Path(__file__).parents[2] / "scripts" / "runs" / "config_example.yaml"

    cfg = _build_from_argv(["--tt_config", str(path)])

    assert cfg.ops.overrides == {"rms_norm": "torch_rms_norm"}
    assert cfg.backend_config.cpu_offload.enabled is False
    assert cfg.optimizer.name == "muon"


def test_cli_use_deterministic_algorithms_overrides_file(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("use_deterministic_algorithms: true\n"),
    )

    from_file = _build_from_argv(["--tt_config", path])
    from_cli = _build_from_argv(
        ["--tt_config", path, "--tt_use_deterministic_algorithms", "false"]
    )

    assert from_file.use_deterministic_algorithms is True
    assert from_cli.use_deterministic_algorithms is False


# ---------------------------------------------------------------------------
# Freeze policy layering
# ---------------------------------------------------------------------------


def test_tt_freeze_flags_kept_without_file_or_cli():
    cfg = build_from_args(TorchTurboArgs(freeze_language_model=True))

    assert cfg.freeze_policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_file_freeze_parameter_groups_raise(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("freeze_policy:\n  parameter_groups: [vision_tower]\n"),
    )

    with pytest.raises(ValueError, match="Unknown key: freeze_policy.parameter_groups"):
        build_from_args(TorchTurboArgs(freeze_language_model=True, config_path=path))


def test_explicit_false_freeze_clears_all_groups(tmp_path):
    tt, _ = parse_torchturbo_args(["--tt_freeze_vision_tower", "false"])

    cfg = build_from_args(tt)

    assert cfg.freeze_policy.parameter_groups == ()


# ---------------------------------------------------------------------------
# parallel_plan.plan parsing from files
# ---------------------------------------------------------------------------


def test_file_parallel_plan_parsed_into_fsdp_plan_specs(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  sp_size: 4
  plan:
    "model.layers.*":
      activation_checkpoint: true
    "model.visual":
      standalone_world_shard: true
      vision_balance:
        mode: patch_sq
"""
        ),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert isinstance(cfg.parallel_plan.plan["model.layers.*"], FSDPPlanSpec)
    assert cfg.parallel_plan.plan["model.layers.*"].activation_checkpoint is True
    assert cfg.parallel_plan.plan["model.visual"].standalone_world_shard is True
    assert (
        cfg.parallel_plan.plan["model.visual"].vision_balance.mode
        is VisionBalanceMode.PATCH_SQ
    )


def test_file_parallel_plan_rejects_runtime_only_tp_size(tmp_path):
    path = _write(
        tmp_path, "cfg.yaml", _engine_config("parallel_plan:\n  tp_size: 2\n"),
    )
    with pytest.raises(ValueError, match="Unknown key: parallel_plan.tp_size"):
        _build_from_argv(["--tt_config", path])


def test_file_plan_without_sp_ep_keeps_tt_values(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.layers.*":
      activation_checkpoint: true
"""
        ),
    )

    cfg = build_from_args(
        TorchTurboArgs(sp_size=2, ep_size=3, config_path=path)
    )

    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 3
    assert "model.layers.*" in cfg.parallel_plan.plan


# ---------------------------------------------------------------------------
# Model default parallel plan vs config file conflicts
# ---------------------------------------------------------------------------


def test_config_sp_ep_override_model_default_plan(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("parallel_plan:\n  sp_size: 2\n  ep_size: 8\n"),
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.sp_size == 2
    assert plan.ep_size == 8


def test_config_disables_model_default_activation_checkpoint(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.language_model.layers.*":
      activation_checkpoint: false
"""
        ),
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.language_model.layers.*"].activation_checkpoint is False
    # Untouched FQNs must keep their model defaults (MoE replacement, kwargs).
    mlp = plan.plan["model.language_model.layers.*.mlp"]
    assert mlp.replace_module is not None
    assert mlp.module_kwargs


def test_config_disables_model_default_standalone_world_shard(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.visual":
      standalone_world_shard: false
"""
        ),
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.visual"].standalone_world_shard is False


@pytest.mark.parametrize("source", ["yaml", "python", "dict"])
@pytest.mark.parametrize("mode", ["patch", "patch_sq"])
def test_parallel_plan_vision_balance_keeps_bailing_sharding(
    tmp_path, source, mode
):
    spec = {"vision_balance": {"mode": mode}}

    if source == "yaml":
        path = _write(
            tmp_path,
            "cfg.yaml",
            _engine_config(
                yaml.safe_dump(
                    {
                        "parallel_plan": {"plan": {"model.visual": spec}},
                    }
                )
            ),
        )
        cfg = _build_from_argv(["--tt_config", path])
    elif source == "dict":
        cfg = EngineConfig.from_dict(
            {"parallel_plan": {"plan": {"model.visual": spec}}}
        )
        assert "parallel_plan" not in cfg.extra
    else:
        cfg = EngineConfig(
            parallel_plan=FSDPParallelPlan(
                plan={
                    "model.visual": FSDPPlanSpec(
                        standalone_world_shard=True,
                        vision_balance=VisionBalanceSpec(mode=VisionBalanceMode(mode)),
                    )
                }
            ),
        )

    plan = _resolve_engine_plan(cfg, _FakeBailingVL)

    vision_spec = plan.plan["model.visual"].vision_balance
    assert vision_spec.mode is VisionBalanceMode(mode)
    assert isinstance(vision_spec.io, BailingVisionBalanceIO)
    assert plan.plan["model.visual"].standalone_world_shard is True
    assert plan.plan["model.visual.blocks.*"].activation_checkpoint is True


@pytest.mark.parametrize("override_mode", [None, VisionBalanceMode.PATCH])
@pytest.mark.parametrize("replace_io", [False, True])
def test_parallel_plan_merges_vision_spec_without_mutating_default(
    override_mode, replace_io
):
    default_io = BailingVisionBalanceIO()
    override_io = BailingVisionBalanceIO() if replace_io else None
    default_vision = VisionBalanceSpec(
        mode=VisionBalanceMode.PATCH_SQ, io=default_io
    )
    default_plan = FSDPParallelPlan(
        plan={"visual": FSDPPlanSpec(vision_balance=default_vision)}
    )

    class ModelWithVisionDefault(TurboModel):
        @classmethod
        def get_default_parallel_plan(cls, **kwargs):
            return default_plan

    cfg = EngineConfig(
        parallel_plan=FSDPParallelPlan(
            plan={
                "visual": FSDPPlanSpec(
                    vision_balance=VisionBalanceSpec(
                        mode=override_mode, io=override_io
                    )
                )
            }
        )
    )

    plan = _resolve_engine_plan(cfg, ModelWithVisionDefault)
    vision_spec = plan.plan["visual"].vision_balance

    assert vision_spec is not default_vision
    assert vision_spec.mode is (override_mode or VisionBalanceMode.PATCH_SQ)
    assert vision_spec.io is (override_io if replace_io else default_io)
    assert default_vision.mode is VisionBalanceMode.PATCH_SQ
    assert default_vision.io is default_io


def test_config_adds_plan_entry_missing_from_model_default(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.new_module":
      activation_checkpoint: true
"""
        ),
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.new_module"].activation_checkpoint is True


def test_config_no_shard_conflicts_with_model_placement(tmp_path):
    """A file-level no_shard on the MoE experts collides with the model's
    default expert placement and must surface as an error, not silently win."""
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.language_model.layers.*.mlp.experts":
      no_shard: true
"""
        ),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_engine_plan(_build_from_argv(["--tt_config", path]))


def test_sample_style_config_patches_model_default_without_losing_defaults(
    tmp_path,
):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  sp_size: 2
  ep_size: 8
  plan:
    "model.language_model.layers.*":
      activation_checkpoint: true
    "model.visual":
      standalone_world_shard: true
    "model.language_model.embed_tokens":
      standalone_world_shard: true
    "lm_head":
      standalone_world_shard: true
    "mtp.layers.*":
      activation_checkpoint: true
"""
        ),
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.sp_size == 2
    assert plan.ep_size == 8
    # Model defaults that the file does not touch must survive the merge.
    assert plan.plan["model.language_model.layers.*.mlp"].replace_module is not None
    assert plan.plan["model.language_model.layers.*.mlp.experts"].placement is not None
    assert plan.plan["mtp.layers.*.mlp"].replace_module is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_env_sp_size_raises(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "abc")

    with pytest.raises(ValueError):
        build_from_args()


def test_invalid_env_backend_raises(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_BACKEND", "bogus")

    with pytest.raises(ValueError):
        build_from_args()


def test_null_mixed_precision_in_file_raises(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("mixed_precision: null\n"))

    with pytest.raises(ValueError):
        _build_from_argv(["--tt_config", path])


def test_flat_op_policy_in_file_raises(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("op_policy: native\n"))

    with pytest.raises(ValueError, match="Unknown key: op_policy"):
        _build_from_argv(["--tt_config", path])


# ---------------------------------------------------------------------------
# Known gaps (documented with strict xfail so fixes flip them to XPASS)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="from_dict supports flat sp_size/ep_size, but the file overlay only "
    "applies EngineConfig fields; flat keys are silently dropped",
)
def test_flat_parallel_keys_in_file_are_applied(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("sp_size: 4\nep_size: 2\n"))

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert cfg.parallel_plan.ep_size == 2


def test_flat_mtp_keys_in_file_raise(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("enable_mtp: true\n"))

    with pytest.raises(ValueError, match="Unknown key: enable_mtp"):
        _build_from_argv(["--tt_config", path])


def test_flat_checkpoint_keys_in_file_raise(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("hf_export: final\n"))

    with pytest.raises(ValueError, match="Unknown key: hf_export"):
        _build_from_argv(["--tt_config", path])


@pytest.mark.xfail(
    strict=True,
    reason="legacy parallel key is accepted by from_dict but the file overlay "
    "only looks up EngineConfig.parallel_plan",
)
def test_legacy_parallel_key_in_file_is_applied(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("parallel:\n  sp_size: 4\n"),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4


@pytest.mark.xfail(
    strict=True,
    reason="legacy fsdp key is accepted by from_dict but the file overlay only "
    "looks up EngineConfig.backend_config",
)
def test_legacy_fsdp_key_in_file_is_applied(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config("fsdp:\n  cpu_offload:\n    enabled: true\n"),
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.backend_config.cpu_offload.enabled is True
    assert cfg.backend_config.cpu_offload.mode.value == "full"


def test_unknown_plan_spec_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.layers.*":
      activation_checkpointing: true
"""
        ),
    )

    with pytest.raises(ValueError, match="activation_checkpointing"):
        _build_from_argv(["--tt_config", path])


def test_unknown_plan_spec_nested_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        _engine_config(
            """\
parallel_plan:
  plan:
    "model.layers.*":
      activation_checkpoint: true
      typo_flag: true
"""
        ),
    )

    with pytest.raises(ValueError, match="typo_flag"):
        _build_from_argv(["--tt_config", path])


# ---------------------------------------------------------------------------
# Global batch size (--tt_gbs -> gbs)
# ---------------------------------------------------------------------------


def test_gbs_defaults_to_none():
    cfg = build_from_args(TorchTurboArgs())

    assert cfg.gbs is None


def test_gbs_from_turboargs():
    cfg = build_from_args(TorchTurboArgs(gbs=32))

    assert cfg.gbs == 32


def test_gbs_from_cli_flag(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config(""))
    cfg = _build_from_argv(["--tt_config", path, "--tt_gbs", "32"])

    assert cfg.gbs == 32


def test_gbs_from_config_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("gbs: 32\n"))
    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.gbs == 32


def test_gbs_cli_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _engine_config("gbs: 8\n"))
    cfg = _build_from_argv(["--tt_config", path, "--tt_gbs", "32"])

    assert cfg.gbs == 32


def test_gbs_env_var_overrides(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_GBS", "16")
    cfg = build_from_args(TorchTurboArgs())

    assert cfg.gbs == 16


def test_gbs_negative_raises():
    with pytest.raises(ValueError):
        build_from_args(TorchTurboArgs(gbs=-1))


def test_gbs_zero_raises():
    with pytest.raises(ValueError):
        build_from_args(TorchTurboArgs(gbs=0))
