"""Tests for TorchTurbo CLI argument parsing."""

import os

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


def test_string_to_bool_matches_transformers_semantics():
    from torchturbo.engine.args import string_to_bool

    assert string_to_bool("y") is True
    assert string_to_bool("f") is False


def test_parse_torchturbo_args_log_level():
    from torchturbo.engine.args import parse_torchturbo_args
    from torchturbo.logging import LogLevel

    args, _ = parse_torchturbo_args([])
    assert args.log_level is LogLevel.INFO

    args, remaining = parse_torchturbo_args(
        ["--tt_log_level", "DEBUG", "--foo", "bar"]
    )
    assert args.log_level is LogLevel.DEBUG
    assert remaining == ["--foo", "bar"]


def test_parse_torchturbo_args_prefetch_depth():
    from torchturbo.engine.args import parse_torchturbo_args

    args, _ = parse_torchturbo_args([])
    assert args.forward_prefetch_depth == 1
    assert args.backward_prefetch_depth == 1

    args, _ = parse_torchturbo_args(
        ["--tt_forward_prefetch_depth", "3", "--tt_backward_prefetch_depth", "2"]
    )
    assert args.forward_prefetch_depth == 3
    assert args.backward_prefetch_depth == 2


def test_parse_torchturbo_args_prefetch_defaults():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert remaining == []
    assert args.prefetch_strategy == "auto"
    assert args.prefetch_direction == "forward_backward"
    assert args.op_policy is None


def test_parse_torchturbo_args_op_policy():
    from torchturbo.engine.args import parse_torchturbo_args

    args, _ = parse_torchturbo_args(["--tt_op_policy", "custom"])

    assert args.op_policy == "custom"


def test_parse_torchturbo_args_optimizer():
    from torchturbo.engine.args import parse_torchturbo_args
    from torchturbo.engine.config import build_from_args

    args, remaining = parse_torchturbo_args(
        ["--tt_optimizer", "muon", "--framework-flag"]
    )

    assert args.optimizer_name == "muon"
    assert remaining == ["--framework-flag"]
    assert build_from_args(args).optimizer.name == "muon"


def test_parse_torchturbo_args_rejects_unsupported_optimizer():
    from torchturbo.engine.args import parse_torchturbo_args

    with pytest.raises(SystemExit, match="2"):
        parse_torchturbo_args(["--tt_optimizer", "adamw"])


def test_parse_torchturbo_args_use_deterministic_algorithms():
    from torchturbo.engine.args import parse_torchturbo_args

    default_args, _ = parse_torchturbo_args([])
    assert default_args.use_deterministic_algorithms is False

    enabled_args, _ = parse_torchturbo_args(["--tt_use_deterministic_algorithms"])
    assert enabled_args.use_deterministic_algorithms is True
    assert enabled_args.config_overrides == {"tt_use_deterministic_algorithms": True}

    disabled_args, _ = parse_torchturbo_args(["--tt_use_deterministic_algorithms=false"])
    assert disabled_args.use_deterministic_algorithms is False
    assert disabled_args.config_overrides == {"tt_use_deterministic_algorithms": False}


def test_parse_torchturbo_args_moe_aux_loss():
    from torchturbo.engine.args import parse_torchturbo_args
    from torchturbo.engine.config import build_from_args

    default_args, remaining = parse_torchturbo_args([])
    assert default_args.moe_aux_loss_coeff == 0.0
    assert build_from_args(default_args).moe.aux_loss_coeff == 0.0
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_moe_aux_loss_coeff", "0.01"]
    )

    assert args.moe_aux_loss_coeff == 0.01
    assert remaining == []

    config = build_from_args(args)
    assert config.moe.aux_loss_coeff == 0.01


def test_moe_dispatcher_config():
    from torchturbo.engine.args import parse_torchturbo_args
    from torchturbo.engine.config import EngineConfig, build_from_args

    args, remaining = parse_torchturbo_args(["--tt_moe_token_dispatcher", "alltoall"])
    assert remaining == []
    config = build_from_args(args)
    assert config.moe.token_dispatcher == "alltoall"
    assert EngineConfig.from_dict(config.to_dict()).moe.token_dispatcher == "alltoall"
    with pytest.raises(ValueError, match="moe.token_dispatcher"):
        EngineConfig.from_dict({"moe": {"token_dispatcher": "invalid"}})


def test_parse_torchturbo_args_moe_z_loss():
    from torchturbo.engine.args import parse_torchturbo_args
    from torchturbo.engine.config import build_from_args

    default_args, remaining = parse_torchturbo_args([])
    assert default_args.moe_z_loss_coeff == 0.0
    assert build_from_args(default_args).moe.z_loss_coeff == 0.0
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_moe_z_loss_coeff", "0.001"]
    )

    assert args.moe_z_loss_coeff == 0.001
    assert remaining == []

    config = build_from_args(args)
    assert config.moe.z_loss_coeff == 0.001


def test_parse_torchturbo_args_gradient_checkpointing():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert args.gradient_checkpointing is True
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_gradient_checkpointing", "true", "--foo", "bar"]
    )
    assert args.gradient_checkpointing is True
    assert remaining == ["--foo", "bar"]

    args, _ = parse_torchturbo_args(["--tt_gradient_checkpointing=false"])
    assert args.gradient_checkpointing is False

    args, _ = parse_torchturbo_args(["--tt_gradient_checkpointing"])
    assert args.gradient_checkpointing is True


def test_parse_torchturbo_args_cpu_offload_overrides_enabled_flag():
    """``--tt_cpu_offload`` is a boolean override, not a JSON policy."""
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert args.cpu_offload.enabled is None
    assert args.cpu_offload_enabled is None
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_cpu_offload", "false", "--foo", "bar"]
    )
    assert args.cpu_offload.enabled is None
    assert args.cpu_offload_enabled is False
    assert args.config_overrides["tt_cpu_offload"] is False
    assert remaining == ["--foo", "bar"]

    args, remaining = parse_torchturbo_args(["--tt_cpu_offload"])
    assert args.cpu_offload_enabled is True
    assert remaining == []


def test_parse_torchturbo_args_freeze_flags():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args(
        [
            "--tt_freeze_vision_tower",
            "--tt_freeze_multi_modal_projector",
            "--foo",
            "bar",
        ]
    )

    assert remaining == ["--foo", "bar"]
    assert args.freeze_vision_tower is True
    assert args.freeze_multi_modal_projector is True
    assert args.freeze_language_model is False


def test_parse_torchturbo_args_chunked_loss():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert args.enable_chunk_loss is False
    assert args.loss_chunk_size is None
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        [
            "--tt_enable_chunk_loss",
            "--tt_loss_chunk_size",
            "384",
            "--other",
        ]
    )
    assert args.enable_chunk_loss is True
    assert args.loss_chunk_size == 384
    assert remaining == ["--other"]


def test_parse_torchturbo_args_mtp_log_loss():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args(
        ["--tt_mtp_log_loss", "--other"]
    )

    assert args.mtp_log_loss is True
    assert remaining == ["--other"]


def test_parse_torchturbo_args_profiling():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args(
        [
            "--tt_profiling",
            "--tt_profiling_config",
            '{"detailed_kernel_info": true, "profiling_step": 20}',
            "--other",
        ]
    )
    assert args.profiling is True
    assert args.profiling_config == {
        "detailed_kernel_info": True,
        "profiling_step": 20,
    }
    # Non-tt args stay in remaining for the training framework parser.
    assert remaining == ["--other"]

    # The config dict parses on its own but does not enable profiling.
    args, _ = parse_torchturbo_args(["--tt_profiling_config", '{"with_stack": true}'])
    assert args.profiling is False
    assert args.profiling_config == {"with_stack": True}

    for invalid in (
        "bogus",
        '{"bad": true}',
        '{"detailed_kernel_info": "yes"}',
        '{"profiling_step": 0}',
        '{"profiling_step": "2"}',
    ):
        with pytest.raises(SystemExit, match="2"):
            parse_torchturbo_args(["--tt_profiling_config", invalid])
