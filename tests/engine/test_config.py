"""Tests for TorchTurbo engine config mapping."""

import json
import os

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


def test_engine_config_env_injects_gradient_checkpointing(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")

    cfg = build_from_args(TorchTurboArgs())

    assert cfg.activation_checkpointing.enabled is False


def test_engine_config_env_injects_op_policy(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args
    from torchturbo.ops.base import OpPolicy

    monkeypatch.setenv("TORCH_TURBO_OP_POLICY", "native")

    cfg = build_from_args(TorchTurboArgs())

    assert cfg.ops.policy is OpPolicy.NATIVE


def test_engine_config_from_dict_rejects_flat_op_policy():
    from torchturbo.engine.config import EngineConfig

    with pytest.raises(ValueError, match="op_policy is no longer supported"):
        EngineConfig.from_dict(
            {
                "backend": "fsdp2",
                "op_policy": "native",
            }
        )


@pytest.mark.parametrize("value", [False, "false", "f", "0"])
def test_engine_config_normalizes_disabled_use_deterministic_algorithms(value):
    from torchturbo.engine.config import EngineConfig

    direct = EngineConfig(use_deterministic_algorithms=value)
    from_dict = EngineConfig.from_dict({"use_deterministic_algorithms": value})

    assert direct.use_deterministic_algorithms is False
    assert from_dict.use_deterministic_algorithms is False


@pytest.mark.parametrize("value", [True, "true", "t", "1"])
def test_engine_config_normalizes_enabled_use_deterministic_algorithms(value):
    from torchturbo.engine.config import EngineConfig

    direct = EngineConfig(use_deterministic_algorithms=value)
    from_dict = EngineConfig.from_dict({"use_deterministic_algorithms": value})

    assert direct.use_deterministic_algorithms is True
    assert from_dict.use_deterministic_algorithms is True


def test_fsdp_cpu_offload_defaults_to_disabled():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.backend_config.cpu_offload.enabled is False


def test_engine_config_to_dict_excludes_runtime_state():
    from torchturbo.engine.config import EngineConfig

    data = EngineConfig().to_dict()

    assert "device" not in data
    assert "device_runtime" not in data


@pytest.mark.parametrize("through_json", [False, True])
def test_engine_config_round_trip_preserves_all_settings(through_json):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import DiffusionConfig, EngineConfig, build_from_args
    from torchturbo.parallel import FSDPPlanSpec, VisionBalanceMode, VisionBalanceSpec

    original = build_from_args(TorchTurboArgs(sp_size=2, ep_size=4))
    original.parallel_plan.plan = {
        "model.layers.*": FSDPPlanSpec(activation_checkpoint=True),
        "model.visual": FSDPPlanSpec(
            standalone_world_shard=True,
            vision_balance=VisionBalanceSpec(mode=VisionBalanceMode.PATCH_SQ),
        ),
    }
    original.moe.aux_loss_coeff = 0.01
    original.diffusion = DiffusionConfig(
        cap_lambda=2.0, bd_mask_low=0.2, bd_mask_high=0.9, bd_block_size=16,
    )
    original.plugin_paths = ["custom_plugin.py"]
    original.extra = {"run_name": "round-trip", "metadata": {"seed": 7}}
    restored = original
    for _ in range(2):
        serialized = restored.to_dict()
        if through_json:
            serialized = json.loads(json.dumps(serialized))
        restored = EngineConfig.from_dict(serialized)
        assert restored == original


def test_engine_config_dict_round_trip_restores_runtime_plan_fields():
    from torch.distributed.tensor import Shard

    from torchturbo.engine.config import EngineConfig
    from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.module import ModuleSpec
    from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
    from torchturbo.parallel import (
        FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec,
        VisionBalanceMode, VisionBalanceSpec,
    )

    original = EngineConfig(parallel_plan=FSDPParallelPlan(
        sp_size=2, ep_size=4, tp_size=2,
        plan={"model.visual": FSDPPlanSpec(
            placement=FSDPPlacementSpec(placement=Shard(0), mesh_dim="ep"),
            replace_module=MoELayer,
            module_kwargs={"moe_submodules": MoESubmodules(
                experts=ModuleSpec(module=GroupedMLP),
            )},
            vision_balance=VisionBalanceSpec(
                mode=VisionBalanceMode.PATCH, io=BailingVisionBalanceIO(),
            ),
        )},
    ))
    serialized = original.to_dict()
    restored = EngineConfig.from_dict(serialized)
    spec = restored.parallel_plan.plan["model.visual"]

    assert restored.parallel_plan.tp_size == 2
    assert isinstance(spec.placement, FSDPPlacementSpec)
    assert spec.placement.placement == Shard(0)
    assert spec.placement.mesh_dim == "ep"
    assert spec.vision_balance.mode is VisionBalanceMode.PATCH
    assert isinstance(spec.vision_balance.io, BailingVisionBalanceIO)
    assert spec.vision_balance.io is serialized["parallel_plan"]["plan"]["model.visual"]["vision_balance"]["io"]
    assert spec.replace_module is MoELayer
    submodules = spec.module_kwargs["moe_submodules"]
    assert isinstance(submodules, MoESubmodules)
    assert isinstance(submodules.experts, ModuleSpec)
    assert submodules.experts.module is GroupedMLP


@pytest.mark.parametrize("plan", [
    {"typo": 1},
    {"plan": {"model.layers.*": {"typo": True}}},
    {"plan": {"model.visual": {"vision_balance": {"typo": "patch"}}}},
    {"plan": {"model.visual": {"placement": {"placement": None, "typo": "ep"}}}},
])
def test_dict_parallel_plan_rejects_unknown_fields(plan):
    from torchturbo.engine.config import EngineConfig

    with pytest.raises((TypeError, ValueError), match="typo"):
        EngineConfig.from_dict({"parallel_plan": plan})


@pytest.mark.parametrize("nested_sizes", [False, True])
def test_dict_parallel_plan_keeps_legacy_sizes_unless_overridden(nested_sizes):
    from torchturbo.engine.config import EngineConfig
    from torchturbo.parallel import VisionBalanceMode

    plan = {"plan": {"model.visual": {"vision_balance": {"mode": "patch_sq"}}}}
    if nested_sizes:
        plan.update(sp_size=2, ep_size=4)
    cfg = EngineConfig.from_dict(
        {"sp_size": 8, "parallel": {"ep_size": 16}, "parallel_plan": plan}
    )

    assert cfg.parallel_plan.sp_size == (2 if nested_sizes else 8)
    assert cfg.parallel_plan.ep_size == (4 if nested_sizes else 16)
    vision_spec = cfg.parallel_plan.plan["model.visual"].vision_balance
    assert vision_spec.mode is VisionBalanceMode.PATCH_SQ
    assert "parallel_plan" not in cfg.extra


def test_dict_config_retains_typed_parallel_plan_and_io_instance():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
    from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec, VisionBalanceSpec

    io = BailingVisionBalanceIO()
    plan = FSDPParallelPlan(
        plan={"model.visual": FSDPPlanSpec(vision_balance=VisionBalanceSpec(io=io))}
    )
    cfg = EngineConfig.from_dict({"parallel_plan": plan})

    assert cfg.parallel_plan is plan
    assert cfg.parallel_plan.plan["model.visual"].vision_balance.io is io


@pytest.mark.parametrize("mode", ["invalid", False])
def test_dict_config_rejects_invalid_vision_mode(mode):
    from torchturbo.engine.config import EngineConfig

    plan = {"model.visual": {"vision_balance": {"mode": mode}}}
    with pytest.raises(ValueError, match="VisionBalanceMode"):
        EngineConfig.from_dict({"parallel_plan": {"plan": plan}})


def test_engine_config_from_dict_parses_moe_config():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"moe": {"aux_loss_coeff": 0.01}})

    assert cfg.moe.aux_loss_coeff == 0.01


def test_engine_config_from_dict_parses_muon_runtime_policy():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "optimizer": {
                "name": "muon",
                "muon": {
                    "lr": 0.002,
                    "weight_decay": 0.1,
                    "momentum": 0.9,
                    "ns_steps": 3,
                    "overlap_comm_compute": "false",
                    "bucket": {
                        "strategy": "size",
                        "bucket_cap_mb": 32,
                    },
                    "adamw_betas": [0.8, 0.95],
                    "adamw_eps": 1e-6,
                },
            }
        }
    )

    assert cfg.optimizer.name == "muon"
    assert cfg.optimizer.muon.lr == 0.002
    assert cfg.optimizer.muon.weight_decay == 0.1
    assert cfg.optimizer.muon.momentum == 0.9
    assert cfg.optimizer.muon.ns_steps == 3
    assert cfg.optimizer.muon.overlap_comm_compute is False
    assert cfg.optimizer.muon.bucket.strategy == "size"
    assert cfg.optimizer.muon.bucket.bucket_cap_mb == 32
    assert cfg.optimizer.muon.adamw_betas == (0.8, 0.95)
    assert cfg.optimizer.muon.adamw_eps == 1e-6


def test_engine_optimizer_defaults_to_framework_owned():
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import EngineConfig, build_from_args

    assert EngineConfig().optimizer.name is None
    assert EngineConfig().optimizer.muon is None
    muon_config = build_from_args(TorchTurboArgs(optimizer_name="muon"))
    assert muon_config.optimizer.muon.lr is None
    assert muon_config.optimizer.muon.weight_decay is None
    assert muon_config.optimizer.muon.bucket.strategy == "parameter"


def test_engine_optimizer_rejects_unsupported_name():
    from torchturbo.engine.config import OptimizerConfig

    for name in ("future_optimizer", "adamw", "muno", ""):
        with pytest.raises(ValueError, match="unsupported engine-managed optimizer"):
            OptimizerConfig(name=name)


def test_engine_optimizer_name_is_normalized():
    from torchturbo.engine.config import OptimizerConfig

    assert OptimizerConfig(name=" MUON ").name == "muon"


def test_engine_config_parses_structured_cpu_offload():
    from torchturbo.engine.config import EngineConfig

    structured = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "backend_config": {
                "cpu_offload": {
                    "enabled": True,
                    "mode": "optimizer",
                    "optimizer_memory_fraction": 0,
                }
            },
        }
    )
    assert structured.backend_config.cpu_offload.enabled is True
    assert structured.backend_config.cpu_offload.mode.value == "optimizer"
    assert (
        structured.backend_config.cpu_offload.optimizer_memory_fraction == 0
    )


def test_cpu_offload_workspace_only_applies_to_optimizer_mode():
    from torchturbo.engine import CPUOffloadConfig

    workspace_config = CPUOffloadConfig(
        enabled=True,
        mode="optimizer",
        optimizer_memory_fraction=0.5,
    )
    assert workspace_config.enabled is True
    assert workspace_config.mode.value == "optimizer"
    assert workspace_config.optimizer_memory_fraction > 0

    no_device_staging = CPUOffloadConfig(
        enabled=True,
        mode="optimizer", optimizer_memory_fraction=0
    )
    assert no_device_staging.optimizer_memory_fraction == 0
    assert CPUOffloadConfig(enabled=True).mode.value == "full"


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_cpu_offload_config_rejects_invalid_memory_fraction(fraction):
    from torchturbo.engine import CPUOffloadConfig

    with pytest.raises(ValueError, match="optimizer_memory_fraction must be in"):
        CPUOffloadConfig(
            enabled=True, mode="optimizer", optimizer_memory_fraction=fraction
        )


def test_cpu_offload_config_rejects_invalid_mode():
    from torchturbo.engine import CPUOffloadConfig

    with pytest.raises(ValueError):
        CPUOffloadConfig(enabled=True, mode="invalid")


def test_disabled_cpu_offload_ignores_mode_and_memory_fraction():
    from torchturbo.engine import CPUOffloadConfig

    config = CPUOffloadConfig(
        enabled=False, mode="invalid", optimizer_memory_fraction=-1
    )

    assert config.enabled is False
    assert config.mode.value == "full"
    assert config.optimizer_memory_fraction == 0.8


def test_cpu_offload_config_defaults_to_unspecified():
    from torchturbo.engine import CPUOffloadConfig

    assert CPUOffloadConfig().enabled is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_cpu_offload_config_parses_string_enabled(value, expected):
    from torchturbo.engine import CPUOffloadConfig
    from torchturbo.engine.config import EngineConfig

    direct = CPUOffloadConfig(enabled=value)
    from_dict = EngineConfig.from_dict(
        {"backend_config": {"cpu_offload": {"enabled": value}}}
    )

    assert direct.enabled is expected
    assert from_dict.backend_config.cpu_offload.enabled is expected


def test_engine_config_from_dict_parses_freeze_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import ParameterGroup

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "freeze_policy": {
                "parameter_groups": ["vision_tower", "language_model"],
            },
        }
    )

    assert cfg.freeze_policy.parameter_groups == (
        ParameterGroup.VISION_TOWER,
        ParameterGroup.LANGUAGE_MODEL,
    )


def test_engine_config_from_dict_parses_ops_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import OpsConfig
    from torchturbo.ops.base import OpPolicy

    with pytest.raises(TypeError, match="OpsConfig.policy must be an OpPolicy"):
        OpsConfig(policy="custom")

    from_string = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {"policy": "custom"},
        }
    )

    from_enum = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {"policy": OpPolicy.CUSTOM},
        }
    )

    assert from_string.ops.policy is OpPolicy.CUSTOM
    assert from_enum.ops.policy is OpPolicy.CUSTOM


def test_engine_config_from_dict_parses_per_op_overrides():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.ops.base import OpPolicy

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {
                "policy": "auto",
                "overrides": {
                    "rms_norm": "torch_rms_norm",
                    "grouped_matmul": "grouped_gemm",
                },
            },
        }
    )

    assert cfg.ops.policy is OpPolicy.AUTO
    assert cfg.ops.overrides == {
        "rms_norm": "torch_rms_norm",
        "grouped_matmul": "grouped_gemm",
    }


@pytest.mark.parametrize(
    "overrides",
    [None, [], {"rms_norm": 1}, {1: "torch_rms_norm"}],
)
def test_ops_config_rejects_invalid_overrides(overrides):
    from torchturbo.engine.config import OpsConfig

    with pytest.raises(TypeError, match="OpsConfig.overrides"):
        OpsConfig(overrides=overrides)


def test_ops_config_copies_overrides():
    from torchturbo.engine.config import OpsConfig

    overrides = {"rms_norm": "torch_rms_norm"}
    config = OpsConfig(overrides=overrides)
    overrides["rms_norm"] = "other"

    assert config.overrides == {"rms_norm": "torch_rms_norm"}


@pytest.mark.parametrize("overrides", [None, []])
def test_build_ops_config_rejects_falsey_invalid_overrides(overrides):
    from torchturbo.engine.config import _build_ops_config

    with pytest.raises(TypeError, match="OpsConfig.overrides"):
        _build_ops_config({"ops": {"overrides": overrides}})


def test_engine_config_json_round_trip_restores_ops_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import OpsConfig
    from torchturbo.ops.base import OpPolicy

    original = EngineConfig(ops=OpsConfig(policy=OpPolicy.NATIVE))
    serialized = json.loads(json.dumps(original.to_dict()))

    restored = EngineConfig.from_dict(serialized)

    assert restored.ops.policy is OpPolicy.NATIVE


def test_engine_config_json_round_trip_restores_per_op_overrides():
    from torchturbo.engine.config import EngineConfig, OpsConfig

    original = EngineConfig(
        ops=OpsConfig(
            overrides={
                "rms_norm": "torch_rms_norm",
                "swiglu": "torch_swiglu",
            }
        )
    )
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = EngineConfig.from_dict(serialized)

    assert restored.ops.overrides == original.ops.overrides


def test_freeze_policy_direct_constructor_accepts_parameter_groups():
    from torchturbo.engine.config import FreezePolicy
    from torchturbo.engine.config import ParameterGroup

    policy = FreezePolicy((ParameterGroup.LANGUAGE_MODEL,))

    assert policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_freeze_policy_from_list_parses_string_groups():
    from torchturbo.engine.config import FreezePolicy
    from torchturbo.engine.config import ParameterGroup

    policy = FreezePolicy.from_list(["language_model", "language_model"])

    assert policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_engine_config_from_dict_rejects_unknown_freeze_policy_fields():
    from torchturbo.engine.config import EngineConfig

    with pytest.raises(ValueError, match="Unknown freeze_policy field"):
        EngineConfig.from_dict(
            {
                "backend": "fsdp2",
                "freeze_policy": {"groups": ["vision_tower"]},
            }
        )


def test_engine_config_from_dict_parses_mtp_config():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "mtp": {
                "enabled": True,
                "loss_scaling_factor": 0.25,
                "log_loss": True,
            },
        }
    )

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.25
    assert cfg.mtp.log_loss is True


def test_engine_config_from_dict_parses_flat_mtp_fields():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "enable_mtp": "true",
            "mtp_loss_scaling_factor": None,
            "mtp_log_loss": "true",
        }
    )

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.1
    assert cfg.mtp.log_loss is True


def test_mtp_config_rejects_negative_loss_scale():
    from torchturbo.engine.config import MTPConfig

    with pytest.raises(ValueError, match="loss_scaling_factor must be non-negative"):
        MTPConfig(loss_scaling_factor=-0.1)


def test_loss_chunk_config_disabled_by_default():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.loss_chunk.enabled is False
    assert cfg.loss_chunk.chunk_size == 1024


def test_loss_chunk_config_explicit_and_default_sizes():
    from torchturbo.engine.config import LossChunkConfig

    explicit = LossChunkConfig(enabled=True, chunk_size=256)
    default = LossChunkConfig(enabled=True)

    assert explicit.chunk_size == 256
    assert default.chunk_size == 1024


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_loss_chunk_config_rejects_non_positive_size(chunk_size):
    from torchturbo.engine.config import LossChunkConfig

    with pytest.raises(ValueError, match="positive"):
        LossChunkConfig(enabled=True, chunk_size=chunk_size)


def test_loss_chunk_config_allows_size_while_disabled():
    from torchturbo.engine.config import LossChunkConfig

    config = LossChunkConfig(enabled=False, chunk_size=256)

    assert config.enabled is False
    assert config.chunk_size == 256


def test_chunk_size_cli_mapping_does_not_enable_chunked_loss():
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    config = build_from_args(TorchTurboArgs(loss_chunk_size=256))

    assert config.loss_chunk.enabled is False
    assert config.loss_chunk.chunk_size == 256


def test_engine_config_from_dict_parses_chunked_loss():
    from torchturbo.engine.config import EngineConfig

    nested = EngineConfig.from_dict(
        {"backend": "fsdp2", "loss_chunk": {"enabled": True}}
    )
    flat = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "enable_chunk_loss": "true",
            "loss_chunk_size": 320,
        }
    )

    assert nested.loss_chunk.enabled is True
    assert nested.loss_chunk.chunk_size == 1024
    assert flat.loss_chunk.enabled is True
    assert flat.loss_chunk.chunk_size == 320


def test_loss_chunk_has_no_env_overrides():
    """Chunked loss is CLI/dict only: no env var may switch it on."""
    from torchturbo.engine.args import get_arg_specs

    assert not [
        spec
        for spec in get_arg_specs()
        if spec.env_name is not None
        and spec.config_path is not None
        and spec.config_path.startswith("loss_chunk")
    ]
