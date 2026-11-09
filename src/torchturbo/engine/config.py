"""Structured configuration dataclasses for torchturbo engine.

This module contains **common** (cross-backend) configuration.  Backend-specific
configs live in their own packages, e.g. ``backends.fsdp2.config.FSDPConfig``.

Usage::

    from torchturbo.engine.config import EngineConfig
    from torchturbo.parallel import ParallelPlan, FSDPParallelPlan

    config = EngineConfig(
        mixed_precision="bf16",
    )
    engine = TorchTurboEngine(config)
"""

from __future__ import annotations

import dataclasses
import os
from argparse import ArgumentTypeError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..logging import get_logger
from ..ops.base import OpPolicy
from .args import TorchTurboArgs, string_to_bool

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Env var overrides — applies to ALL adapters and direct usage
# ---------------------------------------------------------------------------

ENV_MAP: dict[str, str] = {
    "TORCH_TURBO_BACKEND": "backend",
    "TORCH_TURBO_MEGA_MOE": "mega_moe",
    "TORCH_TURBO_MIXED_PRECISION": "mixed_precision",
    "TORCH_TURBO_OP_POLICY": "ops.policy",
    "TORCH_TURBO_GRADIENT_CHECKPOINTING": "activation_checkpointing.enabled",
    "TORCH_TURBO_AC_USE_REENTRANT": "activation_checkpointing.use_reentrant",
    "TORCH_TURBO_RESHARD_AFTER_FWD": "backend_config.reshard_after_forward",
    "TORCH_TURBO_PREFETCH_STRATEGY": "backend_config.prefetch_strategy",
    "TORCH_TURBO_PREFETCH_DIRECTION": "backend_config.prefetch_direction",
    "TORCH_TURBO_FORWARD_PREFETCH_DEPTH": "backend_config.forward_prefetch_depth",
    "TORCH_TURBO_BACKWARD_PREFETCH_DEPTH": "backend_config.backward_prefetch_depth",
    "TORCH_TURBO_CPU_OFFLOAD": "backend_config.cpu_offload",
    "TORCH_TURBO_SP_SIZE": "parallel",
    "TORCH_TURBO_EP_SIZE": "parallel",
}


def _set_field(obj: Any, field_name: str, value: str) -> None:
    """Set a field on a config object, coercing the string value to the right type."""
    current = getattr(obj, field_name, None)
    if isinstance(current, Enum):
        setattr(obj, field_name, type(current)(value))
    elif isinstance(current, int) and not isinstance(current, bool):
        setattr(obj, field_name, int(value))
    elif isinstance(current, bool):
        setattr(obj, field_name, string_to_bool(value))
    else:
        try:
            setattr(obj, field_name, string_to_bool(value))
        except ArgumentTypeError:
            setattr(obj, field_name, value)


# ---------------------------------------------------------------------------
# Enums (common)
# ---------------------------------------------------------------------------


class DistributedBackend(str, Enum):
    """Supported distributed backends."""

    FSDP2 = "fsdp2"
    # Future:
    # MEGATRON = "megatron"


class ParameterGroup(str, Enum):
    """Semantic model parameter groups understood by engine policies."""

    ALL = "all"
    VISION_TOWER = "vision_tower"
    MULTI_MODAL_PROJECTOR = "multi_modal_projector"
    LANGUAGE_MODEL = "language_model"


# ---------------------------------------------------------------------------
# Shared config classes — re-exported for backward compatibility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training optimizations (common)
# ---------------------------------------------------------------------------


class MixedPrecisionPolicy(str, Enum):
    """Mixed precision dtype policy."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8 = "fp8"


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration."""

    policy: str = "bf16"
    param_dtype: str | None = None
    reduce_dtype: str | None = None


@dataclass
class OpsConfig:
    """Op dispatch configuration."""

    policy: OpPolicy = OpPolicy.AUTO

    def __post_init__(self) -> None:
        if not isinstance(self.policy, OpPolicy):
            raise TypeError("OpsConfig.policy must be an OpPolicy.")


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing (gradient checkpointing) configuration.

    Example::

        ActivationCheckpointingConfig(enabled=True)
    """

    enabled: bool = True
    use_reentrant: bool = False


# ---------------------------------------------------------------------------
# Communication optimization (common)
# ---------------------------------------------------------------------------


@dataclass
class CommConfig:
    """Communication optimization settings.

    Attributes
    ----------
    overlap_comm_compute:
        Overlap allgather / reduce-scatter with compute.
    bucket_cap_mb:
        Size of allreduce buckets (MB) for gradient sync.
    reduce_dtype:
        Dtype for gradient reduction.  ``None`` = same as param dtype.
    """

    overlap_comm_compute: bool = False
    bucket_cap_mb: float = 25.0
    reduce_dtype: MixedPrecisionPolicy | None = None


# ---------------------------------------------------------------------------
# Checkpoint (common)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Model checkpoint loading / saving settings.

    Attributes
    ----------
    dcp_path:
        Path to a distributed checkpoint (DCP) for efficient sharded load.
    hf_model_path:
        HuggingFace model path (local or repo id) for weight loading.
    save_format:
        Format for saving: ``"dcp"`` (distributed) or ``"hf"`` (HuggingFace).
    """

    dcp_path: str | None = None
    hf_model_path: str | None = None
    save_format: str = "dcp"


@dataclass
class MTPConfig:
    """Multi-Token-Prediction configuration."""

    enabled: bool = False
    loss_scaling_factor: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        self.loss_scaling_factor = float(self.loss_scaling_factor)


@dataclass
class DiffusionConfig:
    """Configuration for block-diffusion (BD) training.

    Consumed by ``FSDP2Backend.compute_diffusion_loss`` (CAP weight) and by
    the engine's BD dataloader (per-sample noise range). Gated by the model's
    ``is_diffusion_model`` class attribute — these fields are inert for
    non-diffusion models.
    """

    cap_lambda: float = 0.0
    """CAP/DPARALLEL confidence-loss weight (entropy on correct tokens).
    0 = disabled, behavior unchanged. Recommended: 2.0."""

    bd_mask_low: float = 0.3
    """Per-sample noise ratio sigma lower bound (``sigma ~ U[low, high]``)."""

    bd_mask_high: float = 0.8
    """Per-sample noise ratio sigma upper bound."""

    def __post_init__(self) -> None:
        self.cap_lambda = float(self.cap_lambda)
        self.bd_mask_low = float(self.bd_mask_low)
        self.bd_mask_high = float(self.bd_mask_high)


@dataclass(frozen=True)
class FreezePolicy:
    """Engine config for freezing semantic model parameter groups."""

    parameter_groups: tuple[ParameterGroup, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.parameter_groups)

    @classmethod
    def from_list(
        cls,
        groups: Iterable[ParameterGroup | str] | ParameterGroup | str | None,
    ) -> FreezePolicy:
        if groups is None:
            return cls()
        if isinstance(groups, (ParameterGroup, str)):
            groups = (groups,)
        normalized = []
        seen = set()
        for group in groups:
            parameter_group = cls._coerce_group(group)
            if parameter_group in seen:
                continue
            normalized.append(parameter_group)
            seen.add(parameter_group)
        return cls(tuple(normalized))

    @staticmethod
    def _coerce_group(group: ParameterGroup | str) -> ParameterGroup:
        if isinstance(group, ParameterGroup):
            return group
        return ParameterGroup(group)


# ---------------------------------------------------------------------------
# Top-level engine config
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Top-level configuration for :class:`TorchTurboEngine`.

    Backend-specific settings are stored in ``backend_config`` as an ``Any``
    type — each backend defines its own config dataclass (e.g.
    ``FSDPConfig`` from ``backends.fsdp2.config``).

    Example::

        from torchturbo.engine.backends.fsdp2.config import FSDPConfig

        config = EngineConfig(
            backend=DistributedBackend.FSDP2,
            mixed_precision=MixedPrecisionPolicy.BF16,
            backend_config=FSDPConfig(reshard_after_forward=True),
            activation_checkpointing=ActivationCheckpointingConfig(enabled=True),
            comm=CommConfig(overlap_comm_compute=True),
        )
    """

    backend: DistributedBackend = DistributedBackend.FSDP2
    """Distributed backend."""

    parallel_plan: Any = None
    """Layer-specific parallel plan (``ParallelPlan``).
    Maps FQN patterns to mesh-dim placements for EP, SP, etc."""

    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.BF16
    """Global mixed precision policy."""

    mega_moe: bool = False
    """Use the opt-in BF16 NPU MegaMoE/MoonEP backend for grouped experts."""

    ops: OpsConfig = field(default_factory=OpsConfig)
    """Op dispatch settings."""

    backend_config: Any = None
    """Backend-specific config (e.g. ``FSDPConfig``, ``MegatronConfig``).
    If ``None``, the backend uses its own defaults."""

    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=ActivationCheckpointingConfig
    )
    """Activation checkpointing settings."""

    comm: CommConfig = field(default_factory=CommConfig)
    """Communication optimization settings."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint loading / saving settings."""

    mtp: MTPConfig = field(default_factory=MTPConfig)
    """Multi-Token-Prediction settings."""

    freeze_policy: FreezePolicy = field(default_factory=FreezePolicy)
    """Policy for freezing semantic model parameter groups."""

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    """Block-diffusion training settings (only active for ``is_diffusion_model`` models)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Catch-all for user-defined settings."""

    def __post_init__(self) -> None:
        """Auto-create default backend_config, then apply env var overrides."""
        self.mega_moe = string_to_bool(self.mega_moe)
        if self.backend_config is None:
            self.backend_config = self._default_backend_config()
        if self.ops is None:
            self.ops = OpsConfig()
        self.apply_env_overrides()

    def _set_top_level(self, field_name: str, value: str) -> None:
        """Set a top-level config field from a string value."""
        if field_name == "backend":
            self.backend = DistributedBackend(value)
        elif field_name == "mega_moe":
            self.mega_moe = string_to_bool(value)
        elif field_name == "mixed_precision":
            self.mixed_precision = MixedPrecisionPolicy(value)
        elif field_name == "parallel" and "=" in value:
            # Handles TORCH_TURBO_SP_SIZE, etc manually in apply_env_overrides
            pass

    def apply_env_overrides(self) -> None:
        """Apply ``TORCH_TURBO_*`` env vars to this config."""
        for env_key, config_path in ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            if env_key in ("TORCH_TURBO_SP_SIZE", "TORCH_TURBO_EP_SIZE"):
                if self.parallel_plan is None:
                    from torchturbo.parallel import FSDPParallelPlan

                    self.parallel_plan = FSDPParallelPlan()
                attr_name = env_key.replace("TORCH_TURBO_", "").lower()
                setattr(self.parallel_plan, attr_name, int(value))
                continue

            parts = config_path.split(".")
            if len(parts) == 1:
                self._set_top_level(parts[0], value)
            else:
                obj = getattr(self, parts[0], None)
                if obj is not None:
                    _set_field(obj, parts[1], value)

        post_init = getattr(self.backend_config, "__post_init__", None)
        if callable(post_init):
            post_init()

    def _default_backend_config(self) -> Any:
        """Create the default config for the selected backend."""
        if self.backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            return FSDPConfig()
        return {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineConfig:
        """Create an ``EngineConfig`` from a flat or nested dict.

        Supports both flat keys (``{"backend": "fsdp2", "dp_size": 4}``) and
        nested keys (``{"parallel": {"dp_size": 4}}``).
        """
        backend = DistributedBackend(d.get("backend", "fsdp2"))
        mp = MixedPrecisionPolicy(d.get("mixed_precision", "bf16"))

        if "op_policy" in d:
            raise ValueError("op_policy is no longer supported; use ops.policy.")

        # Parallel Plan (parse sp, ep, tp sizes)
        parallel_d = d.get("parallel", {})

        sp_size = parallel_d.get("sp_size", d.get("sp_size", 1))
        ep_size = parallel_d.get("ep_size", d.get("ep_size", 1))

        from torchturbo.parallel import FSDPParallelPlan

        parallel_plan = FSDPParallelPlan(
            sp_size=sp_size,
            ep_size=ep_size,
        )

        # Backend-specific config
        backend_config = d.get("backend_config", None)
        if isinstance(backend_config, dict) and backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            backend_config = FSDPConfig(**backend_config)

        if backend_config is None and backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            fsdp_d = d.get("fsdp", {})
            if not fsdp_d:
                fsdp_d = {
                    k: d[k]
                    for k in (
                        "sharding_strategy",
                        "reshard_after_forward",
                        "prefetch_strategy",
                        "prefetch_direction",
                        "forward_prefetch_depth",
                        "backward_prefetch_depth",
                        "cpu_offload",
                        "pin_memory",
                    )
                    if k in d
                }
            # FSDPConfig.__post_init__ coerces str values to the right enums.
            backend_config = FSDPConfig(**fsdp_d) if fsdp_d else FSDPConfig()

        # Activation checkpointing
        ac_d = d.get("activation_checkpointing", {})
        ac = ActivationCheckpointingConfig(**ac_d)

        # Communication
        comm_d = d.get("comm", {})
        if not comm_d:
            if "comm_overlap" in d:
                comm_d["overlap_comm_compute"] = d["comm_overlap"]
        comm = CommConfig(**comm_d)

        # Ops
        ops_d = d.get("ops", {})
        if isinstance(ops_d, OpsConfig):
            ops = ops_d
        else:
            if ops_d is None:
                ops_d = {}
            if not isinstance(ops_d, dict):
                raise TypeError("ops must be a dict or OpsConfig.")
            ops_d = dict(ops_d)
            if "policy" in ops_d:
                ops_d["policy"] = OpPolicy(ops_d["policy"])
            ops = OpsConfig(**ops_d) if ops_d else OpsConfig()

        # Checkpoint
        ckpt_d = d.get("checkpoint", {})
        if not ckpt_d:
            ckpt_d = {
                k: d[k] for k in ("dcp_path", "hf_model_path", "save_format") if k in d
            }
        ckpt = CheckpointConfig(**ckpt_d)

        # MTP
        mtp_config = cls._parse_mtp_config(d)

        # Freeze policy
        freeze_policy = cls._parse_freeze_policy(d.get("freeze_policy", {}))

        # Extra
        known_keys = {
            "backend",
            "mixed_precision",
            "mega_moe",
            "parallel",
            "fsdp",
            "backend_config",
            "activation_checkpointing",
            "comm",
            "ops",
            "checkpoint",
            "dp_size",
            "tp_size",
            "ep_size",
            "pp_size",
            "sp_size",
            "sharding_strategy",
            "reshard_after_forward",
            "prefetch_strategy",
            "prefetch_direction",
            "forward_prefetch_depth",
            "backward_prefetch_depth",
            "cpu_offload",
            "pin_memory",
            "sync_module_states",
            "comm_overlap",
            "dcp_path",
            "hf_model_path",
            "save_format",
            "mtp",
            "enable_mtp",
            "mtp_loss_scaling_factor",
            "freeze_policy",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            backend=backend,
            parallel_plan=parallel_plan,
            mixed_precision=mp,
            mega_moe=d.get("mega_moe", False),
            ops=ops,
            backend_config=backend_config,
            activation_checkpointing=ac,
            comm=comm,
            checkpoint=ckpt,
            mtp=mtp_config,
            freeze_policy=freeze_policy,
            extra=extra,
        )

    @staticmethod
    def _parse_mtp_config(d: dict[str, Any]) -> MTPConfig:
        mtp_d = d.get("mtp", {})
        if isinstance(mtp_d, MTPConfig):
            return mtp_d
        if mtp_d is None:
            mtp_d = {}
        if not isinstance(mtp_d, dict):
            raise TypeError("mtp must be a dict or MTPConfig.")

        mtp_d = dict(mtp_d)
        if "enable_mtp" in d:
            mtp_d["enabled"] = d["enable_mtp"]
        if "mtp_loss_scaling_factor" in d:
            mtp_d["loss_scaling_factor"] = d["mtp_loss_scaling_factor"]
        if mtp_d.get("loss_scaling_factor") is None:
            mtp_d["loss_scaling_factor"] = 0.1
        return MTPConfig(**mtp_d)

    @staticmethod
    def _parse_freeze_policy(value: Any) -> FreezePolicy:
        if isinstance(value, FreezePolicy):
            return value
        if value is None or value == {}:
            return FreezePolicy()
        if not isinstance(value, dict):
            raise TypeError("freeze_policy must be a dict or FreezePolicy.")

        allowed_keys = {"parameter_groups"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown freeze_policy field(s): {keys}.")
        if "parameter_groups" not in value:
            raise ValueError("freeze_policy requires 'parameter_groups'.")

        return FreezePolicy.from_list(value["parameter_groups"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict (for logging, etc.)."""

        data = {}
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            data[field_info.name] = value
        return data


# ---------------------------------------------------------------------------
# EngineConfig builder — single mapping from ``TorchTurboArgs`` → EngineConfig
# ---------------------------------------------------------------------------


def build_from_args(torchturbo_args: TorchTurboArgs | None) -> EngineConfig:
    """Assemble an :class:`EngineConfig` purely from a :class:`TorchTurboArgs`.

    Every slice of the engine config is reconstructed from the parsed
    ``--tt_*`` flags (the single source of truth for turbo-owned settings).
    Both adapters (llamafactory_v0, msswift) — and future ones — fold any
    adapter-resolved bits (mixed-precision seed from the adapter's bf16/fp16,
    checkpoint paths from the adapter's model/output_dir, CPU-offload
    reconciliation, gradient-checkpointing takeover) onto the
    :class:`TorchTurboArgs` they pass in, so this function needs only that one
    argument. Adding a new ``--tt_*`` flag touches only ``engine/args.py`` +
    this builder, and every adapter picks it up.

    ``FSDPConfig`` / ``FSDPParallelPlan`` are imported lazily (mirroring
    :meth:`EngineConfig.__post_init__`) to avoid an import cycle: they live in
    ``torchturbo.engine.backends.fsdp2.config`` / ``torchturbo.parallel``,
    which import nothing from this module.

    Parameters
    ----------
    torchturbo_args:
        Parsed ``--tt_*`` args (the single source of truth for turbo flags).
        ``None`` is treated as all-defaults.
    """
    from torchturbo.engine.backends.fsdp2.config import FSDPConfig, ShardingStrategy
    from torchturbo.parallel import FSDPParallelPlan

    tt = torchturbo_args or TorchTurboArgs()

    # --- Backend ---
    backend = DistributedBackend.FSDP2
    if tt.backend is not None:
        backend = DistributedBackend(tt.backend)

    # --- Mixed precision (adapter may have seeded tt.mixed_precision) ---
    mixed_precision = MixedPrecisionPolicy.FP32
    if tt.mixed_precision is not None:
        mixed_precision = MixedPrecisionPolicy(tt.mixed_precision)

    # --- Parallel plan (sp / ep) ---
    parallel_plan = FSDPParallelPlan(
        sp_size=tt.sp_size if tt.sp_size is not None else 1,
        ep_size=tt.ep_size if tt.ep_size is not None else 1,
    )

    # --- Backend config (prefetch from tt; cpu_offload resolved by adapter) ---
    fsdp_config = _build_fsdp_config(tt)

    # --- Activation checkpointing (enabled + reentrant both from tt) ---
    ac_config = ActivationCheckpointingConfig(
        enabled=bool(tt.gradient_checkpointing),
        use_reentrant=bool(tt.ac_use_reentrant),
    )

    config = EngineConfig(
        backend=backend,
        parallel_plan=parallel_plan,
        mixed_precision=mixed_precision,
        mega_moe=tt.mega_moe,
        backend_config=fsdp_config,
        activation_checkpointing=ac_config,
        comm=CommConfig(overlap_comm_compute=False),
        checkpoint=_build_checkpoint_config(tt),
        ops=_build_ops_config(tt),
        mtp=_build_mtp_config(tt),
        freeze_policy=_build_freeze_policy(tt),
        diffusion=_build_diffusion_config(tt),
        extra={},
    )
    logger.info_rank0(
        "Built EngineConfig from TorchTurboArgs: backend=%s, mp=%s, sp=%d, ep=%d",
        config.backend.value,
        config.mixed_precision.value,
        parallel_plan.sp_size,
        parallel_plan.ep_size,
    )
    return config


def _build_fsdp_config(tt: TorchTurboArgs) -> "FSDPConfig":
    """FSDPConfig from tt prefetch fields + tt.cpu_offload."""
    from torchturbo.engine.backends.fsdp2.config import FSDPConfig, ShardingStrategy

    return FSDPConfig(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        reshard_after_forward=True,
        cpu_offload=bool(tt.cpu_offload),
        pin_memory=True,
        prefetch_strategy=tt.prefetch_strategy,
        prefetch_direction=tt.prefetch_direction,
        forward_prefetch_depth=tt.forward_prefetch_depth,
        backward_prefetch_depth=tt.backward_prefetch_depth,
    )


def _build_checkpoint_config(tt: TorchTurboArgs) -> CheckpointConfig:
    """CheckpointConfig from the adapter-seeded model/output paths on tt."""
    output_dir = tt.output_dir
    return CheckpointConfig(
        hf_model_path=tt.model_path,
        dcp_path=output_dir if output_dir else None,
    )


def _build_ops_config(tt: TorchTurboArgs) -> OpsConfig:
    """OpsConfig from the ``--tt_op_policy`` flag (default AUTO when unset)."""
    if tt.op_policy is None:
        return OpsConfig()
    return OpsConfig(policy=OpPolicy(str(tt.op_policy).lower()))


def _build_mtp_config(tt: TorchTurboArgs) -> MTPConfig:
    loss_scaling_factor = (
        tt.mtp_loss_scaling_factor
        if tt.mtp_loss_scaling_factor is not None
        else MTPConfig().loss_scaling_factor
    )
    return MTPConfig(
        enabled=bool(tt.enable_mtp),
        loss_scaling_factor=float(loss_scaling_factor),
    )


def _build_freeze_policy(tt: TorchTurboArgs) -> FreezePolicy:
    groups: list[ParameterGroup] = []
    if tt.freeze_vision_tower:
        groups.append(ParameterGroup.VISION_TOWER)
    if tt.freeze_multi_modal_projector:
        groups.append(ParameterGroup.MULTI_MODAL_PROJECTOR)
    if tt.freeze_language_model:
        groups.append(ParameterGroup.LANGUAGE_MODEL)
    return FreezePolicy(parameter_groups=tuple(groups))


def _build_diffusion_config(tt: TorchTurboArgs) -> DiffusionConfig:
    # BD/CAP config for block-diffusion (dLLM) training. Gated by the model's
    # ``is_diffusion_model`` attribute — inert (cap_lambda=0) for non-diffusion
    # models, so constructing it unconditionally here is safe for all adapters.
    return DiffusionConfig(
        cap_lambda=float(tt.cap_lambda or 0.0),
        bd_mask_low=float(tt.bd_mask_low),
        bd_mask_high=float(tt.bd_mask_high),
    )
