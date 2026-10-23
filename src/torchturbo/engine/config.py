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
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, get_args, get_type_hints

from omegaconf import OmegaConf

from ..logging import get_logger
from ..ops.base import OpPolicy
from .args import TorchTurboArgs, get_arg_specs, string_to_bool

logger = get_logger(__name__)


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
    """Supported mixed precision policies."""

    BF16 = "bf16"
    FP32 = "fp32"


def _parse_mixed_precision_policy(value: Any) -> MixedPrecisionPolicy:
    if value in ("fp16", "fp8"):
        raise ValueError(
            f"TorchTurbo engine does not support mixed_precision={value!r}; "
            "use bf16 or fp32."
        )
    return MixedPrecisionPolicy(value)


def _parse_ns_coefficients(
    value: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[float, float, float] | tuple[tuple[float, float, float], ...]:
    """Normalize Newton-Schulz coefficients to a flat triple or per-step triples.
    A flat (a, b, c) triple applies to every iteration; a sequence of triples
    applies one per iteration and must provide exactly one triple per
    ``ns_steps`` iteration.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "Muon ns_coefficients must be a triple or a non-empty list of triples"
        )
    nested_entries = [entry for entry in value if isinstance(entry, (list, tuple))]
    if nested_entries and len(nested_entries) != len(value):
        raise ValueError("Muon ns_coefficients must not mix scalars and triples")
    if nested_entries:
        triples = tuple(
            tuple(float(coefficient) for coefficient in entry) for entry in value
        )
        if any(len(triple) != 3 for triple in triples):
            raise ValueError("Muon ns_coefficients entries must contain three values")
        return triples
    flat = tuple(float(coefficient) for coefficient in value)
    if len(flat) != 3:
        raise ValueError("Muon ns_coefficients must contain three values")
    return flat


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration."""

    policy: str = "bf16"
    param_dtype: str | None = None
    reduce_dtype: str | None = None


@dataclass
class OpsConfig:
    """Op dispatch policy and optional per-op overrides."""

    policy: OpPolicy = OpPolicy.AUTO
    overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, OpPolicy):
            raise TypeError("OpsConfig.policy must be an OpPolicy.")
        if not isinstance(self.overrides, dict):
            raise TypeError("OpsConfig.overrides must be a dict.")
        if any(
            not isinstance(op_name, str) or not isinstance(impl_name, str)
            for op_name, impl_name in self.overrides.items()
        ):
            raise TypeError(
                "OpsConfig.overrides must map op names to implementation names."
            )
        self.overrides = dict(self.overrides)


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


@dataclass
class OptimizerBucketPolicy:
    """Engine policy for combining optimizer parameters into buckets."""

    strategy: str = "parameter"
    bucket_cap_mb: float | None = None

    def __post_init__(self) -> None:
        self.strategy = str(self.strategy).lower()
        if self.strategy not in {"parameter", "size", "single"}:
            raise ValueError(
                "optimizer bucket strategy must be parameter, size, or single"
            )
        if self.bucket_cap_mb is not None:
            self.bucket_cap_mb = float(self.bucket_cap_mb)
            if self.bucket_cap_mb <= 0:
                raise ValueError("optimizer bucket_cap_mb must be positive")


@dataclass
class QKClipConfig:
    """Post-step MLA weight rescaling based on maximum attention logits.

    Args:
        threshold: Maximum target logit, after the attention softmax scale.
        alpha: Fraction of the NoPE correction applied to the query projection.
    """

    threshold: float = 100.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        self.threshold = float(self.threshold)
        self.alpha = float(self.alpha)
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("QK clip threshold must be finite and positive")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("QK clip alpha must be in [0, 1]")


@dataclass
class MuonOptimizerConfig:
    """Runtime hyperparameters for TorchTurbo's native distributed Muon."""

    lr: float | None = None
    weight_decay: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    # Newton-Schulz iteration coefficients: a flat (a, b, c) triple applies to
    # every iteration; a sequence of triples applies one per iteration and
    # must match ns_steps exactly
    ns_coefficients: tuple[float, float, float] | tuple[tuple[float, float, float], ...] = (
        3.4445,
        -4.7750,
        2.0315,
    )
    eps: float = 1e-7
    ns_steps: int = 5
    adjust_lr_fn: str | None = "match_rms_adamw"
    overlap_comm_compute: bool = True
    # Learning rate for the fallback AdamW
    adamw_lr: float | None = None
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    adamw_fused: bool | None = None
    bucket: OptimizerBucketPolicy = field(default_factory=OptimizerBucketPolicy)
    qk_clip: QKClipConfig | None = None

    def __post_init__(self) -> None:
        if self.lr is not None:
            self.lr = float(self.lr)
        if self.adamw_lr is not None:
            self.adamw_lr = float(self.adamw_lr)
        if self.weight_decay is not None:
            self.weight_decay = float(self.weight_decay)
        self.momentum = float(self.momentum)
        self.nesterov = string_to_bool(self.nesterov)
        self.ns_coefficients = _parse_ns_coefficients(self.ns_coefficients)
        self.eps = float(self.eps)
        self.ns_steps = int(self.ns_steps)
        if (
            isinstance(self.ns_coefficients[0], tuple)
            and len(self.ns_coefficients) != self.ns_steps
        ):
            raise ValueError(
                "Muon ns_coefficients must provide one triple per ns_step: "
                f"{len(self.ns_coefficients)} triples for "
                f"ns_steps={self.ns_steps}"
            )
        self.overlap_comm_compute = string_to_bool(self.overlap_comm_compute)
        self.adamw_betas = tuple(float(value) for value in self.adamw_betas)
        self.adamw_eps = float(self.adamw_eps)
        if self.adamw_fused is not None:
            self.adamw_fused = string_to_bool(self.adamw_fused)
        if self.lr is not None and self.lr < 0:
            raise ValueError("Muon lr must be non-negative")
        if self.adamw_lr is not None and self.adamw_lr < 0:
            raise ValueError("Muon AdamW lr must be non-negative")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError("Muon weight_decay must be non-negative")
        if self.momentum < 0:
            raise ValueError("Muon momentum must be non-negative")
        if not 0 <= self.ns_steps < 100:
            raise ValueError("Muon ns_steps must be in [0, 100)")
        if len(self.adamw_betas) != 2 or not all(
            0 <= beta < 1 for beta in self.adamw_betas
        ):
            raise ValueError("Muon AdamW betas must contain two values in [0, 1)")
        if self.adamw_eps < 0:
            raise ValueError("Muon AdamW eps must be non-negative")
        if self.adjust_lr_fn not in {
            None,
            "original",
            "match_rms_adamw",
            "spectral_unclamped",
        }:
            raise ValueError(f"unsupported Muon adjust_lr_fn {self.adjust_lr_fn!r}")
        if self.bucket is None:
            self.bucket = OptimizerBucketPolicy()
        elif isinstance(self.bucket, dict):
            self.bucket = OptimizerBucketPolicy(**self.bucket)
        elif not isinstance(self.bucket, OptimizerBucketPolicy):
            raise TypeError("Muon bucket must be an OptimizerBucketPolicy or dict")
        if isinstance(self.qk_clip, dict):
            self.qk_clip = QKClipConfig(**self.qk_clip)
        elif self.qk_clip is not None and not isinstance(self.qk_clip, QKClipConfig):
            raise TypeError("Muon qk_clip must be a QKClipConfig, dict, or None")


@dataclass
class OptimizerConfig:
    """Engine-managed optimizer selection and runtime policy."""

    name: str | None = None
    muon: MuonOptimizerConfig | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            self.name = str(self.name).strip().lower()

        if self.name == "muon":
            if self.muon is None:
                self.muon = MuonOptimizerConfig()
            elif isinstance(self.muon, dict):
                self.muon = MuonOptimizerConfig(**self.muon)
            elif not isinstance(self.muon, MuonOptimizerConfig):
                raise TypeError(
                    "optimizer.muon must be a MuonOptimizerConfig, dict, or None"
                )
        elif self.name is not None:
            raise ValueError(
                f"unsupported engine-managed optimizer {self.name!r}; "
                "TorchTurbo currently supports 'muon'"
            )


# ---------------------------------------------------------------------------
# Checkpoint (common)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Checkpoint settings.

    Training checkpoints always store exact model state for resuming. The
    on-disk format belongs to the backend checkpoint manager.

    Attributes:
        hf_export: When training checkpoints also export HF safetensors:
            ``"final"`` exports with the final save only, ``"every"`` with
            every checkpoint. Defaults to ``"every"``.
    """

    hf_export: Literal["final", "every"] = "every"

    def __post_init__(self) -> None:
        if self.hf_export not in ("final", "every"):
            raise ValueError(
                "checkpoint.hf_export must be 'final' or 'every', got "
                f"{self.hf_export!r}"
            )


@dataclass
class MTPConfig:
    """Multi-Token-Prediction configuration."""

    enabled: bool = False
    loss_scaling_factor: float = 0.1
    log_loss: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        self.loss_scaling_factor = float(self.loss_scaling_factor)
        if self.loss_scaling_factor < 0:
            raise ValueError(
                "MTP loss_scaling_factor must be non-negative, got "
                f"{self.loss_scaling_factor}"
            )
        if isinstance(self.log_loss, str):
            self.log_loss = string_to_bool(self.log_loss)
        else:
            self.log_loss = bool(self.log_loss)


@dataclass
class DSparkTrainingConfig:
    """DeepSeek-V4 DSpark training recipe.

    Defaults follow the official DeepSpec DSpark configs for Qwen3 and Gemma4:
    https://github.com/deepseek-ai/DeepSpec

    ``enabled`` independently controls the DSpark training branch.  MTP layers
    are still gated by ``mtp.enabled``, while the DSpark architecture is
    determined by the model config's ``dspark_block_size``.

    These values are set onto the model config during ``init_context``; the
    model config itself is not expected to provide them.
    """

    enabled: bool = True
    loss_scaling_factor: float = 0.1
    num_anchors: int = 512
    loss_decay_gamma: float = 4.0
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_loss_alpha: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        self.loss_scaling_factor = float(self.loss_scaling_factor)
        if self.loss_scaling_factor < 0:
            raise ValueError(
                "dspark loss_scaling_factor must be non-negative, got "
                f"{self.loss_scaling_factor}"
            )
        if self.num_anchors <= 0:
            raise ValueError(
                f"dspark num_anchors must be positive, got {self.num_anchors}"
            )
        if self.loss_decay_gamma < 0:
            raise ValueError(
                f"dspark loss_decay_gamma must be non-negative, "
                f"got {self.loss_decay_gamma}"
            )
        loss_alphas = {
            "ce_loss_alpha": self.ce_loss_alpha,
            "l1_loss_alpha": self.l1_loss_alpha,
            "confidence_loss_alpha": self.confidence_loss_alpha,
        }
        for name, value in loss_alphas.items():
            if value < 0:
                raise ValueError(f"dspark {name} must be non-negative, got {value}")


@dataclass
class LossChunkConfig:
    """Compute ``lm_head`` + cross-entropy in token chunks, avoiding the full
    ``[tokens, vocab]`` logits tensor.

    Off by default; ``chunk_size`` defaults to 1024.
    """

    enabled: bool = False
    """Enable chunked linear cross-entropy during training."""

    chunk_size: int = 1024
    """Flattened tokens per chunk."""

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        self.chunk_size = int(self.chunk_size)
        self.validate()

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(
                f"loss_chunk.chunk_size must be positive, got {self.chunk_size}"
            )


@dataclass
class MoEConfig:
    """MoE training configuration."""

    aux_loss_coeff: float = 0.0
    z_loss_coeff: float = 0.0
    token_dispatcher: str = "alltoall"
    """Expert token dispatcher type."""

    def __post_init__(self) -> None:
        if self.token_dispatcher != "alltoall":
            raise ValueError("moe.token_dispatcher must be 'alltoall'.")


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

    bd_block_size: int = 32
    """Block size (``BLOCK_SIZE``) for the block-diagonal attention mask; the
    unit ``cutoff_len`` is ceil-aligned to. Must be a positive integer (default
    ``32``, dFactory/LLaDA2 default). Only affects the BD dataset/loss path."""

    def __post_init__(self) -> None:
        self.cap_lambda = float(self.cap_lambda)
        self.bd_mask_low = float(self.bd_mask_low)
        self.bd_mask_high = float(self.bd_mask_high)
        self.bd_block_size = int(self.bd_block_size)
        if self.bd_block_size < 1:
            raise ValueError(
                f"DiffusionConfig.bd_block_size must be a positive integer, "
                f"got {self.bd_block_size}."
            )


@dataclass
class MfuConfig:
    """MFU/HFU reporting: ``enabled`` + ``card``."""

    enabled: bool = False
    card: str | None = None

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

    use_deterministic_algorithms: bool = False
    """Require deterministic algorithms for the selected device runtime."""

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

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer selection and distributed optimizer runtime policy."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint loading / saving settings."""

    mtp: MTPConfig = field(default_factory=MTPConfig)
    """Multi-Token-Prediction settings."""

    dspark: DSparkTrainingConfig = field(default_factory=DSparkTrainingConfig)
    """DeepSeek-V4 DSpark training overrides."""

    moe: MoEConfig = field(default_factory=MoEConfig)
    """MoE training settings."""

    loss_chunk: LossChunkConfig = field(default_factory=LossChunkConfig)
    """Chunked fused ``lm_head`` + CE settings."""

    freeze_policy: FreezePolicy = field(default_factory=FreezePolicy)
    """Policy for freezing semantic model parameter groups."""

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    """Block-diffusion training settings (only active for ``is_diffusion_model`` models)."""

    mfu: MfuConfig = field(default_factory=MfuConfig)
    """MFU/HFU reporting settings."""

    gbs: int | None = None
    """Global batch size in micro-batch units (``--tt_gbs``). When set, the
    engine derives ``gradient_accumulation_steps = gbs * sp_size / world_size``
    """

    plugin_paths: list[str] = field(default_factory=list)
    """``--tt_plugin`` paths."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Catch-all for user-defined settings."""

    def __post_init__(self) -> None:
        """Normalize values and create missing nested configuration objects."""
        if not isinstance(self.mixed_precision, MixedPrecisionPolicy):
            self.mixed_precision = _parse_mixed_precision_policy(
                self.mixed_precision
            )
        self.use_deterministic_algorithms = string_to_bool(
            self.use_deterministic_algorithms
        )
        if self.backend_config is None:
            self.backend_config = self._default_backend_config()
        if self.ops is None:
            self.ops = OpsConfig()
        if self.loss_chunk is None:
            self.loss_chunk = LossChunkConfig()
        if self.dspark is None:
            self.dspark = DSparkTrainingConfig()
        elif isinstance(self.dspark, dict):
            self.dspark = DSparkTrainingConfig(**self.dspark)
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        elif isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        elif not isinstance(self.optimizer, OptimizerConfig):
            raise TypeError("optimizer must be an OptimizerConfig or dict")
        if self.gbs is not None and self.gbs <= 0:
            raise ValueError(
                f"gbs (--tt_gbs) must be a positive integer, got {self.gbs}."
            )

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
        mp = _parse_mixed_precision_policy(d.get("mixed_precision", "bf16"))
        use_deterministic_algorithms = d.get("use_deterministic_algorithms", False)

        if "op_policy" in d:
            raise ValueError("op_policy is no longer supported; use ops.policy.")

        from torchturbo.parallel import FSDPParallelPlan

        legacy_parallel = d.get("parallel", {})
        parallel_sizes = {
            "sp_size": legacy_parallel.get("sp_size", d.get("sp_size", 1)),
            "ep_size": legacy_parallel.get("ep_size", d.get("ep_size", 1)),
        }
        parallel_d = d.get("parallel_plan")
        if isinstance(parallel_d, FSDPParallelPlan):
            parallel_plan = parallel_d
        elif parallel_d is not None:
            if not isinstance(parallel_d, Mapping):
                raise ValueError("parallel_plan must be a mapping or FSDPParallelPlan.")
            parallel_plan = FSDPParallelPlan.from_dict({**parallel_sizes, **parallel_d})
        else:
            parallel_plan = FSDPParallelPlan(**parallel_sizes)

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

        optimizer_d = dict(d.get("optimizer") or {})
        optimizer = OptimizerConfig(**optimizer_d)

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
        ckpt = CheckpointConfig(**ckpt_d)

        # MTP
        mtp_config = cls._parse_mtp_config(d)

        # DeepSeek-V4 DSpark
        dspark_config = cls._parse_dspark_training_config(d.get("dspark", {}))

        # MoE
        moe_config = cls._parse_moe_config(d.get("moe", {}))

        # Chunked fused lm_head + CE
        loss_chunk_config = cls._parse_loss_chunk_config(d)

        # Freeze policy
        freeze_policy = cls._parse_freeze_policy(d.get("freeze_policy", {}))

        # MFU/HFU reporting
        mfu_config = cls._parse_mfu_config(d.get("mfu", {}))

        # Extra
        known_keys = {
            "backend",
            "mixed_precision",
            "use_deterministic_algorithms",
            "parallel",
            "parallel_plan",
            "fsdp",
            "backend_config",
            "activation_checkpointing",
            "comm",
            "optimizer",
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
            "mtp",
            "dspark",
            "moe",
            "enable_mtp",
            "mtp_loss_scaling_factor",
            "mtp_log_loss",
            "loss_chunk",
            "enable_chunk_loss",
            "loss_chunk_size",
            "freeze_policy",
            "diffusion",
            "mfu",
            "gbs",
            "plugin_paths",
            "extra",
        }
        extra = dict(d.get("extra") or {})
        extra.update({k: v for k, v in d.items() if k not in known_keys})

        return cls(
            backend=backend,
            parallel_plan=parallel_plan,
            mixed_precision=mp,
            use_deterministic_algorithms=use_deterministic_algorithms,
            ops=ops,
            backend_config=backend_config,
            activation_checkpointing=ac,
            comm=comm,
            optimizer=optimizer,
            checkpoint=ckpt,
            mtp=mtp_config,
            dspark=dspark_config,
            moe=moe_config,
            loss_chunk=loss_chunk_config,
            freeze_policy=freeze_policy,
            diffusion=_build_diffusion_config(d),
            mfu=mfu_config,
            gbs=d.get("gbs"),
            plugin_paths=list(d.get("plugin_paths") or []),
            extra=extra,
        )

    @staticmethod
    def _parse_loss_chunk_config(d: dict[str, Any]) -> LossChunkConfig:
        chunk_d = d.get("loss_chunk", {})
        if isinstance(chunk_d, LossChunkConfig):
            return chunk_d
        if chunk_d is None:
            chunk_d = {}
        if not isinstance(chunk_d, dict):
            raise TypeError("loss_chunk must be a dict or LossChunkConfig.")

        chunk_d = dict(chunk_d)
        if "enable_chunk_loss" in d:
            chunk_d["enabled"] = d["enable_chunk_loss"]
        if "loss_chunk_size" in d:
            chunk_d["chunk_size"] = d["loss_chunk_size"]
        return LossChunkConfig(**chunk_d)

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
        if "mtp_log_loss" in d:
            mtp_d["log_loss"] = d["mtp_log_loss"]
        if mtp_d.get("loss_scaling_factor") is None:
            mtp_d["loss_scaling_factor"] = 0.1
        return MTPConfig(**mtp_d)

    @staticmethod
    def _parse_dspark_training_config(value: Any) -> DSparkTrainingConfig:
        if isinstance(value, DSparkTrainingConfig):
            return value
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("dspark must be a dict or DSparkTrainingConfig.")
        return DSparkTrainingConfig(**value)

    @staticmethod
    def _parse_moe_config(value: Any) -> MoEConfig:
        if isinstance(value, MoEConfig):
            return value
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("moe must be a dict or MoEConfig.")
        return MoEConfig(**value)

    @staticmethod
    def _parse_mfu_config(value: Any) -> MfuConfig:
        if isinstance(value, MfuConfig):
            return value
        if value is None or value == {}:
            return MfuConfig()
        if not isinstance(value, dict):
            raise TypeError("mfu must be a dict or MfuConfig.")
        kwargs = dict(value)
        if kwargs.get("enabled") is not None:
            kwargs["enabled"] = string_to_bool(kwargs["enabled"])
        return MfuConfig(**kwargs)

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
        """Serialize configuration fields for logging or reconstruction."""
        from torchturbo.parallel import FSDPParallelPlan

        data = {}
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, FSDPParallelPlan):
                value = value.to_dict()
            elif dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            data[field_info.name] = value
        return data


# ---------------------------------------------------------------------------
# EngineConfig builder — single mapping from ``TorchTurboArgs`` → EngineConfig
# ---------------------------------------------------------------------------


def build_from_args(
    torchturbo_args: TorchTurboArgs | None = None,
) -> EngineConfig:
    """Assemble an :class:`EngineConfig` purely from a :class:`TorchTurboArgs`.

    Every slice of the engine config is reconstructed from one
    :class:`TorchTurboArgs` object. Callers may seed its base configuration
    fields before passing it here; ``--tt_*`` settings are merged uniformly.

    The final config is built once from a merged dict with four layers:
    ``TorchTurboArgs`` < ``TORCH_TURBO_*`` env < YAML/JSON file < explicit
    ``--tt_*`` CLI flags.

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
    from torchturbo.parallel import FSDPParallelPlan

    tt = torchturbo_args or TorchTurboArgs()
    arg_specs = get_arg_specs()

    # Layer 1: TorchTurboArgs values.
    fields = OmegaConf.create()
    for arg_spec in arg_specs:
        if arg_spec.args_name is None or arg_spec.config_path is None:
            continue
        value = getattr(tt, arg_spec.args_name)
        if value is not None:
            # Merge dataclasses as plain dicts. A typed OmegaConf node matches
            # enum strings by member name, so it would reject file values
            # such as "cpu_offload.mode: optimizer".
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            OmegaConf.update(fields, arg_spec.config_path, value, force_add=True)
    # Defaults without a --tt_* flag (tt values above win via merge).
    fields = OmegaConf.merge(
        OmegaConf.create(
            {
                # --- Backend ---
                "backend": "fsdp2",
                # --- Mixed precision ---
                "mixed_precision": "fp32",
                # --- Parallel plan (sp / ep) ---
                "parallel_plan": {"sp_size": 1, "ep_size": 1},
                # --- Backend config ---
                "backend_config": {"reshard_after_forward": True},
            }
        ),
        fields,
    )
    # Layer 2: TORCH_TURBO_* environment variables.
    for arg_spec in arg_specs:
        if arg_spec.env_name is None or arg_spec.config_path is None:
            continue
        value = os.environ.get(arg_spec.env_name)
        if value is None:
            continue
        if arg_spec.type is not None:
            value = arg_spec.type(value)
        OmegaConf.update(fields, arg_spec.config_path, value, force_add=True)

    # Layer 3: YAML/JSON config file.
    config_from_file = OmegaConf.create()
    if tt.config_path is not None:
        raw_config_from_file = OmegaConf.load(tt.config_path)
        unknown_root_keys = set(raw_config_from_file) - {"engine_config"}
        if unknown_root_keys:
            key = sorted(unknown_root_keys)[0]
            raise ValueError(
                f"Unknown key outside engine_config: {key}. "
                "Put TorchTurbo config fields under engine_config."
            )
        config_from_file = raw_config_from_file["engine_config"]
        if config_from_file is None:
            # A section whose entries are all commented out loads as null.
            config_from_file = OmegaConf.create()
        elif not OmegaConf.is_dict(config_from_file):
            raise ValueError("engine_config must be a mapping.")
        config_data = OmegaConf.to_container(config_from_file, resolve=False)
        _validate_config_file(config_data)

    # Layer 4: explicitly supplied --tt_* CLI flags.
    config_from_cli = OmegaConf.create()
    for destination, value in tt.config_overrides.items():
        for arg_spec in arg_specs:
            if (
                arg_spec.cli_name is None
                or arg_spec.cli_name.lstrip("-") != destination
                or arg_spec.config_path is None
            ):
                continue
            OmegaConf.update(
                config_from_cli, arg_spec.config_path, value, force_add=True
            )
            break
    # Merge all layers into one dict, then build the config once.
    fields = OmegaConf.to_container(
        OmegaConf.merge(fields, config_from_file, config_from_cli), resolve=True
    )

    parallel_plan = FSDPParallelPlan.from_dict(fields.get("parallel_plan") or {})

    ac_d = dict(fields.get("activation_checkpointing") or {})
    if "enabled" in ac_d:
        ac_d["enabled"] = string_to_bool(ac_d["enabled"])
    if "use_reentrant" in ac_d:
        ac_d["use_reentrant"] = string_to_bool(ac_d["use_reentrant"])
    comm_d = dict(fields.get("comm") or {})
    if "overlap_comm_compute" in comm_d:
        comm_d["overlap_comm_compute"] = string_to_bool(
            comm_d["overlap_comm_compute"]
        )

    config = EngineConfig(
        backend=DistributedBackend(fields["backend"]),
        mixed_precision=_parse_mixed_precision_policy(fields["mixed_precision"]),
        use_deterministic_algorithms=fields.get("use_deterministic_algorithms", False),
        parallel_plan=parallel_plan,
        ops=_build_ops_config(fields),
        backend_config=_build_fsdp_config(fields),
        activation_checkpointing=ActivationCheckpointingConfig(**ac_d),
        comm=CommConfig(**comm_d),
        optimizer=_build_optimizer_config(fields),
        checkpoint=_build_checkpoint_config(fields),
        mtp=_build_mtp_config(fields),
        dspark=_build_dspark_training_config(fields),
        moe=_build_moe_config(fields),
        loss_chunk=_build_loss_chunk_config(fields),
        freeze_policy=_build_freeze_policy(fields),
        mfu=_build_mfu_config(fields),
        diffusion=_build_diffusion_config(fields),
        gbs=fields.get("gbs"),
        plugin_paths=list(tt.plugins or []),
        extra={},
    )

    logger.info_rank0(
        "Built EngineConfig from TorchTurboArgs: backend=%s, mp=%s, sp=%d, ep=%d",
        config.backend.value,
        config.mixed_precision.value,
        config.parallel_plan.sp_size,
        config.parallel_plan.ep_size,
    )
    return config


def _dataclass_config_paths(
    config_type: type[Any],
    parents: tuple[str, ...],
) -> set[str]:
    """Return dotted paths for all fields in a nested dataclass schema."""
    paths = set()
    type_hints = get_type_hints(config_type)
    for field_info in dataclasses.fields(config_type):
        field_parts = (*parents, field_info.name)
        paths.add(".".join(field_parts))
        field_type = type_hints[field_info.name]
        candidates = get_args(field_type) or (field_type,)
        for candidate in candidates:
            if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
                paths.update(_dataclass_config_paths(candidate, field_parts))
    return paths


def _validate_config_file(data: Mapping[str, Any], parents: tuple[str, ...] = ()) -> None:
    allowed = {
        spec.config_path
        for spec in get_arg_specs()
        if spec.config_path is not None
    }
    # Allow config blocks that do not have corresponding CLI arguments.
    allowed.update(
        {
            "backend_config.cpu_offload.enabled",
            "backend_config.cpu_offload.mode",
            "backend_config.cpu_offload.optimizer_memory_fraction",
        }
    )
    allowed.update(_dataclass_config_paths(OptimizerConfig, ("optimizer",)))

    for key, value in data.items():
        parts = (*parents, key)
        path = ".".join(parts)
        if parts == ("ops", "overrides"):
            # Op names are free-form keys; OpsConfig and the dispatcher
            # validate the mapping itself.
            continue
        if parts == ("parallel_plan", "plan"):
            from torchturbo.parallel import FSDPPlanSpec

            plan_fields = {f.name for f in dataclasses.fields(FSDPPlanSpec)}
            for fqn, spec in value.items():
                if unknown := set(spec) - plan_fields:
                    raise ValueError(
                        "Unknown key: "
                        + ".".join((*parts, str(fqn), sorted(unknown)[0]))
                    )
                vision_config = spec.get("vision_balance")
                if vision_config is not None:
                    vision_path = ".".join((*parts, str(fqn), "vision_balance"))
                    if not isinstance(vision_config, Mapping):
                        raise ValueError(f"{vision_path} must be a mapping.")
                    if unknown := set(vision_config) - {"mode"}:
                        raise ValueError(
                            f"Unknown key: {vision_path}.{sorted(unknown)[0]}"
                        )
            continue
        if not any(path == item or item.startswith(path + ".") for item in allowed):
            raise ValueError(f"Unknown key: {path}")
        if isinstance(value, Mapping):
            _validate_config_file(value, parts)


def _build_fsdp_config(fields: Mapping[str, Any]) -> Any:
    """FSDPConfig from the merged config dict."""
    from torchturbo.engine.backends.fsdp2.config import (
        FSDPConfig,
        ShardingStrategy,
    )

    fsdp_d = fields.get("backend_config") or {}
    # FSDPConfig.__post_init__ accepts None, dict, DictConfig, and
    # CPUOffloadConfig, and normalizes None to the disabled default.
    return FSDPConfig(
        sharding_strategy=fsdp_d.get(
            "sharding_strategy", ShardingStrategy.FULL_SHARD
        ),
        reshard_after_forward=string_to_bool(
            fsdp_d.get("reshard_after_forward", True)
        ),
        cpu_offload=fsdp_d.get("cpu_offload"),
        prefetch_strategy=fsdp_d.get("prefetch_strategy", "auto"),
        prefetch_direction=fsdp_d.get("prefetch_direction", "forward_backward"),
        forward_prefetch_depth=fsdp_d.get("forward_prefetch_depth", 1),
        backward_prefetch_depth=fsdp_d.get("backward_prefetch_depth", 1),
    )


def _build_checkpoint_config(fields: Mapping[str, Any]) -> CheckpointConfig:
    """CheckpointConfig from the merged config dict."""
    ckpt_d = fields.get("checkpoint") or {}
    return CheckpointConfig(**ckpt_d)


def _build_optimizer_config(fields: Mapping[str, Any]) -> OptimizerConfig:
    optimizer_d = dict(fields.get("optimizer") or {})
    return OptimizerConfig(**optimizer_d)


def _build_ops_config(fields: Mapping[str, Any]) -> OpsConfig:
    """OpsConfig from the merged config dict."""
    ops_d = fields.get("ops") or {}
    policy = OpPolicy(str(ops_d.get("policy", OpPolicy.AUTO.value)).lower())
    return OpsConfig(
        policy=policy,
        overrides=ops_d.get("overrides", {}),
    )


def _build_mtp_config(fields: Mapping[str, Any]) -> MTPConfig:
    """MTPConfig from the merged config dict (nested or flat keys)."""
    mtp_d = dict(fields.get("mtp") or {})
    if "enable_mtp" in fields:
        mtp_d["enabled"] = fields["enable_mtp"]
    if "mtp_loss_scaling_factor" in fields:
        mtp_d["loss_scaling_factor"] = fields["mtp_loss_scaling_factor"]
    loss_scaling_factor = mtp_d.get("loss_scaling_factor")
    return MTPConfig(
        enabled=string_to_bool(mtp_d.get("enabled", False)),
        loss_scaling_factor=float(
            loss_scaling_factor if loss_scaling_factor is not None else 0.1
        ),
        log_loss=string_to_bool(mtp_d.get("log_loss", False)),
    )


def _build_dspark_training_config(fields: Mapping[str, Any]) -> DSparkTrainingConfig:
    """DSparkTrainingConfig from the merged config dict."""
    dspark_d = fields.get("dspark") or {}
    return DSparkTrainingConfig(**dspark_d)


def _build_loss_chunk_config(fields: Mapping[str, Any]) -> LossChunkConfig:
    """LossChunkConfig from the merged config dict."""
    chunk_d = fields.get("loss_chunk") or {}
    return LossChunkConfig(
        enabled=chunk_d.get("enabled", fields.get("enable_chunk_loss", False)),
        chunk_size=chunk_d.get("chunk_size", fields.get("loss_chunk_size", 1024)),
    )


def _build_moe_config(fields: Mapping[str, Any]) -> MoEConfig:
    """MoEConfig from the merged config dict."""
    moe_d = fields.get("moe") or {}
    return MoEConfig(
        aux_loss_coeff=float(moe_d.get("aux_loss_coeff", 0.0)),
        z_loss_coeff=float(moe_d.get("z_loss_coeff", 0.0)),
        token_dispatcher=moe_d.get("token_dispatcher", "alltoall"),
    )


def _build_freeze_policy(fields: Mapping[str, Any]) -> FreezePolicy:
    """FreezePolicy from the merged config dict."""
    freeze_d = fields.get("freeze_policy") or {}
    groups = [
        name
        for name, enabled in (
            ("vision_tower", string_to_bool(freeze_d.get("vision_tower", False))),
            (
                "multi_modal_projector",
                string_to_bool(freeze_d.get("multi_modal_projector", False)),
            ),
            ("language_model", string_to_bool(freeze_d.get("language_model", False))),
        )
        if enabled
    ]
    return FreezePolicy.from_list(groups)


def _build_mfu_config(fields: Mapping[str, Any]) -> MfuConfig:
    """MfuConfig from the merged config dict (CLI ``mfu.enabled``/``mfu.card``)."""
    mfu_d = fields.get("mfu") or {}
    return MfuConfig(
        enabled=string_to_bool(mfu_d.get("enabled", False)),
        card=mfu_d.get("card") or None,
    )


def _build_diffusion_config(fields: Mapping[str, Any]) -> DiffusionConfig:
    """DiffusionConfig from the merged config dict."""
    diffusion_d = fields.get("diffusion") or {}
    return DiffusionConfig(
        cap_lambda=float(diffusion_d.get("cap_lambda") or 0.0),
        bd_mask_low=float(diffusion_d.get("bd_mask_low", 0.3)),
        bd_mask_high=float(diffusion_d.get("bd_mask_high", 0.8)),
        bd_block_size=int(diffusion_d.get("bd_block_size", 32)),
    )
