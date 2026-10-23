"""CLI argument definitions for torchturbo, replacing TORCH_TURBO_* env vars.

Usage::

    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    argv = sys.argv[1:]
    tt_args, remaining_argv = parse_torchturbo_args(argv)
    set_torchturbo_args(tt_args)
    # pass remaining_argv to the training framework

The parsed args can later be retrieved::

    from torchturbo.engine.args import get_torchturbo_args

    tt_args = get_torchturbo_args()
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..logging import LogLevel, get_logger
from .backends.fsdp2.config import CPUOffloadConfig

logger = get_logger(__name__)


def string_to_bool(value: str | bool) -> bool:
    """Parse bool values with the same semantics as Transformers."""
    from transformers.hf_argparser import string_to_bool as hf_string_to_bool

    return hf_string_to_bool(value)


def parse_log_level(value: str) -> LogLevel:
    """Parse a TorchTurbo log level name."""
    try:
        return LogLevel[value.strip()]
    except KeyError as exc:
        supported = ", ".join(LogLevel.__members__)
        raise argparse.ArgumentTypeError(
            f"Unsupported log level {value!r}; supported levels: {supported}"
        ) from exc


# ---------------------------------------------------------------------------
# Module-level context for stashing parsed args (single-process training)
# ---------------------------------------------------------------------------

_current_torchturbo_args: TorchTurboArgs | None = None


def set_torchturbo_args(args: TorchTurboArgs) -> None:
    """Store parsed torchturbo args in module-level context."""
    global _current_torchturbo_args
    _current_torchturbo_args = args


def get_torchturbo_args() -> TorchTurboArgs | None:
    """Retrieve previously stashed torchturbo args."""
    return _current_torchturbo_args


# ---------------------------------------------------------------------------
# TorchTurboArgs — mirrors TORCH_TURBO_* env vars
# ---------------------------------------------------------------------------


@dataclass
class TorchTurboArgs:
    """CLI arguments for torchturbo.

    Optional integration fields use ``None`` to mean "not specified" when they
    should not override values from other sources.

    Attributes
    ----------
    backend:
        Distributed backend (``"fsdp2"``).
    mixed_precision:
        Mixed precision policy (``"bf16"``, ``"fp32"``).
    use_deterministic_algorithms:
        Require deterministic algorithms for the selected device runtime.
    op_policy:
        Default op dispatch policy (``"auto"``, ``"native"``, ``"custom"``).
    sp_size:
        Sequence parallelism group size.
    ep_size:
        Expert parallelism group size.
    gbs:
        Global batch size in micro-batch units. When set, the engine derives
        ``gradient_accumulation_steps = gbs * sp_size / world_size`` (overriding
        the framework's own), letting the SP dimension control the effective
        global batch size.
    gradient_checkpointing:
        Whether TorchTurbo should enable engine-owned activation
        checkpointing. Defaults to ``True``.
    cpu_offload:
        Structured CPU offload policy (``CPUOffloadConfig``).
        File configuration uses ``enabled``, ``mode`` and
        ``optimizer_memory_fraction``.
    cpu_offload_enabled:
        Explicit ``--tt_cpu_offload`` boolean override for the file's
        ``cpu_offload.enabled`` value. ``None`` means no CLI override.
    enable_mtp:
        Whether to build Multi-Token-Prediction (MTP) layers.  ``True``
        opts in: build MTP heads, load their weights, run forward, compute
        loss.  ``False`` (default) — MTP is OFF
        (``num_nextn_predict_layers`` is overridden to 0).
    mtp_loss_scaling_factor:
        Weight applied to each MTP head's loss before adding to the main
        loss.  Defaults to ``0.1`` (Megatron-LM default).  Ignored when MTP
        is off.  Set ``0`` to keep MTP forward (for logging) but exclude it
        from gradients.
    mtp_log_loss:
        Log each raw MTP loss at info level. Defaults to ``False``.
    enable_dspark:
        Whether to run the DeepSeek-V4 DSpark training branch. Requires MTP
        layers to be enabled and a DSpark model configuration.
    dspark_loss_scaling_factor:
        Weight applied to the DSpark loss before adding it to the main loss.
        Set ``0`` to keep the DSpark forward but exclude it from gradients.
    dspark_num_anchors:
        DeepSeek-V4 DSpark maximum anchor-block count.
    dspark_loss_decay_gamma:
        DeepSeek-V4 DSpark position-decay gamma. ``None`` leaves the
        official DeepSpec default unchanged.
    dspark_ce_loss_alpha:
        DeepSeek-V4 DSpark cross-entropy loss coefficient. ``None`` leaves the
        official DeepSpec default unchanged.
    dspark_l1_loss_alpha:
        DeepSeek-V4 DSpark distribution-match L1 coefficient. ``None`` leaves
        the official DeepSpec default unchanged.
    dspark_confidence_loss_alpha:
        DeepSeek-V4 DSpark confidence-head BCE coefficient. ``None`` leaves the
        official DeepSpec default unchanged.
    bd_mask_low:
        LLaDA2 block-diffusion per-sample noise ratio (sigma) lower bound.
        Defaults to ``0.3``.
    bd_mask_high:
        As ``bd_mask_low`` but the upper bound. Defaults to ``0.8``.
    bd_block_size:
        LLaDA2 block-diffusion block size (the BD block-diagonal attention
        granularity and the unit ``cutoff_len`` is ceil-aligned to). Defaults to
        ``32`` (dFactory/LLaDA2 default). Only affects the BD dataset path.
    enable_chunk_loss:
        Enable chunked fused ``lm_head`` + cross-entropy during training.
    loss_chunk_size:
        Flattened tokens per loss chunk. Defaults to ``1024`` when chunked
        loss is enabled.
    hf_export:
        Whether to export HF safetensors with the final training checkpoint
        or every training checkpoint. Defaults to every checkpoint.
    log_level:
        TorchTurbo log level. Defaults to ``LogLevel.INFO``.
    """

    backend: str | None = None
    mixed_precision: str | None = None
    op_policy: str | None = None
    optimizer_name: str | None = None
    use_deterministic_algorithms: bool = False
    sp_size: int | None = None
    ep_size: int | None = None
    gbs: int | None = None
    gradient_checkpointing: bool = True
    ac_use_reentrant: bool = False
    cpu_offload: CPUOffloadConfig = field(default_factory=CPUOffloadConfig)
    cpu_offload_enabled: bool | None = None
    enable_mtp: bool = False
    mtp_loss_scaling_factor: float | None = 0.1
    moe_aux_loss_coeff: float = 0.0
    moe_z_loss_coeff: float = 0.0
    moe_token_dispatcher: str | None = None
    mtp_log_loss: bool = False
    enable_dspark: bool = True
    dspark_loss_scaling_factor: float = 0.1
    dspark_num_anchors: int | None = None
    dspark_loss_decay_gamma: float | None = None
    dspark_ce_loss_alpha: float | None = None
    dspark_l1_loss_alpha: float | None = None
    dspark_confidence_loss_alpha: float | None = None
    enable_chunk_loss: bool = False
    loss_chunk_size: int | None = None
    freeze_vision_tower: bool = False
    freeze_multi_modal_projector: bool = False
    freeze_language_model: bool = False
    hf_export: str | None = None
    profiling: bool = False
    memory_snapshot_steps: int | None = None
    log_level: LogLevel = LogLevel.INFO

    # Block-diffusion (BD) fields — threaded into EngineConfig.diffusion by
    # build_config_from_lf_args(); only active for is_diffusion_model models.
    cap_lambda: float = 0.0  # CAP/DPARALLEL confidence loss weight (entropy on correct tokens). 0 = disabled.
    bd_mask_low: float = 0.3  # LLaDA2 BD per-sample sigma lower bound.
    bd_mask_high: float = 0.8  # LLaDA2 BD per-sample sigma upper bound.
    bd_block_size: int = 32  # LLaDA2 BD block size (block-diagonal attn granularity; cutoff ceil-aligns to it).

    # Prefetch strategy (FSDP2 unit prefetch; orthogonal to LLaDA2, kept in sync with mainline).
    prefetch_strategy: str = "auto"
    prefetch_direction: str = "forward_backward"
    forward_prefetch_depth: int = 1
    backward_prefetch_depth: int = 1

    config_path: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)

    # ``--tt_plugin`` paths
    plugins: list[str] = field(default_factory=list)


@dataclass
class TurboArgSpec:
    """Describe one external TorchTurbo argument and its canonical config path."""

    name: str
    """Stable unique name inside the registry."""

    config_path: str | None = None
    """Canonical path in the merged mapping; None means runtime-only."""

    args_name: str | None = None
    """TorchTurboArgs field name; None for environment-only inputs."""

    cli_name: str | None = None
    """CLI option name including the '--' prefix; None means no CLI input."""

    env_name: str | None = None
    """Full TORCH_TURBO_* variable name; None means no ENV input."""

    type: Callable[[str], Any] | None = None
    """String-to-typed-value converter for CLI and env inputs; None means str."""

    metavar: str | None = None
    """Argparse metavar for the CLI option; None means argparse default."""

    choices: tuple[str, ...] | None = None
    """Argparse ``choices`` for the CLI option; None means unrestricted."""

    action: str | None = None
    """Argparse action for the CLI option (e.g. ``"append"`` for list flags);
    None means the default store action."""

    help: str | None = None
    """Help/documentation text for the CLI option."""


_TURBO_ARG_SPECS: tuple[TurboArgSpec, ...] = (
    TurboArgSpec(
        name="config_path",
        args_name="config_path",
        cli_name="--tt_config",
        metavar="PATH",
        help="TorchTurbo YAML or JSON configuration file.",
    ),
    # Engine inputs: top level
    TurboArgSpec(
        name="backend",
        args_name="backend",
        cli_name="--tt_backend",
        env_name="TORCH_TURBO_BACKEND",
        config_path="backend",
        metavar="BACKEND",
        help="Distributed backend (e.g. fsdp2). Replaces TORCH_TURBO_BACKEND.",
    ),
    TurboArgSpec(
        name="mixed_precision",
        args_name="mixed_precision",
        cli_name="--tt_mixed_precision",
        env_name="TORCH_TURBO_MIXED_PRECISION",
        config_path="mixed_precision",
        metavar="POLICY",
        help="Mixed precision policy (bf16, fp32). "
        "Replaces TORCH_TURBO_MIXED_PRECISION.",
    ),
    TurboArgSpec(
        name="op_policy",
        args_name="op_policy",
        cli_name="--tt_op_policy",
        env_name="TORCH_TURBO_OP_POLICY",
        config_path="ops.policy",
        metavar="POLICY",
        help="Default op dispatch policy (auto, native, custom). "
        "Replaces TORCH_TURBO_OP_POLICY.",
    ),
    TurboArgSpec(
        name="optimizer_name",
        args_name="optimizer_name",
        cli_name="--tt_optimizer",
        config_path="optimizer.name",
        metavar="NAME",
        choices=("muon",),
        help="Use an engine-managed optimizer (currently: muon).",
    ),
    TurboArgSpec(
        name="use_deterministic_algorithms",
        args_name="use_deterministic_algorithms",
        cli_name="--tt_use_deterministic_algorithms",
        env_name="TORCH_TURBO_USE_DETERMINISTIC_ALGORITHMS",
        config_path="use_deterministic_algorithms",
        type=string_to_bool,
        metavar="BOOL",
        help="Require deterministic algorithms for the selected device runtime.",
    ),
    TurboArgSpec(
        name="sp_size",
        args_name="sp_size",
        cli_name="--tt_sp_size",
        env_name="TORCH_TURBO_SP_SIZE",
        config_path="parallel_plan.sp_size",
        type=int,
        metavar="SIZE",
        help="Sequence parallelism group size. Replaces TORCH_TURBO_SP_SIZE.",
    ),
    TurboArgSpec(
        name="ep_size",
        args_name="ep_size",
        cli_name="--tt_ep_size",
        env_name="TORCH_TURBO_EP_SIZE",
        config_path="parallel_plan.ep_size",
        type=int,
        metavar="SIZE",
        help="Expert parallelism group size. Replaces TORCH_TURBO_EP_SIZE.",
    ),
    TurboArgSpec(
        name="gbs",
        args_name="gbs",
        cli_name="--tt_gbs",
        env_name="TORCH_TURBO_GBS",
        config_path="gbs",
        type=int,
        metavar="GBS",
        help="Global batch size in micro-batch units. When set, "
        "gradient_accumulation_steps is derived as tt_gbs * sp_size / "
        "world_size (overriding the framework's own ga/gbs), so the SP "
        "dimension controls the effective global batch size. "
        "Replaces TORCH_TURBO_GBS.",
    ),
    TurboArgSpec(
        name="gradient_checkpointing",
        args_name="gradient_checkpointing",
        cli_name="--tt_gradient_checkpointing",
        env_name="TORCH_TURBO_GRADIENT_CHECKPOINTING",
        config_path="activation_checkpointing.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Enable or disable TorchTurbo-owned activation checkpointing. "
        "Replaces TORCH_TURBO_GRADIENT_CHECKPOINTING.",
    ),
    TurboArgSpec(
        name="ac_use_reentrant",
        args_name="ac_use_reentrant",
        cli_name="--tt_ac_use_reentrant",
        env_name="TORCH_TURBO_AC_USE_REENTRANT",
        config_path="activation_checkpointing.use_reentrant",
        type=string_to_bool,
        metavar="BOOL",
        help="Use reentrant activation checkpointing (use_reentrant=True) "
        "instead of the default non-reentrant implementation. Reentrant "
        "checkpointing tears down recomputed intermediates synchronously "
        "within a step, reducing cross-step allocation peak for MoE/linear-"
        "attention workloads. Replaces TORCH_TURBO_AC_USE_REENTRANT.",
    ),
    TurboArgSpec(
        name="cpu_offload",
        args_name="cpu_offload",
        cli_name=None,
        env_name=None,
        config_path="backend_config.cpu_offload",
        help="Structured CPU offload policy (see CPUOffloadConfig).",
    ),
    TurboArgSpec(
        name="cpu_offload_enabled",
        args_name="cpu_offload_enabled",
        cli_name="--tt_cpu_offload",
        config_path="backend_config.cpu_offload.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Override backend_config.cpu_offload.enabled. Accepts "
        "--tt_cpu_offload, --tt_cpu_offload true, or --tt_cpu_offload false.",
    ),
    TurboArgSpec(
        name="prefetch_strategy",
        args_name="prefetch_strategy",
        cli_name="--tt_prefetch_strategy",
        env_name="TORCH_TURBO_PREFETCH_STRATEGY",
        config_path="backend_config.prefetch_strategy",
        metavar="STRATEGY",
        help="FSDP prefetch strategy: native|auto|expert_with_layer|"
        "expert_lookahead. Replaces TORCH_TURBO_PREFETCH_STRATEGY.",
    ),
    TurboArgSpec(
        name="prefetch_direction",
        args_name="prefetch_direction",
        cli_name="--tt_prefetch_direction",
        env_name="TORCH_TURBO_PREFETCH_DIRECTION",
        config_path="backend_config.prefetch_direction",
        metavar="DIRECTION",
        help="Prefetch direction: forward|forward_backward. "
        "Replaces TORCH_TURBO_PREFETCH_DIRECTION.",
    ),
    TurboArgSpec(
        name="forward_prefetch_depth",
        args_name="forward_prefetch_depth",
        cli_name="--tt_forward_prefetch_depth",
        env_name="TORCH_TURBO_FORWARD_PREFETCH_DEPTH",
        config_path="backend_config.forward_prefetch_depth",
        type=int,
        metavar="N",
        help="How many upcoming layers each source prefetches in forward "
        "(1 = next layer only). Higher trades peak memory for more overlap. "
        "Replaces TORCH_TURBO_FORWARD_PREFETCH_DEPTH.",
    ),
    TurboArgSpec(
        name="backward_prefetch_depth",
        args_name="backward_prefetch_depth",
        cli_name="--tt_backward_prefetch_depth",
        env_name="TORCH_TURBO_BACKWARD_PREFETCH_DEPTH",
        config_path="backend_config.backward_prefetch_depth",
        type=int,
        metavar="N",
        help="How many previous layers each layer prefetches in backward "
        "(1 = previous layer only). Only used with forward_backward direction. "
        "Replaces TORCH_TURBO_BACKWARD_PREFETCH_DEPTH.",
    ),
    TurboArgSpec(
        name="enable_mtp",
        args_name="enable_mtp",
        cli_name="--tt_enable_mtp",
        config_path="mtp.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Build Multi-Token-Prediction (MTP) layers. "
        "Accepts --tt_enable_mtp, --tt_enable_mtp true, or --tt_enable_mtp false. "
        "Off by default.",
    ),
    TurboArgSpec(
        name="mtp_loss_scaling_factor",
        args_name="mtp_loss_scaling_factor",
        cli_name="--tt_mtp_loss_scaling_factor",
        config_path="mtp.loss_scaling_factor",
        type=float,
        metavar="FLOAT",
        help="MTP loss weight. Ignored when MTP is off.",
    ),
    TurboArgSpec(
        name="enable_dspark",
        args_name="enable_dspark",
        cli_name="--tt_enable_dspark",
        config_path="dspark.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Enable the DeepSeek-V4 DSpark training branch. "
        "Off skips its loss without stripping MTP modules.",
    ),
    TurboArgSpec(
        name="dspark_loss_scaling_factor",
        args_name="dspark_loss_scaling_factor",
        cli_name="--tt_dspark_loss_scaling_factor",
        config_path="dspark.loss_scaling_factor",
        type=float,
        metavar="FLOAT",
        help="DeepSeek-V4 DSpark loss weight.",
    ),
    TurboArgSpec(
        name="dspark_num_anchors",
        args_name="dspark_num_anchors",
        cli_name="--tt_dspark_num_anchors",
        config_path="dspark.num_anchors",
        type=int,
        metavar="N",
        help="DeepSeek-V4 DSpark maximum anchor-block count. "
        "Unset keeps the official DeepSpec default.",
    ),
    TurboArgSpec(
        name="dspark_loss_decay_gamma",
        args_name="dspark_loss_decay_gamma",
        cli_name="--tt_dspark_loss_decay_gamma",
        config_path="dspark.loss_decay_gamma",
        type=float,
        metavar="FLOAT",
        help="DeepSeek-V4 DSpark position-decay gamma. "
        "Unset keeps the official DeepSpec default.",
    ),
    TurboArgSpec(
        name="dspark_ce_loss_alpha",
        args_name="dspark_ce_loss_alpha",
        cli_name="--tt_dspark_ce_loss_alpha",
        config_path="dspark.ce_loss_alpha",
        type=float,
        metavar="FLOAT",
        help="DeepSeek-V4 DSpark cross-entropy loss coefficient. "
        "Unset keeps the official DeepSpec default.",
    ),
    TurboArgSpec(
        name="dspark_l1_loss_alpha",
        args_name="dspark_l1_loss_alpha",
        cli_name="--tt_dspark_l1_loss_alpha",
        config_path="dspark.l1_loss_alpha",
        type=float,
        metavar="FLOAT",
        help="DeepSeek-V4 DSpark distribution-match L1 coefficient. "
        "Unset keeps the official DeepSpec default.",
    ),
    TurboArgSpec(
        name="dspark_confidence_loss_alpha",
        args_name="dspark_confidence_loss_alpha",
        cli_name="--tt_dspark_confidence_loss_alpha",
        config_path="dspark.confidence_loss_alpha",
        type=float,
        metavar="FLOAT",
        help="DeepSeek-V4 DSpark confidence-head BCE coefficient. "
        "Unset keeps the official DeepSpec default.",
    ),
    TurboArgSpec(
        name="moe_token_dispatcher",
        args_name="moe_token_dispatcher",
        cli_name="--tt_moe_token_dispatcher",
        env_name="TORCH_TURBO_MOE_TOKEN_DISPATCHER",
        config_path="moe.token_dispatcher",
        type=str,
        choices=("alltoall",),
        help="Expert token dispatcher type: alltoall.",
    ),
    TurboArgSpec(
        name="moe_aux_loss_coeff",
        args_name="moe_aux_loss_coeff",
        cli_name="--tt_moe_aux_loss_coeff",
        config_path="moe.aux_loss_coeff",
        type=float,
        metavar="FLOAT",
        help="Token-level MoE auxiliary loss coefficient.",
    ),
    TurboArgSpec(
        name="moe_z_loss_coeff",
        args_name="moe_z_loss_coeff",
        cli_name="--tt_moe_z_loss_coeff",
        config_path="moe.z_loss_coeff",
        type=float,
        metavar="FLOAT",
        help="Z-loss coefficient for MoE router logits.",
    ),
    TurboArgSpec(
        name="mtp_log_loss",
        args_name="mtp_log_loss",
        cli_name="--tt_mtp_log_loss",
        config_path="mtp.log_loss",
        type=string_to_bool,
        metavar="BOOL",
        help="Log each raw MTP loss at info level. Off by default.",
    ),
    TurboArgSpec(
        name="enable_chunk_loss",
        args_name="enable_chunk_loss",
        cli_name="--tt_enable_chunk_loss",
        config_path="loss_chunk.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Enable chunked lm_head + cross-entropy during training.",
    ),
    TurboArgSpec(
        name="loss_chunk_size",
        args_name="loss_chunk_size",
        cli_name="--tt_loss_chunk_size",
        config_path="loss_chunk.chunk_size",
        type=int,
        metavar="TOKENS",
        help="Flattened tokens per loss chunk. Defaults to 1024.",
    ),
    TurboArgSpec(
        name="freeze_vision_tower",
        args_name="freeze_vision_tower",
        cli_name="--tt_freeze_vision_tower",
        config_path="freeze_policy.vision_tower",
        type=string_to_bool,
        metavar="BOOL",
        help="Freeze vision encoder parameters (requires_grad=False). "
        "Accepts --tt_freeze_vision_tower, ... true, or ... false.",
    ),
    TurboArgSpec(
        name="freeze_multi_modal_projector",
        args_name="freeze_multi_modal_projector",
        cli_name="--tt_freeze_multi_modal_projector",
        config_path="freeze_policy.multi_modal_projector",
        type=string_to_bool,
        metavar="BOOL",
        help="Freeze multi-modal projector parameters (requires_grad=False). "
        "Accepts --tt_freeze_multi_modal_projector, ... true, or ... false.",
    ),
    TurboArgSpec(
        name="freeze_language_model",
        args_name="freeze_language_model",
        cli_name="--tt_freeze_language_model",
        config_path="freeze_policy.language_model",
        type=string_to_bool,
        metavar="BOOL",
        help="Freeze language model parameters (requires_grad=False). "
        "Accepts --tt_freeze_language_model, ... true, or ... false.",
    ),
    TurboArgSpec(
        name="cap_lambda",
        args_name="cap_lambda",
        cli_name="--tt_cap_lambda",
        config_path="diffusion.cap_lambda",
        type=float,
        metavar="FLOAT",
        help="CAP/DPARALLEL confidence loss weight (entropy on correctly-"
        "predicted tokens). 0 disables (default). Recommended: 2.0.",
    ),
    TurboArgSpec(
        name="bd_mask_low",
        args_name="bd_mask_low",
        cli_name="--tt_bd_mask_low",
        config_path="diffusion.bd_mask_low",
        type=float,
        metavar="FLOAT",
        help="LLaDA2 BD per-sample noise ratio (sigma) lower bound (default 0.3). "
        "Only affects the BD dataset path.",
    ),
    TurboArgSpec(
        name="bd_mask_high",
        args_name="bd_mask_high",
        cli_name="--tt_bd_mask_high",
        config_path="diffusion.bd_mask_high",
        type=float,
        metavar="FLOAT",
        help="LLaDA2 BD per-sample noise ratio (sigma) upper bound (default 0.8). "
        "Only affects the BD dataset path.",
    ),
    TurboArgSpec(
        name="bd_block_size",
        args_name="bd_block_size",
        cli_name="--tt_bd_block_size",
        config_path="diffusion.bd_block_size",
        type=int,
        metavar="INT",
        help="LLaDA2 BD block size (block-diagonal attention granularity; "
        "cutoff_len is ceil-aligned to it). Default 32. Only affects the "
        "BD dataset path.",
    ),
    # Runtime-only CLI inputs
    TurboArgSpec(
        name="profiling",
        args_name="profiling",
        cli_name="--tt_profiling",
        type=string_to_bool,
        metavar="BOOL",
        help="Enable NPU/CUDA profiling for the second step. "
        "Outputs to {output_dir}/profiling/. "
        "Accepts --tt_profiling, ... true, or ... false.",
    ),
    TurboArgSpec(
        name="memory_snapshot_steps",
        args_name="memory_snapshot_steps",
        cli_name="--tt_memory_snapshot_steps",
        type=int,
        metavar="STEPS",
        help="Number of optimizer steps to capture from training step 0. For "
        "example, 3 captures the first 3 optimizer steps. Outputs to "
        "{output_dir}/memory_snapshot/.",
    ),
    TurboArgSpec(
        name="log_level",
        args_name="log_level",
        cli_name="--tt_log_level",
        type=parse_log_level,
        metavar="LEVEL",
        help="TorchTurbo log level: DEBUG, INFO, WARNING, or ERROR.",
    ),
    # Environment-only Engine input
    TurboArgSpec(
        name="reshard_after_forward",
        env_name="TORCH_TURBO_RESHARD_AFTER_FWD",
        config_path="backend_config.reshard_after_forward",
        type=string_to_bool,
    ),
    TurboArgSpec(
        name="log_mfu",
        cli_name="--tt_log_mfu",
        env_name="TORCH_TURBO_LOG_MFU",
        config_path="mfu.enabled",
        type=string_to_bool,
        metavar="BOOL",
        help="Turn MFU/HFU logging on (off by default)",
    ),
    TurboArgSpec(
        name="card_name",
        cli_name="--tt_card_name",
        env_name="TORCH_TURBO_CARD_NAME",
        config_path="mfu.card",
        metavar="CARD",
        help="Chip family for the device runtime's peak table. Unset (default) "
        "uses the runtime's default card (NPU 910c / CUDA h20); "
        "e.g. 910b / h100 / h200 / a100.",
    ),
    TurboArgSpec(
        name="hf_export",
        args_name="hf_export",
        config_path="checkpoint.hf_export",
        cli_name="--tt_hf_export",
        choices=("final", "every"),
        help=(
            "When training checkpoints also export HF safetensors: 'final' "
            "exports with the final checkpoint only, 'every' (default) with "
            "every checkpoint. Exact model state is always saved in the "
            "backend's native format; FSDP2 stores it as DCP."
        ),
    ),
    # turbo's plugin mechanism loads these modules; each registers a Plugin for a framework
    # (or framework-agnostic), and the framework adapter dispatches it on activation.
    TurboArgSpec(
        name="plugins",
        args_name="plugins",
        cli_name="--tt_plugin",
        metavar="PATH",
        action="append",
        help="Plugin .py to load (repeatable); optionally bind a per-plugin "
        "config as 'plugin:config' (passed to the Plugin's __init__).",
    ),
)


def get_arg_specs() -> tuple[TurboArgSpec, ...]:
    """Return all registered TorchTurbo argument specifications."""
    return _TURBO_ARG_SPECS


# ---------------------------------------------------------------------------
# Argparse-based CLI parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``--tt_*`` flags."""
    parser = argparse.ArgumentParser(
        description="torchturbo arguments",
        add_help=False,
    )
    for spec in get_arg_specs():
        if spec.cli_name is None:
            continue
        kwargs = {
            "type": spec.type,
            "default": argparse.SUPPRESS,
            "metavar": spec.metavar,
            "help": spec.help,
        }
        if spec.choices is not None:
            kwargs["choices"] = spec.choices
        if spec.action is not None:
            kwargs["action"] = spec.action
            # "append" stores a list; don't pass type/metavar semantics that
            # conflict with a store-true/append action.
            kwargs.pop("type", None)
        if spec.type is string_to_bool:
            kwargs["nargs"] = "?"
            kwargs["const"] = True
        parser.add_argument(spec.cli_name, **kwargs)
    return parser


def parse_torchturbo_args(argv: list[str]) -> tuple[TorchTurboArgs, list[str]]:
    """Parse ``--tt_*`` arguments from *argv*, returning parsed args and
    the remaining (unconsumed) arguments.

    This uses ``parse_known_args`` so that non-torchturbo arguments are
    left untouched and can be forwarded to the training framework.

    The parser uses ``argparse.SUPPRESS``, so absent options never enter the
    parsed namespace: ``TorchTurboArgs`` defaults fill in absent fields, while
    ``config_overrides`` only ever contains explicitly supplied flags (the
    highest-priority layer in :func:`build_from_args`).

    Parameters
    ----------
    argv:
        Argument list (e.g. ``sys.argv[1:]``).

    Returns
    -------
    A tuple ``(TorchTurboArgs, remaining_argv)`` where *remaining_argv*
    contains all arguments that are not ``--tt_*`` flags.

    Example::

        tt_args, remaining = parse_torchturbo_args(sys.argv[1:])
        # Pass remaining arguments to the caller's argument parser.
    """
    parser = _build_parser()
    parsed, remaining = parser.parse_known_args(argv)
    explicit = vars(parsed)
    arg_specs = get_arg_specs()

    # Only explicitly supplied CLI args enter TorchTurboArgs; absent fields use dataclass defaults.
    args_values = {
        spec.args_name: explicit[spec.cli_name.lstrip("-")]
        for spec in arg_specs
        if spec.args_name is not None
        and spec.cli_name is not None
        and spec.cli_name.lstrip("-") in explicit
    }
    # Runtime-only args have no config_path; exclude them so they never touch EngineConfig.
    runtime_destinations = {
        spec.cli_name.lstrip("-")
        for spec in arg_specs
        if spec.config_path is None and spec.cli_name is not None
    }
    config_overrides = {
        key: value
        for key, value in explicit.items()
        if key not in runtime_destinations
    }
    args = TorchTurboArgs(**args_values, config_overrides=config_overrides)

    if any(
        v is not None
        for v in [
            args.backend,
            args.mixed_precision,
            args.op_policy,
            args.optimizer_name,
            args.sp_size,
            args.ep_size,
            args.cpu_offload_enabled,
            args.enable_mtp,
            args.mtp_loss_scaling_factor,
            args.moe_aux_loss_coeff,
            args.moe_z_loss_coeff,
            args.moe_token_dispatcher,
            args.mtp_log_loss,
            args.enable_dspark,
            args.dspark_loss_scaling_factor,
            args.dspark_num_anchors,
            args.dspark_loss_decay_gamma,
            args.dspark_ce_loss_alpha,
            args.dspark_l1_loss_alpha,
            args.dspark_confidence_loss_alpha,
            args.enable_chunk_loss,
            args.loss_chunk_size,
        ]
    ):
        logger.info_rank0("torchturbo CLI args: %s", args)
    return args, remaining
