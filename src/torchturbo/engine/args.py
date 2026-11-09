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
from dataclasses import dataclass

from ..logging import LogLevel, get_logger

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
        Mixed precision policy (``"bf16"``, ``"fp16"``, ``"fp32"``, ``"fp8"``).
    op_policy:
        Default op dispatch policy (``"auto"``, ``"native"``, ``"custom"``).
    sp_size:
        Sequence parallelism group size.
    ep_size:
        Expert parallelism group size.
    gradient_checkpointing:
        Whether TorchTurbo should enable engine-owned activation
        checkpointing. Defaults to ``True``.
    cpu_offload:
        Whether CPU offload is applied to FSDP-wrapped units. Defaults to
        ``False``.
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
    bd_mask_low:
        LLaDA2 block-diffusion per-sample noise ratio (sigma) lower bound.
        Defaults to ``0.3``.
    bd_mask_high:
        As ``bd_mask_low`` but the upper bound. Defaults to ``0.8``.
    log_level:
        TorchTurbo log level. Defaults to ``LogLevel.INFO``.
    """

    backend: str | None = None
    mixed_precision: str | None = None
    op_policy: str | None = None
    sp_size: int | None = None
    ep_size: int | None = None
    gradient_checkpointing: bool = True
    ac_use_reentrant: bool = False
    cpu_offload: bool = False
    mega_moe: bool = False
    enable_mtp: bool = False
    mtp_loss_scaling_factor: float | None = 0.1
    freeze_vision_tower: bool = False
    freeze_multi_modal_projector: bool = False
    freeze_language_model: bool = False
    profiling: bool = False
    memory_snapshot_steps: int | None = None
    log_level: LogLevel = LogLevel.INFO

    # Adapter-seeded fields (NOT CLI ``--tt_*`` flags). Adapters fold values
    # sourced from their own args (model path, output_dir, bf16/fp16 -> mixed
    # precision, ...) onto these before calling ``build_from_args``, so the
    # builder can reconstruct an EngineConfig from a single TorchTurboArgs.
    model_path: str | None = None
    output_dir: str | None = None

    # Block-diffusion (BD) fields — threaded into EngineConfig.diffusion by
    # build_config_from_lf_args(); only active for is_diffusion_model models.
    cap_lambda: float = 0.0  # CAP/DPARALLEL confidence loss weight (entropy on correct tokens). 0 = disabled.
    bd_mask_low: float = 0.3  # LLaDA2 BD per-sample sigma lower bound.
    bd_mask_high: float = 0.8  # LLaDA2 BD per-sample sigma upper bound.

    # Prefetch strategy (FSDP2 unit prefetch; orthogonal to LLaDA2, kept in sync with mainline).
    prefetch_strategy: str = "auto"
    prefetch_direction: str = "forward_backward"
    forward_prefetch_depth: int = 1
    backward_prefetch_depth: int = 1


# ---------------------------------------------------------------------------
# Argparse-based CLI parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``--tt_*`` flags."""
    parser = argparse.ArgumentParser(
        description="torchturbo arguments",
        add_help=False,
    )

    parser.add_argument(
        "--tt_backend",
        type=str,
        default=None,
        metavar="BACKEND",
        help="Distributed backend (e.g. fsdp2). Replaces TORCH_TURBO_BACKEND.",
    )
    parser.add_argument(
        "--tt_mixed_precision",
        type=str,
        default=None,
        metavar="POLICY",
        help="Mixed precision policy (bf16, fp16, fp32, fp8). "
        "Replaces TORCH_TURBO_MIXED_PRECISION.",
    )
    parser.add_argument(
        "--tt_op_policy",
        type=str,
        default=None,
        metavar="POLICY",
        help="Default op dispatch policy (auto, native, custom). "
        "Replaces TORCH_TURBO_OP_POLICY.",
    )
    parser.add_argument(
        "--tt_mega_moe",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Enable the BF16 NPU MegaMoE/MoonEP backend for grouped experts. "
        "Replaces TORCH_TURBO_MEGA_MOE. Expert LoRA is unsupported.",
    )
    parser.add_argument(
        "--tt_sp_size",
        type=int,
        default=None,
        metavar="SIZE",
        help="Sequence parallelism group size. Replaces TORCH_TURBO_SP_SIZE.",
    )
    parser.add_argument(
        "--tt_ep_size",
        type=int,
        default=None,
        metavar="SIZE",
        help="Expert parallelism group size. Replaces TORCH_TURBO_EP_SIZE.",
    )
    parser.add_argument(
        "--tt_gradient_checkpointing",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help="Enable or disable TorchTurbo-owned activation checkpointing. "
        "Replaces TORCH_TURBO_GRADIENT_CHECKPOINTING.",
    )
    parser.add_argument(
        "--tt_ac_use_reentrant",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Use reentrant activation checkpointing (use_reentrant=True) "
        "instead of the default non-reentrant implementation. Reentrant "
        "checkpointing tears down recomputed intermediates synchronously "
        "within a step, reducing cross-step allocation peak for MoE/linear-"
        "attention workloads. Replaces TORCH_TURBO_AC_USE_REENTRANT.",
    )
    parser.add_argument(
        "--tt_cpu_offload",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Enable or disable FSDP CPU offload for wrapped units. "
        "Replaces TORCH_TURBO_CPU_OFFLOAD.",
    )
    parser.add_argument(
        "--tt_prefetch_strategy",
        type=str,
        default="auto",
        metavar="STRATEGY",
        help="FSDP prefetch strategy: native|auto|expert_with_layer|expert_lookahead. "
        "Replaces TORCH_TURBO_PREFETCH_STRATEGY.",
    )
    parser.add_argument(
        "--tt_prefetch_direction",
        type=str,
        default="forward_backward",
        metavar="DIRECTION",
        help="Prefetch direction: forward|forward_backward. "
        "Replaces TORCH_TURBO_PREFETCH_DIRECTION.",
    )
    parser.add_argument(
        "--tt_forward_prefetch_depth",
        type=int,
        default=1,
        metavar="N",
        help="How many upcoming layers each source prefetches in forward "
        "(1 = next layer only). Higher trades peak memory for more overlap. "
        "Replaces TORCH_TURBO_FORWARD_PREFETCH_DEPTH.",
    )
    parser.add_argument(
        "--tt_backward_prefetch_depth",
        type=int,
        default=1,
        metavar="N",
        help="How many previous layers each layer prefetches in backward "
        "(1 = previous layer only). Only used with forward_backward direction. "
        "Replaces TORCH_TURBO_BACKWARD_PREFETCH_DEPTH.",
    )
    parser.add_argument(
        "--tt_enable_mtp",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Build Multi-Token-Prediction (MTP) layers. "
        "Accepts --tt_enable_mtp, --tt_enable_mtp true, or --tt_enable_mtp false. "
        "Off by default.",
    )
    parser.add_argument(
        "--tt_mtp_loss_scaling_factor",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="MTP loss weight. Ignored when MTP is off.",
    )
    parser.add_argument(
        "--tt_freeze_vision_tower",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Freeze vision encoder parameters (requires_grad=False). "
        "Accepts --tt_freeze_vision_tower, ... true, or ... false.",
    )
    parser.add_argument(
        "--tt_freeze_multi_modal_projector",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Freeze multi-modal projector parameters (requires_grad=False). "
        "Accepts --tt_freeze_multi_modal_projector, ... true, or ... false.",
    )
    parser.add_argument(
        "--tt_freeze_language_model",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Freeze language model parameters (requires_grad=False). "
        "Accepts --tt_freeze_language_model, ... true, or ... false.",
    )
    parser.add_argument(
        "--tt_cap_lambda",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="CAP/DPARALLEL confidence loss weight (entropy on correctly-"
             "predicted tokens). 0 disables (default). Recommended: 2.0.",
    )
    parser.add_argument(
        "--tt_bd_mask_low",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="LLaDA2 BD per-sample noise ratio (sigma) lower bound (default 0.3). "
             "Only affects the BD dataset path.",
    )
    parser.add_argument(
        "--tt_bd_mask_high",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="LLaDA2 BD per-sample noise ratio (sigma) upper bound (default 0.8). "
             "Only affects the BD dataset path.",
    )
    parser.add_argument(
        "--tt_profiling",
        type=string_to_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Enable NPU/CUDA profiling for the second step. "
        "Outputs to {output_dir}/profiling/. "
        "Accepts --tt_profiling, ... true, or ... false.",
    )
    parser.add_argument(
        "--tt_memory_snapshot_steps",
        type=int,
        default=None,
        metavar="STEPS",
        help="Number of optimizer steps to capture from training step 0. For "
        "example, 3 captures the first 3 optimizer steps. Outputs to "
        "{output_dir}/memory_snapshot/.",
    )
    parser.add_argument(
        "--tt_log_level",
        type=parse_log_level,
        default=LogLevel.INFO,
        metavar="LEVEL",
        help="TorchTurbo log level: DEBUG, INFO, WARNING, or ERROR.",
    )

    return parser


def parse_torchturbo_args(argv: list[str]) -> tuple[TorchTurboArgs, list[str]]:
    """Parse ``--tt_*`` arguments from *argv*, returning parsed args and
    the remaining (unconsumed) arguments.

    This uses ``parse_known_args`` so that non-torchturbo arguments are
    left untouched and can be forwarded to the training framework.

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
        # remaining can be passed to LlamaFactory's run_exp()
    """
    parser = _build_parser()
    parsed, remaining = parser.parse_known_args(argv)

    args = TorchTurboArgs(
        backend=parsed.tt_backend,
        mixed_precision=parsed.tt_mixed_precision,
        op_policy=parsed.tt_op_policy,
        sp_size=parsed.tt_sp_size,
        ep_size=parsed.tt_ep_size,
        gradient_checkpointing=parsed.tt_gradient_checkpointing,
        ac_use_reentrant=parsed.tt_ac_use_reentrant,
        cpu_offload=parsed.tt_cpu_offload,
        mega_moe=parsed.tt_mega_moe,
        enable_mtp=parsed.tt_enable_mtp,
        mtp_loss_scaling_factor=parsed.tt_mtp_loss_scaling_factor,
        freeze_vision_tower=parsed.tt_freeze_vision_tower,
        freeze_multi_modal_projector=parsed.tt_freeze_multi_modal_projector,
        freeze_language_model=parsed.tt_freeze_language_model,
        profiling=parsed.tt_profiling,
        memory_snapshot_steps=parsed.tt_memory_snapshot_steps,
        cap_lambda=parsed.tt_cap_lambda,
        bd_mask_low=parsed.tt_bd_mask_low,
        bd_mask_high=parsed.tt_bd_mask_high,
        prefetch_strategy=parsed.tt_prefetch_strategy,
        prefetch_direction=parsed.tt_prefetch_direction,
        forward_prefetch_depth=parsed.tt_forward_prefetch_depth,
        backward_prefetch_depth=parsed.tt_backward_prefetch_depth,
        log_level=parsed.tt_log_level,
    )

    if any(
        v is not None
        for v in [
            args.backend,
            args.mixed_precision,
            args.op_policy,
            args.sp_size,
            args.ep_size,
            args.enable_mtp,
            args.mtp_loss_scaling_factor,
        ]
    ):
        logger.info_rank0("torchturbo CLI args: %s", args)
    return args, remaining
