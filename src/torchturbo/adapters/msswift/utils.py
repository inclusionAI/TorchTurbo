"""Translate swift args into a turbo ``EngineConfig`` and PEFT config.

Pure helpers, no swift import — usable with swift absent (the args object is
duck-typed).  Driven by the parsed :class:`TurboSftArguments` (swift-native
fields) plus the shared :class:`TorchTurboArgs` (``--tt_*`` flags, stashed by
the launcher via :func:`set_torchturbo_args`).
"""

from __future__ import annotations

from typing import Any

from ...engine.config import EngineConfig, build_from_args
from ...engine.args import TorchTurboArgs, get_torchturbo_args
from ...engine.utils import raise_if_evaluation_enabled
from ...logging import get_logger

logger = get_logger(__name__)


def build_config_from_swift_args(args: Any) -> EngineConfig:
    """Build an :class:`EngineConfig` from a parsed ``TurboSftArguments`` + stashed ``TorchTurboArgs``.

    swift-native fields (bf16/fp16, model path, output_dir) are folded onto the
    shared :class:`TorchTurboArgs` as adapter-seeded values; all
    ``--tt_*``-driven slices (backend, sp/ep, AC, MTP, freeze, FSDP prefetch)
    are read back from it by :func:`build_from_args`. Env-var overrides are
    merged inside ``build_from_args``.
    """
    # Swift's arguments keep the user's generation option; DPOConfig drops it.
    raise_if_evaluation_enabled(
        args.training_args, predict_with_generate=args.predict_with_generate
    )
    _raise_if_unsupported(args)

    tt = get_torchturbo_args()
    if tt is None:
        logger.warning_rank0(
            "No TorchTurboArgs stashed (set_torchturbo_args not called); using "
            "all-turbo defaults. The launcher (run_swift.py) must parse --tt_* "
            "before sft_main()."
        )
        tt = TorchTurboArgs()

    # swift-native mixed-precision seed (bf16/fp16). Only seed tt when the user
    # did not pass --tt_mixed_precision; explicit tt value wins, env overrides
    # still apply inside build_from_args.
    if tt.mixed_precision is None:
        mixed_precision_hint = "fp32"
        if getattr(args, "bf16", False):
            mixed_precision_hint = "bf16"
        elif getattr(args, "fp16", False):
            mixed_precision_hint = "fp16"
        tt.mixed_precision = mixed_precision_hint

    # DPO uses the engine's router loss and metrics. Explicit Turbo settings
    # still override this Swift default when build_from_args merges the config.
    if vars(args).get("rlhf_type") == "dpo" and tt.moe_aux_loss_coeff == 0.0:
        tt.moe_aux_loss_coeff = args.router_aux_loss_coef

    return build_from_args(tt)


def build_lora_config_from_swift_args(args: Any) -> Any | None:
    """Build a ``peft.LoraConfig`` from swift's LoRA fields.

    The engine applies LoRA during ``prepare_model`` (before FSDP2 sharding),
    so swift's own ``TunerMixin.prepare_model`` must be bypassed (the pipeline
    does that).  Returns ``None`` if PEFT is not installed.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        logger.warning_rank0("peft not installed — cannot build LoraConfig.")
        return None

    rank = int(getattr(args, "lora_rank", 8) or 8)
    alpha = int(getattr(args, "lora_alpha", None) or rank * 2)
    target = getattr(args, "target_modules", None) or "all-linear"

    if isinstance(target, str):
        target_modules = "all-linear" if target in ("all", "all-linear") else [
            m.strip() for m in target.split(",") if m.strip()
        ]
    else:
        names = {str(m).strip().lower() for m in target}
        target_modules = "all-linear" if names & {"all", "all-linear"} else list(target)

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        bias="none",
        init_lora_weights=True,
        use_rslora=bool(getattr(args, "use_rslora", False)),
        use_dora=bool(getattr(args, "use_dora", False)),
        task_type=TaskType.CAUSAL_LM,
    )


def _raise_if_unsupported(args: Any) -> None:
    tuner_type = getattr(args, "tuner_type", "full")
    rlhf_type = getattr(args, "rlhf_type", None)
    unsupported: list[str] = []

    if tuner_type not in {"full", "lora"}:
        unsupported.append(f"tuner_type={tuner_type!r}")
    if rlhf_type == "dpo":
        if tuner_type != "full":
            unsupported.append("DPO with tuner_type!=full")
        tt = get_torchturbo_args()
        if tt is not None and getattr(tt, "enable_mtp", False):
            unsupported.append("DPO with MTP")

    if unsupported:
        raise NotImplementedError(
            "torchturbo ms-swift adapter does not support " + ", ".join(unsupported) + " yet."
        )
