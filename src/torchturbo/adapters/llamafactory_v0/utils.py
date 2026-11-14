"""Utility functions for integrating LlamaFactory v0 with torchturbo."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...engine import CPUOffloadConfig, CPUOffloadMode, EngineConfig
from ...engine.args import TorchTurboArgs, get_arg_specs, string_to_bool
from ...engine.utils import raise_if_evaluation_enabled
from ...logging import get_logger


logger = get_logger(__name__)


def build_config_from_lf_args(
    training_args: Any = None,
    finetuning_args: Any = None,
    model_args: Any = None,
    torchturbo_args: TorchTurboArgs | None = None,
    raw_args: Any = None,
) -> tuple[EngineConfig, Any]:
    """Build an ``EngineConfig`` from LlamaFactory args.

    Priority (lowest → highest): defaults → LF args mapping → ``torchturbo_args``
    → env vars (merged by :func:`build_from_args`).

    Parameters
    ----------
    training_args:
        LlamaFactory's ``TrainingArguments``.
    finetuning_args:
        LlamaFactory's ``FinetuningArguments``.
    model_args:
        LlamaFactory's ``ModelArguments``.
    torchturbo_args:
        Parsed ``--tt_*`` CLI arguments.  Applied directly during
        config construction so they override LF-derived values.
    raw_args:
        Raw arguments that will be passed back into LlamaFactory's native
        ``run_exp`` after TorchTurbo-owned fields are consumed.

    Returns
    -------
    The configured ``EngineConfig`` and raw args with LlamaFactory gradient
    checkpointing disabled.
    """
    if training_args is not None:
        raise_if_evaluation_enabled(
            training_args, predict_with_generate=training_args.predict_with_generate
        )
    _raise_if_unsupported_lf_args(finetuning_args, model_args, torchturbo_args)

    config, raw_args = _build_config_from_args(
        training_args, finetuning_args, model_args, torchturbo_args, raw_args
    )
    # Note: env vars are already merged inside build_from_args.

    logger.info_rank0(
        "Built EngineConfig from LlamaFactory args: backend=%s, mp=%s",
        config.backend.value,
        config.mixed_precision.value,
    )
    return config, raw_args


def _build_config_from_args(
    training_args: Any,
    finetuning_args: Any,
    model_args: Any,
    torchturbo_args: TorchTurboArgs | None = None,
    raw_args: Any = None,
) -> tuple[EngineConfig, Any]:
    """Map LlamaFactory args and torchturbo CLI args to EngineConfig.

    Adapter-specific bits (LF mixed-precision seed, FSDP CPU-offload
    reconciliation with LF ``fsdp`` options, gradient-checkpointing takeover
    with LF-arg side effects, checkpoint paths) are resolved here and folded
    onto the :class:`TorchTurboArgs` so the shared
    :func:`build_from_args` can reconstruct the EngineConfig from that
    single argument. All ``--tt_*``-driven slices live there.
    """
    from ...engine.config import build_from_args

    tt = torchturbo_args or TorchTurboArgs()  # shorthand

    # --- Mixed precision seed (bf16/fp16/fp8 from LF training args) ---
    # Only seed tt when the user did not pass --tt_mixed_precision; explicit tt
    # value wins; env overrides are merged later inside build_from_args.
    if tt.mixed_precision is None:
        mixed_precision_hint = "fp32"
        if training_args is not None:
            if getattr(training_args, "bf16", False):
                mixed_precision_hint = "bf16"
            elif getattr(training_args, "fp16", False):
                mixed_precision_hint = "fp16"
            if getattr(training_args, "fp8", False):
                mixed_precision_hint = "fp8"
        tt.mixed_precision = mixed_precision_hint

    # --- FSDP CPU offload: reconcile LF fsdp options with the file/adapter config ---
    # Single source of truth, shared with the early process-group init in the
    # adapter patch so the two never diverge. resolve_cpu_offload raises on an
    # explicit non-full mode that contradicts an LF-requested offload.
    fsdp_options = (
        getattr(training_args, "fsdp", []) if training_args is not None else []
    )
    tt.cpu_offload = resolve_cpu_offload(fsdp_options, tt, strict=True)

    # --- Activation checkpointing takeover (LF-owned, with side effects) ---
    # HF/LF gradient checkpointing wraps model forwards after model loading.
    # TorchTurbo owns FSDP2 wrapping, so disable LF's implementation here. When
    # LF requested gradient checkpointing, the engine enables plan-defined AC
    # targets; otherwise those plan targets are ignored with a warning. If the
    # user explicitly set TorchTurbo's value, require it to agree with LF.
    ac_enabled, raw_args = _take_over_lf_gradient_checkpointing(
        training_args,
        model_args,
        raw_args,
        tt,
    )
    tt.gradient_checkpointing = ac_enabled

    config = build_from_args(tt)
    return config, raw_args


def _take_over_lf_gradient_checkpointing(
    training_args: Any,
    model_args: Any,
    raw_args: Any,
    torchturbo_args: TorchTurboArgs | None = None,
) -> tuple[bool, Any]:
    # LF/HF applies gradient checkpointing after model construction, which is
    # too late for TorchTurbo's FSDP2 path: activation checkpointing must be
    # installed before fully_shard so recompute runs inside each FSDP unit.
    # Treat the LF flags as a request for engine-owned activation checkpointing,
    # then force LF's own gradient checkpointing off before re-entering run_exp.
    adapter_gc = bool(getattr(training_args, "gradient_checkpointing", False))
    torchturbo_gradient_checkpointing = _explicit_torchturbo_bool_arg(
        "gradient_checkpointing",
        torchturbo_args,
        config_path="activation_checkpointing.enabled",
    )

    if (
        torchturbo_gradient_checkpointing is not None
        and torchturbo_gradient_checkpointing != adapter_gc
    ):
        raise ValueError(
            "TorchTurbo --tt_gradient_checkpointing must match the adapter "
            "gradient checkpointing setting: "
            f"adapter={adapter_gc}, torchturbo={torchturbo_gradient_checkpointing}."
        )

    if training_args is not None and hasattr(training_args, "gradient_checkpointing"):
        training_args.gradient_checkpointing = False
    if model_args is not None:
        if hasattr(model_args, "disable_gradient_checkpointing"):
            model_args.disable_gradient_checkpointing = True

    if isinstance(raw_args, dict):
        raw_args = dict(raw_args)
        raw_args["gradient_checkpointing"] = False
        raw_args["disable_gradient_checkpointing"] = True
        return adapter_gc, raw_args

    if not isinstance(raw_args, list):
        return adapter_gc, raw_args

    remove_flags = {
        "--gradient_checkpointing",
        "--disable_gradient_checkpointing",
    }
    result: list[Any] = []
    i = 0
    while i < len(raw_args):
        item = raw_args[i]
        item_str = str(item)
        if item_str.startswith("--") and "=" in item_str:
            flag = item_str.split("=", 1)[0]
            if flag in remove_flags:
                i += 1
                continue
        if item_str in remove_flags:
            i += 1
            if i < len(raw_args) and not str(raw_args[i]).startswith("--"):
                i += 1
            continue
        result.append(item)
        i += 1
    result.append("--disable_gradient_checkpointing")
    return adapter_gc, result


def _explicit_torchturbo_bool_arg(
    name: str,
    torchturbo_args: TorchTurboArgs | None,
    *,
    config_path: str | None = None,
) -> bool | None:
    flag = f"--tt_{name}"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        item = str(args[i])
        if item.startswith(f"{flag}="):
            return string_to_bool(item.split("=", 1)[1])
        if item == flag:
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                return string_to_bool(str(args[i + 1]))
            return True
        i += 1

    if config_path is not None:
        config_file = getattr(torchturbo_args, "config_path", None) if torchturbo_args else None
        if config_file:
            from omegaconf import OmegaConf
            file_conf = OmegaConf.load(config_file)
            file_value = OmegaConf.select(file_conf.get("engine_config", file_conf), config_path)
            if file_value is not None:
                return string_to_bool(file_value)
        for arg_spec in get_arg_specs():
            if arg_spec.env_name is None or arg_spec.config_path != config_path:
                continue
            env_value = os.environ.get(arg_spec.env_name)
            if env_value is not None:
                return string_to_bool(env_value)
            break
    return None


def fsdp_options_from_raw_args(raw_args: Any) -> Any:
    """Peek the LF ``--fsdp`` value from raw (pre-parse) args (dict or argv list).

    Lets the process group be initialized (with a CPU gloo backend for offload)
    before ``get_train_args`` parses the arguments.
    """
    if isinstance(raw_args, dict):
        return raw_args.get("fsdp")
    if isinstance(raw_args, (list, tuple)):
        items = [str(a) for a in raw_args]
        for i, item in enumerate(items):
            if item.startswith("--fsdp="):
                return item.split("=", 1)[1]
            if item == "--fsdp" and i + 1 < len(items):
                values = []
                j = i + 1
                while j < len(items) and not items[j].startswith("--"):
                    values.append(items[j])
                    j += 1
                return values
    return None


def resolve_cpu_offload(
    fsdp_options: Any,
    torchturbo_args: TorchTurboArgs,
    *,
    strict: bool = True,
) -> CPUOffloadConfig:
    """Resolve CPU offload from LF ``fsdp`` options plus the file/adapter config.

    Single source of truth shared by ``_build_config_from_args`` (which feeds
    ``cpu_offload`` into the shared ``build_from_args``) and by early
    process-group initialization, so the gloo-backend decision and the engine
    config can never diverge. ``fsdp_options`` may be the parsed list
    (``training_args.fsdp``) or the raw string from
    :func:`fsdp_options_from_raw_args`.

    With ``strict=True`` (config path) an explicit structured config that disables
    parameter offload while LF requests it raises. Early PG init uses
    ``strict=False`` -- it only needs the policy and must not move where that
    validation surfaces.
    """
    if isinstance(fsdp_options, str):
        fsdp_tokens = fsdp_options.replace(",", " ").split()
    elif isinstance(fsdp_options, (list, tuple)):
        fsdp_tokens = []
        for item in fsdp_options:
            if isinstance(item, str):
                fsdp_tokens.extend(item.replace(",", " ").split())
            else:
                fsdp_tokens.append(str(item))
    else:
        fsdp_tokens = []

    adapter_offload = "offload" in fsdp_tokens
    if strict and adapter_offload and torchturbo_args.cpu_offload_enabled is False:
        raise ValueError(
            "TorchTurbo cpu_offload must use mode='full' when the adapter "
            "enables FSDP parameter offload"
        )
    cpu_offload = torchturbo_args.cpu_offload
    if cpu_offload.enabled is None:
        cpu_offload = CPUOffloadConfig(enabled=adapter_offload)
    if strict and adapter_offload and (
        not cpu_offload.enabled or cpu_offload.mode is not CPUOffloadMode.FULL
    ):
        raise ValueError(
            "TorchTurbo cpu_offload must use mode='full' when the adapter "
            "enables FSDP parameter offload"
        )
    return cpu_offload


def _raise_if_unsupported_lf_args(
    finetuning_args: Any,
    model_args: Any | None = None,
    torchturbo_args: TorchTurboArgs | None = None,
) -> None:
    stage = getattr(finetuning_args, "stage", None)
    finetuning_type = getattr(finetuning_args, "finetuning_type", None)
    unsupported: list[str] = []

    if finetuning_type == "oft":
        unsupported.append("finetuning_type=oft")

    if stage == "dpo":
        if finetuning_type != "full":
            unsupported.append(f"DPO with finetuning_type={finetuning_type!r}")
        if not getattr(finetuning_args, "use_ref_model", False):
            unsupported.append("DPO with use_ref_model=False")
        if torchturbo_args is not None and getattr(torchturbo_args, "enable_mtp", False):
            unsupported.append("DPO with MTP")
        if getattr(model_args, "use_kt", False):
            unsupported.append("DPO with use_kt=True")
        if getattr(finetuning_args, "ref_model_adapters", None):
            unsupported.append("DPO with ref_model_adapters")
        if getattr(finetuning_args, "ref_model_quantization_bit", None):
            unsupported.append("DPO with ref_model_quantization_bit")

    if unsupported:
        raise NotImplementedError(
            "TorchTurbo LlamaFactory adapter does not support "
            f"{', '.join(unsupported)} yet."
        )


def _get_unsupported_lora_options(
    finetuning_args: Any,
    model_args: Any | None = None,
) -> list[str]:
    if getattr(finetuning_args, "finetuning_type", None) != "lora":
        return []

    checks = {
        "adapter_name_or_path": getattr(model_args, "adapter_name_or_path", None)
        is not None,
        "additional_target": getattr(finetuning_args, "additional_target", None)
        is not None,
        "loraplus_lr_ratio": getattr(finetuning_args, "loraplus_lr_ratio", None)
        is not None,
        "pissa_init": bool(getattr(finetuning_args, "pissa_init", False)),
        "pissa_convert": bool(getattr(finetuning_args, "pissa_convert", False)),
        "create_new_adapter": bool(getattr(finetuning_args, "create_new_adapter", False)),
        "use_llama_pro": bool(getattr(finetuning_args, "use_llama_pro", False)),
    }
    return [name for name, enabled in checks.items() if enabled]


def build_lora_config_from_lf_args(
    finetuning_args: Any,
    model_args: Any | None = None,
) -> Any | None:
    """Build a ``peft.LoraConfig`` from LlamaFactory's ``FinetuningArguments``.

    Maps supported LlamaFactory LoRA fields (``lora_rank``, ``lora_alpha``,
    ``lora_target``, ``lora_dropout``, ``use_rslora``, ``use_dora``) to a
    :class:`peft.LoraConfig` and registers the
    :class:`~torchturbo.models.modules.moe.experts.GroupedMLP` custom module
    handler so that MoE expert layers are included in LoRA.

    Call :func:`strip_lora_args` before re-entering LlamaFactory so that
    LlamaFactory does not apply LoRA itself.  TorchTurbo passes this config
    into ``engine.prepare_model`` and owns the LoRA lifecycle from there.

    Returns ``None`` if PEFT is not installed.
    """
    unsupported = _get_unsupported_lora_options(finetuning_args, model_args)
    if unsupported:
        raise NotImplementedError(
            "TorchTurbo LlamaFactory adapter does not support "
            f"LoRA options {', '.join(unsupported)} yet."
        )

    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        logger.warning_rank0("peft not installed — cannot build LoraConfig from LF args.")
        return None

    # Map LlamaFactory finetuning args to LoraConfig fields.
    # LlamaFactory uses: finetuning_type="lora", lora_rank, lora_alpha,
    # lora_target, lora_dropout, etc.
    lora_rank = getattr(finetuning_args, "lora_rank", 8)
    lora_alpha = getattr(finetuning_args, "lora_alpha", lora_rank * 2)
    lora_target = getattr(finetuning_args, "lora_target", "all-linear")
    lora_dropout = getattr(finetuning_args, "lora_dropout", 0.0)

    # Optional LoRA init / scaling flags from LlamaFactory.
    use_rslora = getattr(finetuning_args, "use_rslora", False)
    use_dora = getattr(finetuning_args, "use_dora", False)

    # Handle target_modules: LlamaFactory uses comma-separated string or list.
    # "all" in LlamaFactory means "all linear modules" — same as PEFT's "all-linear".
    if isinstance(lora_target, str):
        if lora_target in ("all", "all-linear"):
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",") if m.strip()]
    elif isinstance(lora_target, (list, set)):
        # Handle "all" / "all-linear" appearing as list elements — they are
        # supersets of any specific target, so just use the string form.
        normalized = {str(m).strip().lower() for m in lora_target}
        if normalized & {"all", "all-linear"}:
            target_modules = "all-linear"
        else:
            target_modules = list(lora_target)
    else:
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        init_lora_weights=True,
        use_rslora=use_rslora,
        use_dora=use_dora,
        task_type=TaskType.CAUSAL_LM,
    )

    return lora_config


# LlamaFactory LoRA CLI flags that must be stripped from ``remaining_argv``
# when torchturbo manages LoRA.  HF's ``HfArgumentParser`` accepts both
# ``--flag value`` and ``--flag=value`` forms; boolean flags (``--use_rslora``)
# are consumed without a value.
_LORA_CLI_FLAGS: list[str] = [
    # LoRA hyper-parameters (each takes a value)
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "lora_dropout",
    "additional_target",
    "loraplus_lr_ratio",
    "loraplus_lr_embedding",
    "pissa_iter",
    # Boolean flags (no value consumed)
    "use_rslora",
    "use_dora",
    "pissa_init",
    "pissa_convert",
    "create_new_adapter",
]


def strip_lora_args(args: Any) -> Any:
    """Strip LoRA-related flags from LlamaFactory CLI args.

    When torchturbo manages the LoRA lifecycle, LlamaFactory must not
    apply LoRA itself.  This function removes all ``--lora_*``,
    ``--use_rslora``, ``--use_dora``, ``--pissa_*``, etc. flags from the
    arg list and changes ``--finetuning_type lora`` to
    ``--finetuning_type full``.

    Handles ``list[str]`` (CLI argv), ``dict`` (pre-parsed), and ``None``.
    Returns a **new** object; the input is never mutated.
    """
    if args is None:
        return None

    # --- dict form ---
    if isinstance(args, dict):
        stripped = dict(args)
        if stripped.get("finetuning_type") == "lora":
            stripped["finetuning_type"] = "full"
        for flag in _LORA_CLI_FLAGS:
            stripped.pop(flag, None)
        return stripped

    # --- list[str] form (CLI argv) ---
    if isinstance(args, (list, tuple)):
        bool_flags = {
            "use_rslora",
            "use_dora",
            "pissa_init",
            "pissa_convert",
            "create_new_adapter",
        }
        val_flags = set(_LORA_CLI_FLAGS) - bool_flags
        out: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]

            # --finetuning_type=lora  →  --finetuning_type=full
            if tok.startswith("--finetuning_type=lora"):
                out.append("--finetuning_type=full")
                i += 1
                continue

            # --finetuning_type lora  (separate tokens)
            if tok == "--finetuning_type" and i + 1 < len(args) and args[i + 1] == "lora":
                out.append("--finetuning_type")
                out.append("full")
                i += 2
                continue

            # --flag=value  form (single token with =)
            if tok.startswith("--") and "=" in tok:
                key_eq = tok.split("=", 1)[0][2:]  # strip leading --
                if key_eq in val_flags or key_eq in bool_flags:
                    i += 1  # skip --flag=value
                    continue

            # --flag value  (separate tokens) or --bool-flag (no value)
            if tok.startswith("--"):
                key = tok[2:]  # strip leading --
                if key in val_flags and i + 1 < len(args):
                    i += 2  # skip --flag and value
                    continue
                if key in bool_flags:
                    i += 1  # skip --bool-flag
                    continue

            out.append(tok)
            i += 1

        logger.info_rank0(
            "torchturbo: stripped LoRA CLI flags and set finetuning_type=full "
            "(LoRA managed by torchturbo engine)"
        )
        return out

    # Unknown type — return as-is
    return args


def compute_sft_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    state: Any = None,
) -> torch.Tensor:
    """Standard SFT (supervised fine-tuning) loss.

    Computes cross-entropy on the shifted labels, compatible with
    HuggingFace-style models.
    """
    device = next(model.parameters()).device
    model_inputs = {
        k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"
    }
    labels = batch["labels"].to(device, non_blocking=True)

    outputs = model(**model_inputs)
    logits = outputs.logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ---------------------------------------------------------------------------
# Vision-boundary-aware packing for multimodal pre-training
# ---------------------------------------------------------------------------

_IGNORE_INDEX = -100


def pack_with_vision_boundaries(
    tokenized_examples: dict[str, list[list[int]]],
    image_list: list[list[str]],
    cutoff_len: int,
    vision_start_id: int,
    vision_end_id: int,
) -> tuple[dict[str, list[list[int]]], list[list[str]]]:
    """Pack tokenized sequences into fixed-length blocks, keeping vision blocks intact.

    A vision block is the span [vision_start_id ... vision_end_id].  If the
    last block in a chunk window straddles the boundary, the chunk is shortened
    to exclude it — the block appears at the start of the next chunk instead.
    """
    from itertools import chain

    concat = {k: list(chain(*v)) for k, v in tokenized_examples.items()}
    ids = concat["input_ids"]
    total = len(ids)
    if total == 0:
        return {}, []

    flat_images = list(chain(*image_list))
    starts = [i for i, t in enumerate(ids) if t == vision_start_id]
    ends = [i for i, t in enumerate(ids) if t == vision_end_id]

    assert len(starts) == len(ends), (
        f"Vision boundary mismatch: {len(starts)} starts vs {len(ends)} ends"
    )
    assert len(starts) == len(flat_images), (
        f"Vision block count ({len(starts)}) != image count ({len(flat_images)})"
    )

    start_to_end = dict(zip(starts, ends))
    if starts:
        max_block = max(e - s + 1 for s, e in start_to_end.items())
        assert max_block <= cutoff_len, (
            f"Vision block length {max_block} exceeds cutoff_len {cutoff_len}"
        )

    result = {k: [] for k in concat}
    packed_images: list[list[str]] = []
    pos = 0
    img_cursor = 0

    while pos < total:
        end = min(pos + cutoff_len, total)

        window_starts = [s for s in starts if pos <= s < end]
        if window_starts:
            last_s = window_starts[-1]
            if start_to_end[last_s] >= end:
                end = last_s

        n_imgs = ids[pos:end].count(vision_end_id)
        packed_images.append(flat_images[img_cursor:img_cursor + n_imgs])
        img_cursor += n_imgs

        for k in concat:
            result[k].append(concat[k][pos:end])
        pos = end

    assert img_cursor == len(flat_images)
    return result, packed_images


def preprocess_multimodal_pretrain(
    processor_self: Any,
    examples: dict[str, list[Any]],
    img_examples: list,
) -> dict[str, Any]:
    """Multimodal pretrain preprocessing with vision-boundary-aware packing.

    Called by PretrainProcessorPatch when the batch contains images.
    """
    import re

    tokenizer = processor_self.tokenizer
    data_args = processor_self.data_args
    template = processor_self.template
    mm_processor = getattr(processor_self, "processor", None)
    cutoff_len = data_args.cutoff_len

    eos_token = (
        "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    )
    text_examples = [
        msg[0]["content"] + eos_token for msg in examples["_prompt"]
    ]

    # Validate and normalize image placeholders
    IMAGE_PLACEHOLDER = "<image>"
    texts, images = [], []
    for img_ex, txt in zip(img_examples, text_examples):
        try:
            paths = img_ex[0].split(",")
            if len(paths) != txt.count("<image"):
                continue
        except Exception:
            continue
        images.append(paths)
        texts.append(re.sub(r"<image:\d+>", IMAGE_PLACEHOLDER, txt))

    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # Derive image token counts from the same media path used by the collator.
    vision_bos = "<|vision_start|>"
    vision_eos = "<|vision_end|>"
    image_token = getattr(template.mm_plugin, "image_token", None) or "<|image_pad|>"

    plugin = template.mm_plugin

    expanded = []
    for txt, img_paths in zip(texts, images):
        result_txt = txt
        for path in img_paths:
            media_inputs = plugin.get_mm_inputs(
                images=[path.strip()],
                videos=[],
                audios=[],
                imglens=[1],
                vidlens=[0],
                audlens=[0],
                batch_ids=[[]],
                processor=mm_processor,
            )
            grid_thw = media_inputs["image_grid_thw"][0]
            merge_length = mm_processor.image_processor.merge_size ** 2
            seqlen = int(grid_thw.prod().item()) // merge_length
            replacement = f"{vision_bos}{image_token * seqlen}{vision_eos}"
            result_txt = result_txt.replace(IMAGE_PLACEHOLDER, replacement, 1)
        expanded.append(result_txt)

    if not expanded:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # Tokenize and pack with boundary protection
    tokenized = tokenizer(expanded, add_special_tokens=False)
    vision_start_id = tokenizer.convert_tokens_to_ids(vision_bos)
    vision_end_id = tokenizer.convert_tokens_to_ids(vision_eos)

    packed, packed_images = pack_with_vision_boundaries(
        tokenized, images, cutoff_len, vision_start_id, vision_end_id,
    )
    if not packed.get("input_ids"):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    if getattr(tokenizer, "add_bos_token", False):
        for seq in packed["input_ids"]:
            seq[0] = tokenizer.bos_token_id

    # Pad and build labels
    batch = tokenizer.pad(
        packed, return_tensors="pt", padding="max_length", max_length=cutoff_len
    )
    labels = batch["input_ids"].clone()

    ignore_ids = {vision_start_id, vision_end_id}
    if tokenizer.pad_token_id is not None:
        ignore_ids.add(tokenizer.pad_token_id)
    image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is not None:
        ignore_ids.add(image_token_id)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for tid in ignore_ids:
        mask |= (labels == tid)
    labels[mask] = _IGNORE_INDEX

    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
        "images": packed_images,
        "videos": [None] * len(batch["input_ids"]),
        "audios": [None] * len(batch["input_ids"]),
    }
