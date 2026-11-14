"""Monkey-patches for LlamaFactory v0 — version-aware BasePatch classes.

Each patch declares ``supported_versions`` — incompatible patches are
automatically skipped by the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from ...logging import get_logger
from ...registry import BasePatch

if TYPE_CHECKING:
    import torch

    from ...engine import TorchTurboEngine

logger = get_logger(__name__)

# Version range for LlamaFactory v0
_LF_V0 = "==0.9.5"

# ---------------------------------------------------------------------------
# DictFieldPatch — fix dict-typed CLI args in HfArgumentParser
# ---------------------------------------------------------------------------

# Field names whose type resolves to dict in HfArgumentParser and need
# the str→dict fix.  Append more names here as needed.
_DICT_FIELD_NAMES = frozenset({
    "lr_scheduler_kwargs",
})


class DictFieldPatch(BasePatch):
    """Fix dict-typed CLI args in ``HfArgumentParser``.

    ``HfArgumentParser._parse_dataclass_field`` resolves ``Union[dict, str]``
    to ``type=dict``, so argparse calls ``dict(cli_string)`` which cannot
    parse JSON and raises::

        error: invalid dict value: '{min_lr:1e-7}'

    Fix: for fields listed in ``_DICT_FIELD_NAMES``, temporarily set
    ``field.type = str`` so argparse stores the raw string.  The existing
    ``TrainingArguments.__post_init__`` pipeline
    (``_VALID_DICT_FIELDS`` → ``json.loads`` → ``_convert_str_dict``)
    then converts it to a proper dict.

    To cover more dict-typed fields, add their names to ``_DICT_FIELD_NAMES``.

    CLI usage::

        --lr_scheduler_kwargs '{"min_lr":1e-7}'
    """

    name = "dict_field"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original = None
        try:
            from transformers.hf_argparser import HfArgumentParser
        except ImportError:
            return

        original = HfArgumentParser._parse_dataclass_field
        self._original = original
        field_names = _DICT_FIELD_NAMES

        @staticmethod
        def _patched(parser, field):
            if field.name in field_names:
                saved = field.type
                field.type = str
                try:
                    original(parser, field)
                finally:
                    field.type = saved
            else:
                original(parser, field)

        HfArgumentParser._parse_dataclass_field = _patched
        logger.info_rank0(
            "Patched HfArgumentParser for dict-typed fields: %s",
            sorted(field_names),
        )

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        pass  # already applied in __init__

    def revert(self) -> None:
        if self._original is None:
            return
        try:
            from transformers.hf_argparser import HfArgumentParser
        except ImportError:
            return
        HfArgumentParser._parse_dataclass_field = self._original
        self._original = None
        logger.info_rank0("Reverted HfArgumentParser dict-field patch")


# ---------------------------------------------------------------------------
# RunExpPatch — intercepts run_exp to build engine from LF args
# ---------------------------------------------------------------------------


class RunExpPatch(BasePatch):
    """Intercept ``run_exp`` to build the engine and apply inner patches."""

    name = "run_exp"
    supported_versions = _LF_V0

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._original = None
        self._module = None
        self._launcher_module = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        try:
            from llamafactory.train import tuner as tuner_module
        except ImportError:
            raise ImportError("tuner in llamafactory.train not found.")

        self._module = tuner_module
        self._original = tuner_module.run_exp
        adapter = self.adapter

        def _patched_run_exp(
            args: dict[str, Any] | None = None,
            callbacks: list[Any] | None = None,
        ):
            from llamafactory.hparams import get_train_args, read_args
            from .utils import (
                build_config_from_lf_args,
                build_lora_config_from_lf_args,
                strip_lora_args,
            )
            from ...engine import TorchTurboEngine
            from ...engine.args import get_torchturbo_args
            from ...logging import logging_configure

            raw_args = read_args(args)

            # Retrieve CLI-parsed torchturbo args (stashed by the entry point).
            tt_args = get_torchturbo_args()
            logging_configure(tt_args.log_level)
            logger.info_rank0("torchturbo CLI args: %s", tt_args)

            model_args, data_args, training_args, finetuning_args, _generating_args = (
                get_train_args(raw_args)
            )
            stage = getattr(finetuning_args, "stage", None)

            if stage not in {"sft", "pt", "dpo"}:
                logger.warning_rank0(
                    "torchturbo: LlamaFactory stage %r is not enabled for "
                    "TorchTurbo yet; running native LlamaFactory.",
                    stage,
                )
                return self._original(args, callbacks)

            if stage == "sft" and not data_args.ignore_pad_token_for_loss:
                raise ValueError(
                    "TorchTurbo packed SFT requires ignore_pad_token_for_loss=True; "
                    "padding tokens are removed before loss computation."
                )

            peft_config = None
            if (
                stage != "dpo"
                and getattr(finetuning_args, "finetuning_type", None) == "lora"
            ):
                peft_config = build_lora_config_from_lf_args(
                    finetuning_args,
                    model_args=model_args,
                )
                if peft_config is None:
                    raise RuntimeError(
                        "TorchTurbo LlamaFactory LoRA requires PEFT to build "
                        "the adapter config before engine initialization."
                    )
                logger.info_rank0(
                    "torchturbo: LoRA mode detected — r=%d, alpha=%d, "
                    "target_modules=%s, dropout=%.2f",
                    peft_config.r,
                    peft_config.lora_alpha,
                    peft_config.target_modules,
                    peft_config.lora_dropout,
                )
            else:
                logger.info_rank0(
                    "torchturbo: Full-parameter training mode (finetuning_type=%s)",
                    getattr(finetuning_args, "finetuning_type", "full"),
                )

            config, raw_args = build_config_from_lf_args(
                training_args=training_args,
                finetuning_args=finetuning_args,
                model_args=model_args,
                torchturbo_args=tt_args,
                raw_args=raw_args,
            )

            engine = TorchTurboEngine(config)

            adapter.engine = engine
            adapter.stage = stage
            adapter.peft_config = peft_config

            # Stash template name for downstream use (e.g. collator upgrade)
            engine._lf_template_name = getattr(data_args, "template", None)
            # Stash model/data args for downstream use (e.g. BDDataset in get_train_dataloader)
            engine._lf_model_args = model_args
            engine._lf_data_args = data_args

            # Keep the current run's LoRA config on the adapter context.
            # LoadModelPatch reads it when LF calls from_pretrained.
            if peft_config is not None:
                logger.info_rank0(
                    "torchturbo: LoRA config set for current LF run "
                    "(r=%d, target_modules=%s)",
                    peft_config.r,
                    peft_config.target_modules,
                )

                # Strip LoRA CLI flags and change --finetuning_type to "full"
                # so that LlamaFactory's own init_adapter won't try to apply
                # LoRA again when it re-parses the args.
                raw_args = strip_lora_args(raw_args)

            # apply_patches checks version compat for each patch
            adapter.apply_patches(engine)

            logger.info_rank0(
                "torchturbo: engine ready, entering run_exp for stage: %s",
                stage,
            )
            return self._original(raw_args, callbacks)

        tuner_module.run_exp = _patched_run_exp
        logger.info_rank0("Patched tuner.run_exp")

        # launcher.py does `from .train.tuner import run_exp`
        # meaning it holds a reference to the unpatched version if imported before torchturbo
        try:
            import llamafactory.launcher as launcher_module

            if hasattr(launcher_module, "run_exp"):
                self._launcher_module = launcher_module
                launcher_module.run_exp = _patched_run_exp
                logger.debug_rank0("Patched launcher.run_exp")
        except ImportError as e:
            raise ImportError("Failed to patch llamafactory.launcher")

    def revert(self) -> None:
        if self._module and self._original:
            self._module.run_exp = self._original
        if self._launcher_module and self._original:
            self._launcher_module.run_exp = self._original
            self._launcher_module = None
        self._original = None


# ---------------------------------------------------------------------------
# PackedInputCollatorPatch - normalize training batches
# ---------------------------------------------------------------------------


def _normalize_training_features(
    batch: dict[str, Any],
    *,
    cu_seqlens: torch.Tensor,
    max_length: int,
    num_dummy_tokens: int,
    original_first_length: int,
    padding_side: str,
) -> None:
    """Flatten LF rows while preserving masked dummy media at the tail."""
    import torch
    import torch.nn.functional as F

    from ...engine.utils import assert_packed_inputs

    attention_mask = batch.pop("attention_mask")
    # No Turbo model consumes token_type_ids; drop it rather than carry a
    # stale [batch, seq] tensor alongside the flattened packed rows.
    batch.pop("token_type_ids", None)
    batch_size, seq_len = batch["input_ids"].shape
    if attention_mask.shape != (batch_size, seq_len):
        # LF's dummy-image mRoPE path appends an extra zero mask tail.
        if (
            num_dummy_tokens == 0
            or attention_mask.size(0) != batch_size
            or attention_mask.size(1) < seq_len
            or torch.any(attention_mask[:, seq_len:] != 0)
        ):
            raise ValueError("LF packed attention_mask must match input_ids.")
        attention_mask = attention_mask[:, :seq_len]

    position_ids = batch.get("position_ids")
    if position_ids is not None and position_ids.dim() == 2 and position_ids.size(1) < seq_len:
        # LF does not extend text position_ids for collator or dummy padding.
        missing = seq_len - position_ids.size(1)
        if padding_side == "right":
            padding = (0, missing)
            missing_mask = attention_mask[:, -missing:]
        else:
            padding = (missing, 0)
            missing_mask = attention_mask[:, :missing]
        if torch.any(missing_mask != 0):
            raise ValueError("LF position_ids are missing valid token positions.")
        position_ids = F.pad(position_ids, padding)
    if position_ids is not None and (
        position_ids.dim() not in (2, 3) or position_ids.shape[-2:] != (batch_size, seq_len)
    ):
        raise ValueError("LF position_ids must align with the collated token rows.")

    valid_indices = torch.nonzero(attention_mask.flatten() != 0, as_tuple=False).flatten()
    if valid_indices.numel() != int(cu_seqlens[-1]):
        raise ValueError("LF collation must preserve the original valid tokens.")

    selected = valid_indices
    if num_dummy_tokens:
        # LF injects fake image/audio tokens into the first row with mask zero.
        # Keep their media placeholders, but place them after all valid tokens.
        if padding_side == "right":
            dummy_start = original_first_length
        else:
            dummy_start = seq_len - original_first_length - num_dummy_tokens
        dummy_indices = torch.arange(dummy_start, dummy_start + num_dummy_tokens)
        selected = torch.cat([selected, dummy_indices])

    if position_ids is None:
        # Text positions would be wrong for media tokens, so multimodal batches
        # must bring the collator's RoPE coordinates.
        if batch.get("mm_token_type_ids") is None:
            positions = torch.cat([
                torch.arange(length) for length in cu_seqlens.diff().tolist()
            ])
            batch["position_ids"] = F.pad(positions, (0, num_dummy_tokens)).unsqueeze(0)
    elif position_ids.dim() == 3:
        batch["position_ids"] = position_ids.flatten(1).index_select(1, selected).unsqueeze(1)
    else:
        batch["position_ids"] = position_ids.flatten().index_select(0, selected).unsqueeze(0)

    for key in ("input_ids", "labels", "mm_token_type_ids", "cross_attention_mask"):
        value = batch.get(key)
        if value is not None:
            if value.shape[:2] != (batch_size, seq_len):
                raise ValueError(f"LF {key} must align with the collated token rows.")
            batch[key] = value.flatten(0, 1).index_select(0, selected).unsqueeze(0)

    if "labels" in batch:
        # Trainer counts supervision before the backend shifts labels. Sample
        # starts have no preceding prediction within their attention segment.
        batch["labels"][0, cu_seqlens[:-1]] = -100

    batch["cu_seq_lens_q"] = cu_seqlens
    batch["cu_seq_lens_k"] = cu_seqlens
    batch["max_length_q"] = max_length
    batch["max_length_k"] = max_length
    assert_packed_inputs(batch)


def _collate_training_features(
    collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    padding_side: str,
) -> dict[str, Any]:
    """Normalize native LF collation, retaining injected dummy media tokens."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from ...engine.utils import get_packed_boundaries

    # LF can overwrite packed text positions when computing mRoPE. Capture
    # sample boundaries before collation; model-specific coordinates stay intact.
    attention_masks = []
    text_positions = []
    for feature in features:
        num_tokens = len(feature["input_ids"])
        # Pretraining features may contain only input_ids.
        attention_masks.append(torch.as_tensor(feature.get("attention_mask", [1] * num_tokens)))
        text_positions.append(torch.as_tensor(feature.get("position_ids", range(num_tokens))))
    _, cu_seqlens, max_length = get_packed_boundaries(
        pad_sequence(attention_masks, batch_first=True),
        pad_sequence(text_positions, batch_first=True),
    )
    original_first_length = len(features[0]["input_ids"])
    batch = collator(features)
    batch.pop("has_dummy_image", None)
    _normalize_training_features(
        batch,
        cu_seqlens=cu_seqlens,
        max_length=max_length,
        num_dummy_tokens=len(features[0]["input_ids"]) - original_first_length,
        original_first_length=original_first_length,
        padding_side=padding_side,
    )
    return batch


class PackedInputCollatorPatch(BasePatch):
    """Normalize LF training batches to one row with cu/max metadata."""

    name = "packed_input_collator"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original_call = None
        self._cls = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        import torch

        from llamafactory.data.collator import (
            MultiModalDataCollatorForSeq2Seq,
            SFTDataCollatorWith4DAttentionMask,
        )

        self._cls = SFTDataCollatorWith4DAttentionMask
        self._original_call = SFTDataCollatorWith4DAttentionMask.__call__

        def _call_with_packed_inputs(collator_self, features):
            batch = _collate_training_features(
                partial(MultiModalDataCollatorForSeq2Seq.__call__, collator_self),
                features,
                padding_side=collator_self.tokenizer.padding_side,
            )
            for key, value in batch.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    batch[key] = value.to(collator_self.compute_dtype)
            return batch

        SFTDataCollatorWith4DAttentionMask.__call__ = _call_with_packed_inputs
        logger.info_rank0("PackedInputCollatorPatch applied")

    def revert(self) -> None:
        if self._cls and self._original_call:
            self._cls.__call__ = self._original_call
            self._original_call = None
            self._cls = None


# ---------------------------------------------------------------------------
# LoadModelPatch
# ---------------------------------------------------------------------------


class LoadModelPatch(BasePatch):
    """Intercept ``from_pretrained`` inside LF's ``load_model``.

    Temporarily monkey-patches ``AutoModelForCausalLM.from_pretrained``
    (and related Auto classes) so that when LF calls ``load_class.from_pretrained()``,
    it goes through ``engine.prepare_model()`` instead.

    When LlamaFactory is using LoRA (``finetuning_type == "lora"``), this
    patch extracts the LoRA config from LlamaFactory's ``FinetuningArguments``
    and passes it to ``engine.prepare_model(peft_config=...)`` so that LoRA
    is applied **after** module replacement but **before** FSDP2 sharding —
    the correct ordering for torchturbo's MoE models.

    All other LF logic (config patching, adapter init, value head, etc.) is preserved.
    """

    name = "load_model"
    supported_versions = _LF_V0

    # Auto classes that we intercept
    _AUTO_CLASSES = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForSeq2SeqLM",
        "AutoModelForTextToWaveform",
    )

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._originals: dict[str, Any] = {}  # class_name → original from_pretrained
        self._module = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            import transformers
        except ImportError:
            return

        self._module = transformers

        for cls_name in self._AUTO_CLASSES:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue

            original_fp = cls.from_pretrained
            self._originals[cls_name] = original_fp

            def _make_patched(orig, klass_name):
                def _patched_from_pretrained(
                    pretrained_model_name_or_path, *args, **kwargs
                ):
                    logger.info_rank0(
                        "torchturbo: intercepting %s.from_pretrained → "
                        "engine.prepare_model(%r)",
                        klass_name,
                        pretrained_model_name_or_path,
                    )

                    # Determine dtype from kwargs or default
                    torch_dtype = kwargs.get("torch_dtype", None)
                    if torch_dtype == "auto" or torch_dtype is None:
                        import torch

                        torch_dtype = torch.bfloat16

                    trust_remote_code = kwargs.get("trust_remote_code", False)

                    model = engine.prepare_model(
                        model_name_or_path=pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        init_kwargs=kwargs,
                        peft_config=getattr(self.adapter, "peft_config", None),
                    )

                    return model

                return _patched_from_pretrained

            cls.from_pretrained = classmethod(
                lambda klass, *a, _fn=_make_patched(original_fp, cls_name), **kw: _fn(
                    *a, **kw
                )
            )

            logger.info_rank0("Patched %s.from_pretrained", cls_name)

    def revert(self) -> None:
        if self._module is None:
            return

        for cls_name, original in self._originals.items():
            cls = getattr(self._module, cls_name, None)
            if cls is not None:
                cls.from_pretrained = original
                logger.info_rank0("Reverted %s.from_pretrained", cls_name)

        self._originals.clear()


# ---------------------------------------------------------------------------
# CreateRefModelPatch
# ---------------------------------------------------------------------------


class CreateRefModelPatch(BasePatch):
    """Patch LlamaFactory's algorithm-level reference model factory."""

    name = "create_ref_model"
    supported_versions = _LF_V0

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._originals: list[tuple[Any, str, Any]] = []

    def patch(self, engine: TorchTurboEngine) -> None:
        def _patched_create_ref_model(
            model_args,
            finetuning_args,
            *args,
            add_valuehead: bool = False,
            **kwargs,
        ):
            if add_valuehead:
                raise NotImplementedError(
                    "TorchTurbo DPO does not support value-head reference models."
                )
            if getattr(finetuning_args, "ref_model_adapters", None):
                raise NotImplementedError(
                    "TorchTurbo DPO does not support ref_model_adapters yet."
                )
            if getattr(finetuning_args, "ref_model_quantization_bit", None):
                raise NotImplementedError(
                    "TorchTurbo DPO does not support ref_model_quantization_bit yet."
                )

            from llamafactory.hparams import ModelArguments
            from llamafactory.model import load_config, load_tokenizer
            from llamafactory.model.loader import _get_init_kwargs
            from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
            from llamafactory.model.patcher import patch_config, patch_model

            ref_path = getattr(finetuning_args, "ref_model", None)
            if ref_path is not None:
                ref_model_args = ModelArguments.copyfrom(
                    model_args,
                    model_name_or_path=ref_path,
                    adapter_name_or_path=None,
                    quantization_bit=None,
                )
            else:
                ref_model_args = ModelArguments.copyfrom(model_args)

            tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
            init_kwargs = _get_init_kwargs(ref_model_args)
            config = load_config(ref_model_args)
            patch_config(
                config,
                tokenizer,
                ref_model_args,
                init_kwargs,
                is_trainable=False,
            )
            apply_liger_kernel(
                config,
                ref_model_args,
                is_trainable=False,
                require_logits=True,
            )
            init_kwargs["config"] = config

            torch_dtype = getattr(ref_model_args, "compute_dtype", None)
            if torch_dtype is None:
                import torch

                torch_dtype = torch.bfloat16

            ref_model = engine.build_model(
                model_name_or_path=ref_model_args.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=getattr(ref_model_args, "trust_remote_code", False),
                init_kwargs=init_kwargs,
                load_weights=ref_path is not None,
            )
            patch_model(
                ref_model,
                tokenizer,
                ref_model_args,
                is_trainable=False,
                add_valuehead=False,
            )

            if ref_path is not None:
                ref_model.requires_grad_(False)
                ref_model.eval()
                logger.info_rank0(
                    "torchturbo: created reference model from %s",
                    ref_model_args.model_name_or_path,
                )
            else:
                logger.info_rank0(
                    "torchturbo: created empty reference model from policy config; "
                    "weights will be copied from the initial policy."
                )

            return ref_model

        self._replace_create_ref_model(
            "llamafactory.train.trainer_utils",
            _patched_create_ref_model,
            required=True,
        )
        self._replace_create_ref_model(
            "llamafactory.train.dpo.workflow",
            _patched_create_ref_model,
            required=getattr(self.adapter, "stage", None) == "dpo",
        )

    def _replace_create_ref_model(
        self,
        module_path: str,
        patched: Any,
        *,
        required: bool,
    ) -> None:
        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError:
            if required:
                raise
            logger.debug_rank0(
                "torchturbo: %s not found; skipping create_ref_model patch.",
                module_path,
            )
            return

        if not hasattr(module, "create_ref_model"):
            if required:
                raise AttributeError(f"{module_path} has no create_ref_model.")
            logger.debug_rank0(
                "torchturbo: %s has no create_ref_model; skipping patch.",
                module_path,
            )
            return

        current = getattr(module, "create_ref_model")
        original = getattr(current, "_torchturbo_original", current)
        patched._torchturbo_original = original
        already_recorded = any(
            item[0] is module and item[1] == "create_ref_model"
            for item in self._originals
        )
        if not already_recorded:
            self._originals.append((module, "create_ref_model", original))
        setattr(module, "create_ref_model", patched)
        logger.info_rank0(
            "torchturbo: replaced %s.create_ref_model",
            getattr(module, "__name__", repr(module)),
        )

    def revert(self) -> None:
        for module, attr_name, original in reversed(self._originals):
            setattr(module, attr_name, original)
        self._originals.clear()


# ---------------------------------------------------------------------------
# TrainerClassPatch — replace LF workflow trainer classes with TT subclasses
# ---------------------------------------------------------------------------


class TrainerClassPatch(BasePatch):
    """Replace LlamaFactory workflow trainer references with TT subclasses."""

    name = "trainer_class"
    supported_versions = _LF_V0

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._originals: list[tuple[Any, str, Any]] = []

    def patch(self, engine: TorchTurboEngine) -> None:
        self._patch_sft(engine)
        self._patch_pt(engine)
        self._patch_dpo(engine)

    def _replace(self, module: Any, attr_name: str, patched_cls: type) -> None:
        from .trainer import unwrap_torchturbo_trainer_class

        original = unwrap_torchturbo_trainer_class(getattr(module, attr_name))
        self._originals.append((module, attr_name, original))
        setattr(module, attr_name, patched_cls)
        logger.info_rank0(
            "torchturbo: replaced %s.%s with %s",
            getattr(module, "__name__", repr(module)),
            attr_name,
            patched_cls.__name__,
        )

    def _patch_sft(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.sft import workflow as sft_workflow
            from .trainer import (
                make_torchturbo_trainer,
                unwrap_torchturbo_trainer_class,
            )
        except ImportError:
            logger.warning_rank0("LlamaFactory SFT trainer not found; skipping trainer patch.")
            return

        base_cls = unwrap_torchturbo_trainer_class(sft_workflow.CustomSeq2SeqTrainer)
        patched_cls = make_torchturbo_trainer(
            base_cls,
            engine=engine,
            stage="sft",
        )
        self._replace(sft_workflow, "CustomSeq2SeqTrainer", patched_cls)

    def _patch_pt(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.pt import workflow as pt_workflow
            from .trainer import (
                make_torchturbo_trainer,
                unwrap_torchturbo_trainer_class,
            )
        except ImportError:
            logger.warning_rank0("LlamaFactory PT trainer not found; skipping trainer patch.")
            return

        base_cls = unwrap_torchturbo_trainer_class(pt_workflow.CustomTrainer)
        patched_cls = make_torchturbo_trainer(
            base_cls,
            engine=engine,
            stage="pt",
        )
        self._replace(pt_workflow, "CustomTrainer", patched_cls)

    def _patch_dpo(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.dpo import trainer as dpo_trainer
            from .trainer import (
                make_torchturbo_trainer,
                unwrap_torchturbo_trainer_class,
            )
        except ImportError:
            if getattr(self.adapter, "stage", None) == "dpo":
                raise ImportError("LlamaFactory DPO trainer not found.")
            logger.debug_rank0("LlamaFactory DPO trainer not found; skipping trainer patch.")
            return

        base_cls = unwrap_torchturbo_trainer_class(dpo_trainer.CustomDPOTrainer)
        patched_cls = make_torchturbo_trainer(
            base_cls,
            engine=engine,
            stage="dpo",
        )
        self._replace(dpo_trainer, "CustomDPOTrainer", patched_cls)

    def revert(self) -> None:
        for module, attr_name, original in reversed(self._originals):
            setattr(module, attr_name, original)
        self._originals.clear()


# ---------------------------------------------------------------------------
# AcceleratorPatch — skip FSDP model prep in accelerator
# ---------------------------------------------------------------------------


class AcceleratorPatch(BasePatch):
    """Skip accelerator's FSDP model preparation and patch clip_grad_norm.

    When torchturbo has already applied FSDP2 sharding via ``LoadModelPatch``,
    the accelerator's ``prepare_model`` would try to re-shard the model.
    This patch:

    1. Guards ``Accelerator.prepare`` — raises if ``is_fsdp2`` is ``True``
       (accelerate's FSDP2 param-fixup conflicts with torchturbo's sharding).
    2. Makes ``Accelerator.prepare_model`` a lightweight pass-through
       (preserves ``native_amp`` autocast wrapping).
    3. Keeps optimizer wrapping but leaves optimizer-state placement to the
       TorchTurbo backend.
    4. Overrides ``Accelerator.clip_grad_norm`` if backend provides a custom
       implementation
    5. Overrides ``Accelerator.backward`` with plain ``loss.backward()``
    """

    name = "accelerator"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._orig_prepare_model = None
        self._orig_prepare_optimizer = None
        self._orig_prepare = None
        self._orig_clip_grad_norm = None
        self._orig_backward = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from accelerate import Accelerator
        except ImportError:
            logger.warning_rank0("accelerate not found — skipping AcceleratorPatch.")
            return

        # ---- 1. Guard prepare: error out if accelerate thinks it owns FSDP2 ----
        self._orig_prepare = Accelerator.prepare

        def _guarded_prepare(accel_self, *args, device_placement=None):
            if getattr(accel_self, "is_fsdp2", False):
                raise RuntimeError(
                    "torchturbo already handles FSDP2 sharding. "
                    "Do not configure accelerate with FSDP2 — remove "
                    "fsdp_plugin / FSDP settings from your accelerate config."
                )
            return self._orig_prepare(
                accel_self, *args, device_placement=device_placement
            )

        Accelerator.prepare = _guarded_prepare

        # ---- 2. Patch prepare_model: skip FSDP, keep native_amp ----
        self._orig_prepare_model = Accelerator.prepare_model

        def _patched_prepare_model(
            accel_self,
            model,
            device_placement=None,
            evaluation_mode=False,
        ):
            from types import MethodType
            from accelerate.utils import convert_outputs_to_fp32

            accel_self._models.append(model)

            logger.info_rank0(
                "torchturbo: prepare_model skipped FSDP wrapping "
                "(already sharded by engine)"
            )
            return model

        Accelerator.prepare_model = _patched_prepare_model

        # ---- 3. Keep Accelerate training semantics without moving state ----
        self._orig_prepare_optimizer = Accelerator.prepare_optimizer

        def _patched_prepare_optimizer(
            accel_self,
            optimizer,
            device_placement=None,
        ):
            if optimizer is engine.optimizer:
                device_placement = False
            return self._orig_prepare_optimizer(
                accel_self,
                optimizer,
                device_placement=device_placement,
            )

        Accelerator.prepare_optimizer = _patched_prepare_optimizer

        # ---- 4. Patch clip_grad_norm_ if backend supports it ----
        self._orig_clip_grad_norm = Accelerator.clip_grad_norm_

        def _patched_clip_grad_norm(accel_self, parameters, max_norm, norm_type=2):
            return engine.backend.clip_grad_norm_(parameters, max_norm, norm_type)

        Accelerator.clip_grad_norm_ = _patched_clip_grad_norm
        logger.info_rank0(
            "torchturbo: Patched Accelerator.clip_grad_norm_ to use backend's implementation"
        )

        # ---- 5. Patch backward ----
        self._orig_backward = Accelerator.backward

        def _patched_backward(accel_self, loss, **kwargs):
            loss.backward()

        Accelerator.backward = _patched_backward
        logger.info_rank0(
            "torchturbo: Patched Accelerator.backward to plain"
        )

        # ---- 6. Patch prepare_data_loader for SP-aware sampling ----
        # sp_size = engine.get_sp_info()["sp_size"]
        # if sp_size > 1:
        #     self._orig_prepare_data_loader = Accelerator.prepare_data_loader
        #
        #     def _patched_prepare_data_loader(accel_self, data_loader, device_placement=None, slice_fn_for_dispatch=None):
        #         dp_info = engine.get_dp_info()
        #         # Temporarily override process topology for SP
        #         orig_num = accel_self.state.num_processes
        #         orig_idx = accel_self.state.process_index
        #         accel_self.state.num_processes = dp_info["dp_size"]
        #         accel_self.state.process_index = dp_info["dp_rank"]
        #         try:
        #             result = self._orig_prepare_data_loader(accel_self, data_loader, device_placement, slice_fn_for_dispatch)
        #         finally:
        #             accel_self.state.num_processes = orig_num
        #             accel_self.state.process_index = orig_idx
        #         return result

        #     Accelerator.prepare_data_loader = _patched_prepare_data_loader
        #     logger.info(
        #         "torchturbo: Patched Accelerator.prepare_data_loader to aware SP"
        #     )

        logger.info_rank0("AcceleratorPreparePatch applied")

    def revert(self) -> None:
        try:
            from accelerate import Accelerator
        except ImportError:
            return

        if self._orig_prepare_model is not None:
            Accelerator.prepare_model = self._orig_prepare_model
            self._orig_prepare_model = None
        if self._orig_prepare_optimizer is not None:
            Accelerator.prepare_optimizer = self._orig_prepare_optimizer
            self._orig_prepare_optimizer = None
        if self._orig_prepare is not None:
            Accelerator.prepare = self._orig_prepare
            self._orig_prepare = None
        if self._orig_clip_grad_norm is not None:
            Accelerator.clip_grad_norm_ = self._orig_clip_grad_norm
            self._orig_clip_grad_norm = None
        if self._orig_backward is not None:
            Accelerator.backward = self._orig_backward
            self._orig_backward = None


# ---------------------------------------------------------------------------
# RegisterAutoclassPatch — fix register_autoclass for wrapped models
# ---------------------------------------------------------------------------


def _find_base_model_class(model: Any) -> type | None:
    """Walk through PeftModel / FSDP2 wrappers to find the inner
    ``PreTrainedModel`` subclass.

    ``register_for_auto_class`` must be called on the *original* model class
    (e.g. ``LlamaForCausalLM``) so that ``custom_object_save`` stores the
    correct ``module.ClassName`` in ``config.auto_map`` — not the wrapper
    class name (e.g. ``FSDPPeftModelForCausalLM``).

    Unwrapping strategy (in order):

    1. **MRO walk**: FSDP2 uses ``fully_shard`` which mutates ``__class__``
       in-place, creating a subclass like ``FSDP<PfxModelForCausalLM>`` whose
       MRO includes the original class.  We check the MRO for any class with
       ``register_for_auto_class``.

    2. **PeftModel unwrapping**: If the MRO doesn't contain a
       ``PreTrainedModel``, the outermost layer is likely a ``PeftModel``.
       Calling ``get_base_model()`` returns the underlying transformer model
       (e.g. ``LlamaForCausalLM``), whose MRO *will* contain it.

    We intentionally do **not** follow the ``.module`` attribute here.
    FSDP2 does not create wrapper objects with ``.module``; it mutates
    ``__class__`` in-place.  Following ``.module`` on an arbitrary
    ``nn.Module`` risks descending into unrelated sub-modules.
    """
    seen: set[int] = set()
    cur = model
    while cur is not None:
        if id(cur) in seen:
            break
        seen.add(id(cur))

        # Check MRO of the current object for a PreTrainedModel subclass.
        # FSDP2 creates a new class: type("FSDP" + cls.__name__, (FSDPModule, cls), {})
        # so the original cls is in the MRO.  PeftModelForCausalLM does NOT
        # inherit from PreTrainedModel, so this won't match at that layer —
        # we need to unwrap first.
        for cls in type(cur).__mro__:
            if hasattr(cls, "register_for_auto_class") and cls is not object:
                return cls

        # PeftModel → get_base_model() returns the original transformer model
        try:
            from peft import PeftModel

            if isinstance(cur, PeftModel):
                cur = cur.get_base_model()
                continue
        except ImportError:
            pass

        # No more wrappers to peel — give up
        break

    return None


class RegisterAutoclassPatch(BasePatch):
    """Fix ``register_autoclass`` for PeftModel / FSDP-wrapped models.

    ``register_for_auto_class`` is a ``@classmethod`` on
    ``PreTrainedModel`` that sets ``cls._auto_class`` so that
    ``save_pretrained`` / ``custom_object_save`` writes the correct
    ``auto_map`` entry in ``config.json``.

    When torchturbo applies LoRA + FSDP2, ``model.__class__`` becomes a
    wrapper (e.g. ``FSDPPeftModelForCausalLM``) that lacks this method.
    LlamaFactory's ``register_autoclass()`` then crashes with
    ``AttributeError``.

    The fix: replace ``register_autoclass`` so that when the model class
    lacks ``register_for_auto_class``, we walk through the wrapper layers
    to find the inner ``PreTrainedModel`` subclass (e.g.
    ``LlamaForCausalLM``) and call ``register_for_auto_class`` on *that*
    class.  This guarantees ``save_pretrained`` stores the original class
    name in ``auto_map``, not the wrapper's.
    """

    name = "register_autoclass"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._patched_modules: dict[Any, Any] = {}
        self._original = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        try:
            from llamafactory.model.model_utils import misc as misc_module
        except ImportError:
            logger.debug_rank0(
                "llamafactory.model.model_utils.misc not found — "
                "skipping RegisterAutoclassPatch."
            )
            return

        if not hasattr(misc_module, "register_autoclass"):
            logger.debug_rank0(
                "register_autoclass not found in misc module — "
                "skipping RegisterAutoclassPatch."
            )
            return

        original_fn = misc_module.register_autoclass
        self._original = original_fn
        self._module = misc_module

        def _patched_register_autoclass(config, model, tokenizer):
            if "AutoConfig" in getattr(config, "auto_map", {}):
                config.__class__.register_for_auto_class()

            if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
                model_cls = model.__class__
                if not hasattr(model_cls, "register_for_auto_class"):
                    # Wrapper class (e.g. FSDPPeftModelForCausalLM) lacks
                    # register_for_auto_class — find the inner model class.
                    base_cls = _find_base_model_class(model)
                    if base_cls is not None:
                        logger.info_rank0(
                            "torchturbo: register_autoclass — using %s "
                            "instead of %s for register_for_auto_class",
                            base_cls.__name__,
                            model_cls.__name__,
                        )
                        model_cls = base_cls
                    else:
                        logger.warning_rank0(
                            "torchturbo: register_autoclass — could not find "
                            "a PreTrainedModel subclass on %s; skipping "
                            "register_for_auto_class",
                            model_cls.__name__,
                        )
                if hasattr(model_cls, "register_for_auto_class"):
                    model_cls.register_for_auto_class()

            if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
                tokenizer.__class__.register_for_auto_class()

        # Patch the source module
        misc_module.register_autoclass = _patched_register_autoclass
        self._patched_modules[misc_module] = original_fn

        # Also patch importers that bound the function by name, e.g.
        #   from .model_utils.misc import register_autoclass
        import sys

        for mod in list(sys.modules.values()):
            if mod is None or mod is misc_module:
                continue
            if getattr(mod, "register_autoclass", None) is original_fn:
                self._patched_modules[mod] = original_fn
                mod.register_autoclass = _patched_register_autoclass

        logger.info_rank0("RegisterAutoclassPatch applied")

    def revert(self) -> None:
        for mod, original in self._patched_modules.items():
            mod.register_autoclass = original
        self._patched_modules.clear()
        self._original = None


# ---------------------------------------------------------------------------
# LogCallbackPatch — inject grad_norm into trainer_log.jsonl
# ---------------------------------------------------------------------------


class LogCallbackPatch(BasePatch):
    """Inject ``grad_norm`` into ``LogCallback.on_log`` JSON output.

    LlamaFactory's ``LogCallback.on_log`` writes a JSON dict to
    ``trainer_log.jsonl`` every step, but does not include ``grad_norm``.
    This patch wraps ``_write_log`` to add it.

    ``_write_log`` is submitted to a thread pool by ``on_log``, so we
    cannot reference ``state`` from the patched ``_write_log`` — the
    ``state`` object may have been mutated by the next step by the time
    the thread pool worker runs.  Instead we stash the scalar
    ``grad_norm`` value on the callback instance *before* calling the
    original ``on_log``, so the thread pool worker reads a stable copy.
    """

    name = "log_callback"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original_on_log = None
        self._original_write_log = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        try:
            from llamafactory.train.callbacks import LogCallback
        except ImportError:
            logger.debug_rank0("LogCallback not found; skipping LogCallbackPatch.")
            return

        self._original_on_log = LogCallback.on_log
        self._original_write_log = LogCallback._write_log
        _orig_write_log = self._original_write_log

        def _patched_write_log(callback_self, output_dir, logs):
            gn = getattr(callback_self, "_tt_grad_norm", None)
            mtp_loss = getattr(callback_self, "_tt_mtp_loss", None)
            has_loss = "loss" in logs
            if has_loss:
                new_logs = dict(logs)
                if gn is not None:
                    new_logs["grad_norm"] = round(gn, 6)
                if mtp_loss is not None:
                    new_logs["mtp_loss"] = round(mtp_loss, 6)
                logs = new_logs
            return _orig_write_log(callback_self, output_dir, logs)

        def _patched_on_log(callback_self, args, state, control, **kwargs):
            callback_self._tt_grad_norm = state.log_history[-1].get("grad_norm")
            callback_self._tt_mtp_loss = state.log_history[-1].get("mtp_loss/0")
            self._original_on_log(callback_self, args, state, control, **kwargs)

        LogCallback.on_log = _patched_on_log
        LogCallback._write_log = _patched_write_log
        logger.info_rank0(
            "LogCallbackPatch applied — grad_norm and mtp_loss will be written to trainer_log.jsonl"
        )

    def revert(self) -> None:
        if self._original_on_log is not None:
            from llamafactory.train.callbacks import LogCallback
            LogCallback.on_log = self._original_on_log
            self._original_on_log = None
        if self._original_write_log is not None:
            from llamafactory.train.callbacks import LogCallback
            LogCallback._write_log = self._original_write_log
            self._original_write_log = None


# ---------------------------------------------------------------------------
# PretrainProcessorPatch — multimodal CPT packing with image-boundary protection
# ---------------------------------------------------------------------------


class PretrainProcessorPatch(BasePatch):
    """Patch PretrainDatasetProcessor to support multimodal CPT packing.

    Open-source LlamaFactory's PretrainDatasetProcessor ignores images.
    When the batch contains ``_images``, this patch delegates to
    ``pretrain_mllm.preprocess_multimodal_pretrain`` which performs
    image-boundary-aware packing.
    """

    name = "pretrain_processor"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        try:
            from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
        except ImportError:
            logger.debug_rank0("PretrainDatasetProcessor not found; skipping.")
            return

        original = PretrainDatasetProcessor.preprocess_dataset
        self._original = original

        def _patched(self, examples):
            img_examples = examples.get("_images", [None] * len(examples["_prompt"]))
            if not any(x is not None for x in img_examples):
                return original(self, examples)

            from .utils import preprocess_multimodal_pretrain
            return preprocess_multimodal_pretrain(self, examples, img_examples)

        PretrainDatasetProcessor.preprocess_dataset = _patched
        logger.info_rank0("PretrainProcessorPatch applied")

    def revert(self) -> None:
        if self._original is not None:
            from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
            PretrainDatasetProcessor.preprocess_dataset = self._original
            self._original = None
