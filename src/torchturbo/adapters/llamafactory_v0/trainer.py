"""TorchTurbo trainer wrappers for LlamaFactory v0."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ...engine.distributed import assert_finite
from ...engine.checkpoint import CheckpointFormat
from ...engine.checkpoint.layout import validate_output_dir
from ...logging import get_logger

logger = get_logger(__name__)


@dataclass
class LlamaFactoryInputCollator:
    """Normalize native pretraining collation before device transfer."""

    collator: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        from .patches import _collate_training_features

        return _collate_training_features(
            self.collator,
            features,
            padding_side=self.collator.tokenizer.padding_side,
        )


class TorchTurboTrainerBase:
    """Shared trainer behavior injected above LlamaFactory trainer classes.

    ``make_torchturbo_trainer`` creates a temporary class with this base
    first in the MRO, followed by the concrete LlamaFactory trainer.  Native
    LlamaFactory behavior remains available through ``super()``.
    """

    engine: Any = None
    stage: str | None = None
    torchturbo_wrapped: bool = False
    base_trainer_cls: type | None = None
    custom_optimizer_options: tuple[str, ...] = (
        "use_galore",
        "use_apollo",
        "use_badam",
        "use_adam_mini",
        "use_muon",
    )

    def __init__(self, *args, **kwargs):
        self._raise_if_unsupported_trainer_options(
            kwargs.get("finetuning_args", args[0] if args else None)
        )
        self._mm_processor = kwargs.get("processor", None)
        super().__init__(*args, **kwargs)
        if self.optimizer is not None:
            raise ValueError(
                "TorchTurbo does not support externally supplied optimizers"
            )
        self._maybe_upgrade_collator_for_mllm_pt()
        if self.stage == "pt":
            self.data_collator = LlamaFactoryInputCollator(self.data_collator)

    def _move_model_to_device(self, model, _device):
        # TorchTurbo already materializes and shards the model before Trainer init.
        return model

    def _maybe_upgrade_collator_for_mllm_pt(self) -> None:
        """Swap to MultiModalDataCollatorForSeq2Seq for multimodal PT datasets."""
        if self.stage != "pt":
            return

        dataset = self.train_dataset
        if dataset is None or not hasattr(dataset, "features"):
            return

        if not any(k in dataset.features for k in ("images", "videos", "audios")):
            return

        from transformers import DataCollatorForLanguageModeling
        if not isinstance(self.data_collator, DataCollatorForLanguageModeling):
            return

        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
        from llamafactory.extras.constants import IGNORE_INDEX

        # The existing data_collator was constructed with a tokenizer in run_pt.
        # MultiModalDataCollatorForSeq2Seq needs a template — retrieve it from
        # the engine where RunExpPatch stashed the template name.
        engine = self._engine()
        template_name = getattr(engine, "_lf_template_name", None)
        if not template_name:
            logger.warning_rank0("No template name on engine; skipping MLLM PT collator upgrade.")
            return

        from llamafactory.data.template import TEMPLATES
        template = TEMPLATES.get(template_name)
        if template is None:
            logger.warning_rank0(
                "Template %r not in registry; skipping MLLM PT collator upgrade.",
                template_name,
            )
            return

        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            model=self.model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX,
            tokenizer=self.processing_class,
            processor=self._mm_processor,
        )
        logger.info_rank0("Upgraded PT collator to MultiModalDataCollatorForSeq2Seq")

    def _raise_if_unsupported_trainer_options(self, finetuning_args: Any) -> None:
        unsupported = self._get_unsupported_trainer_options(finetuning_args)
        if unsupported:
            raise NotImplementedError(
                "TorchTurbo LlamaFactory trainer does not support "
                f"{', '.join(unsupported)} yet."
            )

    def _get_unsupported_trainer_options(self, finetuning_args: Any) -> list[str]:
        return (
            self._get_enabled_custom_optimizer_options(finetuning_args)
            + self._get_enabled_custom_loss_options(finetuning_args)
        )

    def _get_enabled_custom_optimizer_options(self, finetuning_args: Any) -> list[str]:
        enabled: list[str] = []
        for name in self.custom_optimizer_options:
            if bool(getattr(finetuning_args, name, None)):
                enabled.append(name)
        return enabled

    def _get_enabled_custom_loss_options(self, finetuning_args: Any) -> list[str]:
        if self.stage == "sft" and getattr(finetuning_args, "use_dft_loss", False):
            return ["use_dft_loss"]
        return []

    def _engine(self) -> Any:
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("TorchTurbo trainer is missing its engine binding.")
        return engine

    # Checkpoint resume hooks — route to the engine; fall back to HF when unbound.
    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        self.engine.save_optimizer_and_scheduler(
            self.model,
            self.engine.optimizer,
            self.lr_scheduler,
            output_dir,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.engine.load_from_checkpoint(self.model, resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint:
            self.engine.load_optimizer_and_scheduler(
                self.model,
                self.engine.optimizer,
                self.lr_scheduler,
                checkpoint,
            )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        engine = self._engine()

        if engine.config.optimizer.name is not None:
            self.optimizer = engine.prepare_optimizer()
            logger.info_rank0(
                "torchturbo: created engine-managed optimizer %s",
                engine.config.optimizer.name,
            )
            return self.optimizer

        from transformers import Trainer

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args,
            self.model,
        )
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        optimizer_kwargs.pop("optimizer_dict", None)
        optimizer_kwargs["weight_decay"] = self.args.weight_decay

        self.optimizer = engine.prepare_optimizer(optimizer_cls, **optimizer_kwargs)
        logger.info_rank0(
            "torchturbo: created optimizer via engine.prepare_optimizer "
            "(%s, %s)",
            optimizer_cls.__name__,
            {k: v for k, v in optimizer_kwargs.items() if k != "params"},
        )
        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        engine = self._engine()
        if engine.config.optimizer.name is not None:
            # Engine-managed optimizers take their LRs from engine config, so
            # the engine also builds their LR schedule.
            if self.lr_scheduler is None:
                from llamafactory.train.trainer_utils import create_custom_scheduler

                # Preserve LlamaFactory's scheduler argument defaults.
                create_custom_scheduler(self.args, num_training_steps)
                if optimizer is None:
                    optimizer = self.optimizer
                self.lr_scheduler = engine.prepare_lr_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def save_model(self, output_dir=None, _internal_call=False):
        engine = self._engine()
        if output_dir is None:
            output_dir = self.args.output_dir

        logger.info_rank0("torchturbo: saving model via engine.save_model(%s)", output_dir)
        # Intermediate checkpoints come through the HF checkpoint machinery
        # (_internal_call=True) and store native model state. LF's final save is
        # an explicit save_model() that exports HF weights, and a model-only
        # checkpoint cannot be resumed, so it is an export too.
        if _internal_call and not self.args.save_only_model:
            engine.save_model(
                self.model,
                output_dir,
            )
        else:
            engine.save_model(self.model, output_dir, format=CheckpointFormat.HF)

        if getattr(self.args, "push_to_hub", False) and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def get_train_dataloader(self):
        if self.stage not in {"sft", "pt", "dpo"}:
            return super().get_train_dataloader()

        # Diffusion (BD) models have their own per-sample dataset (BDDataset),
        # built from the LF data/model args the engine stashed at run_exp time.
        if self.model.is_diffusion_model:
            return self._get_bd_train_dataloader()

        engine = self._engine()
        args = self.args

        args.gradient_accumulation_steps = (
            engine.resolve_gradient_accumulation_steps(
                args.gradient_accumulation_steps
            )
        )

        finetuning_args = getattr(self, "finetuning_args", None)
        disable_shuffling = bool(
            getattr(finetuning_args, "disable_shuffling", False)
        )

        return engine.create_dataloader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            drop_last=args.dataloader_drop_last,
            shuffle=not disable_shuffling,
            seed=args.seed,
        )

    def _get_bd_train_dataloader(self):
        """Build the SP-aware DataLoader for block-diffusion (``BDDataset``).

        BD needs per-sample ``[noisy|clean]`` beams + a 4D block mask, so it
        cannot use LlamaFactory's tokenized dataset. Noise range and block size
        come from ``engine.config.diffusion`` (``--tt_bd_mask_low/high``,
        ``--tt_bd_block_size``); the rest (cutoff_len, data file, model path,
        block_diag_attn) is read from the LF args the engine stashed at run_exp
        time. SP folds ``gradient_accumulation_steps *= sp_size`` so the backend's
        ``compute_diffusion_loss`` ``/ ga`` lands on the right scale.
        """
        from ...models.llada2_moe import bd

        engine = self._engine()
        args = self.args
        sp_size = engine.get_sp_info()["sp_size"]
        diff = engine.config.diffusion

        data_args = getattr(engine, "_lf_data_args", None)
        model_args = getattr(engine, "_lf_model_args", None)
        if data_args is None or model_args is None:
            raise RuntimeError(
                "BD dataloader needs the LF data/model args; engine has none. "
                "Ensure RunExpPatch stashed them (engine._lf_data_args/_lf_model_args)."
            )

        cutoff_len = getattr(data_args, "cutoff_len", None)
        if cutoff_len:
            # Undo LF neat_packing's `cutoff_len -= 1` and ceil to a block_size multiple.
            # Use diff.bd_block_size (from --tt_bd_block_size, default 32) so this alignment
            # and the SP divisibility check below match the alignment BDDataset re-applies.
            cutoff_len = bd._align_cutoff_to_block(cutoff_len, diff.bd_block_size)
            beam_len = 2 * cutoff_len
            if sp_size > 1 and beam_len % sp_size != 0:
                raise ValueError(
                    f"BD beam_len=2*cutoff_len={beam_len} must be divisible by "
                    f"sp_size={sp_size} (SP slices the beam, see "
                    "backend.compute_diffusion_loss). "
                    f"Adjust --cutoff_len or --tt_bd_block_size "
                    f"(aligned value {cutoff_len}, block_size {diff.bd_block_size})."
                )

        bd_data_file, bd_columns, bd_tags = bd.resolve_bd_data_file(data_args)
        dataset = bd.BDDataset(
            getattr(model_args, "model_name_or_path", None),
            bd_data_file,
            block_diag_attn=bool(getattr(model_args, "block_diag_attn", False)),
            cutoff_len=cutoff_len,
            mask_low=diff.bd_mask_low,
            mask_high=diff.bd_mask_high,
            block_size=diff.bd_block_size,
            columns=bd_columns,
            tags=bd_tags,
        )

        args.gradient_accumulation_steps = (
            engine.resolve_gradient_accumulation_steps(
                args.gradient_accumulation_steps
            )
        )

        return engine.create_dataloader(
            dataset=dataset,
            batch_size=1,
            collate_fn=lambda feats: feats[0],
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            drop_last=args.dataloader_drop_last,
            shuffle=False,
            seed=args.seed,
        )

    def train(self, *args, **kwargs):
        from ...engine.args import get_torchturbo_args

        engine = self._engine()
        # Fail before training instead of at the first checkpoint save.
        validate_output_dir(self.args.output_dir, engine.model_name_or_path)
        logger.info_rank0(
            "LlamaFactory training started: %d epochs, max_steps=%s",
            int(getattr(self.args, "num_train_epochs", 1)),
            getattr(self.args, "max_steps", None),
        )

        tt_args = get_torchturbo_args()
        profiling_enabled = tt_args is not None and tt_args.profiling
        memory_snapshot_steps = (
            tt_args.memory_snapshot_steps if tt_args is not None else None
        )
        profiling_config = (
            tt_args.profiling_config if tt_args is not None else None
        )
        if profiling_enabled or memory_snapshot_steps is not None:
            profiling_callback = self._make_torchturbo_profiling_callback()
            self.add_callback(profiling_callback)
            engine.start_profiling(
                self.args.output_dir,
                profiling_enabled,
                memory_snapshot_steps,
                profiling_config,
            )

        try:
            result = super().train(*args, **kwargs)
        finally:
            if profiling_enabled or memory_snapshot_steps is not None:
                engine.stop_profiling()

        return result

    def _make_torchturbo_profiling_callback(self):
        from transformers import TrainerCallback
        engine = self._engine()

        class TorchTurboProfilingCallback(TrainerCallback):
            """Bridge HF optimizer-step boundaries to TorchTurbo profiling."""

            def on_step_end(self, args, state, control, **kwargs):
                engine.step_profiling()
                return control

        return TorchTurboProfilingCallback()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.stage in {"sft", "pt"}:
            if not hasattr(model, "is_diffusion_model"):
                raise AttributeError(
                    f"{type(model).__name__} is not a TurboModel — "
                    "is_diffusion_model is required (TurboModel default = False)."
                )
            if model.is_diffusion_model:
                return self._compute_diffusion_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )
            return self._compute_causal_lm_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )

    def training_step(self, model, inputs, *args, **kwargs):
        """Check loss and collect engine metrics after backward."""
        result = super().training_step(model, inputs, *args, **kwargs)
        assert_finite(result, "loss")
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.collect_metrics(model)
        return result

    def _report_engine_metrics(self, logs) -> None:
        from transformers.trainer import nested_gather

        def gather_fn(tensors):
            return nested_gather(tensors, self.args.parallel_mode)

        self._engine().report_metrics(logs, gather_fn)

    def log(self, logs, *args, **kwargs):
        self._report_engine_metrics(logs)
        return super().log(logs, *args, **kwargs)

    def _compute_causal_lm_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "labels" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )

        engine = self._engine()
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)

        loss, outputs = engine.compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
            gradient_accumulation_steps=(
                self.current_gradient_accumulation_steps if model.training else 1
            ),
            **kwargs,
        )

        sp_size = engine.get_sp_info()["sp_size"]
        if num_items_in_batch is None and sp_size > 1:
            loss = loss * sp_size
            outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    def _compute_diffusion_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Diffusion loss path — delegates to ``engine.compute_diffusion_loss``.

        Routed around ``_compute_causal_lm_loss`` (see ``compute_loss``) so BD never
        hits the shifted-CE ``* sp_size`` rescale. The batch is built by the engine's
        BD dataloader (``BDDataset``); here we only forward + compute loss.
        """
        engine = self._engine()
        ga = getattr(self.args, "gradient_accumulation_steps", 1) or 1
        loss, outputs = engine.compute_diffusion_loss(
            model, inputs, ga=ga, return_outputs=True, **kwargs
        )
        return (loss, outputs) if return_outputs else loss


class TorchTurboDPOTrainer(TorchTurboTrainerBase):
    """DPO-specific trainer behavior layered above LlamaFactory's DPO trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LlamaFactory's DPO workflow still owns the ref_model data flow:
        # create_ref_model(...) returns it and passes it into the trainer
        # constructor.  TorchTurbo only patches that factory; by this point
        # the base DPO trainer has already stored it on self.ref_model.
        self._initialize_dpo_ref_model()

    def _initialize_dpo_ref_model(self) -> None:
        ref_model = getattr(self, "ref_model", None)
        if ref_model is None:
            return
        if ref_model is self.model:
            return

        if getattr(self.finetuning_args, "ref_model", None) is None:
            self._copy_model_state(self.model, ref_model)
            logger.info_rank0(
                "torchturbo: initialized DPO reference model from policy weights."
            )

        ref_model.requires_grad_(False)
        ref_model.eval()

    def _copy_model_state(self, src_model: Any, dst_model: Any) -> None:
        src_state = dict(src_model.named_parameters(remove_duplicate=False))
        src_state.update(self._named_persistent_buffers(src_model))
        dst_state = dict(dst_model.named_parameters(remove_duplicate=False))
        dst_state.update(self._named_persistent_buffers(dst_model))

        src_keys = set(src_state)
        dst_keys = set(dst_state)
        if src_keys != dst_keys:
            missing = sorted(src_keys - dst_keys)[:5]
            unexpected = sorted(dst_keys - src_keys)[:5]
            raise RuntimeError(
                "Cannot initialize TorchTurbo DPO reference model: policy and "
                f"reference state keys differ (missing={missing}, "
                f"unexpected={unexpected})."
            )

        with torch.no_grad():
            for name in sorted(src_state):
                src = src_state[name].detach()
                dst = dst_state[name]
                if src.shape != dst.shape:
                    raise RuntimeError(
                        "Cannot initialize TorchTurbo DPO reference model: "
                        f"{name} shape differs ({tuple(src.shape)} vs "
                        f"{tuple(dst.shape)})."
                    )
                dst.copy_(src)

    @staticmethod
    def _named_persistent_buffers(model: Any) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        for module_name, module in model.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", set())
            for name, buffer in module.named_buffers(
                recurse=False,
                remove_duplicate=False,
            ):
                if buffer is None or name in non_persistent:
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                buffers[full_name] = buffer
        return buffers

    def concatenated_forward(self, model, batch, is_ref_model: bool = False):
        from llamafactory.train.trainer_utils import nested_detach
        from torchturbo.models.modules.attention.ulysses import slice_input_tensor

        sp_info = self._engine().get_sp_info()
        sp_size = sp_info["sp_size"]
        sp_group = sp_info["sp_group"]
        sp_rank = sp_info["sp_rank"]

        if getattr(self.finetuning_args, "use_ref_model", False):
            batch = nested_detach(batch, clone=True)
        else:
            batch = dict(batch)

        labels = batch.pop("labels")
        all_logits = model(
            **batch,
            return_dict=True,
            use_cache=False,
        ).logits.to(torch.float32)

        labels = nn.functional.pad(labels, (0, 1), value=self.label_pad_token_id)
        labels = labels[..., 1:].contiguous()
        if sp_size > 1:
            labels = slice_input_tensor(
                labels,
                dim=1,
                group=sp_group,
                rank=sp_rank,
            )

        all_logps, valid_length = self.get_batch_logps_with_sp(
            logits=all_logits,
            labels=labels,
            label_pad_token_id=self.label_pad_token_id,
            ld_alpha=(self.ld_alpha if not is_ref_model else None),
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_length,
        )

    def compute_reference_log_probs(self, model, batch):
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = (
                self.concatenated_forward(
                    ref_model,
                    batch,
                    is_ref_model=True,
                )
            )

        return reference_chosen_logps, reference_rejected_logps

    def get_batch_loss_metrics(self, model, inputs, train_eval: str = "train"):
        from torch.distributed.nn.functional import all_reduce
        from torchturbo.models.modules.attention.ulysses import ulysses_pad_inputs

        inputs = dict(inputs)
        sp_info = self._engine().get_sp_info()
        sp_size = sp_info["sp_size"]
        sp_group = sp_info["sp_group"]

        if inputs.get("position_ids", None) is None:
            inputs["position_ids"] = torch.arange(
                inputs["input_ids"].size(1),
                device=inputs["input_ids"].device,
            ).unsqueeze(0)

        inputs["attention_mask"] = None
        inputs = ulysses_pad_inputs(
            inputs, model.get_sequence_parallel_input_slicing_spec(), sp_size
        )

        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_length,
        ) = self.concatenated_forward(model, inputs)

        reference_chosen_logps, reference_rejected_logps = (
            self.compute_reference_log_probs(model, inputs)
        )

        if sp_size > 1:
            policy_chosen_logps = all_reduce(policy_chosen_logps, group=sp_group)
            policy_rejected_logps = all_reduce(policy_rejected_logps, group=sp_group)
            policy_chosen_length = all_reduce(policy_chosen_length, group=sp_group)
            if reference_chosen_logps is not None:
                reference_chosen_logps = all_reduce(reference_chosen_logps, group=sp_group)
            if reference_rejected_logps is not None:
                reference_rejected_logps = all_reduce(
                    reference_rejected_logps,
                    group=sp_group,
                )

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        policy_chosen_logps_avg = policy_chosen_logps / policy_chosen_length
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (
            chosen_rewards > rejected_rewards
        ).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            chosen_rewards - rejected_rewards
        ).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = (
                (losses - sft_loss) / self.beta
            ).mean().item()

        # HF Trainer divides by the expanded SP accumulation window
        return losses.mean(), metrics

    def get_batch_logps_with_sp(
        self,
        logits,
        labels,
        label_pad_token_id: int | None = None,
        ld_alpha: float | None = None,
    ):
        if label_pad_token_id is None:
            label_pad_token_id = self.label_pad_token_id

        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batchsize x seqlen) and labels must have the same shape."
            )

        labels = labels.clone()
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)

        valid_length = loss_mask.sum(-1)
        if ld_alpha is not None:
            num_examples = labels.shape[0] // 2
            chosen_lengths = valid_length[:num_examples]
            rejected_lengths = valid_length[num_examples:]
            min_lengths = torch.min(chosen_lengths, rejected_lengths)
            start_positions = torch.argmax(loss_mask.int(), dim=1)
            public_lengths = start_positions + torch.cat(
                [min_lengths, min_lengths],
                dim=0,
            )

            seq_len = labels.shape[-1]
            position_ids = torch.arange(
                seq_len,
                device=per_token_logps.device,
            ).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            front_mask = (ld_mask * loss_mask).float()
            rear_mask = (~ld_mask * loss_mask).float()

            front_logps = (per_token_logps * front_mask).sum(-1)
            rear_logps = (per_token_logps * rear_mask).sum(-1)
            logps = front_logps + ld_alpha * rear_logps
        else:
            logps = (per_token_logps * loss_mask).sum(-1)

        return logps, valid_length


def make_torchturbo_trainer(
    base_cls: type,
    *,
    engine: Any,
    stage: str,
) -> type:
    """Create a stage/engine-bound trainer subclass."""

    base_cls = unwrap_torchturbo_trainer_class(base_cls)
    trainer_engine = engine
    trainer_stage = stage
    wrapper_cls = TorchTurboDPOTrainer if stage == "dpo" else TorchTurboTrainerBase

    class TorchTurboTrainer(wrapper_cls, base_cls):
        engine = trainer_engine
        stage = trainer_stage
        torchturbo_wrapped = True
        base_trainer_cls = base_cls
        torchturbo_wrapper_cls = wrapper_cls

    name = f"TorchTurbo{stage.upper()}{base_cls.__name__}"
    TorchTurboTrainer.__name__ = name
    TorchTurboTrainer.__qualname__ = name
    TorchTurboTrainer.__module__ = base_cls.__module__

    return TorchTurboTrainer


def unwrap_torchturbo_trainer_class(cls: type) -> type:
    """Return the original LlamaFactory trainer behind a TT wrapper."""

    if not getattr(cls, "torchturbo_wrapped", False):
        return cls

    base_cls = getattr(cls, "base_trainer_cls", None)
    if isinstance(base_cls, type):
        return base_cls

    mro = getattr(cls, "__mro__", ())
    if len(mro) > 2:
        return mro[2]

    return cls
