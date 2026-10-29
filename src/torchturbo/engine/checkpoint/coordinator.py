"""Backend-independent checkpoint policy and lifecycle coordination."""

from __future__ import annotations

import os
import shutil
import weakref
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from torchturbo.logging import get_logger

from ..optimizers import OptimizersContainer
from ..plugins import CheckpointExportPlugin, Plugin, PluginConsumer
from ..utils import is_peft_adapter_param
from . import layout
from .distributed import collective_check, rank_zero_call
from .format import CheckpointFormat, ComponentStatus
from .metadata import build_metadata, load_metadata, save_metadata, validate_peft_configs

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from ...models.weight_converter import BaseWeightConverter
    from ..backends.base import BaseCheckpointManager
    from ..config import EngineConfig

logger = get_logger(__name__)


class CheckpointExportPluginConsumer(PluginConsumer):
    """Register export plugins and notify them on global rank zero."""

    def __init__(self) -> None:
        self._export_plugins: list[CheckpointExportPlugin] = []

    def matches(self, plugin: Plugin) -> bool:
        return isinstance(plugin, CheckpointExportPlugin)

    def consume(self, plugin: Plugin) -> None:
        if not isinstance(plugin, CheckpointExportPlugin):
            raise TypeError("Expected a CheckpointExportPlugin.")
        self._export_plugins.append(plugin)

    def notify_hf_export(self, export_dir: str) -> None:
        """Notify plugins after an HF export, waiting on all ranks."""
        if not self._export_plugins:
            return

        def notify() -> None:
            for plugin in self._export_plugins:
                try:
                    plugin.on_hf_export(export_dir)
                except Exception:
                    logger.warning_rank0(
                        "HF export plugin %s failed for %s; continuing training.",
                        type(plugin).__name__, export_dir, exc_info=True,
                    )

        rank_zero_call(notify)


class CheckpointCoordinator:
    """Coordinate checkpoint policy above a backend checkpoint manager."""

    def __init__(
        self,
        config: EngineConfig,
        manager: BaseCheckpointManager,
        *,
        parallel_context: dict[str, Any],
        export_plugin_consumer: CheckpointExportPluginConsumer,
    ) -> None:
        self._export_plugin_consumer = export_plugin_consumer
        self._config = config
        self._manager = manager
        self._parallel_context = parallel_context
        self._weight_converters: weakref.WeakKeyDictionary[
            nn.Module, tuple[str, BaseWeightConverter]
        ] = weakref.WeakKeyDictionary()

    def load_pretrained(self, model: nn.Module, model_path: str) -> None:
        """Load source HF weights into a model prepared on meta."""
        self._manager.load_hf(
            model, model_path,
            weight_converter=self._weight_converter(model, model_path),
            assign=True,
        )

    def save_model(
        self,
        model: nn.Module,
        output_dir: str,
        *,
        source_path: str,
        format: CheckpointFormat | None = None,
        final: bool = False,
    ) -> tuple[CheckpointFormat, ...]:
        """Save a training checkpoint or an explicit model export."""
        if format is not None:
            if not isinstance(format, CheckpointFormat):
                raise TypeError(
                    "format must be a CheckpointFormat or None, got "
                    f"{type(format).__name__}."
                )
            formats = (format,)
        elif final or self._config.checkpoint.hf_export == "every":
            formats = (CheckpointFormat.NATIVE, CheckpointFormat.HF)
        else:
            formats = (CheckpointFormat.NATIVE,)

        metadata = collective_check(
            lambda: build_metadata(model, self._config, source_path)
        )

        # A model save starts a new checkpoint generation. Reset once before
        # writing any component, including when the new save is model-only.
        def start_save() -> None:
            self._reset_checkpoint(output_dir, source_path)
            for model_format in formats:
                component_dir = (
                    self._manager.native_model_dir(output_dir)
                    if model_format is CheckpointFormat.NATIVE
                    else os.path.join(output_dir, layout.HF_MODEL_DIR)
                )
                os.makedirs(component_dir)
            save_metadata(metadata, output_dir)

        rank_zero_call(start_save)
        for model_format in formats:
            if model_format is CheckpointFormat.NATIVE:
                self._manager.save_native(
                    model, self._manager.native_model_dir(output_dir)
                )
            else:
                self._manager.save_hf(
                    model,
                    os.path.join(output_dir, layout.HF_MODEL_DIR),
                    source_path=source_path,
                    weight_converter=self._weight_converter(model, source_path),
                )
                self._export_plugin_consumer.notify_hf_export(
                    os.path.join(output_dir, layout.HF_MODEL_DIR)
                )
        return formats

    def load_model(
        self,
        model: nn.Module,
        checkpoint_dir: str,
        *,
        source_path: str,
    ) -> None:
        """Restore model weights from a complete checkpoint."""
        model_format = rank_zero_call(
            lambda: layout.resolve_model_format(
                checkpoint_dir,
                self._manager.native_model_status(checkpoint_dir),
                self._manager.native_model_dir(checkpoint_dir),
            )
        )
        if model_format is CheckpointFormat.NATIVE:
            metadata = rank_zero_call(lambda: load_metadata(checkpoint_dir))
            if metadata is not None:
                collective_check(lambda: validate_peft_configs(model, metadata))
            self._manager.load_native(
                model, self._manager.native_model_dir(checkpoint_dir)
            )
        else:
            self._warn_hf_resume(model, checkpoint_dir)
            component_dir = os.path.join(checkpoint_dir, layout.HF_MODEL_DIR)
            self._manager.load_hf(
                model, component_dir,
                weight_converter=self._weight_converter(model, source_path),
            )

    def save_optimizer_and_scheduler(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
        output_dir: str,
    ) -> None:
        """Save optimizer and scheduler state."""
        collective_check(self._manager.validate_optimizer_checkpoint)
        rank_zero_call(lambda: self._reset_optimizer_checkpoint(output_dir))
        self._manager.save_optimizer(
            model, optimizer, self._manager.optimizer_dir(output_dir)
        )
        if lr_scheduler is not None:
            rank_zero_call(
                lambda: layout.save_scheduler(lr_scheduler.state_dict(), output_dir)
            )

    def load_optimizer_and_scheduler(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
        checkpoint_dir: str,
    ) -> None:
        """Restore optimizer and scheduler state from a checkpoint."""
        collective_check(self._manager.validate_optimizer_checkpoint)
        optimizer_dir = self._manager.optimizer_dir(checkpoint_dir)

        def preflight() -> dict[str, Any] | None:
            status = self._manager.optimizer_status(checkpoint_dir)
            if status is ComponentStatus.MISSING:
                hf_dir = os.path.join(checkpoint_dir, layout.HF_MODEL_DIR)
                raise FileNotFoundError(
                    f"No optimizer state found in {checkpoint_dir}, so it cannot "
                    "be used with resume_from_checkpoint. Either its save was "
                    "interrupted before the optimizer state was written (resume "
                    "from an earlier complete checkpoint), or it holds model "
                    "weights only, for example from save_only_model. To continue "
                    "training from its weights, start a new run in a new "
                    f"output_dir with model_name_or_path={hf_dir} (if that "
                    "directory is missing, export it first with "
                    "scripts/checkpoint/dcp_to_hf.py)."
                )
            if status is ComponentStatus.INCOMPLETE:
                raise RuntimeError(
                    f"Optimizer state in {optimizer_dir} is incomplete; the save was "
                    "interrupted. Resume from an earlier checkpoint."
                )
            metadata = load_metadata(checkpoint_dir)
            if metadata is not None:
                saved_ep_size = metadata["engine_config"]["parallel_plan"]["ep_size"]
                current_ep_size = self._parallel_context["ep"]["ep_size"]
                if saved_ep_size != current_ep_size:
                    raise ValueError(
                        f"Checkpoint {checkpoint_dir} saved EP size {saved_ep_size}, "
                        f"but the current run uses EP size {current_ep_size}. "
                        "Optimizer state requires matching EP sizes."
                    )
            scheduler_state = None
            if lr_scheduler is not None:
                scheduler_state = layout.load_scheduler(checkpoint_dir)

            return scheduler_state

        scheduler_state = rank_zero_call(preflight)

        self._manager.load_optimizer(model, optimizer, optimizer_dir)
        if isinstance(optimizer, OptimizersContainer):
            optimizer.refresh_param_groups()
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(scheduler_state)

    def _weight_converter(
        self, model: nn.Module, source_path: str
    ) -> BaseWeightConverter:
        cached = self._weight_converters.get(model)
        if cached is not None and cached[0] == source_path:
            return cached[1]
        converter = collective_check(
            lambda: model.get_weight_converter(self._parallel_context, source_path)
        )
        self._weight_converters[model] = (source_path, converter)
        return converter

    @staticmethod
    def _warn_hf_resume(model: nn.Module, checkpoint_dir: str) -> None:
        message = (
            "Resuming model weights from HF safetensors in %s. They are "
            "restored in the checkpoint dtype (usually bf16), which rounds "
            "away fp32 updates below its precision. Checkpoints with exact "
            "model state resume exactly."
        )
        has_lora = any(
            is_peft_adapter_param(name) for name, _ in model.named_parameters()
        )
        if has_lora:
            message += (
                " LoRA adapters were merged into the frozen base weights and "
                "restart from their initial values, while the optimizer "
                "restores the moments of the previous adapters."
            )
        logger.warning_rank0(message, checkpoint_dir)

    def _reset_checkpoint(self, checkpoint_dir: str, source_path: str) -> None:
        """Remove stale components after checking every destructive path."""
        paths = (
            self._manager.native_model_dir(checkpoint_dir),
            self._manager.optimizer_dir(checkpoint_dir),
            os.path.join(checkpoint_dir, layout.HF_MODEL_DIR),
            os.path.join(checkpoint_dir, layout.METADATA_DIR),
            os.path.join(checkpoint_dir, layout.SCHEDULER_FILE),
        )
        source = os.path.realpath(source_path)
        for path in paths:
            target = os.path.realpath(path)
            if os.path.commonpath((source, target)) in (source, target):
                raise ValueError(
                    f"Checkpoint destination {path} overlaps source model "
                    f"{source_path}. HF exports copy the source model directory "
                    "tree and resetting a checkpoint deletes files, so the "
                    "source must stay separate. Choose a separate output "
                    "directory."
                )
        for path in paths:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            elif os.path.lexists(path):
                os.remove(path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _reset_optimizer_checkpoint(self, checkpoint_dir: str) -> None:
        """Start an optimizer save without retaining stale scheduler state."""
        optimizer_dir = self._manager.optimizer_dir(checkpoint_dir)
        scheduler_path = os.path.join(checkpoint_dir, layout.SCHEDULER_FILE)
        if os.path.isdir(optimizer_dir):
            shutil.rmtree(optimizer_dir)
        if os.path.isfile(scheduler_path):
            os.remove(scheduler_path)
        os.makedirs(optimizer_dir)
