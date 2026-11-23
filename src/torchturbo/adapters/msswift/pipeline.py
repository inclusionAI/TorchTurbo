"""Turbo-backed swift pipelines: a stage class declares ``args_class`` +
``turbo_stage`` and shares :class:`TurboEnginePipelineMixin`; :mod:`plugin`
swaps it onto the swift module global per :mod:`stages`. Classes stay
module-level (dill pickles by reference; closures trip on ABCMeta).
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from swift.pipelines.base import SwiftPipeline
from swift.pipelines.train.pretrain import SwiftPretrain
from swift.pipelines.train.rlhf import SwiftRLHF
from swift.pipelines.train.sft import SwiftSft

from .args import TurboPretrainArguments, TurboRLHFArguments, TurboSftArguments
from . import engine_registry
from .utils import build_config_from_swift_args, build_lora_config_from_swift_args
from ...logging import LogLevel, get_logger, logging_configure

logger = get_logger(__name__)


class TurboEnginePipelineMixin:
    """Mixed ahead of the swift pipeline base: ``__init__`` builds the engine
    before the model loads; ``prepare_model`` bypasses swift's TunerMixin
    (LoRA/freeze are engine-owned).
    """

    # Stage label recorded in :mod:`engine_registry` (logging/telemetry only);
    # each concrete pipeline class overrides this ("sft" / "pt" / ...).
    turbo_stage = "sft"

    def __init__(self, args: Optional[Union[List[str], Any]] = None) -> None:
        # Parse args without loading the model (skip SwiftSft.__init__).
        SwiftPipeline.__init__(self, args)
        self.train_msg: dict = {}

        self._build_turbo_engine()

        # swift's normal prepare sequence (SP forced off on the args).
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_flash_ckpt()

    def _build_turbo_engine(self) -> None:
        """Build the engine and stash it for :class:`TurboModelLoader` (runs before
        the model loads).
        """
        from ...engine import TorchTurboEngine
        from ...engine.args import get_torchturbo_args

        tt_args = get_torchturbo_args()
        logging_configure(tt_args.log_level if tt_args is not None else LogLevel.INFO)

        config = build_config_from_swift_args(self.args)
        if config.mega_moe:
            from ...optimizations.mega_moe.integration import configure_token_capacity

            length = self.args.packing_length if self.args.packing else self.args.max_length
            configure_token_capacity(
                config, length or self.args.max_length, self.args.per_device_train_batch_size,
            )
        engine = TorchTurboEngine(config)

        from .plugins import SwiftPluginHost

        # Plugin host for --tt_plugin plugins; reused by _prepare_dataset and
        # handed to the trainer via _get_trainer_kwargs.
        self._plugin_host = SwiftPluginHost()
        self._plugin_host.install_plugins(engine.plugin_manager)
        if getattr(self.args, "tuner_type", "full") == "lora":
            engine.peft_config = build_lora_config_from_swift_args(self.args)
        engine_registry.set_engine(engine, stage=type(self).turbo_stage)
        logger.info_rank0(
            "%s: engine ready (backend=%s, mp=%s)",
            type(self).__name__, config.backend.value, config.mixed_precision.value,
        )

    def prepare_model(self, args=None, model=None, template=None, train_dataset=None, **kwargs):
        # The engine already built + sharded + applied freeze/LoRA inside
        # TurboModelLoader; skip swift's TunerMixin.prepare_model.
        return model if model is not None else self.model

    def _prepare_dataset(self):
        """A plugin-provided dataset takes over via ``try_takeover``; else swift's
        native path."""
        datasets = self._plugin_host.dataset_bridge.try_takeover(
            args=self.args,
            template=self.template,
            show_dataset=getattr(self, "_show_dataset", None),
        )
        return datasets if datasets is not None else super()._prepare_dataset()

    def _get_trainer_kwargs(self):
        """Hand the routed plugin's collator to the trainer through swift's kwargs seam.
         swift's pipelines pass trainer constructor kwargs the same way (``SwiftRLHF`` 
         ships ``ref_model`` and friends through this hook)."""
        return {
            **super()._get_trainer_kwargs(),
            'plugin_collator': self._plugin_host.dataset_bridge.collator_for(self.args),
        }


class TurboSwiftSft(TurboEnginePipelineMixin, SwiftSft):  # type: ignore[misc]
    """SwiftSft that builds the turbo engine up front (module-level for dill
    by-reference pickling -- see module docstring).
    """

    args_class = TurboSftArguments
    turbo_stage = "sft"


class TurboSwiftPretrain(TurboEnginePipelineMixin, SwiftPretrain):  # type: ignore[misc]
    """SwiftPretrain that builds the turbo engine up front. SwiftPretrain
    only swaps in ``PretrainArguments``; everything else is the SFT takeover
    path. Module-level for the same dill/ABCMeta reason as TurboSwiftSft.
    """

    args_class = TurboPretrainArguments
    turbo_stage = "pt"

    def _get_dataset(self):
        from .data_check import validate_raw_dataset

        train_dataset, val_dataset = super()._get_dataset()
        validate_raw_dataset(train_dataset, "train")
        validate_raw_dataset(val_dataset, "val")
        return train_dataset, val_dataset


class TurboSwiftRLHF(TurboEnginePipelineMixin, SwiftRLHF):  # type: ignore[misc]
    """Swift DPO pipeline backed by TorchTurbo."""

    args_class = TurboRLHFArguments
    turbo_stage = "dpo"

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        if (origin_key or key) != "ref":
            return super()._prepare_single_model(key, origin_key, model_type, model_revision)

        with engine_registry.scoped_loading_role("reference"):
            return super()._prepare_single_model(key, origin_key, model_type, model_revision)
