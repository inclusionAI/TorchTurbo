"""Turbo wrapper for DeepSeek V4.1 (registration, weight converter, hooks)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from transformers import AutoModelForCausalLM

from ..base import TurboModel as TurboVLMModel
from ...engine.config import ParameterGroup
from ...models.modules.attention.ulysses import SequencePadding
from dataclasses import replace
from contextlib import contextmanager
from ...utils.flash_attn_utils import cu_seqlens_to_segment_ids
from ...engine.args import get_torchturbo_args
from ...engine.metrics.flops import BaseModelFlops
from ...logging import get_logger
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .configuration_deepseek_v41 import DeepseekV41Config, text_config_of
from .modeling_deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41Indexer,
    HostOffloadEngram,
)

if TYPE_CHECKING:
    from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
    from ..modules.moe.moe_layer import MoERuntimeInputs

logger = get_logger(__name__)


class DeepseekV41WeightConverter(BaseWeightConverter):
    """Checkpoint names are the native inference format; expert weights and
    the router bias are the only renames needed."""

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        self.config = text_config_of(self.config)
        self.model_type = self.config.model_type

    def get_expert_gate_name(self):
        return "w1"

    def get_expert_up_name(self):
        return "w3"

    def get_expert_down_name(self):
        return "w2"

    def get_hf_expert_mlp_name(self):
        return "ffn"

    def get_hf_gate_bias_name(self):
        return "gate.bias"

    def get_param_load_entries(self, name, numel, param_to_filename=None):
        # Engram params sit in the 197GiB table shards; loading them through
        # the generic path would materialize whole shards on the host.
        # HostOffloadEngram loads them itself in on_materialized (the parallel
        # plan always swaps the engram modules on the engine path).
        if ".engram." in name:
            return [], None, {}
        return super().get_param_load_entries(name, numel, param_to_filename)


@register_model(
    "deepseek_v41", config_cls=DeepseekV41Config, auto_class=AutoModelForCausalLM
)
class TurboDeepseekV41ForCausalLM(TurboVLMModel, DeepseekV41ForCausalLM):
    config_class = DeepseekV41Config
    base_model_prefix = ""
    # The packed RL/SFT path slices input_ids before embedding (v4-style).

    def __init__(self, config: DeepseekV41Config):
        self._pre_init_patches(config)
        super().__init__(config)
        self._post_init_patches()
        self._model_path = None

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context, model_path):
        # Stash the path for on_materialized (engram hash + row-sharded
        # tables), which runs after checkpoint loading.
        self._model_path = model_path
        return DeepseekV41WeightConverter(self, parallel_context, model_path)

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def get_flops_class(self) -> type[BaseModelFlops]:
        """DeepSeek-V4.1 flops recipe (CSA2 sparse latent attention + MoE + HC)."""
        from .mfu import DeepseekV41Flops

        return DeepseekV41Flops

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # FSDP2Backend's chunked-loss path shadows this module's ``head``
        # attribute with nn.Identity via object.__setattr__ (the _modules
        # registration keeps the real head). When shadowed, hand the no-op
        # head the normed fp32 hidden states in a single argument so the
        # vocab projection can run chunked outside the model.
        if self.head is self._modules["head"]:
            return super()._compute_logits(hidden_states)
        return self.head(self.norm(hidden_states).float())

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        if self.config.vision_enabled:
            groups[ParameterGroup.VISION_TOWER] = self.vision.named_parameters(
                prefix="vision"
            )
            groups[ParameterGroup.MULTI_MODAL_PROJECTOR] = (
                self.aligner.named_parameters(prefix="aligner")
            )
        groups[ParameterGroup.LANGUAGE_MODEL] = (
            (name, param)
            for name, param in self.named_parameters()
            if not name.startswith(("vision.", "aligner."))
        )
        return groups

    @staticmethod
    def _pre_init_patches(config):
        config._attn_implementation = "eager"
        # --tt_indexer_kl_loss_coeff overrides the checkpoint's
        # indexer_kl_loss_coeff (smoke runs against configs that lack or
        # zero the field); every KL decision reads the merged field.
        args = get_torchturbo_args()
        if args is not None and args.indexer_kl_loss_coeff is not None:
            config.text_config.indexer_kl_loss_coeff = args.indexer_kl_loss_coeff

    def _post_init_patches(self):
        self._use_flash_attention_2 = True
        self.config._attn_implementation = "flash_attention_2"
        # Phase 1 is text-only: the vision path is never exercised, so the
        # tower/projector stay frozen (weights still load for exchange).
        if self.config.vision_enabled:
            for module in (self.vision, self.aligner):
                for param in module.parameters():
                    param.requires_grad_(False)
            for param in (self.image_start, self.image_end, self.image_newline):
                param.requires_grad_(False)
        # Top-k indices do not carry grad into the indexer. Freeze before
        # sharding so those weights stay out of the optimizer; on_materialized
        # repeats this after assign-style checkpoint loading.
        self._npu_sfa_li_patched = False
        self._freeze_indexers()
        # Install the instance mHC helper before activation checkpointing,
        # so both the initial pass and recomputation use the same dispatch.
        # LI/KL patches target submodules and run in on_parallelized.
        from torchturbo.runtime import is_npu_available

        if is_npu_available():
            from .npu import (
                patch_deepseek_v41_npu_hc_post,
                patch_deepseek_v41_npu_mhc,
                patch_deepseek_v41_npu_rms_norm,
            )

            # mHC first: it validates the config constraints the hc_post
            # patch relies on.
            patch_deepseek_v41_npu_mhc(self)
            patch_deepseek_v41_npu_rms_norm(self)
            patch_deepseek_v41_npu_hc_post(self)

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        self._sp_group = sp_info["sp_group"]
        self._sp_size = sp_info["sp_size"]
        self._sp_rank = sp_info["sp_rank"]
        from torchturbo.runtime import is_npu_available

        if self._sp_size > 1 and not is_npu_available():
            raise NotImplementedError(
                "DeepSeek V4.1 sequence parallelism requires the NPU patches "
                f"(got sp_size={self._sp_size})."
            )
        # publish_shared runs outside decoder_layer.forward, so FSDP2's
        # forward pre-hook never all-gathers that layer. Register it as a
        # forward method or the first RMSNorm sees a DTensor weight.
        self._register_publish_shared_for_fsdp()

        if is_npu_available():
            from .npu import patch_deepseek_v41_npu_sfa

            count = patch_deepseek_v41_npu_sfa(
                self,
                sp_group=self._sp_group,
                sp_size=self._sp_size,
                sp_rank=self._sp_rank,
            )
            logger.info_rank0(
                "DeepSeek V4.1 NPU fused MLA enabled, patched_layers=%d", count
            )

            from .npu import patch_deepseek_v41_npu_engram

            patch_deepseek_v41_npu_engram(
                self,
                sp_group=self._sp_group,
                sp_size=self._sp_size,
                sp_rank=self._sp_rank,
            )

            from .npu import patch_deepseek_v41_npu_sfa_li

            patch_deepseek_v41_npu_sfa_li(self)
            self._npu_sfa_li_patched = True
            self._freeze_indexers()

    def _register_publish_shared_for_fsdp(self):
        try:
            from torch.distributed.fsdp._fully_shard import (
                FSDPModule,
                register_fsdp_forward_method,
            )
        except ImportError:
            from torch.distributed._composable.fsdp.fully_shard import (
                FSDPModule,
                register_fsdp_forward_method,
            )
        for layer in self.layers:
            if not isinstance(layer, FSDPModule):
                continue
            if layer._tt_publish_shared_fsdp_registered:
                continue
            register_fsdp_forward_method(layer, "publish_shared")
            layer._tt_publish_shared_fsdp_registered = True

    def _freeze_indexers(self):
        """Freeze the indexer parameters unless the KL loss is active.

        Top-k indices do not carry grad into the indexer, so they are
        frozen before sharding (kept out of the optimizer);
        on_materialized repeats this after assign-style checkpoint loading.
        The indexer trains only when the fused SFA/LI bundle is installed
        AND a positive KL coefficient is configured
        (config.text_config.indexer_kl_loss_coeff, overridable at build
        time via --tt_indexer_kl_loss_coeff) — with coeff 0 the bundle is
        the pure forward path and the indexer stays frozen.

        Loud by contract: ``_npu_sfa_li_patched`` is initialized in
        _post_init_patches and ``indexer_kl_loss_coeff`` is defined on
        DeepseekV41TextConfig; missing either raises instead of silently
        falling back to a default.
        """
        from .npu import _kl_loss_coeff

        kl_active = self._npu_sfa_li_patched and (
            _kl_loss_coeff(self.config.text_config) > 0
        )
        for module in self.modules():
            if isinstance(module, DeepseekV41Indexer):
                for param in module.parameters():
                    param.requires_grad_(kl_active)

    def on_materialized(self):
        """Build engram runtime state after parameters are on a real device.

        Host-offload tables materialize their owner-local CPU shard on the
        EP group and stream the rows in from the checkpoint; the sibling
        params (wkv/q/k) share the giant table shard files, so they are read
        here too (the converter skips them - see
        DeepseekV41WeightConverter). Table training is opt-in.
        """
        from torchturbo.runtime import get_device_runtime

        device = get_device_runtime().device
        args = get_torchturbo_args()
        train_engram = args is not None and args.train_engram
        if self.engram_hash is not None:
            self.engram_hash.build(device, self._model_path)
        for module in self.modules():
            if isinstance(module, HostOffloadEngram):
                module.materialize()
                if self._model_path is not None:
                    module.embed.load_rows(self._model_path, module.table_key)
                    module.load_params(self._model_path, device)
                else:
                    module.embed.init_rows_random()
                    module.init_params_random(device)
                # PEFT or an explicit freeze may precede host materialization.
                # The training flag must not unfreeze an existing parameter.
                module.embed.weight.requires_grad_(
                    train_engram and module.embed.weight.requires_grad
                )
                if self.engram_hash is not None:
                    module.embed.hash_fingerprint = self.engram_hash.hash_fingerprint
        # Checkpoint assign can replace parameters and restore requires_grad.
        self._freeze_indexers()

    def set_aux_loss_scale(self, scale: float) -> None:
        """Set the backward scale for the V4.1 indexer auxiliary loss."""
        from .npu import V41IndexerLossAutoScaler

        V41IndexerLossAutoScaler.set_loss_scale(scale)

    @classmethod
    @contextmanager
    def init_context(cls, config, engine_config):
        # The public base exposes dtype/context setup through this hook.
        with super().init_context(config, engine_config):
            states = {
                name: cls._force_config_dtype_float32(getattr(config, name))
                for name in ("text_config", "vision_config")
            }
            try:
                yield
            finally:
                for name, state in states.items():
                    cls._restore_config_dtype(getattr(config, name), state)

    def on_prepare_inputs(self, inputs, *, source_inputs=None):
        # Public packed batches carry cumulative lengths; the internal model
        # consumes the same boundaries as a global sequence-id mask.
        prepared = dict(inputs)
        cu = prepared.get("cu_seq_lens_q")
        if cu is not None:
            length = prepared["input_ids"].shape[-1]
            segments = cu_seqlens_to_segment_ids(cu, length)
            prepared["attention_mask"] = segments
            if prepared.get("position_ids") is None:
                positions = segments.new_tensor(range(length)) - cu[(segments - 1).clamp_min(0)]
                prepared["position_ids"] = positions.masked_fill(segments == 0, 0)
        return prepared

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        spec = super().get_sequence_parallel_input_slicing_spec()
        return replace(
            spec,
            sliced_inputs=(*spec.sliced_inputs, ("inputs_embeds", 1)),
            padded_inputs=(*spec.padded_inputs, ("attention_mask", SequencePadding.ZERO)),
        )

    def get_moe_runtime_inputs(
        self, inputs: dict[str, Any], parallel_context: dict[str, Any],
    ) -> "MoERuntimeInputs":
        if inputs.get("cu_seq_lens_q") is not None:
            return super().get_moe_runtime_inputs(inputs, parallel_context)
        from ..modules.attention.ulysses import slice_input_tensor
        from ..modules.moe.moe_layer import MoERuntimeInputs

        mask = inputs.get("attention_mask")
        if mask is None:
            return MoERuntimeInputs()
        padding_mask = ~mask.bool()
        sp = parallel_context["sp"]
        if sp["sp_size"] > 1:
            padding_mask = slice_input_tensor(
                padding_mask, dim=1, group=sp["sp_group"], rank=sp["sp_rank"],
            )
        return MoERuntimeInputs(padding_mask=padding_mask)
