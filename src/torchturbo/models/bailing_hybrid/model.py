from typing import Any

import torch
from typing_extensions import override

from transformers import AutoModelForCausalLM

from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.runtime import get_device_runtime

from ..registry import register_model
from ..base import TurboModel
from ..weight_converter import BaseWeightConverter
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from .modeling_bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
    MoEV3CausalLMOutputWithPast,
    roll_tensor,
)

class BailingV3WeightConverter(BaseWeightConverter):
    @override
    def get_hf_gate_bias_name(self):
        return "gate.expert_bias"


@register_model(
    "bailing_hybrid", config_cls=BailingMoeV3Config, auto_class=AutoModelForCausalLM
)
class TurboBailingMoeV3ForCausalLM(TurboModel, BailingMoeV3ForCausalLM):
    config_class = BailingMoeV3Config
    base_model_prefix = "model"

    @override
    def get_weight_converter(self, parallel_context, model_path):
        return BailingV3WeightConverter(self, parallel_context, model_path)

    @override
    def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
        if optimizer_name == "muon":
            from .optimizer_recipe import get_muon_recipe

            return get_muon_recipe(self)
        return super().get_optimizer_recipe(optimizer_name)

    def get_flops_class(self):
        """Bailing v3 flops recipe (hybrid MLA+KDA attention, dense+MoE FFN, MTP)."""
        from .mfu import BailingMoeV3Flops

        return BailingMoeV3Flops

    def __init__(self, config: BailingMoeV3Config) -> None:
        self._pre_init_patches(config)
        super().__init__(config)

    @override
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> tuple | MoEV3CausalLMOutputWithPast:
        """Run Bailing forward with selected positions projected by the LM head."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None
        all_mtp_loss = None
        aux_loss = None
        hidden_states = outputs[0]
        # Select prediction positions before the full-vocabulary projection.
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)

        all_mtp_logits = None
        if self.num_nextn_predict_layers > 0:
            mtp_hidden_states = outputs.mtp_hidden_states
            shift_labels_mtp = None
            for i in range(self.num_nextn_predict_layers):
                mtp_hidden_states = mtp_hidden_states[i]
                mtp_logits = self.lm_head(mtp_hidden_states[:, slice_indices, :]).float()
                if all_mtp_logits is None:
                    all_mtp_logits = []
                all_mtp_logits.append(mtp_logits)
                if labels is not None:
                    if shift_labels_mtp is None:
                        shift_labels_mtp = labels.clone()
                    shift_labels_mtp, _ = roll_tensor(shift_labels_mtp, shifts=-1, dims=-1, fill_value=-100)
                    mtp_logits_ = mtp_logits.view(-1, self.config.vocab_size)
                    mtp_loss = self.loss_function(
                        mtp_logits_, shift_labels_mtp.to(mtp_logits_.device).view(-1), self.config.vocab_size, **kwargs
                    )
                    if loss is not None:
                        loss += self.mtp_loss_scaling_factor * mtp_loss
                    else:
                        loss = self.mtp_loss_scaling_factor * mtp_loss

                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoEV3CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=all_mtp_loss,
            aux_loss=aux_loss,
            logits=logits,
            mtp_logits=all_mtp_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        assert sp_size >= 1, "must be"

        if self.config._attn_implementation != "flash_attention_2":
            raise NotImplementedError(
                "BailingMoeV3 sequence parallelism requires flash_attention_2 "
                "for MLA layers. Pass `--flash_attn fa2` to enable it."
            )

        # MLA layers dispatch through the registered turbo_ulysses interface;
        # KDA layers rewrite their forward instead.
        self.register_sp_attention_interface(sp_info)
        self.patch_kda_forward(sp_group, sp_size, sp_rank)

    @classmethod
    def patch_kda_forward(cls, sp_group, sp_size, sp_rank):
        from .attention import patch_kda_forward as install_kda_forward
        from .modeling_bailing_moe_v3 import BailingMoeV3KimiDeltaAttention

        install_kda_forward(BailingMoeV3KimiDeltaAttention, sp_group, sp_size, sp_rank)

    @staticmethod
    def _pre_init_patches(config):
        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model
        from torchturbo.ops import ops

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        bailing_moe_v3_model.BailingMoeV3RMSNorm.forward = rms_norm_forward

    def on_materialized(self):
        """Initialize runtime buffers after weights are on the real device."""

        device = get_device_runtime().device
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config, device
        )
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.original_inv_freq = inv_freq
