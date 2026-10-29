from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM

from torchturbo.engine.metrics.flops import BaseModelFlops
from torchturbo.models.qwen3_5_moe.model import (
    Qwen3_5MoeWeightConverter,
    TurboQwen3_5MoeForConditionalGeneration as _qwen3_5_moe_vlm,
)
from torchturbo.runtime import get_device_runtime

from ..base import TurboModel
from ..modules.attention.gdn import build_gdn_sp_forward
from ..registry import register_model
from .configuration_qwen3_5_moe_text import Qwen3_5MoeTextConfig
from .modeling_qwen3_5_moe_text import Qwen3_5MoeTextForCausalLM


@register_model(
    "qwen3_5_moe_text", config_cls=Qwen3_5MoeTextConfig, auto_class=AutoModelForCausalLM
)
class TurboQwen3_5MoeTextForCausalLM(TurboModel, Qwen3_5MoeTextForCausalLM):
    config_class = Qwen3_5MoeTextConfig
    base_model_prefix = "model"

    patch_mtp_for_sp = _qwen3_5_moe_vlm.patch_mtp_for_sp

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context, model_path):
        return Qwen3_5MoeWeightConverter(self, parallel_context, model_path)

    def get_flops_class(self) -> type[BaseModelFlops]:
        from ..qwen3_5_moe.mfu import Qwen3_5MoeFlops

        return Qwen3_5MoeFlops

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeGatedDeltaNet,
            Qwen3_5MoeMultiTokenPredictor,
        )

        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        Qwen3_5MoeGatedDeltaNet.forward = build_gdn_sp_forward(
            sp_group, sp_size, sp_rank
        )

        if sp_size > 1:
            self.register_sp_attention_interface(sp_info)

            if self.mtp is not None:
                self.patch_mtp_for_sp(
                    Qwen3_5MoeMultiTokenPredictor, sp_group, sp_size, sp_rank
                )

    def on_materialized(self):
        device = get_device_runtime().device
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        rotary_emb = self.model.rotary_emb
        rope_init_fn = rotary_emb.compute_default_rope_parameters
        if rotary_emb.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )
