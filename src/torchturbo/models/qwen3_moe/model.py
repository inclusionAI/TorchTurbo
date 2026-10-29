import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeConfig,
)

from torchturbo.runtime import is_npu_available, get_device_runtime
from ..base import TurboModel
from ..registry import register_model
from transformers import AutoModelForCausalLM


class TurboQwen3MoeRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    """Optional swap-in; inv_freq buffers are rebuilt in on_materialized."""


@register_model("qwen3_moe", config_cls=Qwen3MoeConfig, auto_class=AutoModelForCausalLM)
class TurboQwen3MoeForCausalLM(TurboModel, Qwen3MoeForCausalLM):

    config_class = Qwen3MoeConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3MoeConfig) -> None:
        self._pre_init_patches()
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        """Return the default parallel plan for this model."""
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    @staticmethod
    def _pre_init_patches():
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_model

        if is_npu_available():
            from .npu import apply_rotary_pos_emb_npu, rms_norm_forward_npu

            qwen3_moe_model.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
            qwen3_moe_model.Qwen3MoeRMSNorm.forward = rms_norm_forward_npu

    def _post_init_patches(self):
        pass

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        self.register_sp_attention_interface(sp_info)

    def on_materialized(self):
        """Initialize runtime buffers after weights are on real device."""
        device = get_device_runtime().device
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        # patch to rotary_emb
        rotary_emb = self.model.rotary_emb
        rope_init_fn = rotary_emb.compute_default_rope_parameters
        if rotary_emb.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(
            rotary_emb.config, device
        )
        inv_freq = inv_freq.to(dtype=torch.bfloat16)
        rotary_emb.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )
        rotary_emb.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )
