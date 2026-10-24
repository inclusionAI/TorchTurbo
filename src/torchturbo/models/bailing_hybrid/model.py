from typing_extensions import override
from torch.distributed import ProcessGroup

from transformers import AutoModelForCausalLM

from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.runtime import get_device_runtime

from ..registry import register_model
from ..base import TurboModel
from ..weight_converter import BaseWeightConverter
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM

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
        self.patch_kda_forward(sp_group, sp_size, sp_rank, sp_info["use_fla_kcp"])

    @classmethod
    def patch_kda_forward(
        cls,
        sp_group: ProcessGroup | None,
        sp_size: int,
        sp_rank: int,
        use_fla_kcp: bool = False,
    ) -> None:
        from .attention import patch_kda_forward as install_kda_forward
        from .modeling_bailing_moe_v3 import BailingMoeV3KimiDeltaAttention

        install_kda_forward(
            BailingMoeV3KimiDeltaAttention, sp_group, sp_size, sp_rank, use_fla_kcp
        )

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
