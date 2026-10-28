try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Older versions

import torch

from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.runtime import is_npu_available, get_device_name
from .modeling_kimi_k25 import KimiK25ForConditionalGeneration
from .configuration_kimi_k25 import KimiK25Config
from torchturbo.models.modules.moe.moe_layer import MoELayer

from ..base import TurboVLMModel
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from transformers import AutoModelForCausalLM


class KimiK25WeightConverter(BaseWeightConverter):
    """Weight converter for Kimi-K2.5.

    Kimi-K2.5 is a VLM whose real transformer dims live under
    ``config.text_config`` (DeepseekV3Config), not on the top-level
    ``KimiK25Config`` wrapper. BaseWeightConverter reads ``hidden_size`` /
    ``moe_intermediate_size`` and ``model_type`` off ``self.config``,
    so we re-point it at ``text_config``.

    The language model is DeepSeek-V3 with per-expert checkpoint format
    (``language_model.model.layers.N.mlp.experts.{i}.gate_proj/up_proj/
    down_proj.weight``), so the BaseWeightConverter defaults for the
    expert (``gate_proj``/``up_proj``/``down_proj``) and router
    (``router`` -> ``gate``, ``gate.e_score_correction_bias``) conversions
    already match; no name overrides are needed.
    """

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type


@register_model("kimi_k25", config_cls=KimiK25Config, auto_class=AutoModelForCausalLM)
class TurboKimiK25ForConditionalGeneration(
    TurboVLMModel, KimiK25ForConditionalGeneration
):
    multimodal_input_keys: tuple[str, ...] = ("pixel_values", "grid_thws")

    def get_weight_converter(self, parallel_context, model_path):
        # Transformers 5.x creates `all_tied_weights_keys` unconditionally in
        # PreTrainedModel.post_init(); 4.x never does. Seed an empty mapping
        # when it is missing so BaseWeightConverter's strict
        # dict(model.all_tied_weights_keys) read does not crash if the
        # environment runs an older transformers than lf_deps.sh pins (5.5.4).
        # On 5.x this is a no-op: the attribute already exists.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return KimiK25WeightConverter(self, parallel_context, model_path)

    @override
    def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
        if optimizer_name == "muon":
            from .optimizer_recipe import get_muon_recipe

            return get_muon_recipe(self)
        return super().get_optimizer_recipe(optimizer_name)

    def get_norm(self):
        # Final text-model RMSNorm is nested at ``language_model.model.norm``;
        # the base default ``<base_model>.norm`` does not find it. Expose it so
        # the FSDP2 tail optimizer merges it into the lm_head resident unit
        # instead of the auto root unit (whose sole managed norm breaks CPU
        # offload — ``_validate_cpu_offload_params`` fails at the first forward).
        return self.language_model.model.norm

    def get_lm_head(self):
        # kimi wraps the text model in ``language_model`` (a ForCausalLM); the
        # lm_head lives there, not at the top level. Override the base default
        # (``self.lm_head``) so the tail unit / output-embedding resolve correctly.
        return self.language_model.lm_head

    def __init__(self, config: KimiK25Config) -> None:
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
        from . import modeling_deepseek as ds
        from torchturbo.ops import ops

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        ds.DeepseekV3RMSNorm.forward = rms_norm_forward

        if is_npu_available():
            from .npu import apply_rotary_pos_emb_kimi_moe

            ds.apply_rotary_pos_emb = apply_rotary_pos_emb_kimi_moe

        # patch moe_layer forward
        origin_moe_forward = MoELayer.forward

        def deepseek_moe_forward(self, hidden_states: torch.Tensor):
            output, _ = origin_moe_forward(self, hidden_states)
            return output

        MoELayer.forward = deepseek_moe_forward

    def _post_init_patches(self):
        pass

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        cls = type(self.language_model)
        return SequenceParallelInputSlicingSpec(
            self.language_model,
            cls.forward,
        )

    def on_materialized(self):
        """Initialize runtime buffers after weights are on real device."""

        device = get_device_name()
        self._register_buffer_for_kimi_k25(device)

    def _register_buffer_for_kimi_k25(self, device):
        # deal with vision tower
        self._register_buffer_for_kimi_k25_vision(self.vision_tower, device)
        # deal with language_model
        self._register_buffer_for_kimi_k2(self.language_model, device)

    def _register_buffer_for_kimi_k2(self, model, device):
        for layer in model.model.layers:
            attn_layer = layer.self_attn
            rotary_emb = attn_layer.rotary_emb

            init_kwargs = {
                "dim": rotary_emb.dim,
                "max_position_embeddings": rotary_emb.max_position_embeddings,
                "base": rotary_emb.base,
                "device": device,
            }

            if hasattr(rotary_emb, "scaling_factor"):
                init_kwargs["scaling_factor"] = rotary_emb.scaling_factor
            if hasattr(rotary_emb, "original_max_position_embeddings"):
                init_kwargs["original_max_position_embeddings"] = (
                    rotary_emb.original_max_position_embeddings
                )
            if hasattr(rotary_emb, "beta_fast"):
                init_kwargs["beta_fast"] = rotary_emb.beta_fast
            if hasattr(rotary_emb, "beta_slow"):
                init_kwargs["beta_slow"] = rotary_emb.beta_slow
            if hasattr(rotary_emb, "mscale"):
                init_kwargs["mscale"] = rotary_emb.mscale
            if hasattr(rotary_emb, "mscale_all_dim"):
                init_kwargs["mscale_all_dim"] = rotary_emb.mscale_all_dim

            rotary_emb.__init__(**init_kwargs)

    def _register_buffer_for_kimi_k25_vision(self, vision_tower, device):
        from .modeling_kimi_k25 import get_1d_sincos_pos_embed

        pos_emb = vision_tower.patch_embed.pos_emb
        time_weight = (
            torch.from_numpy(get_1d_sincos_pos_embed(pos_emb.dim, pos_emb.num_frames))
            .float()
            .unsqueeze(1)
        )
        pos_emb.register_buffer("time_weight", time_weight, persistent=False)

    @override
    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        flat_image_features = torch.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        ignore_index: int = self.config.ignore_index

        mask = input_ids == image_token_index
        num_placeholders = mask.sum().item()
        num_features = flat_image_features.shape[0]

        if num_placeholders != num_features:
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {num_placeholders} while"
                f" the number of image features given to the model is {num_features}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding = inputs_embeds.clone()
        final_embedding[mask] = flat_image_features.to(
            final_embedding.device, dtype=final_embedding.dtype
        )

        final_labels = labels
        if labels is not None:
            final_labels = labels.clone()
            final_labels[mask] = ignore_index

        return final_embedding, attention_mask, final_labels
