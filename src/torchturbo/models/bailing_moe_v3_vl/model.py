"""TorchTurbo lifecycle and parallel input contract for Bailing V3 VL."""

from collections.abc import Iterable, Iterator
import functools
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from torchturbo.engine.config import EngineConfig, ParameterGroup
from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.parallel import FSDPParallelPlan
from torchturbo.runtime import get_device_runtime

from ..bailing_hybrid.attention import (
    patch_kda_forward,
    patch_rms_norm,
)
from ..bailing_hybrid.model import BailingV3WeightConverter
from ..base import TurboVLMModel
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from .configuration_bailing_moe_v3_vl import BailingMoeV3VLConfig
from .modeling_bailing_moe_v3_vl import (
    BailingMoeV3KimiDeltaAttention,
    BailingMoeV3RMSNorm,
    BailingMoeV3VLForConditionalGeneration,
    BailingMoeV3VLModel,
    Qwen3VLMoeVisionRotaryEmbedding,
)


class BailingVLWeightConverter(BailingV3WeightConverter):
    """Convert Bailing experts using the nested language configuration."""

    def __init__(
        self,
        model: nn.Module,
        parallel_context: dict[str, Any],
        model_path: str,
    ) -> None:
        super().__init__(model, parallel_context, model_path)
        self.config = model.config.text_config


@register_model(
    "bailing_moe_v3_vl",
    config_cls=BailingMoeV3VLConfig,
    auto_class=AutoModelForCausalLM,
)
class TurboBailingMoeV3VLForConditionalGeneration(
    TurboVLMModel, BailingMoeV3VLForConditionalGeneration
):
    """Preserve the checkpoint module tree while adding TorchTurbo training."""

    config_class = BailingMoeV3VLConfig
    base_model_prefix = "model"
    vision_tower_module_names = ("model.visual",)
    multi_modal_projector_module_names = ("linear_proj",)

    def __init__(self, config: BailingMoeV3VLConfig) -> None:
        # trust_remote_code may supply the checkpoint's original config class.
        if not isinstance(config, BailingMoeV3VLConfig):
            config = BailingMoeV3VLConfig.from_dict(
                config.to_dict(), attn_implementation=config._attn_implementation
            )
        super().__init__(config)
        self._patch_get_rope_index()

    def _patch_get_rope_index(self) -> None:
        """Install the dispatched mRoPE builder on the inner model.
        """
        original_get_rope_index = self.model.get_rope_index

        @functools.wraps(original_get_rope_index)
        def _wrapped_get_rope_index(
            input_ids,
            mm_token_type_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            segment_ids=None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            from torchturbo.ops import ops

            vl_config = self.model._vl_config
            return ops.bailing_vl_get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                segment_ids,
                image_token_id=vl_config.image_token_id,
                video_token_id=vl_config.video_token_id,
                spatial_merge_size=vl_config.vision_config.spatial_merge_size,
                use_interleaved_frame_timestamp=(
                    vl_config.text_config.use_interleaved_frame_timestamp
                ),
            )

        self.model.get_rope_index = _wrapped_get_rope_index

    @classmethod
    @contextmanager
    def init_context(
        cls, config: BailingMoeV3VLConfig, engine_config: EngineConfig
    ) -> Iterator[None]:
        if not engine_config.mtp.enabled:
            config.text_config.num_nextn_predict_layers = 0
        with super().init_context(config, engine_config):
            yield

    @classmethod
    def get_default_parallel_plan(cls) -> FSDPParallelPlan:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
        if optimizer_name == "muon":
            from .optimizer_recipe import get_muon_recipe

            return get_muon_recipe(self)
        return super().get_optimizer_recipe(optimizer_name)

    def get_parameter_groups(
        self,
    ) -> dict[ParameterGroup, Iterable[tuple[str, nn.Parameter]]]:
        groups = super().get_parameter_groups()
        groups[ParameterGroup.LANGUAGE_MODEL] = (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("lm_head.")
            or (name.startswith("model.") and not name.startswith("model.visual."))
        )
        return groups

    def get_weight_converter(
        self, parallel_context: dict[str, Any], model_path: str
    ) -> BailingVLWeightConverter:
        return BailingVLWeightConverter(self, parallel_context, model_path)

    def get_flops_class(self):
        """Bailing v3 VL flops recipe (text hybrid MLA+KDA + Qwen-style ViT)."""
        from .mfu import BailingMoeV3VLFlops

        return BailingMoeV3VLFlops

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        return SequenceParallelInputSlicingSpec(
            self.model,
            BailingMoeV3VLModel._forward_text,
            forward_name="_forward_text",
            sliced_inputs=(
                ("inputs_embeds", 1),
                ("position_ids", -1),
                ("text_position_ids", -1),
                ("mtp_shifted_input_ids", 1),
            ),
        )

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        text_config = self.config.text_config
        patch_rms_norm(BailingMoeV3RMSNorm)
        patch_kda_forward(
            BailingMoeV3KimiDeltaAttention, sp_group, sp_size, sp_rank
        )
        # The shared interface records QK-clip observations at any SP size;
        # the vision tower keeps its own subconfig attention implementation.
        self.register_sp_attention_interface(sp_info)

        if sp_size > 1:
            if text_config.num_attention_heads % sp_size:
                raise ValueError(
                    "Bailing VL attention heads must be divisible by sp_size."
                )

    def on_materialized(self) -> None:
        device = get_device_runtime().device
        self.model.rotary_emb._ensure_inv_freq_initialized(device)
        visual_rotary = self.model.visual.rotary_pos_emb
        head_dim = (
            self.config.vision_config.hidden_size // self.config.vision_config.num_heads
        )
        with torch.device(device):
            initialized = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)
        visual_rotary.register_buffer(
            "inv_freq", initialized.inv_freq, persistent=False
        )
