import torch
from torchturbo.engine.metrics.flops import BaseModelFlops
from transformers import AutoModelForImageTextToText
from typing import Any
from typing_extensions import override

from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from torchturbo.runtime import get_device_runtime

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..modules.attention.gdn import build_gdn_sp_forward
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .inputs import prepare_packed_position_ids


class Qwen3_5WeightConverter(BaseWeightConverter):
    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type


@register_model(
    "qwen3_5",
    config_cls=Qwen3_5Config,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5ForConditionalGeneration(
    TurboVLMModel, Qwen3_5ForConditionalGeneration
):

    config_class = Qwen3_5Config
    base_model_prefix = "model"
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    @property
    def visual(self):
        return self.model.visual

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, param)
            for name, param in self.model.visual.named_parameters(prefix="model.visual")
            if name != "model.visual.merger"
            and not name.startswith("model.visual.merger.")
        )
        return groups

    def get_norm(self):
        return self.model.language_model.norm

    def get_flops_class(self) -> type[BaseModelFlops]:
        """Qwen3.5 dense flops recipe (hetero attention + gated-MLP FFN + vision)."""
        from .mfu import Qwen3_5DenseFlops

        return Qwen3_5DenseFlops

    def __init__(self, config: Qwen3_5Config) -> None:
        self._pre_init_patches()
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    @staticmethod
    def _pre_init_patches():
        # No MoE token-dispatcher patching for dense (no experts).
        pass

    def _post_init_patches(self):
        pass

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        from .modeling_qwen3_5 import Qwen3_5TextModel

        return SequenceParallelInputSlicingSpec(
            self.model.language_model,
            Qwen3_5TextModel.forward,
        )

    def on_prepare_inputs(
        self,
        inputs: dict[str, Any],
        *,
        source_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build per-sample mRoPE from the packed row's media grids."""
        inputs = super().on_prepare_inputs(inputs, source_inputs=source_inputs)
        return prepare_packed_position_ids(self.model, inputs)

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            Qwen3_5MultiTokenPredictor,
        )

        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        Qwen3_5GatedDeltaNet.forward = build_gdn_sp_forward(
            sp_group, sp_size, sp_rank
        )

        if sp_size > 1:
            # Vision keeps flash_attention_2 on its own subconfig and stays
            # on the native flash path; only text attention switches to the
            # registered SP interface.
            self.register_sp_attention_interface(sp_info)

            if self.mtp is not None:
                self.patch_mtp_for_sp(
                    Qwen3_5MultiTokenPredictor, sp_group, sp_size, sp_rank
                )

    # ------------------------------------------------------------------
    # MTP: SP input slicing
    # ------------------------------------------------------------------

    @classmethod
    def patch_mtp_for_sp(cls, mtp_cls, sp_group, sp_size, sp_rank):
        """Keep MTP next-token inputs correct under sequence parallelism.

        The VLM path provides global IDs, while the text path provides SP-local
        IDs. The text path gathers local IDs before MTP builds next-token
        embeddings so SP-boundary tokens use the next rank's first ID.
        """
        from ..modules.attention.ulysses import slice_input_tensor

        original_forward = mtp_cls.forward

        def _mtp_sp_forward(self_mtp, input_ids, hidden_states, embed_tokens, **kwargs):
            position_embeddings = kwargs.pop("position_embeddings", None)
            attention_mask = kwargs.pop("attention_mask", None)
            position_ids = kwargs.pop("position_ids", None)
            pad_token_id = int(kwargs.pop("pad_token_id", 0))
            # The backbone is already SP-local. Reconstruct only IDs so MTP
            # rolling sees the next token across an SP boundary.
            local_seq_len = hidden_states.size(1)
            if input_ids.size(1) == local_seq_len:
                input_ids_list = [torch.empty_like(input_ids) for _ in range(sp_size)]
                torch.distributed.all_gather(input_ids_list, input_ids, group=sp_group)
                input_ids = torch.cat(input_ids_list, dim=1)

            def localize(tensor, dim):
                if tensor is None or tensor.size(dim) == local_seq_len:
                    return tensor
                return slice_input_tensor(
                    tensor, dim=dim, group=sp_group, rank=sp_rank
                )

            position_ids = localize(position_ids, -1)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                position_embeddings = (localize(cos, 1), localize(sin, 1))

            def sp_embed_tokens(ids):
                return localize(embed_tokens(ids), 1)

            return original_forward(
                self_mtp,
                input_ids,
                hidden_states,
                sp_embed_tokens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pad_token_id=pad_token_id,
                **kwargs,
            )

        mtp_cls.forward = _mtp_sp_forward

    def get_weight_converter(self, parallel_context, model_path):
        return Qwen3_5WeightConverter(self, parallel_context, model_path)

    def on_materialized(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

        device = get_device_runtime().device
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        lm = self.model.language_model
        rotary_emb = lm.rotary_emb
        rope_init_fn = rotary_emb.compute_default_rope_parameters
        if rotary_emb.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )

        vis = self.model.visual
        if hasattr(vis, "rotary_pos_emb"):
            vis_rope = vis.rotary_pos_emb
            vis_inv_freq = 1.0 / (
                vis_rope.theta
                ** (torch.arange(0, vis_rope.dim, 2, dtype=torch.float) / vis_rope.dim)
            ).to(device)
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
