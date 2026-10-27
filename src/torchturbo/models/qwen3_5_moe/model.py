from typing import Any

import torch
from transformers import AutoModelForImageTextToText

from torchturbo.engine.metrics.flops import BaseModelFlops
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)
from torchturbo.runtime import get_device_runtime

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..modules.attention.gdn import build_gdn_sp_forward
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..qwen3_5.inputs import prepare_packed_position_ids
from ..registry import register_model
from ..weight_converter import FusedExpertWeightConverter


class Qwen3_5MoeWeightConverter(FusedExpertWeightConverter):

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type


@register_model(
    "qwen3_5_moe",
    config_cls=Qwen3_5MoeConfig,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5MoeForConditionalGeneration(
    TurboVLMModel, Qwen3_5MoeForConditionalGeneration
):

    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    @property
    def visual(self):
        # swift's qwen _post_encode (forward pre-hook on this outer model) calls
        # ``model.visual``; the tower lives on the inner Qwen3_5MoeModel.
        return self.model.visual

    def get_norm(self):
        # Final text-model RMSNorm lives on the inner language model (nested
        # below base_model); the base default ``<base_model>.norm`` does not find
        # it. Expose it so the FSDP2 tail optimizer merges it into the lm_head
        # resident unit instead of the auto root unit (the root unit's lone norm
        # breaks CPU offload).
        return self.model.language_model.norm

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, param)
            for name, param in self.model.visual.named_parameters(prefix="model.visual")
            if name != "model.visual.merger"
            and not name.startswith("model.visual.merger.")
        )
        return groups

    def get_flops_class(self) -> type[BaseModelFlops]:
        """Qwen3.5-MoE flops recipe (gated GQA + GatedDeltaNet + MoE FFN + vision)."""
        from .mfu import Qwen3_5MoeFlops

        return Qwen3_5MoeFlops

    def __init__(self, config: Qwen3_5MoeConfig) -> None:
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def _post_init_patches(self):
        import functools
        import itertools

        import torch

        # Wrap get_rope_index:
        # 1. auto-fill mm_token_type_ids when not provided (the open-source LF
        #    collator never passes this arg);
        # 2. run a mask-aware variant whenever a mask is given. Mask values
        #    label blocks (1, 2, 3, ... per packed sub-sequence, 0 = pad), so
        #    positions restart at every block, and image placeholders cut by
        #    truncation keep aligned positions instead of failing the upstream
        #    shape check. The LlamaFactory collator passes binary masks, so
        #    its calls take this path too. This keeps modeling_qwen3_5_moe.py
        #    unchanged (upstream-faithful).
        original_get_rope_index = self.model.get_rope_index
        get_vision_position_ids = self.model.get_vision_position_ids
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)

        def _build_mm_token_type_ids(input_ids):
            mm_token_type_ids = torch.zeros_like(input_ids)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
            return mm_token_type_ids

        @functools.wraps(original_get_rope_index)
        def _wrapped_get_rope_index(
            input_ids,
            mm_token_type_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            **kwargs,
        ):
            # ``mm_token_type_ids`` is a *positional* parameter of the upstream
            # ``get_rope_index`` (right after ``input_ids``), so a caller may
            # pass it (and ``input_ids``) positionally OR by keyword — and the
            # open-source LF collator never passes ``mm_token_type_ids`` at all.
            # Bind these to real parameters here instead of ``*args`` /
            # ``**kwargs``: reading them from ``kwargs`` breaks when they're
            # passed positionally (which the signature invites — caller 2 in
            # modeling_qwen3_5_moe.py passes ``input_ids`` positionally, so the
            # old ``kwargs["input_ids"]`` would KeyError).
            if mm_token_type_ids is None:
                mm_token_type_ids = _build_mm_token_type_ids(input_ids)

            # Only mask-less calls keep the upstream implementation.
            if attention_mask is None or attention_mask.numel() == 0:
                return original_get_rope_index(
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            grid_iters = {
                1: iter(image_grid_thw) if image_grid_thw is not None else None,
                2: iter(video_grid_thw) if video_grid_thw is not None else None,
            }

            mrope_position_deltas = []
            for batch_idx in range(input_ids.shape[0]):
                row_mask = attention_mask[batch_idx]
                input_token_type = mm_token_type_ids[batch_idx]

                # Valid (non-padding) tokens and their packed-block labels.
                nonzero_idx = row_mask != 0
                token_types = input_token_type[nonzero_idx].tolist()
                block_labels = row_mask[nonzero_idx].tolist()

                # Group consecutive tokens by (block_label, modality): a change in
                # EITHER the packed-block label OR the modality starts a new group,
                # so two images in adjacent packed sequences consume two separate
                # grid_thw entries and each block restarts positions from 0.
                input_type_group = []
                for (block_label, modality_type), group in itertools.groupby(
                    range(len(token_types)),
                    key=lambda i: (block_labels[i], token_types[i]),
                ):
                    grp = list(group)
                    input_type_group.append((modality_type, grp[0], grp[-1] + 1, block_label))

                current_pos = 0
                prev_block = None
                llm_pos_ids_list = []
                for modality_type, start_idx, end_idx, block_label in input_type_group:
                    # Reset position at each packed-sequence boundary.
                    if block_label != prev_block:
                        current_pos = 0
                        prev_block = block_label
                    # text == 0
                    if modality_type == 0:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(
                            torch.arange(text_len, device=input_ids.device)
                            .view(1, -1)
                            .expand(3, -1)
                            + current_pos
                        )
                        current_pos += text_len
                    # image == 1, video == 2
                    else:
                        grid_thw = next(grid_iters[modality_type])
                        vision_position_ids = get_vision_position_ids(
                            current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                        )
                        # Packing can truncate the image placeholder span at a block
                        # boundary while image_grid_thw still describes the full image.
                        # Surviving tokens are the patch prefix -> align positions to them.
                        placeholder_len = end_idx - start_idx
                        grid_len = int(vision_position_ids.shape[1])
                        if grid_len != placeholder_len:
                            if grid_len > placeholder_len:
                                vision_position_ids = vision_position_ids[:, :placeholder_len]
                            else:
                                last = vision_position_ids[:, -1:].expand(3, placeholder_len - grid_len)
                                vision_position_ids = torch.cat([vision_position_ids, last], dim=1)
                            import warnings as _warnings

                            _warnings.warn(
                                f"[qwen3_5_mrope packing] batch_idx={batch_idx} group "
                                f"modality={'image' if modality_type == 1 else 'video'}: start_idx {start_idx},  end_idx {end_idx} "
                                f"placeholder_len={placeholder_len} != grid_len={grid_len} "
                                f"(grid_thw={grid_thw.tolist()}, spatial_merge_size="
                                f"{spatial_merge_size}). Truncated/aligned to {placeholder_len}; "
                                f"neat packing likely cut image tokens at a block boundary.",
                                stacklevel=2,
                            )
                        llm_pos_ids_list.append(vision_position_ids)
                        current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

                if llm_pos_ids_list:
                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[:, batch_idx, nonzero_idx] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(int(llm_positions.max()) + 1 - len(token_types))
                else:
                    # All-padding row: nothing to assign, delta is 0.
                    mrope_position_deltas.append(0)

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas

        self.model.get_rope_index = _wrapped_get_rope_index


    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        from .modeling_qwen3_5_moe import Qwen3_5MoeTextModel

        return SequenceParallelInputSlicingSpec(
            self.model.language_model,
            Qwen3_5MoeTextModel.forward,
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
            # Vision keeps flash_attention_2 on its own subconfig and stays
            # on the native flash path; only text attention switches to the
            # registered SP interface.
            self.register_sp_attention_interface(sp_info)

            if self.mtp is not None:
                self.patch_mtp_for_sp(
                    Qwen3_5MoeMultiTokenPredictor, sp_group, sp_size, sp_rank
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
        return Qwen3_5MoeWeightConverter(self, parallel_context, model_path)

    def on_materialized(self):
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
                ** (
                    torch.arange(0, vis_rope.dim, 2, dtype=torch.float)
                    / vis_rope.dim
                )
            ).to(device)
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
