from __future__ import annotations

import math
from functools import partial
from types import MethodType
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoModelForImageTextToText
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from torchturbo.engine.config import ParameterGroup
from torchturbo.ops import ops
from torchturbo.runtime import get_device_runtime

from ..base import TurboVLMModel
from ..modules.attention.gdn import build_gdn_sp_forward
from ..modules.attention.ulysses import (
    SequencePadding,
    SequenceParallelInputSlicingSpec,
)
from ..registry import register_model
from ..weight_converter import FusedExpertWeightConverter
from .configuration_qwen4_exp import Qwen4ExpConfig
from .modeling_qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpPreTrainedModel,
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpVisionModel,
    Qwen4ExpVisionRotaryEmbedding,
)
from .ple import (
    NGramEmbedding,
    install_ple_prefetch,
    ple_forward,
)
from .qsa import qsa_forward


def _group_rms_norm(
    module: Qwen4ExpTextRMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Normalize each group while leaving learned FP32 scaling to forward."""
    rows = hidden_states.reshape(-1, module.group_size)
    weight = hidden_states.new_ones(module.group_size)
    return ops.rms_norm(rows, weight, module.eps).reshape_as(hidden_states)


def _filter_vision_packing_kwargs(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Keep text packing lengths out of the vision tower's own varlen call."""
    for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        kwargs.pop(name, None)
    return args, kwargs


class Qwen4ExpWeightConverter(FusedExpertWeightConverter):
    """Convert fused Qwen4 expert checkpoint tensors for expert parallelism."""

    _PLE_BUFFER_SUFFIXES = (
        ".ple.ple_embedding.layer_multipliers",
        ".ple.ple_embedding.ngram_heads_vocab_sizes",
        ".ple.ple_embedding.ngram_heads_offsets",
    )

    def __init__(
        self,
        model: nn.Module,
        parallel_context: dict[str, Any],
        model_path: str,
    ) -> None:
        super().__init__(model, parallel_context, model_path)
        self.config = model.config.text_config
        self.model_type = self.config.model_type

    def _ngram_checkpoint_entries(
        self, name: str
    ) -> tuple[list[str], dict[str, int]] | None:
        if not name.endswith("ngram_embedding.weight"):
            return None
        infos = self.checkpoint_info.get_tensor_infos()
        hf_name = self.to_hf_name(name)
        if hf_name in infos:
            keys = [hf_name]
        else:
            prefix = hf_name.removesuffix(".weight")
            keys = [
                f"{prefix}.shard_{index}.weight"
                for index in range(self.config.split_ngram_parts)
            ]
        sizes = {}
        for key in keys:
            shape, dtype = infos[key]
            sizes[key] = (
                math.prod(shape)
                * torch.empty((), dtype=dtype, device="meta").element_size()
            )
        return keys, sizes

    def get_param_load_entries(
        self, name: str, numel: int, param_to_filename: dict[str, str] | None = None
    ) -> tuple[list[str], Any, dict[str, int]]:
        entries = self._ngram_checkpoint_entries(name)
        if entries is not None:
            keys, sizes = entries
            return keys, None, sizes
        return super().get_param_load_entries(name, numel, param_to_filename)

    def get_param_save_entries(
        self, name: str, numel: int, param_to_filename: dict[str, str] | None = None
    ) -> tuple[list[str], list[str], dict[str, int]]:
        entries = self._ngram_checkpoint_entries(name)
        if entries is not None:
            keys, sizes = entries
            return keys, [], sizes
        return super().get_param_save_entries(name, numel, param_to_filename)

    def is_persist_buffer(self, name: str) -> bool:
        return super().is_persist_buffer(name) or name.endswith(
            self._PLE_BUFFER_SUFFIXES
        )


@register_model(
    "qwen4_exp",
    config_cls=Qwen4ExpConfig,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen4ExpForConditionalGeneration(
    TurboVLMModel,
    Qwen4ExpForConditionalGeneration,
):
    """Qwen4 experimental VLM integrated with TorchTurbo."""

    config_class = Qwen4ExpConfig
    base_model_prefix = "model"
    vision_tower_module_names = ("model.visual",)
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    def __init__(self, config: Qwen4ExpConfig) -> None:
        AutoModel.register(type(config.text_config), Qwen4ExpTextModel, exist_ok=True)
        AutoModel.register(
            type(config.vision_config), Qwen4ExpVisionModel, exist_ok=True
        )
        # HF rejects flash_attention_2 on every submodel whose class does not
        # advertise support. The vendored dense-mask forward cannot run flash;
        # TorchTurbo's packed QSA forward drives varlen flash itself.
        Qwen4ExpPreTrainedModel._supports_flash_attn = True
        # HF's generic rotary initialization resolves every non-default rope
        # type through this registry, which has no entry for the vision tower's
        # axial RoPE. The runtime buffers are rebuilt in on_materialized, so a
        # native entry from a future Transformers release takes precedence.
        ROPE_INIT_FUNCTIONS.setdefault(
            "axial", Qwen4ExpVisionRotaryEmbedding.compute_axial_rope_parameters
        )
        super().__init__(config)
        self.model.visual.register_forward_pre_hook(
            _filter_vision_packing_kwargs, with_kwargs=True
        )
        for module in self.modules():
            if isinstance(module, Qwen4ExpTextRMSNorm) and module.group_size is not None:
                module._norm = MethodType(_group_rms_norm, module)
        for layer in self.model.language_model.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.forward = MethodType(qsa_forward, layer.self_attn)
            if layer.ple is not None:
                ple = layer.ple
                ple.forward = MethodType(ple_forward, ple)
                ngram = ple.ple_embedding
                embedding = ngram.ngram_embedding
                ngram.ngram_embedding = NGramEmbedding(embedding.weight)
        self._ple_prefetch_hook = install_ple_prefetch(
            self.model.language_model, sp_size=1, sp_rank=0
        )

    @property
    def visual(self) -> nn.Module:
        return self.model.visual

    def get_norm(self) -> nn.Module:
        return self.model.language_model.hyper_connection_mixer.hc_norm

    def get_flops_class(self):
        """Return the Qwen3.8 heterogeneous-MoE MFU recipe."""
        from .mfu import Qwen4ExpFlops

        return Qwen4ExpFlops

    def get_parameter_groups(self) -> dict[ParameterGroup, Any]:
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("model.visual.")
            and not name.startswith("model.visual.merger.")
        )
        return groups

    @classmethod
    def get_default_parallel_plan(cls, **kwargs: Any) -> Any:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(
        self,
        parallel_context: dict[str, Any],
        model_path: str,
    ) -> Qwen4ExpWeightConverter:
        return Qwen4ExpWeightConverter(self, parallel_context, model_path)

    def get_sequence_parallel_input_slicing_spec(
        self,
    ) -> SequenceParallelInputSlicingSpec:
        return SequenceParallelInputSlicingSpec(
            self.model.language_model,
            Qwen4ExpTextModel.forward,
            sliced_inputs=(("inputs_embeds", 1), ("position_ids", -1)),
            padded_inputs=(
                ("input_ids", SequencePadding.ZERO),
                ("mm_token_type_ids", SequencePadding.ZERO),
                ("labels", SequencePadding.IGNORE_INDEX),
                ("position_ids", SequencePadding.POSITION),
            ),
        )

    def on_materialized(self) -> None:
        """Rebuild nonpersistent RoPE buffers after loading."""
        device = get_device_runtime().device

        text_rotary = self.model.language_model.rotary_emb
        rope_init_fn = text_rotary.compute_default_rope_parameters
        if text_rotary.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[text_rotary.rope_type]
        inv_freq, text_rotary.attention_scaling = rope_init_fn(
            text_rotary.config, device
        )
        text_rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        text_rotary.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )

        vision_rotary = self.model.visual.rotary_pos_emb
        vision_inv_freq, vision_rotary.attention_scaling = (
            vision_rotary.compute_axial_rope_parameters(vision_rotary.config, device)
        )
        vision_rotary.register_buffer("inv_freq", vision_inv_freq, persistent=False)
        vision_rotary.register_buffer(
            "original_inv_freq", vision_inv_freq.clone(), persistent=False
        )

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        language_model = self.model.language_model
        self._ple_prefetch_hook.remove()
        self._ple_prefetch_hook = install_ple_prefetch(
            language_model, sp_size=sp_size, sp_rank=sp_rank
        )

        gdn_forward = build_gdn_sp_forward(
            sp_group, sp_size, sp_rank
        )

        for layer in language_model.layers:
            if layer.layer_type == "linear_attention":
                linear_attention = layer.linear_attn
                linear_attention.forward = MethodType(gdn_forward, linear_attention)
            else:
                layer.self_attn.forward = MethodType(
                    partial(
                        qsa_forward,
                        sp_group=sp_group,
                        sp_size=sp_size,
                        sp_rank=sp_rank,
                    ),
                    layer.self_attn,
                )

            if layer.ple is not None:
                layer.ple.forward = MethodType(
                    partial(
                        ple_forward,
                        sp_group=sp_group,
                        sp_size=sp_size,
                        sp_rank=sp_rank,
                    ),
                    layer.ple,
                )
