"""TorchTurbo registration and lifecycle hooks for GLM-5.3-Flash."""

import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .configuration_glm5_3_flash import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from .modeling_glm5_3_flash import (
    Glm5NextForConditionalGeneration,
    Glm5NextTextLinearAttention,
    Glm5NextTextAttention,
)


def dequantize_block_fp8(
    weight, scale_inv, block_size=(128, 128), dtype=torch.bfloat16
):
    """Decode one matrix on CPU, keeping FP32 scratch space to one row block."""
    if weight.ndim != 2:
        raise ValueError("Block FP8 weights must be matrices")
    rows, cols = weight.shape
    br, bc = block_size
    if (
        br < 1
        or bc < 1
        or tuple(scale_inv.shape) != ((rows + br - 1) // br, (cols + bc - 1) // bc)
    ):
        raise ValueError(
            f"Invalid FP8 scale shape {scale_inv.shape} for weight {weight.shape}"
        )
    result = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    for row in range(0, rows, br):
        scale = scale_inv[row // br].float().repeat_interleave(bc)[:cols]
        result[row : row + br] = (weight[row : row + br].float() * scale).to(dtype)
    return result


class Glm5NextWeightConverter(BaseWeightConverter):
    def __init__(self, model, parallel_context):
        super().__init__(model, parallel_context)
        self.source_config = model.config
        self.config = model.config.text_config
        self.dtype = self.get_save_dtype(self.config)
        quant = getattr(model.config, "quantization_config", {}) or {}
        if hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        self.block_size = quant.get("weight_block_size", (128, 128))

    def get_save_dtype(self, config):
        config = getattr(config, "text_config", config)
        dtype = (
            getattr(config, "dtype", None)
            or getattr(config, "torch_dtype", None)
            or torch.bfloat16
        )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        return dtype

    def get_default_hf_dtype(self, name):
        if any(
            key in name
            for key in (
                ".hc_",
                ".A_log",
                ".dt_bias",
                "_conv1d.weight",
                ".e_score_correction_bias",
                ".expert_bias",
            )
        ):
            return torch.float32
        return self.dtype

    def is_persist_buffer(self, name):
        return super().is_persist_buffer(name) or name.endswith(
            ".gate.e_score_correction_bias"
        )

    def decode_checkpoint_tensor(self, name, tensor, read_tensor):
        """Called by the distributed loader on the reading rank, before broadcast."""
        if str(tensor.dtype).startswith("torch.float8_"):
            if not name.endswith(".weight"):
                raise ValueError(f"Unsupported non-weight FP8 tensor: {name}")
            scale = read_tensor(name.removesuffix(".weight") + ".weight_scale_inv")
            return dequantize_block_fp8(
                tensor, scale, self.block_size, self.get_default_hf_dtype(name)
            )
        return tensor.to(dtype=self.get_default_hf_dtype(name), copy=True)

    def update_saved_config(self, save_path):
        # The saver copies source metadata; the saved weights are BF16/FP32 now.
        path = save_path / "config.json"
        config = (
            json.loads(path.read_text())
            if path.exists()
            else self.source_config.to_dict()
        )
        config.pop("quantization_config", None)
        if "text_config" in config:
            config["text_config"].pop("quantization_config", None)
            config["text_config"]["num_nextn_predict_layers"] = 0
        config["architectures"] = ["Glm5NextForConditionalGeneration"]
        path.write_text(json.dumps(config, indent=2) + "\n")


@register_model("glm5_3_flash")
@register_model("glm5_next", config_cls=Glm5NextConfig, auto_class=AutoModelForCausalLM)
class TurboGlm5NextForConditionalGeneration(
    TurboVLMModel, Glm5NextForConditionalGeneration
):
    vision_tower_module_names = ("model.visual",)
    multi_modal_projector_module_names = (
        "model.visual.downsample",
        "model.visual.merger",
    )
    language_model_module_names = ("model.language_model", "lm_head")

    @property
    def visual(self):
        return self.model.visual

    def get_norm(self):
        return self.model.language_model.norm

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, parameter)
            for name, parameter in self.model.visual.named_parameters(
                prefix="model.visual"
            )
            if not any(
                name.startswith(prefix + ".")
                for prefix in self.multi_modal_projector_module_names
            )
        )
        return groups

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context):
        return Glm5NextWeightConverter(self, parallel_context)

    def on_parallelized(self, parallel_context):
        info = parallel_context["sp"]
        language = self.model.language_model
        language.sp_size, language.sp_rank = info["sp_size"], info["sp_rank"]
        for module in language.modules():
            if isinstance(module, (Glm5NextTextLinearAttention, Glm5NextTextAttention)):
                module.sp_group = info["sp_group"]
                module.sp_size, module.sp_rank = info["sp_size"], info["sp_rank"]

        from ...runtime import is_npu_available

        if is_npu_available():
            from .npu import patch_model

            patch_model(self)


# Configs and the stable AutoClasses are sufficient; no transformers.models.glm5_next import.
AutoConfig.register("glm5_next_text", Glm5NextTextConfig, exist_ok=True)
AutoConfig.register("glm5_next_vision", Glm5NextVisionConfig, exist_ok=True)
try:
    from transformers import Glm5NextProcessor
except ImportError:
    from .processing_glm5_3_flash import Glm5NextProcessor
AutoProcessor.register(Glm5NextConfig, Glm5NextProcessor, exist_ok=True)
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
AutoModelForImageTextToText.register(
    Glm5NextConfig, TurboGlm5NextForConditionalGeneration, exist_ok=True
)
