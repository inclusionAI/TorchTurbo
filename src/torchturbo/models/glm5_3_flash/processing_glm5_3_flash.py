"""GLM-5.3-Flash processor compatibility for Transformers before glm5_next.

The checkpoint's image patches, CLIP normalization and image boundary tokens
use the GLM-4V layout. Convert its nested token-budget configuration to the
pixel-budget configuration understood by the existing GLM-4V processors.
"""

import json

from transformers import AutoTokenizer
from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
from transformers.utils import cached_file


def _processor_kwargs(config):
    config = dict(config)
    config.pop("image_processor_type", None)
    config.pop("video_processor_type", None)
    expand = config.pop("patch_expand_factor", 1)
    if expand != 1:
        raise ValueError("GLM-5.3-Flash compatibility processor requires patch_expand_factor=1")
    min_tokens = config.pop("min_image_tokens", 16)
    max_tokens = config.pop("max_image_tokens", 8000)
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("Invalid GLM-5.3-Flash image token budget")
    # GLM-4V smart_resize budgets include the repeated temporal frames, whereas
    # the checkpoint budgets count the tokens after spatial/temporal merging.
    pixels_per_token = (
        config.get("patch_size", 14) * config.get("merge_size", 2)
    ) ** 2 * config.get("temporal_patch_size", 2)
    config.setdefault("size", {
        "shortest_edge": min_tokens * pixels_per_token,
        "longest_edge": max_tokens * pixels_per_token,
    })
    return config


class Glm5NextProcessor(Glm4vProcessor):
    """Load the checkpoint's nested processor config on Transformers 4.x."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # ProcessorMixin in 4.x drops nested component configs while reading
        # processor_config.json. Read the original metadata before that step.
        hub_kwargs = {key: value for key, value in kwargs.items() if key in (
            "cache_dir", "force_download", "local_files_only", "revision", "token", "subfolder"
        )}
        config_file = cached_file(pretrained_model_name_or_path, "processor_config.json", **hub_kwargs)
        with open(config_file, encoding="utf-8") as stream:
            config = json.load(stream)
        if "image_processor" not in config:
            # Also accept the separate preprocessor files written by 4.x.
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor = Glm4vImageProcessor(**_processor_kwargs(config["image_processor"]))
        video_processor = Glm4vVideoProcessor(**_processor_kwargs(config.get("video_processor", {})))
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
        if return_unused_kwargs:
            return processor, {}
        return processor
