"""LLamaFactory mm_plugin for Kimi-K3.

Handles image data: process_messages (expand placeholder) + get_mm_inputs (preprocess image).
K3 uses <|kimi_image_placeholder|> as image token, MoonViT-V2 as vision tower.
"""
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
from typing_extensions import override

from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from llamafactory.data.mm_plugin import BasePlugin
from llamafactory.extras.packages import (
    is_pillow_available,
    is_transformers_version_greater_than,
)

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer, ProcessorMixin

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]


class KimiK3Plugin(BasePlugin):
    """mm_plugin for Kimi-K3 VLM.

    K3 image token: <|kimi_image_placeholder|>
    Vision tower: MoonViT-V2 (from scratch, 401M)
    Processor: kimi_k3_processor.py (official)
    """
    image_token: str = "<|kimi_image_placeholder|>"
    video_token: str = "<|kimi_image_placeholder|>"

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list,
        processor: "ProcessorMixin",
        **kwargs,
    ):
        """Expand image placeholders in messages.

        Replace IMAGE_PLACEHOLDER with K3 image_token for each image.
        """
        images = images or []
        num_images = len(images)
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        item["text"] = self.image_token * self._get_image_token_count(
                            processor, **kwargs
                        )
            elif isinstance(content, str):
                # Replace placeholder tokens
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)
                message["content"] = content
        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list,
        videos: list,
        audios: list,
        tokenizer: "PreTrainedTokenizer",
        processor: "ProcessorMixin",
        **kwargs,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Keep token ids unchanged on the text-only path."""
        return input_ids, labels

    def _get_image_token_count(self, processor, **kwargs) -> int:
        """Get the number of image tokens per image.

        K3 uses MoonViT-V2 with PatchMergerMLPV2 projector.
        The exact token count depends on image resolution and patch config.
        Default: 1 (will be replaced by actual token count from processor).
        """
        return 1  # TODO: get actual token count from kimi_k3_processor

    @override
    def get_mm_inputs(
        self,
        images: list,
        videos: list,
        audios: list,
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: "ProcessorMixin",
        **kwargs,
    ):
        """Forward only structured K3 media to the official processor.

        The official kimi_k3 image_processor consumes Moonshot's structured
        media dicts (``media['type']``), not bare PIL objects. LlamaFactory
        0.9.5 injects dummy PIL images into text-only batches; forwarding
        those would crash the processor's ``ensure_media_type``. Skip media
        that is not in K3's protocol; text-only batches yield no mm inputs.
        """
        structured = [m for m in (images or []) if isinstance(m, dict)]
        if not structured:
            return {}
        return super().get_mm_inputs(
            structured,
            videos,
            audios,
            imglens,
            vidlens,
            audlens,
            batch_ids,
            processor,
            **kwargs,
        )
