"""GLM-5.3-Flash image/text SFT with the GLM5 conversation format."""

from dataclasses import replace


def register() -> None:
    """Register text and multimodal templates using LlamaFactory's GLM format."""
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES

    # LlamaFactory 0.9.5 provides the GLM format and history-CoT handling.
    if "glm5_3_flash" not in TEMPLATES:
        TEMPLATES["glm5_3_flash"] = replace(TEMPLATES["glm4_7"])
    if "glm5_3_flash_vl" not in TEMPLATES:
        # Keep the checkpoint's GLM5 formatting and reasoning behavior. Its
        # vision tokens and patch layout use the existing GLM-4V plugin.
        TEMPLATES["glm5_3_flash_vl"] = replace(
            TEMPLATES["glm5_3_flash"],
            mm_plugin=get_mm_plugin(
                name="glm4v", image_token="<|image|>", video_token="<|video|>"
            ),
        )
