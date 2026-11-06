"""GLM-5.3-Flash image/text SFT with the GLM5 conversation format."""

from dataclasses import replace


def register():
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES

    from ..glm_moe_dsa.lf_template import register as register_glm5

    register_glm5()
    if "glm5_3_flash" not in TEMPLATES:
        # Keep the checkpoint's GLM5 formatting and reasoning behavior. Its
        # vision tokens and patch layout use the existing GLM-4V plugin.
        TEMPLATES["glm5_3_flash"] = replace(
            TEMPLATES["glm5"],
            mm_plugin=get_mm_plugin(
                name="glm4v", image_token="<|image|>", video_token="<|video|>"
            ),
        )
