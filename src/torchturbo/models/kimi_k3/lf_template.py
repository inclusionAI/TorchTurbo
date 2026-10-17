"""LlamaFactory template + plugin registration for Kimi K3.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here — the model directory stays
clean if LF isn't installed.
"""
from __future__ import annotations

from ...logging import get_logger

logger = get_logger(__name__)


def register() -> None:
    """Register the kimi_k3 mm_plugin and template with LlamaFactory."""
    from llamafactory.data.mm_plugin import register_mm_plugin, get_mm_plugin, PLUGINS
    from llamafactory.data.template import register_template, TEMPLATES
    from llamafactory.data.formatter import StringFormatter

    from .mm_plugin import KimiK3Plugin

    # 1. Register the mm_plugin
    if "kimi_k3" not in PLUGINS:
        register_mm_plugin("kimi_k3", KimiK3Plugin)
        logger.debug("Registered LF mm_plugin 'kimi_k3'")

    # 2. Register the template
    if "kimi_k3" not in TEMPLATES:
        register_template(
            name="kimi_k3",
            format_user=StringFormatter(
                slots=["<|im_start|>user\n{{content}}<|im_end|>\n"]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(
                slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
            ),
            default_system="You are a helpful assistant.",
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=get_mm_plugin(
                "kimi_k3", image_token="<|kimi_image_placeholder|>"
            ),
        )
        logger.debug("Registered LF template 'kimi_k3'")