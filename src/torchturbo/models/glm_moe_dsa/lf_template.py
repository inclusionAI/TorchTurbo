"""LlamaFactory template for GLM5.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here — the model directory stays
clean if LF isn't installed.
"""

from __future__ import annotations

from llamafactory.data.template import Glm47ReasoningTemplate

from ...logging import get_logger


logger = get_logger(__name__)


def register() -> None:
    """Register the GLM5 template with LlamaFactory."""
    from llamafactory.data.template import (
        register_template,
        TEMPLATES,
    )
    from llamafactory.data.formatter import StringFormatter
    from llamafactory.data.formatter import FunctionFormatter
    from llamafactory.data.formatter import ToolFormatter
    from llamafactory.data.formatter import EmptyFormatter

    # Register the template
    if "glm5" not in TEMPLATES:
        register_template(
            name="glm5",
            format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
            format_assistant=StringFormatter(slots=["\n{{content}}"]),
            format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
            format_function=FunctionFormatter(
                slots=["{{content}}"], tool_format="glm4_moe"
            ),
            format_observation=StringFormatter(
                slots=["<|observation|>\n{{content}}<|assistant|>"]
            ),
            format_tools=ToolFormatter(tool_format="glm4_moe"),
            format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
            stop_words=["<|user|>", "<|observation|>"],
            thought_words=("<think>", "</think>"),
            efficient_eos=True,
            template_class=Glm47ReasoningTemplate,
        )
        logger.debug("Registered LF template 'glm5'")
    if "glm4_7" not in TEMPLATES:
        register_template(
            name="glm4_7",
            format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
            format_assistant=StringFormatter(slots=["\n{{content}}"]),
            format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
            format_function=FunctionFormatter(
                slots=["{{content}}"], tool_format="glm4_moe"
            ),
            format_observation=StringFormatter(
                slots=["<|observation|>\n{{content}}<|assistant|>"]
            ),
            format_tools=ToolFormatter(tool_format="glm4_moe"),
            format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
            stop_words=["<|user|>", "<|observation|>"],
            thought_words=("<think>", "</think>"),
            efficient_eos=True,
            template_class=Glm47ReasoningTemplate,
        )
        logger.debug("Registered LF template 'glm4.7'")
