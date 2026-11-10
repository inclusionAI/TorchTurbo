"""LlamaFactory template for GLM5.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here — the model directory stays
clean if LF isn't installed.
"""

from __future__ import annotations

import json

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any
from typing_extensions import override

from ...logging import get_logger

from llamafactory.data.formatter import FunctionFormatter, StringFormatter
from llamafactory.data.template import ReasoningTemplate, Template

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger(__name__)


class Glm52SystemFormatter(StringFormatter):
    r"""Reorders fused system+tools content to match the GLM-5.2 jinja template.

    The base ``Template._encode`` fuses system and tools into one block as
    ``system + tool_text``, while the jinja template emits the tools block
    first as its own ``<|system|>`` segment.
    """

    @override
    def apply(self, **kwargs) -> list:
        content = kwargs.get("content", "")
        idx = content.find("\n# Tools\n")
        if idx > 0:  # fused "system + tool_text": move the tools block first
            kwargs["content"] = content[idx:] + "<|system|>" + content[:idx]
        return super().apply(**kwargs)


class Glm52AssistantFormatter(StringFormatter):
    r"""GLM-5.2 assistant messages always start with a think block."""

    @override
    def apply(self, **kwargs) -> list:
        content = kwargs.get("content", "")
        if "</think>" not in content:
            kwargs["content"] = "<think></think>" + content
        return super().apply(**kwargs)


class Glm52FunctionFormatter(FunctionFormatter):
    r"""GLM-5.2 tool-call messages always start with a think block."""

    @override
    def apply(self, **kwargs) -> list:
        content = kwargs.get("content", "")
        elements = super().apply(**kwargs)
        if "</think>" not in content and elements and isinstance(elements[0], str):
            elements[0] = "<think></think>" + elements[0]
        return elements


@dataclass
class Glm47ReasoningTemplate(ReasoningTemplate):
    r"""GLM-4.7 uses only the closing </think> tag for empty thinking blocks."""

    @override
    def add_thought(self, content: str = "") -> str:
        if not content:
            return self.thought_words[1]
        return self.thought_words[0] + content + self.thought_words[1]


@dataclass
class Glm52ReasoningTemplate(ReasoningTemplate):
    r"""GLM-5.2 template matching the official jinja chat template.

    Turn 0 is emitted as separate ``<|system|>`` segments (reasoning-effort
    line, tools block, system message) instead of one fused system block.
    """

    reasoning_effort: str = "Max"
    preserve_thinking: bool = False

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if not self.preserve_thinking:  # the jinja template clears historical thinking by default
            for i in range(1, len(messages) - 2, 2):  # preserve the last cot
                messages[i]["content"] = self.remove_thought(messages[i]["content"])
        return super().encode_multiturn(tokenizer, messages, system, tools, **kwargs)

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        system = system or self.default_system
        segments = []
        if self.enable_thinking is not False:
            segments.append(f"Reasoning Effort: {self.reasoning_effort}")
        if tools:
            tool_text = self.format_tools.apply(content=tools)[0]
            if tool_text:
                segments.append(tool_text)
        if system:
            segments.append(system)
        # format_system is "<|system|>{{content}}", so joining the segments with
        # "<|system|>" reproduces the jinja template's separate system blocks.
        return super()._encode(tokenizer, messages, "<|system|>".join(segments), None)


def register() -> None:
    """Register the GLM5 template with LlamaFactory."""
    from llamafactory.data.template import (
        register_template,
        TEMPLATES,
        ReasoningTemplate,
    )
    from llamafactory.data.formatter import StringFormatter
    from llamafactory.data.formatter import FunctionFormatter
    from llamafactory.data.formatter import ToolFormatter
    from llamafactory.data.formatter import EmptyFormatter
    from llamafactory.data.tool_utils import ToolUtils, TOOLS, QwenToolUtils

    GLM5_TOOL_PROMPT = (
        "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
        "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:"
        "\n<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value>"
        "<arg_key>{{arg-key-2}}</arg_key><arg_value>{{arg-value-2}}</arg_value>...</tool_call>"
    )

    class GLM5ToolUtils(QwenToolUtils):
        r"""GLM-5 tool using template, matching the official jinja chat template."""

        @override
        @staticmethod
        def tool_formatter(tools: list[dict[str, Any]]) -> str:
            tool_text = ""
            for tool in tools:
                if "function" in tool:
                    tool = tool["function"]
                tool = {k: v for k, v in tool.items() if k not in ("defer_loading", "strict")}
                tool_text += "\n" + json.dumps(tool, ensure_ascii=False)

            return GLM5_TOOL_PROMPT.format(tool_text=tool_text)

        @override
        @staticmethod
        def function_formatter(functions) -> str:
            function_texts = []
            for name, arguments in functions:
                prompt = "<tool_call>" + name
                for key, value in json.loads(arguments).items():
                    prompt += "<arg_key>" + key + "</arg_key>"
                    if not isinstance(value, str):
                        value = json.dumps(value, ensure_ascii=False)
                    prompt += "<arg_value>" + value + "</arg_value>"
                prompt += "</tool_call>"
                function_texts.append(prompt)

            return "".join(function_texts)

    TOOLS["glm5"] = GLM5ToolUtils()

    # Register the template
    if "glm5" not in TEMPLATES:
        register_template(
            name="glm5",
            format_user=StringFormatter(slots=["<|user|>{{content}}<|assistant|>"]),
            format_assistant=StringFormatter(slots=["{{content}}"]),
            format_system=StringFormatter(slots=["<|system|>{{content}}"]),
            format_function=FunctionFormatter(
                slots=["{{content}}"], tool_format="glm5"
            ),
            # the jinja template wraps string observations in <tool_response> tags
            format_observation=StringFormatter(
                slots=["<|observation|><tool_response>{{content}}</tool_response><|assistant|>"]
            ),
            format_tools=ToolFormatter(tool_format="glm5"),
            format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
            stop_words=["<|user|>", "<|observation|>"],
            thought_words=("<think>", "</think>"),
            efficient_eos=True,
            template_class=Glm52ReasoningTemplate,
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
