"""LlamaFactory template for GLM5.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here — the model directory stays
clean if LF isn't installed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union
from typing_extensions import override

from ...logging import get_logger

from llamafactory.data.template import ReasoningTemplate

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger(__name__)


@dataclass
class Glm47ReasoningTemplate(ReasoningTemplate):
    r"""GLM-4.7 uses only the closing </think> tag for empty thinking blocks."""

    @override
    def add_thought(self, content: str = "") -> str:
        if not content:
            return self.thought_words[1]
        return self.thought_words[0] + content + self.thought_words[1]

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0].strip() not in messages[i + 1]["content"]
                and self.thought_words[1].strip() not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = (
                        self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]
                    )
        return [
            (encoded_messages[i], encoded_messages[i + 1])
            for i in range(0, len(encoded_messages), 2)
        ]


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
