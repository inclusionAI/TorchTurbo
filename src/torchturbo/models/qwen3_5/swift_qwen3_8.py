# Copyright (c) ModelScope Contributors. All rights reserved.
"""Swift template for Qwen3.8 conversations.

Preserve historical thinking, use user/tool boundaries for last-round loss,
and serialize tool arguments with JSON scalar literals. All overrides are
scoped to the registered ``qwen3_8`` template.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from swift.agent_template import BaseAgentTemplate
from swift.agent_template.qwen3_coder import Qwen3_5AgentTemplate
from swift.loss_scale.base import LossScale
from swift.template.template_inputs import StdTemplateInputs, TemplateInputs
from swift.template.templates.qwen import QwenTemplateMeta

from .swift_template import _make_kwarg_preserving_template_cls


def _get_last_user_round(messages: list[dict[str, Any]], *, include_tool: bool = True) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user" or (include_tool and messages[i]["role"] == "tool"):
            return i
    return -1


def _normalize_openai_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI calls and align complete result batches with the call order."""
    normalized = []
    messages = list(messages)
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not tool_calls:
            normalized.append(message)
            continue

        call_order = {call.get("id"): i for i, call in enumerate(tool_calls)}
        end = index + 1
        while end < len(messages) and messages[end].get("role") in {"tool", "tool_response"}:
            end += 1
        responses = messages[index + 1:end]
        if (None not in call_order and len(call_order) == len(tool_calls) == len(responses)
                and set(call_order) == {response.get("tool_call_id") for response in responses}):
            messages[index + 1:end] = sorted(responses, key=lambda response: call_order[response["tool_call_id"]])

        if message.get("content"):
            normalized.append({
                key: value for key, value in message.items()
                if key in {"role", "content", "loss", "loss_scale"}
            })
        for tool_call in tool_calls:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            tool_message = {
                "role": "tool_call",
                "content": {"name": function["name"], "arguments": arguments},
            }
            for key in ("loss", "loss_scale"):
                if key in message:
                    tool_message[key] = message[key]
            normalized.append(tool_message)
    return normalized


class _Qwen3_8AgentTemplate(Qwen3_5AgentTemplate):
    def _format_tool_calls(self, tool_call_messages: list[dict[str, Any]]) -> str:
        calls = []
        for message in tool_call_messages:
            call = self._parse_tool_call(message["content"])
            if call.get("arguments"):
                call ={**call, "arguments": {
                    name: json.dumps(value) if isinstance(value, bool) or value is None else value
                    for name, value in call["arguments"].items()
                }}
            calls.append({**message, "content": call})
        return super()._format_tool_calls(calls)

    def _format_standalone_tool_responses(self, tool_messages: list[dict[str, Any]]) -> list:
        contexts = (self.template_meta.chat_sep or []).copy()
        tool_content = self._get_tool_responses(tool_messages)
        for context in self.template_meta.prompt:
            if isinstance(context, str) and "{{QUERY}}" in context:
                query_prefix = context.split("{{QUERY}}", maxsplit=1)[0]
                if query_prefix:
                    contexts.append(query_prefix)
                contexts.append(tool_content)
                return contexts
            contexts.append(context)
        raise ValueError(f"Template prompt does not contain {{{{QUERY}}}}: {self.template_meta.prompt}")


class _ToolBoundaryLossScale(LossScale):
    """Reuse Swift loss rules with tools treated as last-round boundaries."""

    def __init__(self, delegate: LossScale) -> None:
        super().__init__(delegate.base_strategy)
        self.delegate = delegate

    @property
    def is_binary_loss_scale(self) -> bool:
        return self.delegate.is_binary_loss_scale

    def get_loss_scale(self, context: str, **kwargs: Any) -> tuple[list, list[float]]:
        return self.delegate.get_loss_scale(context, **kwargs)

    def __call__(
        self, context_list: list, context_types: list, messages: list, **kwargs: Any
    ) -> tuple[list, list[float]]:
        # LossScale reads roles only to find the last query. Content and explicit
        # loss/loss_scale fields must remain intact, including list-valued fields.
        queries = [
            {**message, "role": "user"} if message["role"] == "tool" else message
            for message in messages
        ]
        return self.delegate(context_list, context_types, queries, **kwargs)


class _Qwen3_8TemplateInputs(TemplateInputs):
    """Use full response trajectories and canonical tool-call inputs."""

    def __post_init__(self) -> None:
        if isinstance(self.chosen, dict):
            self.chosen = {**self.chosen, "messages": _normalize_openai_tool_calls(self.chosen["messages"])}
        if isinstance(self.rejected, dict):
            self.rejected = {**self.rejected, "messages": _normalize_openai_tool_calls(self.rejected["messages"])}
        # StdTemplateInputs normalizes tool_response to tool and extracts media
        # separately for each branch, after preference construction.
        super().__post_init__()

    @staticmethod
    def _compat_rejected_response(inputs: dict[str, Any]) -> None:
        if "rejected_response" not in inputs:
            return
        messages = inputs["messages"]
        if not messages:
            raise ValueError("messages must not be empty.")
        boundary = _get_last_user_round(messages, include_tool=False) + 1
        rejected = inputs.pop("rejected_response")
        if isinstance(rejected, str):
            rejected = [{"role": "assistant", "content": rejected}]
        elif not isinstance(rejected, list):
            raise ValueError("rejected_response must be a string or a list of messages.")
        if any(message["role"] == "user" for message in rejected):
            raise ValueError("rejected_response must not contain user messages.")
        if len(messages[boundary:]) == 1 and len(rejected) == 1:
            chosen_content = [message["content"] for message in _normalize_openai_tool_calls(messages[boundary:])]
            rejected_content = [message["content"] for message in _normalize_openai_tool_calls(rejected)]
            if chosen_content == rejected_content:
                raise ValueError("The chosen and rejected responses must differ.")
        inputs["rejected_messages"] = deepcopy(messages[:boundary]) + rejected


@dataclass
class Qwen3_8TemplateMeta(QwenTemplateMeta):
    preserve_thinking: bool = True


class Qwen3_8Template(_make_kwarg_preserving_template_cls()):
    """Qwen3.8 template framing without changing Swift's global helpers."""

    reasoning_effort_instructions: dict[str, str] = {
        "xhigh": (
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
            "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
            "and clarity in the final answer."
        ),
        "medium": "",
        "low": (
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
            "conclusion without unnecessary elaboration."
        ),
    }
    default_reasoning_effort: str = "xhigh"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keep encoded dataset caches separate from the native template.
        self._version += "-turbo-qwen3_8-v2"

    def encode(
        self, inputs: Any, return_template_inputs: bool = False, return_length: bool = False,
    ) -> dict[str, Any]:
        if isinstance(inputs, dict):
            inputs = _Qwen3_8TemplateInputs.from_dict(inputs)
        return super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)

    def _preprocess_tool_call(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_tool_call(inputs)
        if self.template_backend != "swift":
            return
        messages = inputs.messages
        i = 0
        while i < len(messages):
            if messages[i]["role"] != "tool" or (i > 0 and messages[i - 1]["role"] in {"assistant", "tool"}):
                i += 1
                continue
            start = i
            while i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                i += 1
            if start == 0 or messages[start - 1]["role"] != "user":
                raise ValueError("A standalone tool message must follow a user message.")
            query = messages[start - 1]
            content = query.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Standalone tool messages currently require text-only user content.")
            agent = self.agent_template
            agent.template_meta = self.template_meta
            contexts = agent._format_standalone_tool_responses(messages[start:i + 1])
            if not all(isinstance(context, str) for context in contexts):
                raise ValueError(f"Standalone tool formatting must produce text contexts: {contexts}")
            query["content"] = content + "".join(contexts)
            del messages[start:i + 1]
            i = start

    @property
    def loss_scale(self) -> LossScale:
        loss_scale = super().loss_scale
        if not isinstance(loss_scale, _ToolBoundaryLossScale):
            loss_scale = _ToolBoundaryLossScale(loss_scale)
            self._loss_scale_cache[self._loss_scale] = loss_scale
        return loss_scale

    @property
    def agent_template(self) -> BaseAgentTemplate:
        agent = super().agent_template
        if self._agent_template == "qwen3_5" and not isinstance(agent, _Qwen3_8AgentTemplate):
            agent = _Qwen3_8AgentTemplate()
            self._agent_template_cache[self._agent_template] = agent
        return agent

    def _get_preserve_thinking(self, inputs: StdTemplateInputs | None = None) -> bool:
        preserve = None if inputs is None else inputs.chat_template_kwargs.get("preserve_thinking")
        if preserve is None:
            preserve = self.preserve_thinking
        if preserve is None:
            preserve = self.template_meta.preserve_thinking
        if preserve is None:
            preserve = super()._get_preserve_thinking(inputs)
        return preserve

    def _add_non_thinking_prefix(
        self, inputs: StdTemplateInputs, thinking_prefix: str = "<think>"
    ) -> None:
        prefix = self.template_meta.non_thinking_prefix
        if not prefix:
            return
        # Qwen3.8 metadata preserves history, so even last_round inserts the
        # non-thinking prefix in every assistant round.
        for i, message in enumerate(inputs.messages):
            if not self._is_add_non_thinking_round(inputs.messages, i, -1):
                continue
            content = message["content"]
            if isinstance(content, list):
                if content and isinstance(content[0], str) and not content[0].startswith((thinking_prefix, prefix)):
                    content[0] = prefix + content[0]
            elif isinstance(content, str) and not content.startswith((thinking_prefix, prefix)):
                message["content"] = prefix + content

    def _remove_history_thinking(self, inputs: StdTemplateInputs) -> None:
        boundary = _get_last_user_round(inputs.messages)
        for i, message in enumerate(inputs.messages):
            if i >= boundary or message["role"] != "assistant":
                continue
            content = message["content"]
            if isinstance(content, list) and content and isinstance(content[0], str):
                content[0] = self._remove_thinking_content(content[0])
            elif isinstance(content, str):
                message["content"] = self._remove_thinking_content(content)

    def _get_reasoning_instructions(self, inputs: StdTemplateInputs | None = None) -> str:
        if not self._get_enable_thinking(inputs):
            return ""
        effort = None if inputs is None else inputs.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.default_reasoning_effort
        if effort not in self.reasoning_effort_instructions:
            raise ValueError(f"Unsupported Qwen3.8 reasoning_effort: {effort!r}")
        return self.reasoning_effort_instructions[effort]

    def _get_system(self, inputs: StdTemplateInputs) -> str | None:
        system = super()._get_system(inputs)
        instruction = self._get_reasoning_instructions(inputs)
        if not instruction:
            return system
        return f"{instruction}\n\n{system}" if system else instruction

    def _jinja_encode(self, inputs: StdTemplateInputs) -> Any:
        for message in inputs.messages:
            content = message.get("content")
            if (message["role"] != "assistant" or not isinstance(content, str)
                    or message.get("reasoning_content") is not None):
                continue
            if content.startswith("<think>") and "</think>" in content:
                reasoning, _, rest = content.partition("</think>")
                message["reasoning_content"] = reasoning[len("<think>"):].strip()
                message["content"] = rest.lstrip("\n")
        return super()._jinja_encode(inputs)
