"""Bailing V3 VL training prompts and media preprocessing for ms-swift."""

import json
from itertools import groupby
from typing import Any, Literal

from PIL import Image
from swift.template import ContextType, Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.template_meta import TemplateMeta
from swift.utils import to_device

from ..bailing_hybrid.swift_template import (
    ASSISTANT_OPEN,
    HUMAN_OPEN,
    OBSERVATION_OPEN,
    ROLE_END,
    SYS_OPEN,
    THINK_END,
    LingV3AgentTemplate,
    _make_ling_v3_template_meta,
)
from .processing import (
    get_last_query_index,
    get_video_inputs,
    normalize_thinking_content,
    read_auto_thinking_env,
    remove_thinking_content,
    resolve_sft_thinking,
)

_TEMPLATE_NAME = "bailing_moe_v3_vl"
_ORDERED_TEMPLATE_NAME = "bailing_moe_v3_vl_agent"

ASSISTANT_HEAD = ASSISTANT_OPEN + "\n"

# Roles accepted by the ordered Ling SFT encoder, in Swift's canonical naming.
_LING_SFT_ROLES = frozenset(
    {"user", "system", "assistant", "tool_call", "tool", "tool_response"}
)
# The ling agent protocol names tool results "observation"; normalize it to
# Swift's "tool", matching the LlamaFactory path's role mapping.
_OBSERVATION_ROLE = "observation"


class BailingVLAgentTemplate(LingV3AgentTemplate):
    """Format tool calls exactly as the checkpoint chat template renders them."""

    def _format_tools(
        self,
        tools: list[dict[str, Any]],
        system: str | None = None,
        user_message: dict[str, Any] | None = None,
    ) -> str:
        descriptions = super()._format_tools(tools, user_message=user_message)
        system = system or ""
        if system and not system.endswith("\n"):
            system += "\n"
        return system + descriptions

    def _format_tool_calls(self, tool_call_messages: list[dict[str, Any]]) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message["content"])
            name, arguments = tool_call["name"], tool_call["arguments"]
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(arguments, dict)
            ):
                raise ValueError(
                    "Ling tool calls require a non-empty name and an arguments object."
                )
            lines = [f"<tool_call>{name}"]
            for key, value in arguments.items():
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                lines.extend([f"<arg_key>{key}</arg_key>", f"<arg_value>{value}</arg_value>"])
            lines.append("</tool_call>")
            tool_calls.append("\n".join(lines))
        return "\n".join(tool_calls)

    def _add_tool_call_prefix(
        self, tool_content: str, pre_message: dict[str, Any] | None = None
    ) -> str:
        # A newline separates the first call from visible assistant content.
        if pre_message is None or pre_message["role"] != "assistant":
            return tool_content
        content = pre_message.get("content") or ""
        if not isinstance(content, str):
            return tool_content
        answer = content.rsplit(THINK_END, 1)[-1].lstrip("\n")
        return "\n" + tool_content if answer else tool_content

    def _format_tool_responses(
        self, assistant_content: Any, tool_messages: list[dict[str, Any]]
    ) -> tuple[Any, list[str]]:
        prompt = self._format_standalone_tool_responses(tool_messages)
        prompt.append("<|role_end|><role>ASSISTANT</role>\n")
        return assistant_content, prompt

    def _format_standalone_tool_responses(
        self, tool_messages: list[dict[str, Any]]
    ) -> list[str]:
        if not tool_messages:
            raise ValueError("Ling observations require at least one tool response.")
        prompt = ["<|role_end|>", "<role>OBSERVATION</role>"]
        for message in tool_messages:
            prompt.extend(["\n<tool_response>\n", message["content"], "\n</tool_response>"])
        return prompt


class BailingVLOrderedAgentTemplate(BailingVLAgentTemplate):
    """Per-message rendering for the ordered template.

    The base class keeps the folding-path serializers untouched; this
    subclass renders each tool call and observation group as standalone
    pieces for the ordered message encoder.
    """

    @staticmethod
    def _observations(tool_messages: list[dict[str, Any]]) -> list[str]:
        """Render the OBSERVATION role body without a leading role_end."""
        if not tool_messages:
            raise ValueError("Ling observations require at least one tool response.")
        parts = [OBSERVATION_OPEN]
        for message in tool_messages:
            parts.extend(["\n<tool_response>\n", message["content"], "\n</tool_response>"])
        return parts

    def render_tool_call(
        self, message: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> str:
        """Render one call, keeping the native inter-call newline."""
        content = self._format_tool_calls([message])
        if previous is not None and previous["role"] == "tool_call":
            return "\n" + content
        return self._add_tool_call_prefix(content, previous)

    def render_observations(self, tool_messages: list[dict[str, Any]]) -> list[str]:
        """Render a consecutive observation group without guessing the next role."""
        return [*self._observations(tool_messages), ROLE_END]


class BailingVLTemplate(Template):
    """Encode Ling SFT prompts and leave real-media fusion to the model."""

    support_padding_free: bool = True
    auto_sft_thinking: bool

    def init_env_args(self) -> None:
        super().init_env_args()
        # Include prompt and image preprocessing semantics in Swift's cache key.
        self._version = "bailing-vl-auto-thinking-qwen-images-v1"
        self.auto_sft_thinking = read_auto_thinking_env()

    def _get_max_pixels(self, inputs: StdTemplateInputs | None = None) -> int | None:
        # Per-sample pixel limits are applied by fetch_image after Swift's resize.
        return self.max_pixels

    def _remove_thinking_content(
        self, content: str, thinking_suffix: str = THINK_END
    ) -> str:
        return remove_thinking_content(content, thinking_suffix)

    def _remove_history_thinking(self, inputs: StdTemplateInputs) -> None:
        messages = inputs.messages
        last_query = get_last_query_index(messages)
        for message in messages[:last_query]:
            if message["role"] != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list) and content and isinstance(content[0], str):
                content[0] = self._remove_thinking_content(content[0])
            elif isinstance(content, str):
                message["content"] = self._remove_thinking_content(content)

    def _get_system(self, inputs: StdTemplateInputs) -> str:
        system = super()._get_system(inputs)
        raw_system = inputs.system
        if raw_system is None:
            raw_system = self.template_meta.default_system
        raw_system = raw_system or ""
        if "detailed thinking on" in raw_system or "detailed thinking off" in raw_system:
            return system
        thinking = "on" if self._get_enable_thinking(inputs) else "off"
        directive = f"detailed thinking {thinking}"
        if not system:
            return directive
        if inputs.tools:
            return system + directive
        return system.rstrip("\n") + "\n" + directive

    def _preprocess_tool_call(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_tool_call(inputs)
        if self.template_backend != "swift":
            return
        # Fold observations without an assistant call into the preceding query
        # before Swift pairs queries and answers.
        messages = []
        for role, group in groupby(inputs.messages, key=lambda message: message["role"]):
            group = list(group)
            if role != "tool" or (messages and messages[-1]["role"] == "assistant"):
                messages.extend(group)
                continue
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("A standalone tool message must follow a user message.")
            query = messages[-1]
            content = query.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Standalone tool messages require text-only user content.")
            self.agent_template.template_meta = self.template_meta
            observations = self.agent_template._format_standalone_tool_responses(group)
            if not all(isinstance(part, str) for part in observations):
                raise ValueError("Standalone tool formatting must produce text contexts.")
            query["content"] = content + "".join(observations)
        inputs.messages[:] = messages

    def _ling_prepare_thinking(self, inputs: StdTemplateInputs) -> None:
        """Resolve per-sample auto-thinking and normalize thinking blocks."""
        if (
            self.auto_sft_thinking
            and self.mode == "train"
            and self.task_type == "causal_lm"
            and self.use_chat_template
        ):
            system = inputs.system
            if system is None:
                system = self.template_meta.default_system
            inputs.chat_template_kwargs["enable_thinking"] = resolve_sft_thinking(
                inputs.messages, system, inputs.chat_template_kwargs.get("enable_thinking")
            )
        if self.use_chat_template and self.template_backend == "swift":
            # Normalize before calls are rendered: arguments can contain literal
            # think tags, and a merged string could lose loss boundaries.
            for message in inputs.messages:
                if message["role"] != "assistant":
                    continue
                message["content"] = normalize_thinking_content(message.get("content"))

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs) -> None:
        self._ling_prepare_thinking(inputs)
        super()._swift_prepare_inputs(inputs)

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ) -> list[str]:
        processor = self.processor
        if media_type == "image":
            from qwen_vl_utils import fetch_image

            inputs.images[index] = fetch_image(
                {"image": inputs.images[index], **inputs.chat_template_kwargs},
                image_patch_size=processor.image_processor.patch_size,
            )
            media_inputs = processor.image_processor(
                images=[inputs.images[index]], return_tensors="pt", do_resize=False
            )
            merge_size = processor.image_processor.merge_size
            num_tokens = int(media_inputs["image_grid_thw"].prod().item()) // merge_size**2
            text = (
                processor.vision_start_token
                + processor.image_token * num_tokens
                + processor.vision_end_token
            )
        elif media_type == "video":
            media_inputs = get_video_inputs([inputs.videos[index]], processor)
            text = (
                processor.video_start_token
                + processor.video_token
                + processor.video_end_token
            )
            text = processor._expand_video_tokens([text], media_inputs)[0]
        else:
            raise ValueError("Bailing V3 VL supports image and video inputs only.")
        inputs.mm_processor_kwargs["bailing_media_inputs"].append(media_inputs)
        return [text]

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        inputs.mm_processor_kwargs["bailing_media_inputs"] = []
        encoded = super()._encode(inputs)
        media_inputs = inputs.mm_processor_kwargs.pop("bailing_media_inputs")
        encoded.update(self._data_collator_mm_data(media_inputs))
        return encoded

    def _post_encode(self, model, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add zero-valued visual and projector dependencies to text-only batches."""
        if (
            not self.is_training
            or inputs.get("pixel_values") is not None
            or inputs.get("pixel_values_videos") is not None
        ):
            return inputs

        input_ids = inputs["input_ids"]
        base_model = self.get_base_model(model)
        inputs_embeds = base_model.get_input_embeddings()(input_ids)
        dummy_image = Image.new("RGB", (32, 32), (0, 0, 0))
        media_inputs = self.processor.image_processor(
            images=[dummy_image], return_tensors="pt"
        )
        media_inputs = to_device(media_inputs, input_ids.device)
        dummy_embeds = base_model.get_image_features(
            pixel_values=media_inputs["pixel_values"],
            image_grid_thw=media_inputs["image_grid_thw"],
        )
        dummy_dependency = dummy_embeds[0].mean().to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs["inputs_embeds"] = inputs_embeds + dummy_dependency * 0
        inputs["mtp_input_ids"] = input_ids
        return inputs


class BailingVLOrderedTemplate(BailingVLTemplate):
    """Agent-format SFT template: ordered rendering, no query/response pairing.

    Renders messages in their original order (native OBSERVATION blocks,
    per-message supervision) instead of Swift's alternating user/assistant
    pairing. Swift's tokenization, media expansion and packing remain
    downstream; its loss policy's per-context hook is reused, not its
    pair-indexed conversation dispatcher. Select it explicitly with
    ``--template bailing_moe_v3_vl_agent``; the base template keeps the
    pairing behavior for inference and ordinary conversations.
    """

    def init_env_args(self) -> None:
        super().init_env_args()
        # Separate cache fingerprint from the pairing template.
        self._version = "bailing-vl-ordered-agent-encoding-v1"

    def _ling_use_message_encoding(self) -> bool:
        return (
            self.mode == "train"
            and self.task_type == "causal_lm"
            and self.use_chat_template
            and self.template_backend == "swift"
        )

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs) -> None:
        self._ling_prepare_thinking(inputs)
        # Keep message order and per-message supervision; no pair folding.
        return

    def _swift_encode(
        self, inputs: StdTemplateInputs
    ) -> tuple[list[Any], list[float], int]:
        if not self._ling_use_message_encoding():
            return super()._swift_encode(inputs)
        messages = inputs.messages
        if not messages:
            raise ValueError("Ling SFT requires at least one message.")
        for message in messages:
            if message["role"] == _OBSERVATION_ROLE:
                message["role"] = "tool"
        for i, message in enumerate(messages):
            if message["role"] not in _LING_SFT_ROLES:
                raise ValueError(
                    f"Unsupported Ling role at message {i}: {message['role']!r}; "
                    f"accepted roles: {sorted(_LING_SFT_ROLES)}"
                )
            if message.get("content") is None:
                raise ValueError(
                    f"Ling SFT requires content at message {i}; null is a generation placeholder."
                )
            if i > 0 and message["role"] == messages[i - 1]["role"] == "assistant":
                # Ordinary sample error: strict=False MAP drops the whole row.
                # Do not merge independent answers or insert a synthetic user.
                # Canonical assistant -> tool_call is intentionally unaffected.
                raise ValueError(
                    "Ling SFT rejects consecutive assistant messages at normalized message indices "
                    f"{i - 1} and {i}; invalid sample (assistant -> assistant)."
                )
        if not any(m["role"] in {"assistant", "tool_call"} for m in messages):
            raise ValueError("Ling SFT requires an assistant answer or tool call.")

        # Do this before rendering calls: arguments can contain literal think tags.
        if self.add_non_thinking_prefix:
            self._add_non_thinking_prefix(inputs)
        if not self._get_preserve_thinking(inputs):
            self._remove_history_thinking(inputs)
        system = self._get_system(inputs)
        contexts: list[Any] = []
        weights: list[float] = []
        # Keep Swift last_round semantics (last user OR tool), independently of
        # Ling's preserve_thinking boundary (last real user).
        last_round = max(
            (i for i, m in enumerate(messages) if m["role"] in {"user", "tool", "tool_response"}),
            default=-1,
        )
        query = None

        def emit(content, kind=ContextType.OTHER, message=None, index=-1):
            message = message or {}
            loss, scale = message.get("loss"), message.get("loss_scale")
            parts = content if isinstance(content, list) and content and not isinstance(content[0], int) else [content]
            for key, value in (("loss", loss), ("loss_scale", scale)):
                if isinstance(value, list) and len(value) != len(parts):
                    raise ValueError(f"Ling {key} length does not match content at message {index}.")
            for j, part in enumerate(parts):
                new_contexts, new_weights = self.loss_scale._inner_call(
                    part,
                    kind,
                    query,
                    loss[j] if isinstance(loss, list) else loss,
                    scale[j] if isinstance(scale, list) else scale,
                    index >= last_round,
                )
                contexts.extend(new_contexts)
                weights.extend(new_weights)

        def finish(message, index, final):
            # Intermediate EOS previously became supervised via dynamic_eos only
            # after a positive-weight tail. Final EOS follows the suffix policy.
            # Explicit loss=False must not be resurrected by dynamic_eos.
            if final:
                content = message["content"]
                tail = (
                    content[-1] if isinstance(content, list) and content and isinstance(content[-1], str) else content
                )
                if not isinstance(tail, str):
                    token_ids = tail["token_ids"] if isinstance(tail, dict) else tail
                    tail = self.tokenizer.decode(token_ids[-20:])
                ends = any(tail.endswith(stop) for stop in self.template_meta.stop_words if isinstance(stop, str))
                add_eos = inputs.extra_kwargs.get("add_eos")
                if add_eos is None:
                    add_eos = not ends
                if not add_eos:
                    return
                loss = message.get("loss")
                if isinstance(loss, list):
                    loss = loss[-1]
                emit(ROLE_END, ContextType.SUFFIX, {"loss": loss}, index)
            else:
                weight = 1.0 if weights and weights[-1] > 0 else 0.0
                contexts.append(ROLE_END)
                weights.append(weight)

        emit(SYS_OPEN + (system or "") + ROLE_END)
        i = 0
        while i < len(messages):
            message = messages[i]
            role = message["role"]
            if role in {"user", "system"}:
                if role == "user":
                    query = message["content"]
                emit(HUMAN_OPEN if role == "user" else SYS_OPEN)
                emit(message["content"])
                emit(ROLE_END)
                i += 1
            elif role in {"tool", "tool_response"}:
                end = i + 1
                while end < len(messages) and messages[end]["role"] in {"tool", "tool_response"}:
                    end += 1
                for part in self.agent_template.render_observations(messages[i:end]):
                    emit(part)
                i = end
            else:
                emit(ASSISTANT_HEAD)
                if role == "assistant":
                    emit(message["content"], ContextType.RESPONSE, message, i)
                    i += 1
                else:
                    # Canonical standalone calls are assistant actions, not a
                    # new user turn. The native protocol requires an empty think.
                    emit("<think></think>", ContextType.RESPONSE, message, i)
                previous = message if role == "assistant" else None
                while i < len(messages) and messages[i]["role"] == "tool_call":
                    message = messages[i]
                    call = self.agent_template.render_tool_call(message, previous)
                    emit(call, ContextType.RESPONSE, message, i)
                    previous = {"role": "tool_call"}
                    i += 1
                finish(message, i - 1, i == len(messages))
        return contexts, weights, 0  # causal LM only; no encoder/decoder split

    def _add_dynamic_eos(
        self,
        input_ids: list[int],
        labels: list[int],
        loss_scale: list[int] | None,
        suffix_tokens_id: list[int],
    ) -> None:
        if not self._ling_use_message_encoding():
            super()._add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id)
            return
        # All assistant boundaries already have explicit supervision. The generic
        # heuristic could unmask literal role_end tokens in user/tool content.
        # Swift only calls this bound to an instance, so the override applies.


def _register_agent_template() -> None:
    from swift.agent_template import agent_template_map

    agent_template_map[_TEMPLATE_NAME] = BailingVLAgentTemplate
    agent_template_map[_ORDERED_TEMPLATE_NAME] = BailingVLOrderedAgentTemplate


def _make_template_meta(template_type: str, template_cls) -> TemplateMeta:
    """Build the ling v3 template meta with the checkpoint tool-call format."""
    # The ordered template pairs with its own per-message agent template;
    # the base template keeps the original folding-path serializer.
    agent_template_type = (
        _ORDERED_TEMPLATE_NAME if template_type == _ORDERED_TEMPLATE_NAME else _TEMPLATE_NAME
    )
    meta = _make_ling_v3_template_meta(template_type, template_cls)
    meta.agent_template = agent_template_type
    return meta


def register() -> None:
    """Attach the Bailing VL templates to the existing Turbo Swift model entry."""
    from swift.model.model_meta import MODEL_MAPPING
    from swift.template.register import register_template

    _register_agent_template()
    templates = [
        (_TEMPLATE_NAME, BailingVLTemplate),
        (_ORDERED_TEMPLATE_NAME, BailingVLOrderedTemplate),
    ]
    for template_type, template_cls in templates:
        register_template(_make_template_meta(template_type, template_cls), exist_ok=True)
    model_meta = MODEL_MAPPING[f"turbo_{_TEMPLATE_NAME}"]
    model_meta.template = _TEMPLATE_NAME
    model_meta.candidate_templates = [template_type for template_type, _ in templates]
