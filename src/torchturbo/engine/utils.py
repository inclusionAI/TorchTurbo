"""Small dependency-light helpers shared by engine modules."""

from __future__ import annotations

import re
from itertools import chain

import torch
import torch.nn as nn

_PEFT_PREFIX = "base_model.model."
_BASE_LAYER_SEGMENT = ".base_layer."
_ADAPTER_KEYWORDS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_A2.",
    ".lora_B2.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
)


def strip_peft_prefix(name: str) -> str:
    if name.startswith(_PEFT_PREFIX):
        name = name[len(_PEFT_PREFIX):]
    return name.replace(_BASE_LAYER_SEGMENT, ".")


def is_peft_adapter_param(name: str) -> bool:
    return any(keyword in name for keyword in _ADAPTER_KEYWORDS)


def _is_meta_tensor(tensor: torch.Tensor) -> bool:
    if getattr(tensor, "is_meta", False):
        return True

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local().is_meta
    return False


def has_meta_tensor(model: nn.Module) -> list[str]:
    """Return the fully qualified names of parameters and buffers on meta."""
    named_tensors = chain(model.named_parameters(), model.named_buffers())
    return [name for name, tensor in named_tensors if _is_meta_tensor(tensor)]


def match_fqn(pattern: str, fqn: str) -> bool:
    """Return whether an FQN matches a pattern.

    The pattern must match the entire FQN. ``*`` matches one or more characters
    other than ``.``, so it stays inside a single segment. ``**`` can also
    match ``.``, so it can span multiple segments.

    Examples:
        ``*`` stays inside one segment, and may match part of it:

        >>> match_fqn("layers.*.weight", "layers.0.weight")
        True

        ``**`` may match across multiple segments:

        >>> match_fqn("layers.**.weight", "layers.0.attention.q_proj.weight")
        True
    """
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", r".*")
    regex = regex.replace(r"\*", r"[^.]+")
    return bool(re.match(f"^{regex}$", fqn))
