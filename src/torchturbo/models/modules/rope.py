"""Shared read-only RoPE tables."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from weakref import WeakValueDictionary

import torch


_ROPE_TABLE_CACHE: WeakValueDictionary[tuple, torch.Tensor] = WeakValueDictionary()


def get_shared_rope_table(
    builder: Callable[..., torch.Tensor],
    parameters: tuple[Hashable, ...],
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a shared read-only RoPE table.

    Args:
        builder: Stable function or class method that builds a deterministic
            table on the current default device. It must not depend on mutable
            model state or compute gradients.
        parameters: Immutable values that fully determine the table's contents.
        device: Target device, or None to use the current default device.
        dtype: Output dtype. The builder retains its own computation precision.

    Returns:
        A table shared within this process while callers retain references.
        Callers must not modify the table in place.
    """
    device = torch.empty(0, device=device).device
    key = (builder, parameters, device, dtype)
    table = _ROPE_TABLE_CACHE.get(key)
    if table is None:
        # Training autograd must be able to save a table first built in inference.
        with torch.inference_mode(False), torch.no_grad(), torch.device(device):
            table = builder(*parameters).to(dtype=dtype)
        _ROPE_TABLE_CACHE[key] = table
    return table
