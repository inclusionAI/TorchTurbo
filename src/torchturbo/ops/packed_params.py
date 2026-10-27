"""Reusable packed-sequence metadata prepared outside operator providers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from torchturbo.utils.functools import cache


class ValidatedOffsets(list[int]):
    """Host offsets validated while building :class:`PackedParams`."""


class ValidatedChunkLists(dict[str, list[int]]):
    """Host chunk indices derived from :class:`ValidatedOffsets`."""


_GDN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


@dataclass(frozen=True)
class PackedParams:
    """Device and host metadata shared by packed causal-conv and GDN calls.

    Construction performs the device-to-host synchronization once. The caller
    owns caching and lifetime; operator providers only consume these values.
    """

    cu_seqlens: torch.Tensor
    cu_seqlens_list: ValidatedOffsets
    chunk_indices: dict[str, torch.Tensor]
    chunk_indices_list: ValidatedChunkLists


def build_packed_params(
    cu_seqlens: torch.Tensor,
    chunk_sizes: tuple[int, ...] = _GDN_CHUNK_SIZES,
    *,
    total_tokens: int | None = None,
    device: torch.device | None = None,
) -> PackedParams:
    """Build metadata, optionally treating a padded tail as a separate sequence."""
    if device is None:
        device = cu_seqlens.device
    if cu_seqlens.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be rank-1, got shape={tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"cu_seqlens must be int32/int64, got dtype={cu_seqlens.dtype}"
        )

    offsets = ValidatedOffsets(
        int(item) for item in cu_seqlens.detach().cpu().tolist()
    )
    if total_tokens is not None and offsets[-1] < total_tokens:
        offsets.append(total_tokens)
        cu_seqlens = torch.nn.functional.pad(
            cu_seqlens, (0, 1), value=total_tokens
        )
    chunk_tensors: dict[str, torch.Tensor] = {}
    chunk_lists = ValidatedChunkLists()
    for chunk_size in dict.fromkeys(chunk_sizes):
        flattened: list[int] = []
        for sequence_index, (left, right) in enumerate(pairwise(offsets)):
            count = (right - left + chunk_size - 1) // chunk_size
            for chunk_index in range(count):
                flattened.extend((sequence_index, chunk_index))
        key = str(chunk_size)
        chunk_tensors[key] = torch.tensor(
            flattened,
            dtype=torch.int64,
            device=device,
        ).reshape(-1, 2)
        chunk_lists[key] = flattened

    return PackedParams(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int64).contiguous(),
        cu_seqlens_list=offsets,
        chunk_indices=chunk_tensors,
        chunk_indices_list=chunk_lists,
    )


@cache
def _cached_packed_params(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    device: torch.device,
    _version: int | None,
) -> PackedParams:
    return build_packed_params(cu_seqlens, total_tokens=total_tokens, device=device)


def get_packed_params(
    cu_seqlens: torch.Tensor | None,
    total_tokens: int,
    *,
    device: torch.device | None = None,
) -> PackedParams | None:
    """Reuse full-length recurrent metadata across layers of one packed batch.

    The input boundaries describe valid tokens and remain unchanged. Operators
    requiring full token coverage receive an isolated padding sequence.
    Device placement is part of the cache key so host boundaries can be shared
    by operators on the same accelerator without repeated metadata transfers.
    """
    if cu_seqlens is None:
        return None
    if device is None:
        device = cu_seqlens.device
    if torch.is_inference_mode_enabled() or cu_seqlens.is_inference():
        return _cached_packed_params.__wrapped__(cu_seqlens, total_tokens, device, None)
    return _cached_packed_params(cu_seqlens, total_tokens, device, cu_seqlens._version)


__all__ = [
    "PackedParams",
    "ValidatedChunkLists",
    "ValidatedOffsets",
    "build_packed_params",
    "get_packed_params",
]
