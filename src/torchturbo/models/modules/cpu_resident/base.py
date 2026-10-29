"""Shared interface for parameters whose storage is managed on CPU."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import nn


class CPUResidentParameter(nn.Parameter):
    """Interface for CPU storage, materialization, and coordinated shard I/O.

    Concrete parameter types own construction, storage, and rank assignment.
    shard_sizes records the last successfully loaded source layout, or None.
    """

    shard_sizes: tuple[int, ...] | None

    def materialize_(self, *, dtype: torch.dtype | None = None) -> Self:
        """Materialize CPU storage collectively across WORLD, preserving identity."""
        raise NotImplementedError

    def load_(
        self,
        shard_sizes: Sequence[int],
        reader: Callable[[int, torch.Tensor], None],
        *,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Materialize and load source shards collectively across WORLD.

        Args:
            shard_sizes: Source row counts covering the parameter; scalars use [1].
            reader: Fills reader(shard_index, detached_target_view).
            dtype: Storage dtype, or None to preserve the declared dtype.

        Returns:
            This parameter, recording the layout only after global success.
        """
        raise NotImplementedError

    def save(self, writer: Callable[[int, torch.Tensor], None]) -> None:
        """Write existing shards collectively across WORLD using writer(index, view)."""
        raise NotImplementedError

    def local_shards(
        self,
        shard_sizes: Sequence[int] | None = None,
        *,
        for_save: bool = False,
    ) -> list[tuple[int, int, torch.Tensor]]:
        """Describe local source indices, row offsets, and views without collectives."""
        raise NotImplementedError

    def shard_owner(self, index: int) -> int:
        """Return the rank assigned to save a source shard."""
        raise NotImplementedError

    def validate_shard_sizes(self, shard_sizes: Sequence[int]) -> tuple[int, ...]:
        """Return source row counts as a tuple after checking they cover this parameter."""
        raise NotImplementedError
