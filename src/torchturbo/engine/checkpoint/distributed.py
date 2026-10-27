"""Coordinate checkpoint failures across ranks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch.distributed as dist

T = TypeVar("T")


def rank_zero_call(action: Callable[[], T]) -> T:
    """Run a filesystem decision on rank zero and broadcast its result or error."""
    if not dist.is_initialized():
        return action()
    result = [None, None]
    if dist.get_rank() == 0:
        try:
            result[0] = action()
        except Exception as exc:
            result[1] = exc
    dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise result[1]
    return result[0]


def collective_check(action: Callable[[], T]) -> T:
    """Run work and propagate failures across WORLD before ranks proceed.

    The action must be local or coordinate its own failures within every group
    it uses. This function cannot recover peers blocked inside the action.
    """
    if not dist.is_initialized():
        return action()
    result = None
    error = None
    try:
        result = action()
    except Exception as exc:
        error = exc
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    for rank_error in errors:
        if rank_error is not None:
            raise rank_error
    return result
