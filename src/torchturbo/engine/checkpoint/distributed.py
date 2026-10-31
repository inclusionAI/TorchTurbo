"""Coordinate checkpoint failures across ranks."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import torch.distributed as dist

from torchturbo.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

_CTRL_GROUP: dist.ProcessGroup | None = None
_CTRL_GROUP_FAILED = False


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


def get_control_plane_group() -> dist.ProcessGroup | None:
    """Return a CPU (gloo) group for DCP object collectives, or None.

    DCP exchanges SavePlan/metadata through object collectives. On device
    backends these stage the pickled plans as device tensors, which can
    stall at high world sizes (see issue #9) and costs ``world_size x
    plan_size`` device memory at the most memory-pressured moment of
    training. Set ``TORCHTURBO_DCP_CTRL_GROUP=gloo`` to route that
    exchange over a CPU gloo group instead. The group is created lazily
    on first save and reused; creation failures fall back to the default
    group.
    """
    global _CTRL_GROUP, _CTRL_GROUP_FAILED
    if _CTRL_GROUP is not None or _CTRL_GROUP_FAILED:
        return _CTRL_GROUP
    enabled = os.environ.get("TORCHTURBO_DCP_CTRL_GROUP", "").lower()
    if enabled not in ("gloo", "1", "true"):
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_world_size() <= 1:
        return None
    try:
        _CTRL_GROUP = dist.new_group(backend="gloo")
    except Exception:
        _CTRL_GROUP_FAILED = True
        return None
    logger.info_rank0("DCP control-plane gloo group created")
    return _CTRL_GROUP
