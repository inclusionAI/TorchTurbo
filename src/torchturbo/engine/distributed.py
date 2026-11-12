"""Distributed training utilities for the TorchTurbo engine."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ..logging import get_logger
from ..runtime import get_device_runtime

logger = get_logger(__name__)

DEFAULT_PROCESS_GROUP_TIMEOUT = timedelta(seconds=1800)


def assert_finite(value: torch.Tensor, name: str) -> None:
    """Raise on all ranks if any local value is non-finite."""
    value = value.detach()
    if isinstance(value, DTensor):
        value = value.to_local()
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
    if not finite.item():
        raise RuntimeError(f"Non-finite {name} on at least one rank; stopping training.")


def init_process_group_if_needed(enable_cpu_backend: bool = False) -> bool:
    """Initialize the default process group if it is absent.

    Idempotent: returns immediately once a default group exists (so it is safe
    to call before the training framework, which will then reuse this group).

    Parameters
    ----------
    enable_cpu_backend:
        When ``True``, register a CPU ``gloo`` backend alongside the device
        backend (required for FSDP CPU-offload collectives).

    Returns
    -------
    ``True`` if this call created the process group, ``False`` otherwise
    (already initialized, or no distributed launch detected).
    """
    if dist.is_initialized():
        return False

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.debug_rank0(
            "No distributed launch detected (RANK/WORLD_SIZE unset); "
            "skipping process-group initialization."
        )
        return False

    runtime = get_device_runtime()
    device_type = runtime.device.type
    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            runtime.set_device(local_rank)
        except Exception:  # pragma: no cover - defensive, device-specific
            logger.warning_rank0(
                "Failed to set %s device to local rank %d before process-group init",
                device_type,
                local_rank,
                exc_info=True,
            )

    backend = runtime.dist_backend

    if enable_cpu_backend and device_type != "cpu":
        backend = f"{device_type}:{backend},cpu:gloo"

    logger.info_rank0(
        "Initializing torch.distributed (backend=%s, timeout=%s, cpu_offload=%s)",
        backend,
        DEFAULT_PROCESS_GROUP_TIMEOUT,
        enable_cpu_backend,
    )
    dist.init_process_group(
        backend=backend,
        timeout=DEFAULT_PROCESS_GROUP_TIMEOUT,
    )
    return True
