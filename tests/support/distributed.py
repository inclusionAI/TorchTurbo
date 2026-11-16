"""Shared teardown for tests that use torch.distributed."""

from __future__ import annotations

import time

import torch.distributed as dist

_GLOO_SHUTDOWN_GRACE_SECONDS = 0.3


def destroy_process_group() -> None:
    """Destroy the default group and let Gloo's run loop shut down."""
    if not dist.is_available() or not dist.is_initialized():
        return

    backend = dist.get_backend()
    dist.destroy_process_group()
    if backend == "gloo":
        # PyTorch 2.10 can abort while Gloo releases Python work references.
        time.sleep(_GLOO_SHUTDOWN_GRACE_SECONDS)
