# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Release expert resources before their Torch process groups are destroyed."""

import atexit
import functools
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from torchturbo.logging import get_logger

logger = get_logger(__name__)


class _Cleanup:
    """Retain each backend until collective teardown succeeds."""

    def __init__(self) -> None:
        self.callbacks: dict[Callable[[], None], Any] = {}
        self.installed = False

    def close(self, group: Any = None) -> None:
        """Remove successful owners; retain failed and unvisited owners for retry."""
        for callback, root_group in tuple(self.callbacks.items()):
            if group is None or group is dist.group.WORLD or group is root_group:
                callback()
                del self.callbacks[callback]

    def at_exit(self) -> None:
        """Best-effort release before torch_npu closes its runtime."""
        try:
            self.close()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(
                "Automatic expert cleanup failed during process exit", exc_info=True
            )

    def register_shutdown_hook(self) -> None:
        """Hook the supported Torch shutdown boundary, including imported aliases.

        TorchTurbo requires Torch >= 2.10, whose ProcessGroup.shutdown is the
        teardown boundary. The hook is process-wide but acts only on registered
        roots and WORLD. Other process groups keep their original behavior.
        """
        if self.installed:
            return
        original = c10d.ProcessGroup.shutdown

        @functools.wraps(original)
        def shutdown(group: Any, *args: Any, **kwargs: Any) -> Any:
            self.close(group)
            return original(group, *args, **kwargs)

        c10d.ProcessGroup.shutdown = shutdown
        atexit.register(self.at_exit)
        self.installed = True


_CLEANUP = _Cleanup()


def register(callback: Callable[[], None], root_group: Any) -> None:
    """Register an idempotent collective close callback for its owning group.

    Args:
        callback: Backend release operation, retaining ownership on failure.
        root_group: The explicit communication root required for release.
    """
    _CLEANUP.register_shutdown_hook()
    _CLEANUP.callbacks[callback] = root_group
