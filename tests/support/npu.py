"""Runtime guards for tests that execute on Ascend NPUs."""

from __future__ import annotations

import pytest


def require_npu():
    """Skip unless torch_npu is installed and at least one NPU is available."""
    pytest.importorskip("torch_npu")
    import torch

    if not torch.npu.is_available():
        pytest.skip("NPU is not available")


def require_npus(count: int):
    """Skip unless at least ``count`` NPUs are visible to the test process."""
    require_npu()
    import torch

    if torch.npu.device_count() < count:
        pytest.skip(f"requires at least {count} NPUs")
