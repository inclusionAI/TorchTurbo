from __future__ import annotations

import importlib

import pytest

from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    """Model tests run without an engine: point the ops proxy at CPU.

    The patched ``get_rope_index`` dispatches through the ops proxy, which
    requires an initialized dispatcher. Engine-driven runs configure it
    during startup; standalone tests use the CPU runtime so the native
    PyTorch baseline of each op is selected.
    """
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.ops"), "_OPS", None
    )
    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.AUTO)
