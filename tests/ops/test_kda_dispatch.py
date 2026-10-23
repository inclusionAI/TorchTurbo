"""Dispatcher-level contracts for the chunk_kda op.

The dynamic ``__getattr__`` facade must route the op to the native fla
provider on CPU (that is the default path everywhere, NPU included), pass
the model kwargs through, and the native provider must accept the
head-first Q/K/V spelling the fused convolution produces and convert it
back to sequence-major for fla's triton kernels.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.ops.providers import iter_provider_modules
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    runtime_impl.is_device_available.cache_clear()
    get_or_create_ops(device_runtime=CpuRuntime())
    yield
    runtime_impl.is_device_available.cache_clear()


@pytest.fixture()
def fla_stub(monkeypatch):
    calls = []

    def chunk_kda(q, k, v, g, beta, **kwargs):
        calls.append({"q": q, "k": k, "v": v, "g": g, "beta": beta, **kwargs})
        return q, None

    monkeypatch.setitem(
        sys.modules, "fla.ops.kda", SimpleNamespace(chunk_kda=chunk_kda)
    )
    return calls


def test_kda_provider_modules_are_discovered():
    cpu_modules = iter_provider_modules(TurboDevice.CPU)
    npu_modules = iter_provider_modules(TurboDevice.NPU)

    assert "torchturbo.ops.providers.native.kda" in cpu_modules
    assert "torchturbo.ops.providers.npu.kda" in npu_modules
    # The vendored adapter package is underscore-prefixed, so provider
    # discovery must keep skipping it.
    assert not any(
        module.startswith("torchturbo.ops.providers.npu._fla_npu_kda")
        for module in npu_modules
    )


def test_kda_routes_to_native_fl_by_default(fla_stub):
    ops = get_or_create_ops(device_runtime=CpuRuntime())
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    ops.chunk_kda(
        q=q, k=k, v=v, g=torch.randn(1, 8, 2, 4), beta=torch.rand(1, 8, 2),
        disable_recompute=True, head_major_qkv=True,
    )

    assert len(fla_stub) == 1
    # head-major inputs are converted back to sequence-major before the
    # fla triton kernels see them.
    assert fla_stub[0]["q"].shape == (1, 8, 2, 4)
    assert fla_stub[0]["disable_recompute"] is True


def test_kda_explicit_ascendc_impl_fails_on_cpu(fla_stub):
    ops = get_or_create_ops(device_runtime=CpuRuntime())
    q = torch.randn(1, 2, 8, 4)

    # Explicit selection never falls back; the AscendC implementation is
    # NPU-only, so a CPU runtime must refuse the call outright.
    with pytest.raises(RuntimeError, match="ascendc_chunk_kda"):
        ops.chunk_kda(
            q=q, k=q.clone(), v=q.clone(),
            g=torch.randn(1, 8, 2, 4), beta=torch.rand(1, 8, 2),
            impl="ascendc_chunk_kda",
        )
