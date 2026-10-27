from __future__ import annotations

import sys
from types import ModuleType

import torch

from torchturbo.ops.providers.npu.matmul_add import (
    TorchNpuMatmulAdd,
)


def test_torch_npu_fallback_uses_in_place_addmm():
    lhs = torch.randn(3, 5)
    rhs = torch.randn(3, 4)
    accumulator = torch.zeros(5, 4)

    TorchNpuMatmulAdd()(
        accumulator,
        lhs,
        rhs,
        transpose_lhs=True,
    )

    assert torch.allclose(accumulator, lhs.t().matmul(rhs))


def test_torch_npu_impl_detects_torch_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    assert TorchNpuMatmulAdd().available()
