from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, registry
from torchturbo.ops.providers.npu.matmul_add import (
    TorchNpuGroupedMatmulAdd,
    TorchNpuMatmulAdd,
)
from torchturbo.runtime.npu import NpuRuntime


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


@pytest.mark.parametrize(
    ("policy", "has_grouped_kernel"),
    [(OpPolicy.NATIVE, True), (OpPolicy.AUTO, False)],
    ids=["native", "auto-without-grouped-kernel"],
)
def test_grouped_wgrad_fallback_preserves_npu_addmm_operand_dtypes(monkeypatch, policy, has_grouped_kernel):
    monkeypatch.setattr(TorchNpuGroupedMatmulAdd, "available", lambda self: has_grouped_kernel)
    monkeypatch.setattr(TorchNpuMatmulAdd, "available", lambda self: True)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry, default_policy=policy)
    x = torch.ones(3, 2, dtype=torch.bfloat16)
    dy = torch.ones(3, 4, dtype=torch.bfloat16)
    accumulator = torch.full((3, 2, 4), 0.25)
    calls = []
    cpu_addmm = torch.Tensor.addmm_

    def emulate_npu_addmm(out, lhs, rhs):
        calls.append((out.dtype, lhs.dtype, rhs.dtype))
        return cpu_addmm(out, lhs.float(), rhs.float())

    monkeypatch.setattr(torch.Tensor, "addmm_", emulate_npu_addmm)
    dispatcher.grouped_matmul_add(accumulator, x, dy, torch.tensor([1, 0, 2]))

    assert calls == [(torch.float32, torch.bfloat16, torch.bfloat16)] * 2
    expected = torch.tensor([1.25, 0.25, 2.25]).view(3, 1, 1).expand_as(accumulator)
    torch.testing.assert_close(accumulator, expected)
