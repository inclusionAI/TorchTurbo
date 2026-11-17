from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_swiglu(dtype: torch.dtype) -> None:
    values = torch.randn(4, 8, 12, dtype=torch.float32)
    x = values.to(device="npu", dtype=dtype)
    gate, up = torch.chunk(values, 2, dim=-1)
    expected = (F.silu(gate) * up).to(dtype=dtype)

    import torch_npu

    actual = torch_npu.npu_swiglu(x, dim=-1)
    # BF16 inputs give the NPU kernel a different rounding trajectory from the
    # float32 CPU reference. Observed absolute differences are one BF16 step.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(
        actual.float().cpu(), expected.float().cpu(), atol=tolerance, rtol=tolerance
    )
