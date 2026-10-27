from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_rms_norm(dtype: torch.dtype) -> None:
    hidden_size = 128
    values = torch.randn(2, 8, hidden_size, dtype=torch.float32)
    x = values.to(device="npu", dtype=dtype)
    rms_norm = Qwen3MoeRMSNorm(hidden_size).to(dtype=dtype)
    expected = rms_norm(values.to(dtype=dtype)).float()
    gamma = nn.Parameter(torch.ones(hidden_size, device="npu", dtype=dtype))

    import torch_npu

    actual = torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)[0]
    tolerance = 4e-3 if dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(
        actual.float().cpu(), expected.cpu(), atol=tolerance, rtol=tolerance
    )
