from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from torchturbo.ops.providers.npu.rotary_position_embedding import (
    NpuRotaryPositionEmbedding,
)


def _reference_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    if interleave:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
    else:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 1])
def test_rotary_position_embedding_matches_reference(
    dtype: torch.dtype, mode: int
) -> None:
    values = torch.rand(2, 16, 2, 16, dtype=torch.float32) * 2 - 1
    cos_values = torch.rand(1, 16, 1, 16, dtype=torch.float32)
    sin_values = torch.rand(1, 16, 1, 16, dtype=torch.float32)

    x = values.to(device="npu", dtype=dtype)
    cos = cos_values.to(device="npu", dtype=dtype)
    sin = sin_values.to(device="npu", dtype=dtype)
    expected = _reference_rotary(values, cos_values, sin_values, mode == 1)
    actual = NpuRotaryPositionEmbedding()(x, cos, sin, mode)

    tolerance = 2e-3 if dtype == torch.float16 else 6e-3
    assert torch.allclose(
        actual.float().cpu(), expected, atol=tolerance, rtol=tolerance
    )


def test_rotary_position_embedding_rejects_invalid_mode() -> None:
    x = torch.randn(1, 2, 2, 16, device="npu", dtype=torch.bfloat16)
    cos = torch.randn(1, 2, 1, 16, device="npu", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)

    with pytest.raises(ValueError, match="RoPE mode must be 0 or 1"):
        NpuRotaryPositionEmbedding()(x, cos, sin, 2)
