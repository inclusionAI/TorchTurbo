from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from torchturbo.ops.providers.npu.moe_token_permutation import (
    NpuMoeTokenPermutation,
)
from torchturbo.ops.providers.npu.moe_token_unpermutation import (
    NpuMoeTokenUnpermutation,
)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [4, 8])
def test_moe_token_permutation_and_unpermutation_round_trip(
    topk: int,
    num_experts: int,
) -> None:
    num_tokens = 16
    hidden_size = 8
    dtype = torch.bfloat16

    values = torch.randn(num_tokens, hidden_size, dtype=torch.float32)
    tokens = values.to(device="npu", dtype=dtype).requires_grad_(True)
    expert_indices = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="npu",
        dtype=torch.int32,
    )
    probs = torch.full(
        (num_tokens, topk),
        1.0 / topk,
        device="npu",
        dtype=dtype,
        requires_grad=True,
    )

    permuted_tokens, sorted_indices = NpuMoeTokenPermutation()(
        tokens, expert_indices
    )
    restored_tokens = NpuMoeTokenUnpermutation()(
        permuted_tokens, sorted_indices, probs=probs
    )

    assert sorted_indices.shape == (num_tokens * topk,)
    assert torch.equal(
        torch.sort(sorted_indices).values,
        torch.arange(num_tokens * topk, device=sorted_indices.device, dtype=torch.int32),
    )
    assert torch.allclose(restored_tokens, tokens, atol=5e-3, rtol=5e-3)

    restored_tokens.backward(torch.ones_like(restored_tokens))
    assert tokens.grad is not None
    assert torch.allclose(tokens.grad, torch.ones_like(tokens), atol=5e-3)
    assert probs.grad is not None
    assert torch.isfinite(probs.grad).all()
