"""Chunked expert MLP: chunk bookkeeping, forward, and first-order grads.

The chunking machinery (row ranges that may cut expert boundaries, per-chunk
overlap sizes, cross-chunk gradient accumulation) is exercised on CPU via
the native grouped_matmul / gated_activation implementations. The
single-chunk call is the reference: identical math, no chunk splits.
"""

import pytest
import torch

from torchturbo.models.modules.moe.chunked_mlp import (
    CHUNK_ROWS,
    _chunks,
    chunked_expert_mlp,
)
from torchturbo.ops import get_or_create_ops


@pytest.fixture(autouse=True)
def _cpu_ops_dispatcher():
    from torchturbo.runtime.cpu import CpuRuntime

    get_or_create_ops(device_runtime=CpuRuntime())


def _make_inputs(tokens: int = 17, experts: int = 3, hidden: int = 6, inter: int = 4):
    generator = torch.Generator().manual_seed(2026)
    counts = torch.tensor([7, 9, 1], dtype=torch.int64)
    assert int(counts.sum()) == tokens
    x = torch.randn(tokens, hidden, generator=generator) * 2.0
    # Scale weights so the SwiGLU clamps at the limit are exercised.
    weight1 = torch.randn(experts, hidden, 2 * inter, generator=generator) * 2.0
    weight2 = torch.randn(experts, inter, hidden, generator=generator) * 2.0
    return x, weight1, weight2, counts


def test_chunks_partition_tokens():
    counts = (3, 5, 2)
    covered = []
    expert_rows = {expert: 0 for expert in range(len(counts))}
    for start, end, first, last, sizes in _chunks(counts, chunk_rows=4):
        assert start < end
        covered.extend(range(start, end))
        assert (last - first) == sizes.numel()
        assert int(sizes.sum()) == end - start
        for offset, expert in enumerate(range(first, last)):
            expert_rows[expert] += int(sizes[offset])
    assert covered == list(range(sum(counts)))
    assert expert_rows == {expert: count for expert, count in enumerate(counts)}


def test_forward_matches_single_chunk():
    x, weight1, weight2, counts = _make_inputs()
    single = chunked_expert_mlp(x, weight1, weight2, counts, limit=0.5, chunk_rows=17)
    chunked = chunked_expert_mlp(x, weight1, weight2, counts, limit=0.5, chunk_rows=3)
    assert single.shape == (17, weight2.shape[-1])
    # Chunk boundaries change the GEMM M-blocking, so allow last-bit noise.
    torch.testing.assert_close(chunked, single, rtol=1e-6, atol=1e-6)


def test_gradients_match_single_chunk():
    x, weight1, weight2, counts = _make_inputs()

    def run(chunk_rows):
        inputs = [
            t.detach().clone().requires_grad_() for t in (x, weight1, weight2)
        ]
        loss = chunked_expert_mlp(
            inputs[0], inputs[1], inputs[2], counts, limit=0.5, chunk_rows=chunk_rows
        ).square().sum()
        loss.backward()
        return [t.grad for t in inputs]

    single = run(17)
    chunked = run(3)
    for reference, actual in zip(single, chunked):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_default_chunk_rows_constant():
    # The module constant must stay a plain positive int so the production
    # call sites (which rely on the default) keep chunking.
    assert isinstance(CHUNK_ROWS, int)
    assert CHUNK_ROWS > 1


def test_invalid_chunk_rows_raises():
    x, weight1, weight2, counts = _make_inputs()
    with pytest.raises(ValueError, match="chunk_rows must be positive"):
        chunked_expert_mlp(x, weight1, weight2, counts, limit=0.5, chunk_rows=0)


def test_zero_token_experts_are_skipped():
    generator = torch.Generator().manual_seed(2026)
    counts = torch.tensor([0, 5, 0, 4], dtype=torch.int64)
    x = torch.randn(9, 6, generator=generator)
    weight1 = torch.randn(4, 6, 8, generator=generator)
    weight2 = torch.randn(4, 4, 6, generator=generator)
    single = chunked_expert_mlp(x, weight1, weight2, counts, limit=1.0, chunk_rows=9)
    chunked = chunked_expert_mlp(x, weight1, weight2, counts, limit=1.0, chunk_rows=2)
    torch.testing.assert_close(chunked, single, rtol=1e-6, atol=1e-6)
