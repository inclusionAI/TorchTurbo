"""Native KKT cache reuse and full GDN autograd equivalence."""

from itertools import accumulate

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

from fla_npu.ops.triton import chunk_scaled_dot_kkt_fwd as upstream_kkt

from torchturbo.ops.packed_params import build_packed_params
from torchturbo.ops.providers.npu._fla_npu_gdn import (
    chunk_gated_delta_rule_function as gdn,
)
from torchturbo.ops.providers.npu._fla_npu_gdn import (
    kkt,
)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kkt_matches_upstream(dtype, packed):
    keys = torch.randn(1, 6, 145, 128, device="npu", dtype=dtype)
    gate = -torch.rand(1, 145, 6, device="npu").cumsum(1) * 0.01
    beta = torch.rand_like(gate)
    metadata = {}
    if packed:
        params = build_packed_params(
            torch.tensor([0, 65, 145], dtype=torch.int64, device="npu")
        )
        metadata = {
            "cu_seqlens": params.cu_seqlens,
            "chunk_indices": params.chunk_indices["64"],
        }
    actual = kkt.chunk_scaled_dot_kkt_fwd(keys, gate, beta, **metadata)
    expected = upstream_kkt(k=keys, g=gate, beta=beta, **metadata)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_chunk_alignment_classes_share_one_kernel():
    kkt._kkt_fwd_kernel.cache.clear()
    for tokens in (960, 1024, 1025, 960):
        params = build_packed_params(
            torch.tensor([0, tokens], dtype=torch.int64, device="npu")
        )
        keys = torch.randn(1, 6, tokens, 128, device="npu", dtype=torch.bfloat16)
        gate = torch.zeros(1, tokens, 6, device="npu")
        beta = torch.ones_like(gate)
        kkt.chunk_scaled_dot_kkt_fwd(
            keys,
            gate,
            beta,
            cu_seqlens=params.cu_seqlens,
            chunk_indices=params.chunk_indices["64"],
        )
        torch.npu.synchronize()
        assert sum(len(cache) for cache in kkt._kkt_fwd_kernel.cache.values()) == 1


def test_profile_packing_shapes_reuse_the_adapter_kernel() -> None:
    assert gdn.chunk_scaled_dot_kkt_fwd is kkt.chunk_scaled_dot_kkt_fwd
    kkt._kkt_fwd_kernel.cache.clear()
    for lengths, num_chunks in [
        ([16340, 16340, 16404, 16404], 1026),
        ([2330] * 27 + [2602], 1040),
    ]:
        tokens = sum(lengths)
        params = build_packed_params(
            torch.tensor([0, *accumulate(lengths)], device="npu", dtype=torch.int64)
        )
        assert len(params.chunk_indices["64"]) == num_chunks
        keys = torch.randn(1, 12, tokens, 128, device="npu", dtype=torch.bfloat16)
        gate = -torch.rand(1, tokens, 12, device="npu") * 0.1
        beta = torch.rand_like(gate)
        metadata = {
            "cu_seqlens": params.cu_seqlens,
            "chunk_indices": params.chunk_indices["64"],
        }
        actual = gdn.chunk_scaled_dot_kkt_fwd(keys, gate, beta, **metadata)
        expected = upstream_kkt(k=keys, g=gate, beta=beta, **metadata)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.npu.synchronize()
        assert sum(len(cache) for cache in kkt._kkt_fwd_kernel.cache.values()) == 1


@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("save_intermediates", [False, True])
def test_full_gdn_outputs_and_gradients(monkeypatch, heads, save_intermediates):
    tokens = 1057
    params = build_packed_params(
        torch.tensor([0, 65, 1024, tokens], dtype=torch.int64, device="npu")
    )
    inputs = [
        torch.randn(1, heads, tokens, 128, device="npu", dtype=torch.bfloat16),
        torch.randn(1, heads, tokens, 128, device="npu", dtype=torch.bfloat16),
        torch.randn(1, heads, tokens, 256, device="npu", dtype=torch.bfloat16),
        -torch.rand(1, tokens, heads, device="npu") * 0.1,
        torch.rand(1, tokens, heads, device="npu"),
    ]
    gradient = torch.randn(1, tokens, heads, 256, device="npu", dtype=torch.bfloat16)

    def run():
        leaves = [value.detach().clone().requires_grad_() for value in inputs]
        output, _ = gdn.ChunkGatedDeltaRuleFunction.apply(
            *leaves,
            128**-0.5,
            None,
            False,
            params.cu_seqlens,
            params.cu_seqlens_list,
            params.chunk_indices,
            params.chunk_indices_list,
            True,
            64,
            save_intermediates,
        )
        gradients = torch.autograd.grad(output, leaves, gradient)
        return output.detach(), gradients

    actual, actual_grads = run()
    with monkeypatch.context() as context:
        context.setattr(gdn, "chunk_scaled_dot_kkt_fwd", upstream_kkt)
        expected, expected_grads = run()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
