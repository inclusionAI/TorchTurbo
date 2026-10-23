"""NPU permutation parity, empty sources, and routing gradients."""

import pytest
import torch

from torchturbo.models.modules.moe.utils import permute, sort_chunks_by_idxs, unpermute
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import is_npu_available
from torchturbo.runtime.npu import NpuRuntime


pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(autouse=True)
def npu_ops():
    if is_npu_available():
        get_or_create_ops(device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)


@pytest.mark.parametrize("num_tokens", [0, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("with_probs", [False, True])
def test_fused_permutation_matches_native(num_tokens, dtype, input_grad, with_probs):
    torch.manual_seed(71)
    tokens = (torch.randint(-3, 4, (num_tokens, 128), device="npu") / 16).to(dtype)
    tokens.requires_grad_(input_grad)
    experts = torch.arange(num_tokens, device="npu") % 4
    indices = torch.stack((experts, (experts + 1) % 4), dim=1)
    probs = torch.zeros(num_tokens, 4, device="npu").scatter(1, indices, .25)
    probs.requires_grad_()
    mask = probs.detach().bool()
    packed, packed_probs, mapping = permute(
        tokens, mask, probs=probs if with_probs else None,
        num_out_tokens=num_tokens * 2, fused=True,
        expert_indices=indices.sort(dim=-1).values,
    )
    native_packed, native_probs, native_mapping = permute(
        tokens, mask, probs=probs if with_probs else None,
    )
    torch.testing.assert_close(packed, native_packed, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(packed_probs, native_probs, rtol=0, atol=0)
        grad = torch.autograd.grad(packed_probs.sum(), probs, retain_graph=True)[0]
        torch.testing.assert_close(grad, mask.float(), rtol=0, atol=0)
    output = unpermute(packed, mapping, tokens.shape, probs=probs, routing_map=mask, fused=True)
    native = unpermute(native_packed, native_mapping, tokens.shape, probs=probs, routing_map=mask)
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    grad_inputs = (tokens, probs) if input_grad else (probs,)
    actual = torch.autograd.grad(output.sum(), grad_inputs, retain_graph=True)
    expected = torch.autograd.grad(native.sum(), grad_inputs, retain_graph=True)
    for grad, reference in zip(actual, expected):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)
    unweighted = unpermute(packed, mapping, tokens.shape, fused=True)
    torch.testing.assert_close(unweighted, tokens * 2, rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_chunks_restore_order_and_gradients(empty, dtype):
    sizes = torch.tensor([0, 3, 0, 2], device="npu")
    if empty:
        sizes.zero_()
    num_tokens = 0 if empty else 5
    tokens = torch.arange(num_tokens * 128, device="npu").reshape(num_tokens, 128).to(dtype)
    tokens.requires_grad_()
    probs = torch.arange(num_tokens, device="npu", dtype=torch.float32).requires_grad_()
    order = torch.tensor([3, 0, 1, 2], device="npu")
    output, weights = sort_chunks_by_idxs(tokens, sizes, order, probs=probs, fused=True)
    native, native_weights = sort_chunks_by_idxs(tokens, sizes.cpu(), order.cpu(), probs=probs)
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    torch.testing.assert_close(weights, native_weights, rtol=0, atol=0)
    restored, _ = sort_chunks_by_idxs(output, sizes[order], order.argsort(), fused=True)
    torch.testing.assert_close(restored, tokens, rtol=0, atol=0)
    actual = torch.autograd.grad((output * weights[:, None]).sum(), (tokens, probs), retain_graph=True)
    expected = torch.autograd.grad((native * native_weights[:, None]).sum(), (tokens, probs))
    for grad, reference in zip(actual, expected):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)
