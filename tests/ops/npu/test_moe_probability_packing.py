"""Hardware contracts for the optional int32 probability packing provider."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.ops import OpDispatcher, registry
from torchturbo.runtime import is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [0, 37])
@pytest.mark.parametrize("flat", [False, True])
def test_private_packing_preserves_strides_gradients_and_checkpoint(dtype, num_tokens, flat):
    pytest.importorskip("triton.backends.ascend.compiler")
    from torch_npu import npu_moe_token_permute
    from torchturbo.ops.providers.npu._kernels.moe_probability import pack_probabilities

    ids = torch.tensor([[0, 2]], device="npu", dtype=torch.int32).expand(num_tokens, -1).contiguous()
    if num_tokens:
        _, inverse = npu_moe_token_permute(torch.ones(num_tokens, 64, device="npu", dtype=dtype), ids)
    else:
        inverse = torch.empty(0, device="npu", dtype=torch.int32)
    storage = torch.arange(num_tokens * 8, device="npu").to(dtype).view(num_tokens * 2, 4)
    probs = storage[::2, ::2]
    if flat:
        probs = storage.flatten()[::4]
    probs = probs.detach().requires_grad_()
    gradient = (torch.arange(inverse.numel() * 2, device="npu").to(dtype) / 16)[::2]
    output = pack_probabilities(probs, inverse)
    reference = torch.empty_like(probs.flatten()).index_put((inverse.long(),), probs.flatten())
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(reference, probs, gradient)[0]
    torch.testing.assert_close(torch.autograd.grad(output, probs, gradient)[0], expected_grad, rtol=0, atol=0)
    for reentrant in (False, True):
        probs.grad = None
        replay = checkpoint(lambda values: pack_probabilities(values, inverse), probs, use_reentrant=reentrant)
        replay.backward(gradient)
        torch.testing.assert_close(probs.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [0, 37])
def test_optional_provider_matches_existing_permutation_and_chunk_gradients(dtype, num_tokens):
    pytest.importorskip("triton.backends.ascend.compiler")
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    ids = torch.tensor([[0, 2]], device="npu").expand(num_tokens, -1).contiguous()
    tokens = ((torch.arange(num_tokens * 128, device="npu") % 17) / 32).view(num_tokens, 128).to(dtype).requires_grad_()
    probs = (torch.arange(num_tokens * 4, device="npu").to(dtype) / 32).view(num_tokens, 4)[:, ::2].detach().requires_grad_()
    mask = torch.zeros(num_tokens, 3, device="npu", dtype=torch.bool).scatter_(1, ids, True)
    candidate = dispatcher.moe_permutation(
        tokens, mask, probs.new_zeros((num_tokens, 3)).scatter(1, ids, probs),
        expert_indices=ids,
        impl="npu_triton_moe_mask_permutation",
    )
    reference = dispatcher.moe_permutation(
        tokens, mask, probs.new_zeros((num_tokens, 3)).scatter(1, ids, probs),
        expert_indices=ids,
        impl="npu_moe_mask_permutation",
    )
    torch.testing.assert_close(candidate.tokens, reference.tokens, rtol=0, atol=0)
    torch.testing.assert_close(candidate.probs, reference.probs, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(candidate.probs.sum(), probs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference.probs.sum(), probs, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    sizes = torch.tensor([num_tokens, 0, num_tokens], device="npu")
    order = torch.tensor([2, 1, 0], device="npu")
    actual, actual_probs = dispatcher.moe_sort_chunks(
        candidate.tokens, sizes, order, candidate.probs,
        impl="npu_triton_moe_sort_chunks",
    )
    expected, expected_probs = dispatcher.moe_sort_chunks(
        reference.tokens, sizes, order, reference.probs, impl="npu_moe_sort_chunks",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)
    actual_grads = torch.autograd.grad((actual * actual_probs[:, None]).sum(), (tokens, probs))
    expected_grads = torch.autograd.grad((expected * expected_probs[:, None]).sum(), (tokens, probs))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("columns", [1, 3])
@pytest.mark.parametrize("implementation", ["npu_moe_sort_chunks", "npu_triton_moe_sort_chunks"])
def test_chunk_probability_rows_keep_shape_and_gradients(dtype, columns, implementation):
    if implementation == "npu_triton_moe_sort_chunks":
        pytest.importorskip("triton.backends.ascend.compiler")
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    tokens = torch.arange(5, device="npu").to(dtype)[:, None].requires_grad_()
    storage = (torch.arange(20 * columns, device="npu").to(dtype) / 32).view(10, columns * 2)
    probs = storage[::2, ::2].detach().requires_grad_()
    output, weights = dispatcher.moe_sort_chunks(
        tokens, torch.tensor([2, 0, 3], device="npu"),
        torch.tensor([2, 1, 0], device="npu"), probs, impl=implementation,
    )
    source_rows = torch.tensor([2, 3, 4, 0, 1], device="npu")
    assert weights.shape == probs.shape
    torch.testing.assert_close(output, tokens[source_rows], rtol=0, atol=0)
    torch.testing.assert_close(weights, probs[source_rows], rtol=0, atol=0)
    (output * weights).sum().backward()
    torch.testing.assert_close(tokens.grad, probs.detach().sum(1, keepdim=True), rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs), rtol=0, atol=0)
