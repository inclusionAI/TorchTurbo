"""NPU permutation parity, empty sources, and routing gradients."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.ops import OpPolicy, get_ops, get_or_create_ops, ops
from torchturbo.ops.moe import MoEPermutation
from torchturbo.runtime import is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(autouse=True)
def npu_ops():
    if is_npu_available():
        get_or_create_ops(device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)


@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unweighted_combine_saves_only_indices(topk, dtype):
    num_tokens, hidden_size = 256, 128
    packed = torch.randn(
        num_tokens * topk, hidden_size, device="npu", dtype=dtype, requires_grad=True,
    )
    rows = torch.arange(
        num_tokens * topk, device="npu", dtype=torch.int32,
    ).view(topk, num_tokens).T
    experts = torch.arange(topk, device="npu").expand(num_tokens, -1)
    mapping = MoEPermutation(packed, None, rows.flip(1), experts.flip(1))
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        output = get_ops().moe_unpermutation(
            mapping, torch.Size([num_tokens, hidden_size]),
            impl="npu_moe_mask_unpermutation",
        )
    assert saved and all(not tensor.is_floating_point() for tensor in saved)
    torch.testing.assert_close(output, packed.view(topk, num_tokens, hidden_size).sum(0))
    gradient = torch.randn_like(output)
    output.backward(gradient)
    torch.testing.assert_close(packed.grad, gradient.repeat(topk, 1), rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [False, True])
def test_permutation_saves_only_indices(chunks):
    tokens = torch.randn(256, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        if chunks:
            sizes = torch.tensor([64, 128, 0, 64], device="npu")
            order = torch.tensor([3, 1, 0, 2], device="npu")
            output, _ = get_ops().moe_sort_chunks(tokens, sizes, order)
            expected = torch.cat([tokens[192:], tokens[64:192], tokens[:64]])
        else:
            indices = torch.tensor([[0, 2]], device="npu").expand(256, -1).contiguous()
            mask = torch.zeros(256, 3, device="npu", dtype=torch.bool).scatter_(1, indices, True)
            output = get_ops().moe_permutation(tokens, mask, expert_indices=indices).tokens
            expected = tokens.repeat(2, 1)
    assert saved and all(not tensor.is_floating_point() for tensor in saved)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradient = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, tokens, gradient)[0]
    expected_grad = torch.autograd.grad(expected, tokens, gradient)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens,topk", [(0, 0), (0, 1), (0, 2), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unpermutation_checkpoint_recomputes_before_returning_input_gradients(
    num_tokens, topk, dtype,
):
    events = []

    def forward(packed):
        events.append("forward")
        received = packed.clone()
        # Returning this gradient starts token-combine's collective backward.
        received.register_hook(lambda grad: events.append("input_gradient"))
        rows = torch.arange(
            num_tokens * topk, device="npu", dtype=torch.int32,
        ).view(topk, num_tokens).T
        experts = torch.arange(topk, device="npu").expand(num_tokens, -1)
        mapping = MoEPermutation(received, None, rows, experts)
        return get_ops().moe_unpermutation(
            mapping, torch.Size([num_tokens, 128]), impl="npu_moe_mask_unpermutation",
        )

    packed = torch.ones(
        num_tokens * topk, 128, device="npu", dtype=dtype, requires_grad=True,
    )
    output = checkpoint(forward, packed, use_reentrant=False)
    events.clear()
    output.sum().backward()
    assert events == ["forward", "input_gradient"]
    torch.testing.assert_close(output, torch.full_like(output, topk))
    torch.testing.assert_close(packed.grad, torch.ones_like(packed))


@pytest.mark.parametrize("num_tokens", [0, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("num_experts", [4, 8])
def test_custom_permutation_matches_native(
    num_tokens, dtype, input_grad, with_probs, topk, num_experts,
):
    torch.manual_seed(71)
    tokens = (torch.randint(-3, 4, (num_tokens, 128), device="npu") / 16).to(dtype)
    tokens.requires_grad_(input_grad)
    experts = torch.arange(num_tokens, device="npu") % num_experts
    indices = (experts[:, None] + torch.arange(topk, device="npu")) % num_experts
    probs = torch.zeros(num_tokens, num_experts, device="npu").scatter(
        1, indices, 1.0 / topk
    )
    probs.requires_grad_()
    mask = probs.detach().bool()
    mapping = get_ops().moe_permutation(
        tokens,
        mask,
        probs=probs if with_probs else None,
        num_out_tokens=num_tokens * topk,
        expert_indices=indices.sort(dim=-1).values,
        impl="npu_moe_mask_permutation",
    )
    packed = mapping.tokens
    packed_probs = mapping.probs
    native_mapping = get_ops().moe_permutation(
        tokens,
        mask,
        probs=probs if with_probs else None,
        impl="torch_moe_mask_permutation",
    )
    native_packed = native_mapping.tokens
    native_probs = native_mapping.probs
    assert packed.shape == (num_tokens * topk, tokens.shape[1])
    assert packed.requires_grad == input_grad
    torch.testing.assert_close(packed, native_packed, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(packed_probs, native_probs, rtol=0, atol=0)
        grad = torch.autograd.grad(packed_probs.sum(), probs, retain_graph=True)[0]
        torch.testing.assert_close(grad, mask.float(), rtol=0, atol=0)
    else:
        assert packed_probs is None
    output = get_ops().moe_unpermutation(
        mapping, tokens.shape, probs=probs, impl="npu_moe_mask_unpermutation"
    )
    native = get_ops().moe_unpermutation(
        native_mapping, tokens.shape, probs=probs, impl="torch_moe_mask_unpermutation"
    )
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    torch.testing.assert_close(output, tokens, rtol=0, atol=0)
    grad_inputs = (tokens, probs) if input_grad else (probs,)
    actual = torch.autograd.grad(output.sum(), grad_inputs, retain_graph=True)
    expected = torch.autograd.grad(native.sum(), grad_inputs, retain_graph=True)
    for grad, reference in zip(actual, expected):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)
    if input_grad:
        torch.testing.assert_close(actual[0], torch.ones_like(tokens), rtol=0, atol=0)
    unweighted = get_ops().moe_unpermutation(
        mapping, tokens.shape, impl="npu_moe_mask_unpermutation"
    )
    torch.testing.assert_close(unweighted, tokens * topk, rtol=0, atol=0)


@pytest.mark.parametrize("use_tokens", [False, True])
def test_permutation_keeps_zero_probability_routes(use_tokens):
    tokens = torch.full((1, 128), 2.0, device="npu", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.tensor([[0.0, 1.0]], device="npu", requires_grad=True)
    mapping = get_ops().moe_permutation(
        tokens, torch.ones(1, 2, device="npu", dtype=torch.bool), probs,
        impl="npu_moe_mask_permutation",
    )
    torch.testing.assert_close(mapping.tokens, tokens.repeat(2, 1), rtol=0, atol=0)
    torch.testing.assert_close(mapping.probs, probs.flatten(), rtol=0, atol=0)
    if use_tokens:
        mapping.tokens.sum().backward()
        torch.testing.assert_close(tokens.grad, torch.full_like(tokens, 2), rtol=0, atol=0)
        assert probs.grad is None
    else:
        mapping.probs.sum().backward()
        torch.testing.assert_close(probs.grad, torch.ones_like(probs), rtol=0, atol=0)
        assert tokens.grad is None


@pytest.mark.parametrize("token_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("prob_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_permutation_preserves_mixed_dtypes_and_probability_gradients(
    token_dtype, prob_dtype,
):
    tokens = torch.arange(5 * 128, device="npu").reshape(5, 128).to(token_dtype)
    tokens.requires_grad_()
    probs = (torch.arange(20, device="npu").reshape(5, 4) / 31).to(prob_dtype)
    probs.requires_grad_()
    indices = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3], [1, 2]], device="npu")
    mask = torch.zeros(5, 4, device="npu", dtype=torch.bool).scatter_(1, indices, True)
    actual = get_ops().moe_permutation(
        tokens, mask, probs, expert_indices=indices,
        impl="npu_moe_mask_permutation",
    )
    expected = get_ops().moe_permutation(
        tokens, mask, probs, expert_indices=indices, impl="torch_moe_mask_permutation",
    )
    torch.testing.assert_close(actual.tokens, expected.tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.probs, expected.probs, rtol=0, atol=0)
    token_grad = torch.arange(10, device="npu", dtype=token_dtype)[:, None].expand(10, 128)
    prob_grad = torch.arange(10, device="npu", dtype=prob_dtype)
    actual_grads = torch.autograd.grad(
        (actual.tokens, actual.probs), (tokens, probs), (token_grad, prob_grad),
    )
    expected_grads = torch.autograd.grad(
        (expected.tokens, expected.probs), (tokens, probs), (token_grad, prob_grad),
    )
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_custom_chunks_restore_order_and_gradients(empty, dtype):
    sizes = torch.tensor([0, 3, 0, 2], device="npu")
    if empty:
        sizes.zero_()
    num_tokens = 0 if empty else 5
    tokens = (
        torch.arange(num_tokens * 128, device="npu").reshape(num_tokens, 128).to(dtype)
    )
    tokens.requires_grad_()
    probs = torch.arange(num_tokens, device="npu", dtype=torch.float32).requires_grad_()
    order = torch.tensor([3, 0, 1, 2], device="npu")
    output, weights = ops.moe_sort_chunks(tokens, sizes, order, probs=probs)
    native, native_weights = get_ops().moe_sort_chunks(
        tokens, sizes, order, probs=probs, impl="torch_moe_sort_chunks"
    )
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    torch.testing.assert_close(weights, native_weights, rtol=0, atol=0)
    restored, _ = ops.moe_sort_chunks(output, sizes[order], order.argsort())
    torch.testing.assert_close(restored, tokens, rtol=0, atol=0)
    actual = torch.autograd.grad(
        (output * weights[:, None]).sum(), (tokens, probs), retain_graph=True
    )
    expected = torch.autograd.grad(
        (native * native_weights[:, None]).sum(), (tokens, probs)
    )
    for grad, reference in zip(actual, expected):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weighted_combine_keeps_packed_activation_dtype(dtype):
    num_tokens, topk, hidden_size = 64, 8, 128
    packed = torch.randn(
        num_tokens * topk, hidden_size, device="npu", dtype=dtype, requires_grad=True,
    )
    rows = torch.randperm(num_tokens * topk, device="npu", dtype=torch.int32).view(
        num_tokens, topk,
    )
    experts = torch.arange(topk, device="npu").expand(num_tokens, -1)
    probs = torch.rand(num_tokens, topk, device="npu", requires_grad=True)
    mapping = MoEPermutation(packed, None, rows, experts)
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        output = get_ops().moe_unpermutation(
            mapping, torch.Size([num_tokens, hidden_size]), probs,
            impl="npu_moe_mask_unpermutation",
        )
    features = [tensor for tensor in saved if tensor.shape == packed.shape]
    assert features and all(tensor.dtype == dtype for tensor in features)
    reference = packed[rows.long()].float().mul(probs[:, :, None]).sum(1).to(dtype)
    torch.testing.assert_close(output, reference)
    gradient = torch.randn_like(output)
    actual = torch.autograd.grad(output, (packed, probs), gradient)
    expected = torch.autograd.grad(reference, (packed, probs), gradient)
    for result, expected_grad in zip(actual, expected):
        torch.testing.assert_close(result, expected_grad, rtol=2e-4, atol=2e-4)
