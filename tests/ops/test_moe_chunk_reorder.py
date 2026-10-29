from __future__ import annotations

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.registry import registry
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime


@pytest.mark.parametrize("runtime", [CpuRuntime, CudaRuntime])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE])
def test_native_dispatch(runtime, policy):
    dispatcher = OpDispatcher(
        device_runtime=runtime(), registry=registry, default_policy=policy
    )
    selected = dispatcher._candidate_impl(
        "moe_chunk_reorder", dispatcher.device_runtime.turbo_device
    )
    assert selected.op_spec().impl_name == "torch_moe_chunk_reorder"


@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_native_forward_backward(with_probs, empty):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    if empty:
        counts.zero_()
    num_tokens = int(counts.sum())
    tokens = torch.randn(num_tokens, 32, requires_grad=True)
    probs = torch.randn(num_tokens, requires_grad=True) if with_probs else None
    # Chunk order [0, 3, 1, 4, 2, 5] gives these independent row indices.
    indices = torch.tensor([] if empty else [0, 1, 3, 4, 5, 2, 6, 7], dtype=torch.long)
    output, scales = dispatcher.moe_chunk_reorder(tokens, counts, probs)
    torch.testing.assert_close(output, tokens[indices], rtol=0, atol=0)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected_grad = torch.empty_like(tokens)
    expected_grad[indices] = grad
    torch.testing.assert_close(tokens.grad, expected_grad, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(scales, probs[indices], rtol=0, atol=0)
        scales.backward(torch.arange(num_tokens, dtype=probs.dtype))
        expected_probs_grad = torch.empty_like(probs)
        expected_probs_grad[indices] = torch.arange(num_tokens, dtype=probs.dtype)
        torch.testing.assert_close(probs.grad, expected_probs_grad, rtol=0, atol=0)
    else:
        assert scales is None


@pytest.fixture
def npu_runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an available NPU")
    return NpuRuntime()


@pytest.fixture
def npu_dispatcher(npu_runtime):
    result = OpDispatcher(device_runtime=npu_runtime, registry=registry)
    selected = result._candidate_impl(
        "moe_chunk_reorder", result.device_runtime.turbo_device
    )
    assert selected.op_spec().impl_name == "npu_moe_re_routing"
    return result


def _reference(tokens, counts, probs=None):
    chunks = torch.split(tokens, counts.reshape(-1).tolist())
    order = torch.arange(counts.numel()).reshape(counts.shape).T.reshape(-1)
    output = torch.cat([chunks[index] for index in order.tolist()])
    if probs is None:
        return output, None
    scales = torch.split(probs, counts.reshape(-1).tolist())
    return output, torch.cat([scales[index] for index in order.tolist()])


def _assert_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("probs_dtype", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "counts", [[[2, 0, 1], [0, 3, 2]], [[0, 0, 0], [0, 0, 0]], [[2], [1], [3]]]
)
def test_npu_forward_backward_and_inverse(npu_dispatcher, dtype, probs_dtype, counts):
    counts = torch.tensor(counts, dtype=torch.int64)
    num_tokens = int(counts.sum())
    tokens = torch.randn(num_tokens, 128, device="npu", dtype=dtype, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    probs = reference_probs = None
    if probs_dtype is not None:
        probs = torch.randn(num_tokens, device="npu", dtype=probs_dtype, requires_grad=True)
        reference_probs = probs.detach().clone().requires_grad_()
    output, scales = npu_dispatcher.moe_chunk_reorder(tokens, counts, probs=probs)
    expected, expected_scales = _reference(reference_tokens, counts, reference_probs)
    _assert_equal(output, expected)
    restored, restored_scales = npu_dispatcher.moe_chunk_reorder(output, counts.T, probs=scales)
    _assert_equal(restored, tokens)
    grad = torch.randn_like(output)
    if probs is not None:
        _assert_equal(scales, expected_scales)
        _assert_equal(restored_scales, probs)
        grad_scales = torch.randn_like(scales)
        torch.autograd.backward((output, scales), (grad, grad_scales))
        torch.autograd.backward((expected, expected_scales), (grad, grad_scales))
        _assert_equal(probs.grad, reference_probs.grad)
    else:
        assert scales is None
        assert restored_scales is None
        output.backward(grad)
        expected.backward(grad)
    _assert_equal(tokens.grad, reference_tokens.grad)


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_checkpoint(npu_dispatcher, use_reentrant, dtype):
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    tokens = torch.randn(8, 128, device="npu", dtype=dtype, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    probs = torch.randn(8, device="npu", requires_grad=True)
    reference_probs = probs.detach().clone().requires_grad_()

    def reorder(x, p):
        return npu_dispatcher.moe_chunk_reorder(x, counts, probs=p)

    output, scales = checkpoint(reorder, tokens, probs, use_reentrant=use_reentrant)
    expected, expected_scales = _reference(reference_tokens, counts, reference_probs)
    _assert_equal(output, expected)
    _assert_equal(scales, expected_scales)
    grad, grad_scales = torch.randn_like(output), torch.randn_like(scales)
    torch.autograd.backward((output, scales), (grad, grad_scales))
    torch.autograd.backward((expected, expected_scales), (grad, grad_scales))
    _assert_equal(tokens.grad, reference_tokens.grad)
    _assert_equal(probs.grad, reference_probs.grad)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_interleaved_microbatches(npu_dispatcher, dtype):
    batches = []
    for counts in ([[2, 0, 1], [0, 3, 2]], [[0, 1, 2], [3, 2, 0]]):
        counts = torch.tensor(counts)
        tokens = torch.randn(8, 128, device="npu", dtype=dtype, requires_grad=True)
        reference_tokens = tokens.detach().clone().requires_grad_()
        output, _ = npu_dispatcher.moe_chunk_reorder(tokens, counts)
        expected, _ = _reference(reference_tokens, counts)
        _assert_equal(output, expected)
        batches.append((tokens, reference_tokens, output, expected))
    for tokens, reference_tokens, output, expected in reversed(batches):
        grad = torch.randn_like(output)
        output.backward(grad)
        expected.backward(grad)
        _assert_equal(tokens.grad, reference_tokens.grad)


@pytest.mark.parametrize("output_index", [0, 1])
def test_npu_unused_output_gradient(npu_dispatcher, output_index):
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    tokens = torch.randn(8, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.randn(8, device="npu", requires_grad=True)
    actual = npu_dispatcher.moe_chunk_reorder(tokens, counts, probs=probs)[output_index]
    expected = _reference(tokens, counts, probs)[output_index]
    leaf = (tokens, probs)[output_index]
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, leaf, grad)[0]
    expected_grad = torch.autograd.grad(expected, leaf, grad)[0]
    _assert_equal(actual_grad, expected_grad)


def test_native_policy_on_npu(npu_runtime):
    dispatcher = OpDispatcher(
        device_runtime=npu_runtime, registry=registry, default_policy=OpPolicy.NATIVE
    )
    selected = dispatcher._candidate_impl(
        "moe_chunk_reorder", dispatcher.device_runtime.turbo_device
    )
    assert selected.op_spec().impl_name == "torch_moe_chunk_reorder"
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    tokens = torch.randn(8, 128, device="npu", dtype=torch.bfloat16)
    output, _ = dispatcher.moe_chunk_reorder(tokens, counts)
    _assert_equal(output, _reference(tokens, counts)[0])


@pytest.mark.parametrize("requires_tokens,requires_probs", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("probs_dtype", [torch.float32, torch.bfloat16])
def test_npu_strided_gradients(npu_dispatcher, requires_tokens, requires_probs, probs_dtype):
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    tokens = torch.randn(8, 128, device="npu", dtype=torch.bfloat16, requires_grad=requires_tokens)
    probs = torch.randn(8, device="npu", dtype=probs_dtype, requires_grad=requires_probs)
    output, scales = npu_dispatcher.moe_chunk_reorder(tokens, counts, probs)
    expected, expected_scales = _reference(tokens, counts, probs)
    dy = torch.randn(8, 256, device="npu", dtype=tokens.dtype)[:, ::2]
    dp = torch.randn(16, device="npu", dtype=probs_dtype)[::2]
    outputs, references, gradients, leaves = [], [], [], []
    if requires_tokens:
        outputs.append(output)
        references.append(expected)
        gradients.append(dy)
        leaves.append(tokens)
    if requires_probs:
        outputs.append(scales)
        references.append(expected_scales)
        gradients.append(dp)
        leaves.append(probs)
    for retain in (True, False):
        actual = torch.autograd.grad(outputs, leaves, gradients, retain_graph=retain)
        reference = torch.autograd.grad(references, leaves, gradients, retain_graph=retain)
        for value, ref in zip(actual, reference):
            _assert_equal(value, ref)
