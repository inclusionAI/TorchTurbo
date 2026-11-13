"""Permutation order, inverse mappings, and operator selection contracts."""

import builtins
import importlib
import sys
from dataclasses import replace
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest
import torch

import torchturbo.runtime.runtime as runtime_module
from torchturbo.ops import OpPolicy, get_or_create_ops, ops
from torchturbo.ops.moe import MoEPermutation
from torchturbo.runtime import is_cuda_available
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime


def _configure_ops(monkeypatch, runtime, default_policy=OpPolicy.AUTO, overrides=None):
    monkeypatch.setattr(runtime_module, "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    return get_or_create_ops(
        device_runtime=runtime,
        default_policy=default_policy,
        overrides=overrides,
    )


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    return _configure_ops(monkeypatch, CpuRuntime())


@pytest.fixture
def routing():
    tokens = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor(
        [
            [0.25, 0.0, 0.5],
            [0.0, 0.75, 0.125],
            [0.125, 0.25, 0.0],
        ],
        requires_grad=True,
    )
    return tokens, probs, probs.detach().bool()


@pytest.fixture
def npu_boundary(monkeypatch):
    # Replace only the NPU kernels; exercise provider selection and real autograd.
    backend = ModuleType("torch_npu")
    calls = []

    def permute_tokens(tokens, indices):
        calls.append("permute")
        order = indices.flatten().argsort(stable=True)
        rows = order // indices.shape[1]
        return tokens.index_select(0, rows), order.argsort().to(torch.int32)

    def unpermute_tokens(tokens, sorted_indices, probs=None, restore_shape=None):
        calls.append("unpermute")
        topk = 1 if probs is None else probs.shape[1]
        restored = tokens.index_select(0, sorted_indices.long()).reshape(
            -1,
            topk,
            tokens.shape[-1],
        )
        if probs is not None:
            restored = restored * probs.unsqueeze(-1)
        return restored.sum(1)

    backend.npu_moe_token_permute = permute_tokens
    backend.npu_moe_token_unpermute = unpermute_tokens
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    _configure_ops(monkeypatch, NpuRuntime())
    return SimpleNamespace(calls=calls)


@pytest.fixture
def cuda_boundary(monkeypatch):
    # Real CPU tensors exercise autograd behind the selected CUDA runtime.
    backend = ModuleType("transformer_engine.pytorch.permutation")

    def unused_kernel(*args, **kwargs):
        pytest.fail("An unexpected Transformer Engine kernel was called.")

    backend.moe_permute = unused_kernel
    backend.moe_permute_with_probs = unused_kernel
    backend.moe_unpermute = unused_kernel
    backend.moe_sort_chunks_by_index = unused_kernel
    backend.moe_sort_chunks_by_index_with_probs = unused_kernel
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(metadata, "version", lambda name: "2.6.0")
    _configure_ops(monkeypatch, CudaRuntime())
    return backend


@pytest.mark.parametrize("policy", list(OpPolicy))
@pytest.mark.parametrize("runtime_cls", [CpuRuntime, CudaRuntime, NpuRuntime])
def test_missing_custom_backend_obeys_policy(monkeypatch, runtime_cls, policy, routing):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.permutation", None)
    _configure_ops(monkeypatch, runtime_cls(), default_policy=policy)
    tokens, probs, mask = routing
    if policy == OpPolicy.CUSTOM:
        with pytest.raises(RuntimeError, match="policy=custom"):
            ops.moe_permutation(tokens, mask, probs=probs)
        mapping = MoEPermutation(
            tokens,
            None,
            torch.zeros(3, 1, dtype=torch.int32),
            torch.zeros(3, 1, dtype=torch.long),
        )
        with pytest.raises(RuntimeError, match="policy=custom"):
            ops.moe_unpermutation(mapping, tokens.shape)
        with pytest.raises(RuntimeError, match="policy=custom"):
            ops.moe_sort_chunks(tokens, torch.tensor([1, 2]), torch.tensor([1, 0]))
        return
    mapping = ops.moe_permutation(tokens, mask, probs=probs)
    output = ops.moe_unpermutation(mapping, tokens.shape, probs=probs)
    torch.testing.assert_close(output, torch.tensor([[7.5], [17.5], [11.25]]))
    output.sum().backward()
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs) * mask)
    sorted_tokens, _ = ops.moe_sort_chunks(
        tokens, torch.tensor([1, 2]), torch.tensor([1, 0])
    )
    torch.testing.assert_close(sorted_tokens, torch.tensor([[20.0], [30.0], [10.0]]))


@pytest.mark.parametrize("policy", [OpPolicy.NATIVE, OpPolicy.AUTO])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_npu_chunk_sort_obeys_policy_and_restores_gradients(
    monkeypatch, npu_boundary, policy, num_tokens
):
    _configure_ops(monkeypatch, NpuRuntime(), default_policy=policy)
    tokens = torch.tensor([[10.0], [20.0], [30.0]])[:num_tokens].requires_grad_()
    probs = torch.tensor([0.1, 0.2, 0.3])[:num_tokens].requires_grad_()
    sizes = torch.tensor([2, 0, 1] if num_tokens else [0, 0, 0])
    order = torch.tensor([2, 0, 1])
    output, weights = ops.moe_sort_chunks(tokens, sizes, order, probs=probs)
    rows = torch.tensor([2, 0, 1] if num_tokens else [], dtype=torch.long)
    torch.testing.assert_close(output, tokens[rows])
    restored, _ = ops.moe_sort_chunks(output, sizes[order], order.argsort())
    torch.testing.assert_close(restored, tokens)
    (output * weights[:, None]).sum().backward()
    torch.testing.assert_close(tokens.grad.flatten(), probs.detach())
    torch.testing.assert_close(probs.grad, tokens.detach().flatten())
    assert bool(npu_boundary.calls) == (policy == OpPolicy.AUTO and num_tokens > 0)


@pytest.mark.parametrize("policy", list(OpPolicy))
@pytest.mark.parametrize("use_override", [False, True])
def test_native_policy_and_explicit_override_select_the_requested_provider(
    monkeypatch,
    npu_boundary,
    policy,
    use_override,
    routing,
):
    prefix = "npu" if use_override else "torch"
    overrides = (
        {
            "moe_permutation": f"{prefix}_moe_mask_permutation",
            "moe_unpermutation": f"{prefix}_moe_mask_unpermutation",
            "moe_sort_chunks": f"{prefix}_moe_sort_chunks",
        }
        if use_override
        else None
    )
    _configure_ops(
        monkeypatch, NpuRuntime(), default_policy=policy, overrides=overrides
    )
    tokens, probs, mask = routing
    mapping = ops.moe_permutation(tokens, mask)
    output = ops.moe_unpermutation(mapping, tokens.shape, probs=probs)
    torch.testing.assert_close(output, torch.tensor([[7.5], [17.5], [11.25]]))
    chunks, _ = ops.moe_sort_chunks(tokens, torch.tensor([1, 2]), torch.tensor([1, 0]))
    torch.testing.assert_close(chunks, torch.tensor([[20.0], [30.0], [10.0]]))
    output.sum().backward()
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs) * mask)
    assert bool(npu_boundary.calls) == (use_override or policy != OpPolicy.NATIVE)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_cuda_empty_permutation_preserves_probability_dtype_and_gradients(
    cuda_boundary, dtype
):
    # An available TE provider must bypass kernels on empty source ranks.
    tokens = torch.empty(0, 4, dtype=torch.bfloat16)
    probs = torch.empty(0, 2, dtype=dtype, requires_grad=True)
    mapping = ops.moe_permutation(tokens, torch.empty(0, 2, dtype=torch.bool), probs=probs)
    packed = mapping.tokens
    packed_probs = mapping.probs
    assert packed.dtype == tokens.dtype
    assert packed_probs.dtype == probs.dtype
    packed_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))
    output = ops.moe_unpermutation(mapping, tokens.shape, probs=probs)
    assert output.dtype == tokens.dtype
    torch.testing.assert_close(output, tokens)


@pytest.mark.parametrize("with_probs", [False, True])
def test_cuda_chunk_sort_preserves_zero_chunks_and_probability_gradients(
    cuda_boundary, with_probs
):
    tokens = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    sizes, indices = torch.tensor([2, 0, 1]), torch.tensor([2, 1, 0])
    rows = torch.tensor([2, 0, 1])

    def sort(inp, split_sizes, sorted_index):
        assert split_sizes is sizes and sorted_index is indices
        return inp.index_select(0, rows)

    def sort_with_probs(inp, weights, split_sizes, sorted_index):
        return sort(inp, split_sizes, sorted_index), weights.index_select(0, rows)

    cuda_boundary.moe_sort_chunks_by_index = sort
    cuda_boundary.moe_sort_chunks_by_index_with_probs = sort_with_probs
    output, weights = ops.moe_sort_chunks(
        tokens, sizes, indices, probs=probs if with_probs else None
    )
    torch.testing.assert_close(output, torch.tensor([[30.0], [10.0], [20.0]]))
    if with_probs:
        torch.testing.assert_close(weights, torch.tensor([0.3, 0.1, 0.2]))
        (output * weights[:, None]).sum().backward()
        torch.testing.assert_close(probs.grad, tokens.detach().flatten())
        torch.testing.assert_close(tokens.grad.flatten(), probs.detach())
    else:
        assert weights is None


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
def test_missing_cuda_shared_libraries_obeys_policy(
    cuda_boundary, monkeypatch, policy, routing
):
    original_import = builtins.__import__

    def import_without_nvrtc(name, *args, **kwargs):
        if name == "transformer_engine.pytorch.permutation":
            raise RuntimeError("NVRTC shared object not found.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_nvrtc)
    _configure_ops(monkeypatch, CudaRuntime(), default_policy=policy)
    tokens, probs, mask = routing
    if policy == OpPolicy.CUSTOM:
        with pytest.raises(RuntimeError, match="No implementation.*te_moe_"):
            ops.moe_permutation(tokens, mask, probs=probs)
    else:
        mapping = ops.moe_permutation(tokens, mask, probs=probs)
        output = ops.moe_unpermutation(mapping, tokens.shape, probs=probs)
        torch.testing.assert_close(output, torch.tensor([[7.5], [17.5], [11.25]]))


@pytest.mark.parametrize("operation", ["unpermute", "chunks"])
def test_cuda_provider_preserves_router_only_gradients(cuda_boundary, operation):
    # TE conditions probability gradients on activation gradients.
    class Backend(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weights):
            ctx.save_for_backward(inp, weights)
            return inp * weights

        @staticmethod
        def backward(ctx, grad):
            inp, weights = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                return grad * weights, grad * inp
            return None, None

    tokens = torch.tensor([[10.0]])
    probs = torch.tensor([[0.5]], requires_grad=True)
    mapping = MoEPermutation(
        tokens,
        None,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.long),
    )

    def unpermute_backend(inp, row_id_map, *, merging_probs, restore_shape):
        return Backend.apply(inp, merging_probs)

    def chunk_backend(inp, weights, sizes, indices):
        return inp, Backend.apply(inp, weights[:, None]).flatten()

    cuda_boundary.moe_unpermute = unpermute_backend
    cuda_boundary.moe_sort_chunks_by_index_with_probs = chunk_backend
    if operation == "unpermute":
        output = ops.moe_unpermutation(mapping, tokens.shape, probs=probs)
    else:
        _, output = ops.moe_sort_chunks(
            tokens,
            torch.ones(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            probs=probs.flatten(),
        )
    output.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[10.0]]))
    assert not tokens.requires_grad


@pytest.mark.parametrize("runtime_cls", [CpuRuntime, CudaRuntime])
def test_chunk_sort_rejects_mismatched_metadata(monkeypatch, runtime_cls):
    _configure_ops(monkeypatch, runtime_cls())
    with pytest.raises(ValueError, match="input device"):
        ops.moe_sort_chunks(
            torch.ones(1, 1),
            torch.ones(1, dtype=torch.long, device="meta"),
            torch.zeros(1, dtype=torch.long),
        )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("dispatch_probs", [False, True])
def test_cuda_roundtrip_matches_native_tokens_and_probability_gradients(
    monkeypatch, dtype, num_tokens, dispatch_probs
):
    pytest.importorskip("transformer_engine.pytorch.permutation")
    runtime = CudaRuntime()
    dispatcher = _configure_ops(monkeypatch, runtime)
    tokens = torch.tensor([[10.0], [20.0], [30.0]], dtype=dtype, device=runtime.device)[
        :num_tokens
    ].requires_grad_()
    probs = torch.tensor(
        [[0.25, 0.0, 0.5], [0.0, 0.75, 0.125], [0.125, 0.25, 0.0]],
        device=runtime.device,
    )[:num_tokens].requires_grad_()
    mask = probs.detach().bool()
    weights = probs if dispatch_probs else None
    native_mapping = dispatcher.moe_permutation(
        tokens, mask, probs=weights, impl="torch_moe_mask_permutation"
    )
    native_packed = native_mapping.tokens
    native_probs = native_mapping.probs
    mapping = ops.moe_permutation(tokens, mask, probs=weights)
    packed = mapping.tokens
    packed_probs = mapping.probs
    torch.testing.assert_close(packed, native_packed)
    if dispatch_probs:
        torch.testing.assert_close(packed_probs, native_probs)
        native_packed = (native_packed * native_probs.unsqueeze(-1)).to(dtype)
        packed = (packed * packed_probs.unsqueeze(-1)).to(dtype)
    merging_probs = None if dispatch_probs else probs
    native = dispatcher.moe_unpermutation(
        replace(native_mapping, tokens=native_packed),
        tokens.shape,
        probs=merging_probs,
        impl="torch_moe_mask_unpermutation",
    )
    output = ops.moe_unpermutation(
        replace(mapping, tokens=packed), tokens.shape, probs=merging_probs
    )
    torch.testing.assert_close(output, native)
    assert output.dtype == dtype
    # Exercise both directions of native/TE interoperability on real hardware.
    torch.testing.assert_close(
        ops.moe_unpermutation(
            replace(native_mapping, tokens=native_packed),
            tokens.shape,
            probs=merging_probs,
        ),
        native,
    )
    torch.testing.assert_close(
        dispatcher.moe_unpermutation(
            replace(mapping, tokens=packed),
            tokens.shape,
            probs=merging_probs,
            impl="torch_moe_mask_unpermutation",
        ),
        native,
    )
    expected_grads = torch.autograd.grad(
        native.sum(), (tokens, probs), retain_graph=True
    )
    actual_grads = torch.autograd.grad(output.sum(), (tokens, probs))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
