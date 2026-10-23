"""Permutation order, inverse mappings, and optional fused backend contracts."""

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from torchturbo.models.modules.moe.utils import permute, sort_chunks_by_idxs, unpermute
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import is_cuda_available
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime
import torchturbo.runtime.runtime as runtime_module


def _configure_ops(monkeypatch, runtime, default_policy=OpPolicy.AUTO, overrides=None):
    monkeypatch.setattr(runtime_module, "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    return get_or_create_ops(
        device_runtime=runtime, default_policy=default_policy, overrides=overrides,
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


@pytest.mark.parametrize("operation", ["permute", "unpermute", "chunks"])
@pytest.mark.parametrize("runtime_cls", [CpuRuntime, NpuRuntime])
def test_fused_requests_require_a_device_backend(monkeypatch, runtime_cls, operation, routing):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    _configure_ops(monkeypatch, runtime_cls())
    tokens, probs, mask = routing
    with pytest.raises(RuntimeError, match="moe_fused_"):
        if operation == "permute":
            permute(tokens, mask, fused=True)
        elif operation == "unpermute":
            unpermute(tokens, torch.arange(3), tokens.shape, fused=True)
        else:
            sort_chunks_by_idxs(
                tokens, torch.tensor([1, 2]), torch.tensor([1, 0]), fused=True
            )


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
            -1, topk, tokens.shape[-1],
        )
        if probs is not None:
            restored = restored * probs.unsqueeze(-1)
        return restored.sum(1)

    backend.npu_moe_token_permute = permute_tokens
    backend.npu_moe_token_unpermute = unpermute_tokens
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    _configure_ops(monkeypatch, NpuRuntime())
    return SimpleNamespace(calls=calls)


@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("with_probs", [False, True])
def test_npu_fused_roundtrip_and_probability_gradients(npu_boundary, num_tokens, with_probs):
    tokens = torch.tensor([[10.0], [20.0], [30.0]])[:num_tokens].requires_grad_()
    probs = torch.tensor([
        [.25, 0., .5], [0., .75, .125], [.125, .25, 0.],
    ])[:num_tokens].requires_grad_()
    mask = probs.detach().bool()
    packed, packed_probs, mapping = permute(
        tokens, mask, probs=probs if with_probs else None,
        num_out_tokens=num_tokens * 2, fused=True,
    )
    reference, reference_probs, _ = permute(
        tokens, mask, probs=probs if with_probs else None,
    )
    torch.testing.assert_close(packed, reference)
    if with_probs:
        torch.testing.assert_close(packed_probs, reference_probs)
        actual_grad = torch.autograd.grad(packed_probs.sum(), probs, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, mask.float())
    else:
        assert packed_probs is None
    output = unpermute(
        packed, mapping, tokens.shape, probs=probs, routing_map=mask, fused=True,
    )
    torch.testing.assert_close(output, tokens * probs.sum(1, keepdim=True))
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, probs.detach().sum(1, keepdim=True))
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs) * mask)
    if num_tokens:
        assert npu_boundary.calls == ["permute", "unpermute"]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_npu_chunk_sort_obeys_fusion_and_restores_gradients(npu_boundary, fused, num_tokens):
    tokens = torch.tensor([[10.0], [20.0], [30.0]])[:num_tokens].requires_grad_()
    probs = torch.tensor([.1, .2, .3])[:num_tokens].requires_grad_()
    sizes = torch.tensor([2, 0, 1] if num_tokens else [0, 0, 0])
    order = torch.tensor([2, 0, 1])
    output, weights = sort_chunks_by_idxs(tokens, sizes, order, probs=probs, fused=fused)
    expected_rows = torch.tensor([2, 0, 1] if num_tokens else [], dtype=torch.long)
    torch.testing.assert_close(output, tokens[expected_rows])
    torch.testing.assert_close(weights, probs[expected_rows])
    restored, _ = sort_chunks_by_idxs(output, sizes[order], order.argsort(), fused=fused)
    torch.testing.assert_close(restored, tokens)
    (output * weights[:, None]).sum().backward()
    torch.testing.assert_close(tokens.grad.flatten(), probs.detach())
    torch.testing.assert_close(probs.grad, tokens.detach().flatten())
    assert bool(npu_boundary.calls) == (fused and num_tokens > 0)


def test_npu_native_permutation_does_not_call_device_kernels(npu_boundary, routing):
    tokens, probs, mask = routing
    packed, _, mapping = permute(tokens, mask, fused=False)
    output = unpermute(packed, mapping, tokens.shape, probs=probs, routing_map=mask)
    output.sum().backward()
    assert npu_boundary.calls == []


def test_native_roundtrip_preserves_routing_order_and_probability_gradients(cpu_ops, routing):
    tokens, probs, mask = routing
    packed, packed_probs, mapping = cpu_ops.moe_mask_permutation(tokens, mask, probs=probs)
    torch.testing.assert_close(
        packed.flatten(), torch.tensor([10.0, 30.0, 20.0, 30.0, 10.0, 20.0])
    )
    torch.testing.assert_close(
        packed_probs, torch.tensor([0.25, 0.125, 0.75, 0.25, 0.5, 0.125])
    )
    torch.testing.assert_close(mapping, torch.tensor([0, 2, 1, 2, 0, 1]))
    output = cpu_ops.moe_mask_unpermutation(
        packed, mapping, tokens.shape, probs=probs, routing_map=mask,
    )
    torch.testing.assert_close(output.flatten(), torch.tensor([7.5, 17.5, 11.25]))
    output.sum().backward()
    torch.testing.assert_close(
        tokens.grad.flatten(), torch.tensor([0.75, 0.875, 0.375])
    )
    torch.testing.assert_close(
        probs.grad,
        torch.tensor([[10.0, 0.0, 10.0], [0.0, 20.0, 20.0], [30.0, 30.0, 0.0]]),
    )


def test_npu_ops_use_native_chunk_sort_without_optional_dependencies(monkeypatch):
    dispatcher = _configure_ops(monkeypatch, NpuRuntime())
    tokens = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    output, output_probs = dispatcher.moe_sort_chunks(
        tokens, torch.tensor([2, 0, 1]), torch.tensor([2, 1, 0]), probs=probs,
    )
    torch.testing.assert_close(output.flatten(), torch.tensor([30.0, 10.0, 20.0]))
    torch.testing.assert_close(output_probs, torch.tensor([0.3, 0.1, 0.2]))
    (output * output_probs[:, None]).sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(tokens.grad.flatten(), probs.detach())


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
    _configure_ops(monkeypatch, CudaRuntime())
    return backend


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("policy", list(OpPolicy))
@pytest.mark.parametrize("use_override", [False, True])
def test_fused_ops_respect_default_policy_and_overrides(
    monkeypatch, request, device, policy, use_override,
):
    backend = request.getfixturevalue(f"{device}_boundary")
    if device == "cuda":
        backend.moe_permute = lambda inp, mask, **kwargs: (inp, torch.zeros(1, dtype=torch.int32))
        backend.moe_unpermute = lambda inp, mapping, **kwargs: inp
        backend.moe_sort_chunks_by_index = lambda inp, sizes, indices: inp
    runtime = CudaRuntime() if device == "cuda" else NpuRuntime()
    if device == "npu":
        monkeypatch.setattr(NpuRuntime, "device", property(lambda self: torch.device("cpu")))
    prefix = "te" if device == "cuda" else "npu"
    overrides = {
        "moe_fused_permutation": f"{prefix}_moe_mask_permutation",
        "moe_fused_unpermutation": f"{prefix}_moe_mask_unpermutation",
        "moe_fused_sort_chunks": f"{prefix}_moe_sort_chunks",
    } if use_override else None
    dispatcher = _configure_ops(
        monkeypatch, runtime, default_policy=policy, overrides=overrides,
    )
    messages = []
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.dispatcher"), "logger",
        SimpleNamespace(info_rank0=lambda message, *args: messages.append(message % args)),
    )
    dispatcher.log_ops_selection()
    tokens = torch.tensor([[2.0]], requires_grad=True)
    if policy == OpPolicy.NATIVE and not use_override:
        with pytest.raises(RuntimeError, match="moe_fused_permutation.*policy=native"):
            permute(
                tokens, torch.ones(1, 1, dtype=torch.bool), num_out_tokens=1, fused=True,
            )
        with pytest.raises(RuntimeError, match="moe_fused_unpermutation.*policy=native"):
            unpermute(tokens, torch.zeros(1, dtype=torch.int32), tokens.shape, fused=True)
        with pytest.raises(RuntimeError, match="moe_fused_sort_chunks.*policy=native"):
            sort_chunks_by_idxs(tokens, torch.tensor([1]), torch.tensor([0]), fused=True)
        for name in ("moe_fused_permutation", "moe_fused_unpermutation", "moe_fused_sort_chunks"):
            assert f"{name}: No implementation" in messages[-1]
        return
    for name in ("moe_fused_permutation", "moe_fused_unpermutation", "moe_fused_sort_chunks"):
        assert f"{name}: No implementation" not in messages[-1]
    packed, _, mapping = permute(
        tokens, torch.ones(1, 1, dtype=torch.bool), num_out_tokens=1, fused=True,
    )
    sorted_tokens, _ = sort_chunks_by_idxs(
        packed, torch.tensor([1]), torch.tensor([0]), fused=True,
    )
    output = unpermute(sorted_tokens, mapping, tokens.shape, fused=True)
    torch.testing.assert_close(output, tokens)
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))


@pytest.mark.parametrize("with_probs", [False, True])
def test_fused_permute_uses_backend_mapping_and_preserves_probabilities(
    cuda_boundary,
    routing,
    with_probs,
):
    tokens, probs, mask = routing
    rows = torch.tensor([0, 2, 1, 2, 0, 1])
    row_id_map = torch.tensor([[0, -1, 1], [-1, 2, 3], [4, 5, -1]], dtype=torch.int32)

    def backend_permute(inp, routing_map, *, num_out_tokens):
        assert routing_map is mask
        assert num_out_tokens == 6
        return inp.index_select(0, rows), row_id_map

    def backend_with_probs(inp, weights, routing_map, *, num_out_tokens):
        assert weights is probs
        output, mapping = backend_permute(
            inp, routing_map, num_out_tokens=num_out_tokens
        )
        return output, weights[[0, 2, 1, 2, 0, 1], [0, 0, 1, 1, 2, 2]], mapping

    cuda_boundary.moe_permute = backend_permute
    cuda_boundary.moe_permute_with_probs = backend_with_probs
    output, output_probs, mapping = permute(
        tokens,
        mask,
        probs=probs if with_probs else None,
        num_out_tokens=6,
        fused=True,
    )
    assert mapping is row_id_map
    torch.testing.assert_close(
        output.flatten(), torch.tensor([10.0, 30.0, 20.0, 30.0, 10.0, 20.0])
    )
    if with_probs:
        torch.testing.assert_close(
            output_probs, torch.tensor([0.25, 0.125, 0.75, 0.25, 0.5, 0.125])
        )
        (output.sum() + output_probs.sum()).backward()
        torch.testing.assert_close(probs.grad, mask.float())
    else:
        assert output_probs is None


@pytest.mark.parametrize("with_probs", [False, True])
def test_fused_unpermute_passes_opaque_mapping_and_original_probability_dtype(
    cuda_boundary,
    with_probs,
):
    packed = torch.tensor(
        [[10.0], [30.0], [20.0], [30.0], [10.0], [20.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    probs = torch.tensor(
        [[0.25, 0.0, 0.5], [0.0, 0.75, 0.125], [0.125, 0.25, 0.0]], requires_grad=True
    )
    mapping = torch.tensor([[0, -1, 1], [-1, 2, 3], [4, 5, -1]], dtype=torch.int32)

    def backend(inp, row_id_map, *, merging_probs, restore_shape):
        assert row_id_map is mapping
        assert restore_shape == torch.Size([3, 1])
        assert merging_probs is (probs if with_probs else None)
        weights = (
            merging_probs[[0, 2, 1, 2, 0, 1], [0, 0, 1, 1, 2, 2]]
            if with_probs
            else torch.ones(6)
        )
        return (
            (inp * weights[:, None])
            .new_zeros(restore_shape)
            .index_add(
                0,
                torch.tensor([0, 2, 1, 2, 0, 1]),
                inp * weights[:, None],
            )
            .to(inp.dtype)
        )

    cuda_boundary.moe_unpermute = backend
    output = unpermute(
        packed,
        mapping,
        torch.Size([3, 1]),
        probs=probs if with_probs else None,
        fused=True,
    )
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.flatten(),
        torch.tensor(
            [7.5, 17.5, 11.25] if with_probs else [20.0, 40.0, 60.0],
            dtype=torch.bfloat16,
        ),
    )
    output.sum().backward()
    if with_probs:
        torch.testing.assert_close(
            probs.grad,
            torch.tensor([[10.0, 0.0, 10.0], [0.0, 20.0, 20.0], [30.0, 30.0, 0.0]]),
        )


def test_fused_permute_requires_explicit_output_count(cuda_boundary, routing):
    tokens, _, mask = routing
    with pytest.raises(ValueError, match="num_out_tokens"):
        permute(tokens, mask, fused=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cuda_empty_permutation_preserves_probability_dtype_and_gradients(cuda_boundary, dtype):
    def legacy_empty_permute(inp, weights, routing_map, *, num_out_tokens):
        # TE 2.12 creates default-dtype empty probabilities in this branch.
        return inp, torch.empty(0, device=inp.device), torch.empty(0, device=inp.device)

    cuda_boundary.moe_permute_with_probs = legacy_empty_permute
    tokens = torch.empty(0, 4, dtype=torch.bfloat16)
    probs = torch.empty(0, 2, dtype=dtype, requires_grad=True)
    packed, packed_probs, _ = permute(
        tokens, torch.empty(0, 2, dtype=torch.bool), probs=probs,
        num_out_tokens=0, fused=True,
    )
    assert packed.dtype == tokens.dtype
    assert packed_probs.dtype == probs.dtype
    packed_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))


@pytest.mark.parametrize("with_probs", [False, True])
def test_fused_chunk_sort_preserves_zero_chunks_and_probability_gradients(
    cuda_boundary,
    with_probs,
):
    tokens = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    sizes, indices = torch.tensor([2, 0, 1]), torch.tensor([2, 1, 0])
    rows = torch.tensor([2, 0, 1])
    calls = []

    def sort(inp, split_sizes, sorted_index):
        calls.append("sort")
        assert split_sizes is sizes
        assert sorted_index is indices
        return inp.index_select(0, rows)

    def sort_with_probs(inp, weights, split_sizes, sorted_index):
        assert weights is probs
        return sort(inp, split_sizes, sorted_index), weights.index_select(0, rows)

    cuda_boundary.moe_sort_chunks_by_index = sort
    cuda_boundary.moe_sort_chunks_by_index_with_probs = sort_with_probs
    output, output_probs = sort_chunks_by_idxs(
        tokens, sizes, indices, probs=probs if with_probs else None, fused=True
    )
    assert calls == ["sort"]
    torch.testing.assert_close(output.flatten(), torch.tensor([30.0, 10.0, 20.0]))
    if with_probs:
        torch.testing.assert_close(output_probs, torch.tensor([0.3, 0.1, 0.2]))
        (output * output_probs[:, None]).sum().backward()
        torch.testing.assert_close(probs.grad, torch.tensor([10.0, 20.0, 30.0]))
        torch.testing.assert_close(tokens.grad.flatten(), probs.detach())
    else:
        assert output_probs is None


def test_missing_fused_chunk_backend_reports_configuration_fix(
    cuda_boundary, monkeypatch
):
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.permutation", None)
    _configure_ops(monkeypatch, CudaRuntime())
    with pytest.raises(RuntimeError, match="No implementation.*te_moe_"):
        sort_chunks_by_idxs(
            torch.ones(1, 1),
            torch.ones(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            fused=True,
        )


@pytest.mark.parametrize("operation", ["permute", "unpermute"])
def test_missing_cuda_permutation_backend_reports_configuration_fix(
    cuda_boundary,
    monkeypatch,
    operation,
):
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.permutation", None)
    _configure_ops(monkeypatch, CudaRuntime())
    with pytest.raises(RuntimeError, match="No implementation.*te_moe_"):
        if operation == "permute":
            permute(
                torch.ones(1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                num_out_tokens=1,
                fused=True,
            )
        else:
            unpermute(
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
                torch.Size([1, 1]),
                fused=True,
            )


@pytest.mark.parametrize("operation", ["permute", "unpermute", "chunks"])
def test_missing_cuda_shared_libraries_report_configuration_fix(
    cuda_boundary,
    monkeypatch,
    operation,
):
    original_import = builtins.__import__

    def import_without_nvrtc(name, *args, **kwargs):
        if name == "transformer_engine.pytorch.permutation":
            raise RuntimeError("NVRTC shared object not found.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_nvrtc)
    _configure_ops(monkeypatch, CudaRuntime())
    with pytest.raises(RuntimeError, match="No implementation.*te_moe_"):
        if operation == "permute":
            permute(
                torch.ones(1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                num_out_tokens=1,
                fused=True,
            )
        elif operation == "unpermute":
            unpermute(
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
                torch.Size([1, 1]),
                fused=True,
            )
        else:
            sort_chunks_by_idxs(
                torch.ones(1, 1),
                torch.ones(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                fused=True,
            )


@pytest.mark.parametrize("operation", ["permute", "unpermute", "chunks"])
def test_fused_backend_preserves_router_only_gradients(cuda_boundary, operation):
    # TE 2.1-2.12 conditions probability gradients on activation gradients.
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
    mapping = torch.zeros(1, 1, dtype=torch.int32)

    def permute_with_probs(inp, weights, routing_map, *, num_out_tokens):
        return inp, Backend.apply(inp, weights).flatten(), mapping

    def unpermute_backend(inp, row_id_map, *, merging_probs, restore_shape):
        return Backend.apply(inp, merging_probs)

    def chunk_backend(inp, weights, sizes, indices):
        return inp, Backend.apply(inp, weights[:, None]).flatten()

    cuda_boundary.moe_permute_with_probs = permute_with_probs
    cuda_boundary.moe_unpermute = unpermute_backend
    cuda_boundary.moe_sort_chunks_by_index_with_probs = chunk_backend
    if operation == "permute":
        _, output, _ = permute(
            tokens,
            torch.ones(1, 1, dtype=torch.bool),
            probs=probs,
            num_out_tokens=1,
            fused=True,
        )
    elif operation == "unpermute":
        output = unpermute(tokens, mapping, tokens.shape, probs=probs, fused=True)
    else:
        _, output = sort_chunks_by_idxs(
            tokens,
            torch.ones(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            probs=probs.flatten(),
            fused=True,
        )
    output.sum().backward()
    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, torch.tensor([[10.0]]))
    assert not tokens.requires_grad


def test_fused_chunk_sort_rejects_mismatched_metadata(cuda_boundary):
    cuda_boundary.moe_sort_chunks_by_index = lambda inp, sizes, indices: inp
    tokens = torch.ones(1, 1)
    sizes = torch.ones(1, dtype=torch.long, device="meta")
    indices = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="input device"):
        sort_chunks_by_idxs(tokens, sizes, indices, fused=True)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("dispatch_probs", [False, True])
def test_cuda_fused_roundtrip_matches_native_tokens_and_probability_gradients(
    monkeypatch, dtype, num_tokens, dispatch_probs,
):
    pytest.importorskip("transformer_engine.pytorch.permutation")
    runtime = CudaRuntime()
    _configure_ops(monkeypatch, runtime)
    tokens = torch.tensor(
        [[10.0], [20.0], [30.0]], dtype=dtype, device=runtime.device,
    )[:num_tokens].requires_grad_()
    probs = torch.tensor(
        [[0.25, 0.0, 0.5], [0.0, 0.75, 0.125], [0.125, 0.25, 0.0]],
        device=runtime.device,
    )[:num_tokens].requires_grad_()
    mask = probs.detach().bool()
    packed_probs = probs if dispatch_probs else None
    native_packed, native_probs, native_mapping = permute(tokens, mask, probs=packed_probs)
    fused_packed, fused_probs, fused_mapping = permute(
        tokens, mask, probs=packed_probs, num_out_tokens=num_tokens * 2, fused=True,
    )
    torch.testing.assert_close(fused_packed, native_packed)
    if dispatch_probs:
        torch.testing.assert_close(fused_probs, native_probs)
        native_packed = (native_packed * native_probs.unsqueeze(-1)).to(dtype)
        fused_packed = (fused_packed * fused_probs.unsqueeze(-1)).to(dtype)
    merging_probs = None if dispatch_probs else probs
    native = unpermute(
        native_packed, native_mapping, tokens.shape, probs=merging_probs, routing_map=mask,
    )
    fused = unpermute(
        fused_packed, fused_mapping, tokens.shape, probs=merging_probs, fused=True,
    )
    torch.testing.assert_close(fused, native)
    assert fused.dtype == dtype
    native_grads = torch.autograd.grad(native.sum(), (tokens, probs), retain_graph=True)
    fused_grads = torch.autograd.grad(fused.sum(), (tokens, probs))
    for actual, expected in zip(fused_grads, native_grads):
        torch.testing.assert_close(actual, expected)
