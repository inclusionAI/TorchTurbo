"""Optional probability packing retains NPU provider and strict policy semantics."""

import importlib
import inspect
import sys
from types import ModuleType

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, registry
from torchturbo.ops.providers.npu.moe_permutation import NpuMoEMaskPermutation, NpuMoESortChunks
from torchturbo.runtime.npu import NpuRuntime


@pytest.fixture
def torch_npu_backend(monkeypatch):
    module = ModuleType("torch_npu")

    def permute(tokens, indices):
        order = indices.flatten().argsort(stable=True)
        return tokens[order // indices.shape[1]], order.argsort().to(torch.int32)

    def unpermute(tokens, inverse, probs=None):
        topk = 1 if probs is None else probs.shape[1]
        restored = tokens[inverse.long()].view(-1, topk, tokens.shape[-1])
        if probs is not None:
            restored = restored * probs[:, :, None]
        return restored.sum(1)

    module.npu_moe_token_unpermute = unpermute
    module.npu_moe_token_permute = permute
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return module


def _triton_available(monkeypatch, pack):
    compiler = ModuleType("triton.backends.ascend.compiler")
    compiler.AscendBackend = object
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    helper = ModuleType("torchturbo.ops.providers.npu._kernels.moe_probability")
    helper.pack_probabilities = pack
    monkeypatch.setitem(sys.modules, helper.__name__, helper)


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE, OpPolicy.CUSTOM])
def test_optional_provider_keeps_existing_policy_selection(monkeypatch, torch_npu_backend, available, policy):
    if available:
        _triton_available(monkeypatch, lambda probs, inverse: probs.flatten()[inverse.long()])
    else:
        monkeypatch.setitem(sys.modules, "triton", None)
        monkeypatch.setitem(sys.modules, "triton.backends.ascend.compiler", None)
        monkeypatch.setitem(sys.modules, "torchturbo.ops.providers.npu._kernels.moe_probability", None)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry, default_policy=policy)
    for name in ("moe_permutation", "moe_sort_chunks"):
        actual = dispatcher._candidate_impl(name, dispatcher.device_runtime.turbo_device)
        prefix = "torch" if policy == OpPolicy.NATIVE else "npu"
        assert actual.op_spec().impl_name == f"{prefix}_{'moe_mask_permutation' if name == 'moe_permutation' else name}"
    assert NpuMoEMaskPermutation().available() and NpuMoESortChunks().available()


@pytest.mark.parametrize("selection", ["explicit", "override"])
def test_missing_triton_optional_provider_fails_strict_selection(monkeypatch, torch_npu_backend, selection):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.compiler", None)
    monkeypatch.setitem(sys.modules, "torchturbo.ops.providers.npu._kernels.moe_probability", None)
    name = "npu_triton_moe_mask_permutation"
    with pytest.raises(RuntimeError, match=name):
        dispatcher = OpDispatcher(
            device_runtime=NpuRuntime(), registry=registry,
            overrides={"moe_permutation": name} if selection == "override" else None,
        )
        dispatcher.moe_permutation(
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1),
            expert_indices=torch.tensor([[0]]),
            impl=name if selection == "explicit" else None,
        )


@pytest.mark.parametrize("empty", [False, True])
def test_optional_permutation_uses_shared_int32_packing_and_preserves_gradients(
    monkeypatch, torch_npu_backend, empty,
):
    calls = []

    def pack(probs, inverse):
        calls.append((probs.shape, inverse.dtype))
        flat = probs.flatten()
        return torch.empty_like(flat).index_put((inverse.long(),), flat)

    _triton_available(monkeypatch, pack)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    length = 0 if empty else 2
    tokens = torch.tensor([[2.0], [3.0]])[:length].requires_grad_()
    ids = torch.tensor([[0, 2], [1, 2]])[:length]
    dense = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.7, 0.3]])[:length].requires_grad_()
    mask = torch.zeros(length, 3, dtype=torch.bool).scatter_(1, ids, True)
    mapping = dispatcher.moe_permutation(
        tokens, mask, dense, expert_indices=ids,
        impl="npu_triton_moe_mask_permutation",
    )
    expected = dispatcher.moe_permutation(tokens, mask, dense, expert_indices=ids)
    torch.testing.assert_close(mapping.tokens, expected.tokens)
    torch.testing.assert_close(mapping.probs, expected.probs)
    expected_grad = torch.autograd.grad(expected.probs.sum(), dense, retain_graph=True)[0]
    mapping.probs.sum().backward()
    torch.testing.assert_close(dense.grad, expected_grad)
    assert calls == ([] if empty else [(torch.Size([length * 2]), torch.int32)])


def test_optional_chunk_sort_reuses_existing_transport_and_pack_hook(monkeypatch, torch_npu_backend):
    _triton_available(
        monkeypatch,
        lambda probs, inverse: torch.empty_like(probs).index_put((inverse.long(),), probs),
    )
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    tokens = torch.arange(5).float()[:, None].requires_grad_()
    probs = torch.tensor([0.0, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
    sizes, order = torch.tensor([2, 0, 3]), torch.tensor([2, 1, 0])
    output, weights = dispatcher.moe_sort_chunks(
        tokens, sizes, order, probs, impl="npu_triton_moe_sort_chunks",
    )
    torch.testing.assert_close(output, tokens[torch.tensor([2, 3, 4, 0, 1])])
    torch.testing.assert_close(weights, probs[torch.tensor([2, 3, 4, 0, 1])])
    (output * weights[:, None]).sum().backward()
    torch.testing.assert_close(tokens.grad, probs.detach()[:, None])
    torch.testing.assert_close(probs.grad, tokens.detach().flatten())


def test_optional_providers_inherit_identical_operator_signatures():
    module = importlib.import_module("torchturbo.ops.providers.npu.moe_permutation_triton")
    assert inspect.signature(module.NpuTritonMoEMaskPermutation.__call__) == inspect.signature(NpuMoEMaskPermutation.__call__)
    assert inspect.signature(module.NpuTritonMoESortChunks.__call__) == inspect.signature(NpuMoESortChunks.__call__)
    assert module.NpuTritonMoEMaskPermutation.op_spec().priority == 90
    assert module.NpuTritonMoESortChunks.op_spec().priority == 90


@pytest.mark.parametrize("class_name", ["NpuTritonMoEMaskPermutation", "NpuTritonMoESortChunks"])
def test_generic_triton_without_ascend_backend_is_unavailable(monkeypatch, torch_npu_backend, class_name):
    triton = ModuleType("triton")
    triton.__path__ = []
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.compiler", None)
    module = importlib.import_module("torchturbo.ops.providers.npu.moe_permutation_triton")
    provider = module.NpuTritonMoEMaskPermutation if class_name == "NpuTritonMoEMaskPermutation" else module.NpuTritonMoESortChunks
    assert not provider().available()


@pytest.mark.parametrize("implementation", ["npu_moe_sort_chunks", "npu_triton_moe_sort_chunks"])
@pytest.mark.parametrize("columns", [1, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_chunk_probability_rows_preserve_shape_and_independent_gradients(
    monkeypatch, torch_npu_backend, implementation, columns, empty,
):
    def vector_pack(probs, inverse):
        assert probs.ndim == 1
        return torch.empty_like(probs).index_put((inverse.long(),), probs)

    _triton_available(monkeypatch, vector_pack)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    length = 0 if empty else 5
    tokens = torch.arange(length).float()[:, None].requires_grad_()
    storage = torch.arange(length * 4 * columns).float().view(length * 2, columns * 2)
    probs = storage[::2, ::2].detach().requires_grad_()
    sizes = torch.tensor([0, 0, 0]) if empty else torch.tensor([2, 0, 3])
    order = torch.tensor([2, 1, 0])
    output, weights = dispatcher.moe_sort_chunks(tokens, sizes, order, probs, impl=implementation)
    source_rows = torch.empty(0, dtype=torch.long) if empty else torch.tensor([2, 3, 4, 0, 1])
    assert weights.shape == probs.shape
    torch.testing.assert_close(weights, probs[source_rows])
    torch.testing.assert_close(output, tokens[source_rows])
    (output * weights).sum().backward()
    torch.testing.assert_close(tokens.grad, probs.detach().sum(1, keepdim=True))
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs))
