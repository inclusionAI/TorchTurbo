"""Shared MoE mappings must survive independently selected providers."""

import sys
from importlib import metadata
from types import ModuleType

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.ops import OpDispatcher, OpPolicy, registry
from torchturbo.ops.moe import MoEPermutation
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime


def test_native_combine_allocates_output_without_copying_packed_activations():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    num_tokens, topk, hidden_size = 256, 8, 128
    packed = torch.ones(num_tokens * topk, hidden_size)
    # Expert-major packed rows; each token contributes once to every expert.
    rows = torch.arange(num_tokens * topk, dtype=torch.int32).view(topk, num_tokens).T
    permutation = MoEPermutation(
        packed, None, rows, torch.arange(topk).expand(num_tokens, -1)
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profiler:
        output = dispatcher.moe_unpermutation(permutation, torch.Size([num_tokens, hidden_size]))
    torch.testing.assert_close(output, torch.full_like(output, topk))
    # Allow output-sized workspaces and index metadata, but no [tokens * topk, hidden] copy.
    allocation_limit = 2 * output.numel() * output.element_size()
    assert max(event.cpu_memory_usage for event in profiler.events()) <= allocation_limit


@pytest.mark.parametrize("dtype,large", [(torch.bfloat16, 256), (torch.float16, 2048)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_combine_preserves_small_contributions_and_gradients(
    dtype, large, weighted, deterministic,
):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    packed = torch.tensor(
        [[large], [-large], [1], [2], [-large], [large]],
        dtype=dtype, requires_grad=True,
    )
    probs = torch.full((2, 3), 0.5, requires_grad=True) if weighted else None
    # Slots can differ from expert order, while rows and experts stay paired.
    permutation = MoEPermutation(
        packed, None, torch.tensor([[0, 2, 4], [5, 1, 3]], dtype=torch.int32),
        torch.tensor([[0, 1, 2], [2, 0, 1]]),
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        output = dispatcher.moe_unpermutation(permutation, torch.Size([2, 1]), probs)
        expected = torch.tensor([[1], [2]], dtype=dtype) * (0.5 if weighted else 1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(
            packed.grad, torch.full_like(packed, 0.5 if weighted else 1),
        )
        if weighted:
            torch.testing.assert_close(
                probs.grad, torch.tensor([[large, 1, -large], [-large, 2, large]]).float(),
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_permutation_returns_a_complete_result():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    result = dispatcher.moe_permutation(
        torch.tensor([[2.0], [3.0]]),
        torch.tensor([[False, True], [True, False]]),
        probs=torch.tensor([[0.0, 0.5], [0.25, 0.0]]),
    )
    assert isinstance(result, MoEPermutation)
    torch.testing.assert_close(result.tokens, torch.tensor([[3.0], [2.0]]))
    torch.testing.assert_close(result.probs, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(
        result.row_indices, torch.tensor([[1], [0]], dtype=torch.int32)
    )


def test_native_empty_permutation_without_cached_experts_keeps_gradients():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    tokens = torch.empty(0, 4, requires_grad=True)
    probs = torch.empty(0, 3, requires_grad=True)
    mapping = dispatcher.moe_permutation(tokens, probs.detach().bool(), probs=probs)
    assert mapping.row_indices.shape == (0, 0)
    output = dispatcher.moe_unpermutation(mapping, tokens.shape, probs)
    output.sum().backward()
    torch.testing.assert_close(output, tokens)
    torch.testing.assert_close(tokens.grad, torch.empty_like(tokens))
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))


@pytest.fixture
def npu_ops(monkeypatch):
    backend = ModuleType("torch_npu")

    def permute(tokens, indices):
        order = indices.flatten().argsort(stable=True)
        return tokens[order // indices.shape[1]], order.argsort().to(torch.int32)

    def unpermute(tokens, inverse, probs=None):
        topk = 1 if probs is None else probs.shape[1]
        restored = tokens[inverse.long()].reshape(-1, topk, tokens.shape[-1])
        if probs is not None:
            restored = restored * probs.unsqueeze(-1)
        return restored.sum(1)

    backend.npu_moe_token_permute = permute
    backend.npu_moe_token_unpermute = unpermute
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    return OpDispatcher(device_runtime=NpuRuntime(), registry=registry)


@pytest.mark.parametrize("topk", [1, 3])
def test_npu_unweighted_combine_saves_only_indices(npu_ops, topk):
    num_tokens, hidden_size = 4, 128
    packed = torch.randn(num_tokens * topk, hidden_size, requires_grad=True)
    rows = torch.arange(num_tokens * topk, dtype=torch.int32).view(topk, num_tokens).T
    experts = torch.arange(topk).expand(num_tokens, -1)
    # Providers may return the token's slots in a different expert order.
    mapping = MoEPermutation(packed, None, rows.flip(1), experts.flip(1))
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        output = npu_ops.moe_unpermutation(
            mapping, torch.Size([num_tokens, hidden_size]),
            impl="npu_moe_mask_unpermutation",
        )
    assert saved and all(not tensor.is_floating_point() for tensor in saved)
    torch.testing.assert_close(output, packed.view(topk, num_tokens, hidden_size).sum(0))
    gradient = torch.randn_like(output)
    output.backward(gradient)
    torch.testing.assert_close(packed.grad, gradient.repeat(topk, 1))


@pytest.mark.parametrize("implementation", ["torch", "npu", "te"])
@pytest.mark.parametrize("topk", [0, 1, 2])
def test_empty_unpermutation_recomputes_before_returning_input_gradients(
    npu_ops, cuda_ops, implementation, topk,
):
    events = []
    dispatcher = cuda_ops if implementation == "te" else npu_ops

    def forward(tokens):
        events.append("forward")
        received = tokens.clone()
        # This is the boundary where token-combine's collective backward begins.
        received.register_hook(lambda grad: events.append("input_gradient"))
        mapping = MoEPermutation(
            received, None, torch.empty(0, topk, dtype=torch.int32),
            torch.empty(0, topk, dtype=torch.int64),
        )
        return dispatcher.moe_unpermutation(
            mapping, tokens.shape, impl=f"{implementation}_moe_mask_unpermutation",
        )

    tokens = torch.empty(0, 4, requires_grad=True)
    output = checkpoint(forward, tokens, use_reentrant=False)
    events.clear()
    output.sum().backward()
    assert events == ["forward", "input_gradient"]
    torch.testing.assert_close(tokens.grad, torch.empty_like(tokens))


@pytest.fixture
def cuda_ops(monkeypatch):
    backend = ModuleType("transformer_engine.pytorch.permutation")

    def permute(tokens, mask, num_out_tokens):
        # TE's documented mask map packs row IDs and expert IDs per token.
        # Reverse the per-token slots to exercise their paired semantics.
        experts, rows = mask.T.nonzero(as_tuple=True)
        num_tokens, num_experts = mask.shape
        mapping = torch.full((num_tokens, num_experts * 2 + 1), -1, dtype=torch.int32)
        counts = mask.sum(1)
        mapping[:, -1] = counts
        used = [0] * num_tokens
        for packed_row, (expert, token) in enumerate(
            zip(experts.tolist(), rows.tolist())
        ):
            slot = int(counts[token]) - used[token] - 1
            mapping[token, slot] = packed_row
            mapping[token, num_experts + slot] = expert
            used[token] += 1
        return tokens[rows], mapping

    def with_probs(tokens, probs, mask, num_out_tokens):
        packed, mapping = permute(tokens, mask, num_out_tokens)
        packed_probs = probs.T.masked_select(mask.T)
        # TE's fused path treats zero probabilities as padding.
        packed = packed.masked_fill(packed_probs[:, None] == 0, 0)
        return packed, packed_probs, mapping

    def unpermute(tokens, mapping, merging_probs, restore_shape):
        num_experts = (mapping.shape[1] - 1) // 2
        output = tokens.new_zeros(restore_shape) + tokens.sum() * 0
        if merging_probs is not None:
            output = output + merging_probs.sum() * 0
        for token in range(restore_shape[0]):
            count = int(mapping[token, -1])
            rows = mapping[token, :count].long()
            values = tokens[rows]
            if merging_probs is not None:
                experts = mapping[token, num_experts : num_experts + count].long()
                values = values * merging_probs[token, experts, None]
            output[token] = values.sum(0).to(tokens.dtype)
        return output

    backend.moe_permute = permute
    backend.moe_permute_with_probs = with_probs
    backend.moe_unpermute = unpermute
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(metadata, "version", lambda name: "2.6.0")
    return OpDispatcher(device_runtime=CudaRuntime(), registry=registry)


@pytest.mark.parametrize("device,prefix", [("npu", "npu"), ("cuda", "te")])
@pytest.mark.parametrize("implementation", ["torch", "custom"])
def test_selected_zero_probability_keeps_tokens_and_router_gradients(
    request, device, prefix, implementation
):
    dispatcher = request.getfixturevalue(f"{device}_ops")
    prefix = prefix if implementation == "custom" else "torch"
    tokens = torch.tensor([[2.0]])
    probs = torch.tensor([[0.0, 1.0]], requires_grad=True)
    result = dispatcher.moe_permutation(
        tokens,
        torch.tensor([[True, True]]),
        probs=probs,
        impl=f"{prefix}_moe_mask_permutation",
    )
    torch.testing.assert_close(result.tokens, torch.tensor([[2.0], [2.0]]))
    torch.testing.assert_close(result.probs, torch.tensor([0.0, 1.0]))
    (result.tokens * result.probs[:, None]).sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[2.0, 2.0]]))
    assert not tokens.requires_grad


def test_te_permutation_does_not_require_fused_probability_kernel(
    cuda_ops, monkeypatch
):
    backend = sys.modules["transformer_engine.pytorch.permutation"]
    monkeypatch.delattr(backend, "moe_permute_with_probs")
    result = cuda_ops.moe_permutation(
        torch.tensor([[2.0]]),
        torch.tensor([[True]]),
        probs=torch.tensor([[0.5]]),
        impl="te_moe_mask_permutation",
    )
    torch.testing.assert_close(result.tokens, torch.tensor([[2.0]]))
    torch.testing.assert_close(result.probs, torch.tensor([0.5]))


@pytest.mark.parametrize("te_version", ["2.1.0", "2.5.0"])
@pytest.mark.parametrize("selection", ["auto", "custom", "explicit"])
@pytest.mark.parametrize("operation", ["permutation", "unpermutation"])
def test_old_te_mapping_uses_native_or_fails_strict_selection(
    cuda_ops, monkeypatch, te_version, selection, operation
):
    monkeypatch.setattr(metadata, "version", lambda name: te_version)
    backend = sys.modules["transformer_engine.pytorch.permutation"]

    def incompatible_kernel(*args, **kwargs):
        pytest.fail("TE versions before 2.6 use an incompatible row map.")

    monkeypatch.setattr(backend, "moe_permute", incompatible_kernel)
    monkeypatch.setattr(backend, "moe_unpermute", incompatible_kernel)
    dispatcher = OpDispatcher(
        device_runtime=CudaRuntime(),
        registry=registry,
        default_policy=OpPolicy.CUSTOM if selection == "custom" else OpPolicy.AUTO,
    )
    tokens = torch.tensor([[2.0]])
    kwargs = {"impl": f"te_moe_mask_{operation}"} if selection == "explicit" else {}

    def run():
        if operation == "permutation":
            return dispatcher.moe_permutation(
                tokens, torch.tensor([[True]]), **kwargs
            ).tokens
        result = MoEPermutation(
            tokens,
            None,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]]),
        )
        return dispatcher.moe_unpermutation(result, tokens.shape, **kwargs)

    if selection == "auto":
        torch.testing.assert_close(run(), tokens)
    else:
        with pytest.raises(RuntimeError, match=f"te_moe_mask_{operation}"):
            run()


@pytest.mark.parametrize("device,prefix", [("npu", "npu"), ("cuda", "te")])
@pytest.mark.parametrize("permutation", ["torch", "custom"])
@pytest.mark.parametrize("unpermutation", ["torch", "custom"])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_mixed_providers_restore_tokens_and_probability_gradients(
    request,
    device,
    prefix,
    permutation,
    unpermutation,
    num_tokens,
):
    dispatcher = request.getfixturevalue(f"{device}_ops")
    permutation = (
        f"{prefix if permutation == 'custom' else 'torch'}_moe_mask_permutation"
    )
    unpermutation = (
        f"{prefix if unpermutation == 'custom' else 'torch'}_moe_mask_unpermutation"
    )
    tokens = torch.tensor([[10.0], [20.0], [30.0]])[:num_tokens].requires_grad_()
    probs = torch.tensor(
        [
            [0.25, 0.0, 0.5],
            [0.0, 0.75, 0.125],
            [0.125, 0.25, 0.0],
        ]
    )[:num_tokens].requires_grad_()
    mask = probs.detach().bool()
    indices = torch.tensor([[0, 2], [1, 2], [0, 1]])[:num_tokens]
    mapping = dispatcher.moe_permutation(
        tokens,
        mask,
        probs=probs,
        expert_indices=indices,
        impl=permutation,
    )
    packed = mapping.tokens
    weights = mapping.probs
    expected_packed = torch.tensor([[10.0], [30.0], [20.0], [30.0], [10.0], [20.0]])
    torch.testing.assert_close(
        packed, expected_packed if num_tokens else expected_packed[:0]
    )
    # Slot order may vary, but each expert must retain its own packed row.
    expert_order = mapping.expert_indices.argsort(dim=1)
    torch.testing.assert_close(mapping.expert_indices.gather(1, expert_order), indices)
    expected_rows = torch.tensor([[0, 4], [2, 5], [1, 3]], dtype=torch.int32)[
        :num_tokens
    ]
    torch.testing.assert_close(
        mapping.row_indices.gather(1, expert_order), expected_rows
    )
    torch.testing.assert_close(
        torch.autograd.grad(weights.sum(), probs, retain_graph=True)[0], mask.float()
    )
    output = dispatcher.moe_unpermutation(
        mapping, tokens.shape, probs=probs, impl=unpermutation
    )
    expected = torch.tensor([[7.5], [17.5], [11.25]])[:num_tokens]
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, probs.detach().sum(1, keepdim=True))
    torch.testing.assert_close(probs.grad, tokens.detach().expand_as(probs) * mask)
    unweighted = dispatcher.moe_unpermutation(mapping, tokens.shape, impl=unpermutation)
    torch.testing.assert_close(unweighted, tokens * 2)


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE])
def test_missing_custom_permutation_falls_back_without_a_fused_flag(
    monkeypatch, policy
):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    dispatcher = OpDispatcher(
        device_runtime=NpuRuntime(), registry=registry, default_policy=policy
    )
    tokens = torch.tensor([[2.0], [3.0]])
    mapping = dispatcher.moe_permutation(tokens, torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(
        dispatcher.moe_unpermutation(mapping, tokens.shape), tokens
    )


def test_native_policy_overrides_available_custom_provider(npu_ops):
    dispatcher = OpDispatcher(
        device_runtime=NpuRuntime(),
        registry=registry,
        default_policy=OpPolicy.NATIVE,
    )
    tokens = torch.tensor([[2.0], [3.0]])
    mapping = dispatcher.moe_permutation(tokens, torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(
        dispatcher.moe_unpermutation(mapping, tokens.shape), tokens
    )


def test_explicit_custom_implementation_does_not_fall_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    with pytest.raises(RuntimeError, match="npu_moe_mask_permutation"):
        dispatcher.moe_permutation(
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            impl="npu_moe_mask_permutation",
        )


@pytest.mark.parametrize("device,prefix", [("npu", "npu"), ("cuda", "te")])
@pytest.mark.parametrize("implementation", ["torch", "custom"])
def test_unpermutation_keeps_probability_precision(
    request, device, prefix, implementation
):
    dispatcher = request.getfixturevalue(f"{device}_ops")
    implementation = prefix if implementation == "custom" else "torch"
    packed = torch.tensor([[1.0], [-1.0]], dtype=torch.bfloat16, requires_grad=True)
    probs = torch.tensor([[1.00390625, 1.0]], requires_grad=True)
    mapping = MoEPermutation(
        packed, None, torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([[0, 1]])
    )
    output = dispatcher.moe_unpermutation(
        mapping,
        torch.Size([1, 1]),
        probs=probs,
        impl=f"{implementation}_moe_mask_unpermutation",
    )
    torch.testing.assert_close(
        output, torch.tensor([[0.00390625]], dtype=torch.bfloat16), rtol=0, atol=0
    )
    output.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[1.0, -1.0]]))


def test_permutation_rejects_nonuniform_routing_and_token_dropping():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    tokens = torch.ones(2, 1)
    with pytest.raises(ValueError, match="same top-k"):
        dispatcher.moe_permutation(tokens, torch.tensor([[True, True], [False, False]]))
    with pytest.raises(ValueError, match="token dropping"):
        dispatcher.moe_permutation(
            tokens, torch.ones(2, 2, dtype=torch.bool), num_out_tokens=3
        )
