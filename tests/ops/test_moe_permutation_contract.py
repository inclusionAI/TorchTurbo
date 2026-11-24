"""Shared MoE mappings must survive independently selected providers."""

import sys
from dataclasses import replace
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


def test_native_round_trip_reuses_order_with_replaced_expert_outputs():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    tokens = torch.tensor([[1.], [2.]], requires_grad=True)
    indices = torch.tensor([[2, 0], [1, 2]])
    gates = torch.tensor([[.2, 9., .3], [8., .4, .5]], requires_grad=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        packed = dispatcher.moe_permutation(tokens, indices)
        experts = replace(packed, tokens=2 * packed.tokens)
        output = dispatcher.moe_unpermutation(experts, tokens.shape, probs=gates)

    operations = {event.key for event in profile.key_averages()}
    assert operations.isdisjoint({"aten::scatter_", "aten::scatter"})
    torch.testing.assert_close(experts.row_indices, torch.tensor([[2, 0], [1, 3]], dtype=torch.int32))
    torch.testing.assert_close(output, torch.tensor([[1.], [3.6]]))
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.tensor([[1.], [1.8]]))
    torch.testing.assert_close(gates.grad, torch.tensor([[2., 0., 2.], [0., 4., 4.]]))


@pytest.mark.parametrize("field,expected", [("row_indices", [14., 15.]), ("expert_indices", [9., 28.])])
def test_replace_one_permutation_index_preserves_the_other(field, expected):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    packed = torch.tensor([[3.], [7.]])
    rows = torch.tensor([[0], [1]], dtype=torch.int32)
    experts = torch.tensor([[0], [1]])
    permutation = dispatcher.moe_permutation(packed, experts)
    if field == "row_indices":
        rows = rows.flip(0)
        changed = replace(permutation, row_indices=rows)
    else:
        experts = experts.flip(0)
        changed = replace(permutation, expert_indices=experts)
    torch.testing.assert_close(changed.row_indices, rows)
    torch.testing.assert_close(changed.expert_indices, experts)
    gates = torch.tensor([[2., 3.], [4., 5.]])
    output = dispatcher.moe_unpermutation(changed, packed.shape, probs=gates)
    torch.testing.assert_close(output.flatten(), torch.tensor(expected))


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
        torch.tensor([[1], [0]]),
        probs=torch.tensor([[0.5], [0.25]]),
    )
    assert isinstance(result, MoEPermutation)
    torch.testing.assert_close(result.tokens, torch.tensor([[3.0], [2.0]]))
    torch.testing.assert_close(result.probs, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(
        result.row_indices, torch.tensor([[1], [0]], dtype=torch.int32)
    )


def test_native_empty_permutation_keeps_topk_and_gradients():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    tokens = torch.empty(0, 4, requires_grad=True)
    probs = torch.empty(0, 3, requires_grad=True)
    mapping = dispatcher.moe_permutation(
        tokens, torch.empty(0, 3, dtype=torch.int64), probs=probs,
    )
    assert mapping.row_indices.shape == (0, 3)
    output = dispatcher.moe_unpermutation(mapping, tokens.shape, probs)
    output.sum().backward()
    torch.testing.assert_close(output, tokens)
    torch.testing.assert_close(tokens.grad, torch.empty_like(tokens))
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))


@pytest.fixture
def npu_ops(monkeypatch):
    backend = ModuleType("torch_npu")

    def permute(tokens, indices, num_out_tokens=None):
        order = indices.flatten().argsort(stable=True)
        return tokens[order[:num_out_tokens] // indices.shape[1]], order.argsort().to(torch.int32)

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

    def permute(*args, **kwargs):
        pytest.fail("TE index permutation shares its mutable workspace across CUDA streams.")

    def with_probs(*args, **kwargs):
        pytest.fail("Selected zero probabilities must not be treated as padding.")

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
        torch.tensor([[0, 1]]),
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
        torch.tensor([[0]]),
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
                tokens, torch.tensor([[0]]), **kwargs
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
        indices,
        probs=probs.gather(1, indices),
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
    mapping = dispatcher.moe_permutation(tokens, torch.zeros(2, 1, dtype=torch.long))
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
    mapping = dispatcher.moe_permutation(tokens, torch.zeros(2, 1, dtype=torch.long))
    torch.testing.assert_close(
        dispatcher.moe_unpermutation(mapping, tokens.shape), tokens
    )


def test_explicit_custom_implementation_does_not_fall_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    with pytest.raises(RuntimeError, match="npu_moe_mask_permutation"):
        dispatcher.moe_permutation(
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.long),
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


@pytest.mark.parametrize(
    "indices,probs,message",
    [
        (torch.ones(2, 2), None, "int32 or int64"),
        (torch.tensor([0, 1]), None, "shape"),
        (torch.tensor([[0, 1]]), None, "shape"),
        (torch.tensor([[0], [1]]), torch.ones(2, 3), "same shape"),
    ],
)
def test_permutation_rejects_invalid_compact_inputs(indices, probs, message):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    with pytest.raises(ValueError, match=message):
        dispatcher.moe_permutation(torch.ones(2, 1), indices, probs)


def test_permutation_rejects_token_dropping():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    with pytest.raises(ValueError, match="token dropping"):
        dispatcher.moe_permutation(
            torch.ones(2, 1), torch.tensor([[0, 1], [0, 1]]), num_out_tokens=3,
        )


@pytest.mark.parametrize("implementation", ["torch", "npu", "te"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_compact_permutation_preserves_unsorted_strided_routes_and_gradients(
    npu_ops, cuda_ops, implementation, index_dtype, num_tokens,
):
    dispatcher = cuda_ops if implementation == "te" else npu_ops
    tokens = torch.tensor([[2., 99., 3.], [5., 99., 7.], [11., 99., 13.]])[
        :num_tokens, ::2
    ].requires_grad_()
    # Large, unused expert IDs must not require a dense token-by-expert map.
    indices = torch.tensor([[100000, -1, 0], [2, -1, 100000], [2, -1, 0]], dtype=index_dtype)[
        :num_tokens, ::2
    ]
    probs = torch.tensor([[0., 99., .5], [.25, 99., .75], [1., 99., .5]])[
        :num_tokens, ::2
    ].requires_grad_()
    result = dispatcher.moe_permutation(
        tokens, indices, probs, impl=f"{implementation}_moe_mask_permutation",
    )
    expected_tokens = torch.tensor([[2., 3.], [11., 13.], [5., 7.], [11., 13.], [2., 3.], [5., 7.]])
    expected_probs = torch.tensor([.5, .5, .25, 1., 0., .75])
    expected_rows = torch.tensor([[4, 0], [2, 5], [3, 1]], dtype=torch.int32)
    torch.testing.assert_close(result.tokens, expected_tokens[:num_tokens * 2])
    torch.testing.assert_close(result.probs, expected_probs[:num_tokens * 2])
    torch.testing.assert_close(result.row_indices, expected_rows[:num_tokens])
    torch.testing.assert_close(result.expert_indices, indices.long())
    (result.tokens * result.probs[:, None]).sum().backward()
    torch.testing.assert_close(tokens.grad, torch.tensor([[.5, .5], [1., 1.], [1.5, 1.5]])[:num_tokens])
    torch.testing.assert_close(probs.grad, torch.tensor([[5., 5.], [12., 12.], [24., 24.]])[:num_tokens])


@pytest.mark.parametrize("dtype,large", [(torch.bfloat16, 256), (torch.float16, 2048)])
def test_te_compact_permutation_backward_preserves_small_contributions(cuda_ops, dtype, large):
    tokens = torch.ones(1, 128, dtype=dtype, requires_grad=True)
    result = cuda_ops.moe_permutation(
        tokens, torch.tensor([[0, 1, 2]]), impl="te_moe_mask_permutation",
    )
    gradient = torch.tensor([[large], [1], [-large]], dtype=dtype).expand(3, 128)
    result.tokens.backward(gradient)
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens), rtol=0, atol=0)


def test_te_compact_permutation_handles_wide_topk(cuda_ops):
    source = torch.tensor([[3.]], requires_grad=True)
    indices = torch.arange(129).flip(0).view(1, 129)
    result = cuda_ops.moe_permutation(source, indices, impl="te_moe_mask_permutation")
    torch.testing.assert_close(result.tokens, torch.full((129, 1), 3.))
    torch.testing.assert_close(result.row_indices, indices.int())
    result.tokens.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[129.]]))


def test_te_dense_permutation_uses_mask_kernel_and_its_slot_order(cuda_ops, monkeypatch):
    mask = torch.tensor([[True, False, True], [False, True, True]])
    ids = torch.tensor([[0, 2], [1, 2]])
    source = torch.tensor([[3.], [5.]], requires_grad=True)
    probs = torch.tensor([[0., 7., .25], [8., .5, 1.]], requires_grad=True)

    def permute(tokens, routing_map, num_out_tokens):
        assert routing_map is mask
        assert num_out_tokens == 4
        # TE may emit slots in any order. Pair destination rows and expert IDs.
        mapping = torch.tensor([[2, 0, -1, 2, 0, -1, 2], [3, 1, -1, 2, 1, -1, 2]], dtype=torch.int32)
        return tokens[[0, 1, 0, 1]], mapping

    monkeypatch.setattr(sys.modules["transformer_engine.pytorch.permutation"], "moe_permute", permute)
    result = cuda_ops.moe_permutation(source, mask, probs, expert_indices=ids, impl="te_moe_mask_permutation")
    torch.testing.assert_close(result.expert_indices, torch.tensor([[2, 0], [2, 1]]))
    torch.testing.assert_close(result.row_indices, torch.tensor([[2, 0], [3, 1]], dtype=torch.int32))
    torch.testing.assert_close(result.probs, torch.tensor([0., .5, .25, 1.]))
    (result.tokens * result.probs[:, None]).sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[.25], [1.5]]))
    torch.testing.assert_close(probs.grad, torch.tensor([[3., 0., 3.], [0., 5., 5.]]))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 2])
def test_dense_permutation_call_preserves_compact_routes_and_gradients(cached, num_tokens):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    tokens = torch.tensor([[2.], [3.]])[:num_tokens].requires_grad_()
    mask = torch.tensor([[True, False, True], [False, True, True]])[:num_tokens]
    ids = torch.tensor([[0, 2], [1, 2]])[:num_tokens]
    probs = torch.tensor([[0., 9., .5], [8., 1., .25]])[:num_tokens].requires_grad_()
    if cached:
        result = dispatcher.moe_permutation(tokens, mask, probs, ids.numel(), ids)
    else:
        result = dispatcher.moe_permutation(tokens, routing_map=mask, probs=probs)
    expected_ids = ids if cached or num_tokens else ids[:, :0]
    torch.testing.assert_close(result.expert_indices, expected_ids)
    torch.testing.assert_close(result.tokens, torch.tensor([[2.], [3.], [2.], [3.]])[:ids.numel()])
    torch.testing.assert_close(result.probs, torch.tensor([0., 1., .5, .25])[:ids.numel()])
    (result.tokens * result.probs[:, None]).sum().backward()
    torch.testing.assert_close(tokens.grad, torch.tensor([[.5], [1.25]])[:num_tokens])
    torch.testing.assert_close(probs.grad, torch.tensor([[2., 0., 2.], [0., 3., 3.]])[:num_tokens])


def test_compact_permutation_accepts_keyword_expert_indices():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    ids = torch.tensor([[2, 0]])
    result = dispatcher.moe_permutation(torch.tensor([[1.]]), expert_indices=ids)
    torch.testing.assert_close(result.row_indices, torch.tensor([[1, 0]], dtype=torch.int32))
    torch.testing.assert_close(result.expert_indices, ids)


@pytest.mark.parametrize("provider", ["torch", "te", "npu"])
def test_permutation_provider_accepts_compact_indices_by_keyword(npu_ops, cuda_ops, provider):
    from torchturbo.ops.providers.cuda.moe_permutation import TransformerEngineMoEMaskPermutation
    from torchturbo.ops.providers.native.moe_permutation import TorchMoEMaskPermutation
    from torchturbo.ops.providers.npu.moe_permutation import NpuMoEMaskPermutation

    impl = {
        "torch": TorchMoEMaskPermutation, "te": TransformerEngineMoEMaskPermutation,
        "npu": NpuMoEMaskPermutation,
    }[provider]()
    ids = torch.tensor([[2, 0]])
    result = impl(torch.tensor([[3.]]), expert_indices=ids, probs=torch.tensor([[0., 1.]]))
    torch.testing.assert_close(result.probs, torch.tensor([1., 0.]))
    torch.testing.assert_close(result.expert_indices, ids)
