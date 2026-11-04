"""Contracts for source packing, EP stages, and dispatcher metadata."""

import importlib
import sys
import contextlib
from types import ModuleType

import numpy as np
import pytest
import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe import token_dispatcher as dispatch_module
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime import TurboDevice
from torchturbo.models.modules.moe.utils import RoutingResult, maybe_move_tensor_to_cpu


@pytest.mark.parametrize("as_numpy", [False, True])
def test_cpu_counts_honor_host_representation(as_numpy):
    counts = torch.tensor([3, 0, 5])
    result = maybe_move_tensor_to_cpu(counts, as_numpy=as_numpy)
    if as_numpy:
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [3, 0, 5])
    else:
        assert result is counts


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def test_six_stages_restore_token_order_and_gradients():
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(
        2, [0, 1], MoEConfig(
            num_moe_experts=2, moe_router_topk=1,
        ), object(),
    )
    x = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor([[0.0, 0.5], [0.25, 0.0], [0.0, 0.75]], requires_grad=True)
    routing = RoutingResult(probs, probs.detach().bool(), torch.tensor([[1], [0], [1]]))
    packed, packed_probs = dispatcher.dispatch_preprocess(x, routing)
    torch.testing.assert_close(packed, torch.tensor([[20.0], [10.0], [30.0]]))
    received, received_probs = dispatcher.token_dispatch(packed, packed_probs)
    local, counts, local_probs = dispatcher.dispatch_postprocess(received, received_probs)
    torch.testing.assert_close(counts, torch.tensor([1, 2]))
    torch.testing.assert_close(packed_probs, torch.tensor([.25, .5, .75]))
    torch.testing.assert_close(local_probs, packed_probs)
    local = local * local_probs.unsqueeze(-1)
    returned = dispatcher.token_combine(dispatcher.combine_preprocess(local))
    output = dispatcher.combine_postprocess(returned)
    torch.testing.assert_close(output, torch.tensor([[5.0], [5.0], [22.5]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0.5], [0.25], [0.75]]))
    torch.testing.assert_close(probs.grad, torch.tensor([[0.0, 10.0], [20.0, 0.0], [0.0, 30.0]]))
    assert dispatcher.permutation_row_indices is None
    assert dispatcher.permutation_expert_indices is None

    other = torch.tensor([[40.0], [50.0]])
    other_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    other_routing = RoutingResult(other_probs, other_probs.bool(), torch.tensor([[0], [1]]))
    other_packed, _ = dispatcher.dispatch_preprocess(other, other_routing)
    torch.testing.assert_close(dispatcher.combine_postprocess(other_packed), other)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_combine_sums_already_weighted_expert_outputs(dtype):
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(
        3, [0, 1, 2], MoEConfig(
            num_moe_experts=3, moe_router_topk=2,
        ), object(),
    )
    x = torch.tensor([[10.], [20.], [30.]], dtype=dtype, requires_grad=True)
    probs = torch.tensor([
        [.25, 0., .5], [0., .75, .125], [.125, .25, 0.],
    ], requires_grad=True)
    routing = RoutingResult(
        probs, probs.detach().bool(), torch.tensor([[0, 2], [1, 2], [0, 1]]),
    )
    packed, packed_probs = dispatcher.dispatch_preprocess(x, routing)
    packed = (packed * packed_probs.unsqueeze(-1)).to(dtype)
    output = dispatcher.combine_postprocess(packed)
    torch.testing.assert_close(output, torch.tensor([[7.5], [17.5], [11.25]], dtype=dtype))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[.75], [.875], [.375]], dtype=dtype))
    torch.testing.assert_close(probs.grad, torch.tensor([
        [10., 0., 10.], [0., 20., 20.], [30., 30., 0.],
    ]))


def test_two_rank_counts_and_receiver_expert_order(monkeypatch):
    # Rank 1 owns E2/E3. Rank 0 sends it four rows, and rank 1 sends three.
    monkeypatch.setattr(
        dispatch_module, "gather_from_sequence_parallel_region",
        lambda counts, group: torch.tensor([1, 1, 2, 2, 1, 2, 1, 2]),
    )
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(
        2, [2, 3], MoEConfig(
            num_moe_experts=4, expert_model_parallel_size=2,
            moe_router_topk=2,
        ), object(),
    )
    x = torch.tensor([[10.0], [20.0], [30.0]])
    routing_map = torch.tensor([
        [False, True, False, True],
        [True, True, False, False],
        [False, False, True, True],
    ])
    routing = RoutingResult(
        routing_map.float(), routing_map, torch.tensor([[1, 3], [0, 1], [2, 3]]),
    )
    packed, _ = dispatcher.dispatch_preprocess(x, routing)
    torch.testing.assert_close(packed.flatten(), torch.tensor([20., 10., 20., 30., 10., 30.]))
    np.testing.assert_array_equal(dispatcher.input_splits, [3, 3])
    np.testing.assert_array_equal(dispatcher.output_splits, [4, 3])
    torch.testing.assert_close(dispatcher.tokens_per_expert, torch.tensor([3, 4]))
    received = torch.tensor([[201.], [202.], [301.], [302.], [203.], [303.], [304.]])
    probs = torch.arange(1., 8.) / 10
    local, _, local_probs = dispatcher.dispatch_postprocess(received, probs)
    torch.testing.assert_close(local.flatten(), torch.tensor([201., 202., 203., 301., 302., 303., 304.]))
    torch.testing.assert_close(local_probs, torch.tensor([.1, .2, .5, .3, .4, .6, .7]))
    torch.testing.assert_close(dispatcher.combine_preprocess(local), received)


@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("policy", [OpPolicy.NATIVE, OpPolicy.AUTO])
def test_npu_dispatch_handles_empty_sources_and_input_probs(
    monkeypatch, num_tokens, on_input, policy,
):
    # Substitute only the device kernels; retain the real dispatcher and autograd.
    class NpuRuntimeWithCpuKernels(CpuRuntime):
        turbo_device = TurboDevice.NPU

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    calls = []

    def npu_permute(tokens, indices):
        calls.append("permute")
        order = indices.flatten().argsort(stable=True)
        return tokens[order // indices.shape[1]], order.argsort().to(torch.int32)

    def npu_unpermute(tokens, reverse_mapping, probs=None):
        calls.append("unpermute")
        topk = 1 if probs is None else probs.shape[1]
        restored = tokens[reverse_mapping.long()].reshape(-1, topk, tokens.shape[-1])
        if probs is not None:
            restored = restored * probs.unsqueeze(-1)
        return restored.sum(1)

    backend = ModuleType("torch_npu")
    backend.npu_moe_token_permute = npu_permute
    backend.npu_moe_token_unpermute = npu_unpermute
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    config = MoEConfig(
        num_moe_experts=2, moe_router_topk=1 if on_input else 2,
        moe_apply_probs_on_input=on_input,
    )
    runtime = NpuRuntimeWithCpuKernels()
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=runtime, default_policy=policy)
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(2, [0, 1], config, object())
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16)[:num_tokens].requires_grad_()
    probs = torch.tensor([[0., .5], [.25, 0.], [0., .75]])[:num_tokens].requires_grad_()
    routing_map = probs.detach().bool() if on_input else torch.ones_like(probs, dtype=torch.bool)
    indices = routing_map.nonzero(as_tuple=True)[1].reshape(num_tokens, config.moe_router_topk)
    routing = RoutingResult(probs, routing_map, indices)
    with monkeypatch.context() as cached_indices:
        cached_indices.setattr(
            torch, "nonzero",
            lambda *args, **kwargs: pytest.fail("Dispatcher must reuse cached expert indices."),
        )
        packed, packed_probs = dispatcher.dispatch_preprocess(x, routing)
    received, received_probs = dispatcher.token_dispatch(packed, packed_probs)
    packed, _, packed_probs = dispatcher.dispatch_postprocess(received, received_probs)
    assert packed_probs is not None
    packed = (packed * packed_probs.unsqueeze(-1)).to(x.dtype)
    returned = dispatcher.token_combine(dispatcher.combine_preprocess(packed))
    output = dispatcher.combine_postprocess(returned)
    torch.testing.assert_close(output, (x * probs.sum(-1, keepdim=True)).to(x.dtype))
    output.sum().backward()
    torch.testing.assert_close(x.grad, probs.detach().sum(-1, keepdim=True).expand_as(x).to(x.dtype))
    expected_probs_grad = x.detach().float().sum(-1, keepdim=True).expand_as(probs) * routing_map
    torch.testing.assert_close(probs.grad, expected_probs_grad)
    assert bool(calls) == (policy == OpPolicy.AUTO and num_tokens > 0)


class _ScaleExperts(torch.nn.Module):
    """Row-wise per-expert scaling stand-in for GroupedMLP in chunked tests."""

    def __init__(self, num_local_experts: int):
        super().__init__()
        self.scales = torch.nn.Parameter(
            torch.arange(1, num_local_experts + 1, dtype=torch.float32)
        )

    def forward(self, tokens, tokens_per_expert, probs):
        assert probs is None, "chunked dispatcher keeps weights for the merge"
        if tokens.numel() == 0:
            return tokens, None
        counts = [int(count) for count in tokens_per_expert.tolist()]
        parts = torch.split(tokens, counts)
        weighted = [part * scale for part, scale in zip(parts, self.scales)]
        return torch.cat(weighted, dim=0), None


def _chunked_dispatcher(
    num_chunks: int,
    num_local_experts: int = 2,
    topk: int = 2,
):
    from torchturbo.models.modules.moe import token_dispatcher_chunked as chunked_module

    kwargs = dict(
        num_moe_experts=num_local_experts,
        moe_router_topk=topk,
        moe_num_chunks=num_chunks,
    )
    return chunked_module.MoEChunkedAlltoAllTokenDispatcher(
        num_local_experts,
        list(range(num_local_experts)),
        MoEConfig(**kwargs),
        object(),
    )


def test_chunked_rejects_apply_probs_on_input():
    from torchturbo.models.modules.moe import token_dispatcher_chunked as chunked_module

    with pytest.raises(ValueError, match="alltoall_chunked"):
        chunked_module.MoEChunkedAlltoAllTokenDispatcher(
            2, [0, 1], MoEConfig(
                num_moe_experts=2, moe_router_topk=1,
                moe_apply_probs_on_input=True,
            ), object(),
        )


def test_chunk_bounds_split_tokens():
    dispatcher = _chunked_dispatcher(3)
    assert dispatcher._chunk_bounds(7) == [(0, 3), (3, 5), (5, 7)]
    assert dispatcher._chunk_bounds(3) == [(0, 1), (1, 2), (2, 3)]
    assert dispatcher._chunk_bounds(2) == [(0, 1), (1, 2), (2, 2)]
    assert dispatcher._chunk_bounds(1) == [(0, 1), (1, 1), (1, 1)]
    assert dispatcher._chunk_bounds(0) == [(0, 0), (0, 0), (0, 0)]


@pytest.mark.parametrize("num_chunks", [1, 2, 5])
def test_chunked_execute_matches_manual_reference(num_chunks):
    dispatcher = _chunked_dispatcher(num_chunks)
    experts = _ScaleExperts(2)
    x = torch.tensor(
        [[10.0], [20.0], [30.0], [40.0], [50.0]], requires_grad=True,
    )
    probs = torch.tensor(
        [
            [0.25, 0.75], [0.5, 0.5], [0.2, 0.8], [1.0, 0.0], [0.4, 0.6],
        ],
        requires_grad=True,
    )
    routing_map = torch.ones(5, 2, dtype=torch.bool)
    expert_indices = torch.tensor([[0, 1]] * 5)
    routing = RoutingResult(probs, routing_map, expert_indices)

    output = dispatcher.execute(x, routing, experts)

    reference = x * (probs[:, :1] + 2.0 * probs[:, 1:])
    torch.testing.assert_close(output, reference)
    output.sum().backward()
    torch.testing.assert_close(x.grad, probs[:, :1] + 2.0 * probs[:, 1:])
    torch.testing.assert_close(probs.grad, torch.cat([x, 2.0 * x], dim=1))


def test_chunked_execute_keeps_hidden_shape():
    dispatcher = _chunked_dispatcher(2)
    experts = _ScaleExperts(2)
    x = torch.randn(2, 3, 8, requires_grad=True)
    probs = torch.rand(6, 2)
    probs = probs / probs.sum(dim=1, keepdim=True)
    probs.requires_grad_(True)
    routing_map = torch.ones(6, 2, dtype=torch.bool)
    routing = RoutingResult(probs, routing_map, torch.tensor([[0, 1]] * 6))
    output = dispatcher.execute(x, routing, experts)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None and probs.grad is not None


def test_chunked_two_rank_metadata_and_host_splits(monkeypatch):
    # Rank 1 owns E2/E3; rank 0's chunk counts are injected through the gather.
    rank0_counts = torch.tensor([[1, 2, 2, 0], [0, 1, 1, 3]])

    def fake_all_gather_into_tensor(output, input, group=None, async_op=False):
        output = output.view(2, 2, 4)
        output[0].copy_(rank0_counts)
        output[1].copy_(input.view(2, 4))

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor,
    )
    from torchturbo.models.modules.moe import token_dispatcher_chunked as chunked_module

    dispatcher = chunked_module.MoEChunkedAlltoAllTokenDispatcher(
        2, [2, 3], MoEConfig(
            num_moe_experts=4, expert_model_parallel_size=2,
            moe_router_topk=2, moe_num_chunks=2,
        ), object(),
    )
    # Counts derive from the routing mask, matching these selected experts.
    routing_map = torch.tensor([
        [True, False, True, False],
        [False, False, True, True],
        [False, True, True, False],
        [True, False, False, True],
    ])
    routing = RoutingResult(
        routing_map.float(), routing_map,
        torch.tensor([[0, 2], [2, 3], [1, 2], [0, 3]]),
    )
    bounds = [(0, 2), (2, 4)]

    metadata = dispatcher._chunk_metadata(routing, bounds)
    torch.testing.assert_close(
        metadata.received_counts,
        torch.tensor([[[2, 0], [2, 1]], [[1, 3], [1, 1]]]),
    )
    torch.testing.assert_close(metadata.receive_counts, torch.tensor([[2, 3], [4, 2]]))
    torch.testing.assert_close(metadata.tokens_per_expert, torch.tensor([[4, 1], [2, 4]]))

    host = dispatcher._host_metadata(metadata)
    assert host.input_splits == [[1, 3], [2, 2]]
    assert host.output_splits == [[2, 3], [4, 2]]
    torch.testing.assert_close(host.tokens_per_expert[0], torch.tensor([4, 1]))
    torch.testing.assert_close(host.tokens_per_expert[1], torch.tensor([2, 4]))


def _imbalanced_routing(num_tokens: int, num_experts: int, topk: int):
    """Skewed routing: half of the tokens target expert 0."""
    gen = torch.Generator().manual_seed(7)
    primary = torch.randint(0, num_experts, (num_tokens,), generator=gen)
    primary[: num_tokens // 2] = 0
    expert_indices = [primary]
    for k in range(1, topk):
        expert_indices.append((primary + k * 3) % num_experts)
    expert_indices = torch.stack(expert_indices, dim=1)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    routing_map.scatter_(1, expert_indices, True)
    topk_probs = torch.rand(num_tokens, topk, generator=gen)
    topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
    topk_probs.requires_grad_(True)
    # Router probs are dense over experts with unselected entries zeroed.
    probs = torch.zeros(num_tokens, num_experts).scatter(
        1, expert_indices, topk_probs,
    )
    return RoutingResult(probs, routing_map, expert_indices)


def test_chunked_execute_imbalanced_routing_matches_closed_form():
    num_experts, num_chunks, num_tokens = 8, 3, 64
    dispatcher = _chunked_dispatcher(
        num_chunks, num_local_experts=num_experts,
    )
    experts = _ScaleExperts(num_experts)
    x = torch.randn(num_tokens, 8, requires_grad=True)
    routing = _imbalanced_routing(num_tokens, num_experts, topk=2)
    routing.probs.retain_grad()

    output = dispatcher.execute(x, routing, experts)

    scales = torch.arange(1, num_experts + 1, dtype=torch.float32)
    weighted = (routing.probs * scales).sum(dim=1, keepdim=True)
    reference = x * weighted
    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
    output.sum().backward()
    assert x.grad is not None and routing.probs.grad is not None
    torch.testing.assert_close(
        x.grad, weighted.expand_as(x.grad), atol=1e-5, rtol=1e-5,
    )


def test_chunked_execute_handles_empty_tokens():
    dispatcher = _chunked_dispatcher(3, num_local_experts=4)
    experts = _ScaleExperts(4)
    x = torch.zeros(0, 4, requires_grad=True)
    routing = RoutingResult(
        torch.zeros(0, 2),
        torch.zeros(0, 4, dtype=torch.bool),
        torch.zeros(0, 2, dtype=torch.long),
    )
    output = dispatcher.execute(x, routing, experts)
    assert output.shape == x.shape


@pytest.mark.parametrize(
    "num_tokens, topk",
    [(1, 1), (2, 1), (3, 2), (5, 1)],
)
def test_chunked_execute_fewer_tokens_than_chunks(num_tokens, topk):
    # More chunks than tokens: some chunks end up empty.
    num_experts, num_chunks = 4, 4
    dispatcher = _chunked_dispatcher(
        num_chunks, num_local_experts=num_experts, topk=topk,
    )
    experts = _ScaleExperts(num_experts)
    x = torch.randn(num_tokens, 4, requires_grad=True)
    routing = _imbalanced_routing(num_tokens, num_experts, topk=topk)
    routing.probs.retain_grad()

    output = dispatcher.execute(x, routing, experts)

    scales = torch.arange(1, num_experts + 1, dtype=torch.float32)
    weighted = (routing.probs * scales).sum(dim=1, keepdim=True)
    torch.testing.assert_close(output, x * weighted, atol=1e-5, rtol=1e-5)
    output.sum().backward()
    assert x.grad is not None and routing.probs.grad is not None


def test_chunked_execute_topk1_single_expert_hotspot():
    # Extreme imbalance: every token routes to expert 0 only.
    num_experts, num_chunks, num_tokens = 4, 3, 48
    dispatcher = _chunked_dispatcher(
        num_chunks, num_local_experts=num_experts, topk=1,
    )
    experts = _ScaleExperts(num_experts)
    x = torch.randn(num_tokens, 4, requires_grad=True)
    expert_indices = torch.zeros(num_tokens, 1, dtype=torch.long)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    routing_map[:, 0] = True
    probs = torch.ones(num_tokens, num_experts, requires_grad=True)
    probs = probs * routing_map  # dense, only expert 0 weighted
    routing = RoutingResult(probs, routing_map, expert_indices)
    routing.probs.retain_grad()

    output = dispatcher.execute(x, routing, experts)
    torch.testing.assert_close(output, x, atol=1e-5, rtol=1e-5)
    output.sum().backward()
    assert x.grad is not None


def test_chunked_backward_autograd_guard_single_recompute():
    """Checkpointed chunked MoE recomputes exactly once per backward.

    Regression for the use_reentrant=False crash: the engine's AC wires
    checkpoint with a ``context_fn`` whose recompute context is a
    one-shot ``@contextmanager``. torch re-enters that context once per
    *distinct* backward graph id a frame sees. Without a whole-backward
    GraphExecGroup the frame sees two ids -- the outer backward id (a
    node deeper than the dispatcher unpacks first) and the dispatcher's
    own group id -- so the second recompute re-enters the one-shot
    context and dies in contextlib. Constructing the chunked dispatcher
    installs an autograd guard that runs every backward under one shared
    group; assert a plain ``backward()`` recomputes once with reference
    grads, and characterize the unguarded failure mode.
    """
    import torch.utils.checkpoint as checkpoint_module
    from torch.utils.checkpoint import checkpoint

    from torchturbo.models.modules.moe import (
        token_dispatcher_chunked as chunked_module,
    )
    from torchturbo.models.modules.moe.token_dispatcher_chunked import (
        shared_recompute_group,
    )

    if not hasattr(checkpoint_module, "GraphExecGroup"):
        pytest.skip("torch build without GraphExecGroup")

    dispatcher = _chunked_dispatcher(2)
    experts = _ScaleExperts(2)
    routing_map = torch.ones(5, 2, dtype=torch.bool)
    expert_indices = torch.tensor([[0, 1]] * 5)
    probs_data = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [0.2, 0.8], [1.0, 0.0], [0.4, 0.6]],
    )

    state = {"recomputes": 0}

    @contextlib.contextmanager
    def one_shot_recompute_context():
        # One-shot, like ``fsdp_recompute_context``: a second __enter__
        # raises inside contextlib.
        state["recomputes"] += 1
        yield

    def context_fn():
        return contextlib.nullcontext(), one_shot_recompute_context()

    def block_forward(x, probs):
        routing = RoutingResult(probs, routing_map, expert_indices)
        merged = dispatcher.execute(x, routing, experts)
        # Saving node *deeper* than the chunked dispatcher: its backward
        # runs first and performs the frame's first unpack/recompute.
        return torch.sigmoid(merged)

    def clone_inputs():
        x = torch.tensor(
            [[10.0], [20.0], [30.0], [40.0], [50.0]], requires_grad=True,
        )
        return x, probs_data.clone().requires_grad_(True)

    # Reference: same block without checkpointing.
    x_ref, probs_ref = clone_inputs()
    block_forward(x_ref, probs_ref).sum().backward()

    # The guard is installed with the dispatcher: a plain backward() must
    # run under one group and recompute exactly once.
    assert chunked_module._ORIGINAL_AUTOGRAD
    assert torch.autograd.backward is not chunked_module._ORIGINAL_AUTOGRAD["backward"]

    x, probs = clone_inputs()
    out = checkpoint(
        block_forward, x, probs, use_reentrant=False, context_fn=context_fn,
    )
    before = state["recomputes"]
    out.sum().backward()
    assert state["recomputes"] - before == 1
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(probs.grad, probs_ref.grad)
    assert experts.scales.grad is not None

    # Explicit wrapping keeps working and stays at one recompute
    # (ambient-group reuse, no nesting).
    x, probs = clone_inputs()
    out = checkpoint(
        block_forward, x, probs, use_reentrant=False, context_fn=context_fn,
    )
    before = state["recomputes"]
    with shared_recompute_group():
        out.sum().backward()
    assert state["recomputes"] - before == 1
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(probs.grad, probs_ref.grad)

    # Characterize the unguarded shape: with the guard removed the frame
    # sees two gid domains. Either torch crashes re-entering the one-shot
    # recompute context (current contextlib behavior) or it tolerates it
    # and pays one recompute per gid domain (>= 2).
    chunked_module._uninstall_shared_backward_guard()
    try:
        x, probs = clone_inputs()
        out = checkpoint(
            block_forward, x, probs, use_reentrant=False, context_fn=context_fn,
        )
        before = state["recomputes"]
        crashed = False
        try:
            out.sum().backward()
        except Exception:
            crashed = True
        assert crashed or state["recomputes"] - before >= 2
    finally:
        chunked_module._install_shared_backward_guard()
    assert torch.autograd.backward is not chunked_module._ORIGINAL_AUTOGRAD["backward"]
