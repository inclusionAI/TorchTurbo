"""Contracts for source packing, EP stages, and dispatcher metadata."""

import importlib
import sys
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


def _chunked_dispatcher(num_chunks: int, num_local_experts: int = 2, topk: int = 2):
    from torchturbo.models.modules.moe import token_dispatcher_chunked as chunked_module

    return chunked_module.MoEChunkedAlltoAllTokenDispatcher(
        num_local_experts,
        list(range(num_local_experts)),
        MoEConfig(
            num_moe_experts=num_local_experts,
            moe_router_topk=topk,
            moe_num_chunks=num_chunks,
        ),
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
