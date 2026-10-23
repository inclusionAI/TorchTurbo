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
            num_moe_experts=2, moe_router_topk=1, moe_permute_fusion=False,
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
    assert dispatcher.routing_map is None

    other = torch.tensor([[40.0], [50.0]])
    other_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    other_routing = RoutingResult(other_probs, other_probs.bool(), torch.tensor([[0], [1]]))
    other_packed, _ = dispatcher.dispatch_preprocess(other, other_routing)
    torch.testing.assert_close(dispatcher.combine_postprocess(other_packed), other)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_combine_sums_already_weighted_expert_outputs(dtype):
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(
        3, [0, 1, 2], MoEConfig(
            num_moe_experts=3, moe_router_topk=2, moe_permute_fusion=False,
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
            moe_router_topk=2, moe_permute_fusion=False,
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
@pytest.mark.parametrize("permute_fusion", [False, True])
def test_npu_dispatch_handles_empty_sources_and_input_probs(
    monkeypatch, num_tokens, on_input, permute_fusion,
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
        restored = tokens[reverse_mapping.long()].reshape(num_tokens, topk, 2)
        if probs is not None:
            restored = restored * probs.unsqueeze(-1)
        return restored.sum(1)

    backend = ModuleType("torch_npu")
    backend.npu_moe_token_permute = npu_permute
    backend.npu_moe_token_unpermute = npu_unpermute
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    config = MoEConfig(
        num_moe_experts=2, moe_router_topk=1 if on_input else 2,
        moe_apply_probs_on_input=on_input, moe_permute_fusion=permute_fusion,
    )
    runtime = NpuRuntimeWithCpuKernels()
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=runtime)
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
    assert bool(calls) == (permute_fusion and num_tokens > 0)
