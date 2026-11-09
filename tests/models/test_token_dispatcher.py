"""Contracts for source packing, EP stages, and dispatcher metadata."""

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe import token_dispatcher as dispatch_module
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime import TurboDevice
from torchturbo.models.modules.moe.utils import maybe_move_tensor_to_cpu


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


@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("dispatcher_type", ["alltoall", "flex"])
def test_compact_routes_preserve_zero_probability_assignments(num_tokens, dispatcher_type):
    dispatcher_cls = (
        dispatch_module.MoEAlltoAllTokenDispatcher if dispatcher_type == "alltoall"
        else dispatch_module.MoEFlexTokenDispatcher
    )
    dispatcher = dispatcher_cls(
        4, [0, 1, 2, 3], MoEConfig(
            num_moe_experts=4, moe_router_topk=2,
            moe_token_dispatcher_type=dispatcher_type,
        ), object(),
    )
    x = torch.tensor([
        [1., 99., 2., 99.], [3., 99., 4., 99.], [5., 99., 6., 99.],
    ])[:num_tokens, ::2].requires_grad_()
    probs = torch.tensor([
        [0., 7., 8., .5], [9., .25, 0., 6.], [.75, 4., .25, 5.],
    ]).T.contiguous().T[:num_tokens].requires_grad_()
    indices = torch.tensor([[0, 3], [1, 2], [0, 2]])[:num_tokens]
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, indices, True)
    tokens, selected_probs = dispatcher.dispatch_preprocess(
        x, mask, probs, expert_indices=indices,
    )
    if dispatcher_type == "flex":
        torch.testing.assert_close(
            selected_probs, torch.tensor([[0., .5], [.25, 0.], [.75, .25]])[:num_tokens],
        )
    tokens, selected_probs = dispatcher.token_dispatch(tokens, selected_probs)
    tokens, counts, selected_probs = dispatcher.dispatch_postprocess(tokens, selected_probs)
    torch.testing.assert_close(
        counts, torch.tensor([2, 1, 2, 1] if num_tokens else [0, 0, 0, 0]),
    )
    tokens = tokens * selected_probs[:, None]
    tokens = dispatcher.token_combine(dispatcher.combine_preprocess(tokens))
    output = dispatcher.combine_postprocess(tokens)
    torch.testing.assert_close(output, torch.tensor([[.5, 1.], [.75, 1.], [5., 6.]])[:num_tokens])
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[.5, .5], [.25, .25], [1., 1.]])[:num_tokens])
    torch.testing.assert_close(probs.grad, torch.tensor([
        [3., 0., 0., 3.], [0., 7., 7., 0.], [11., 0., 11., 0.],
    ])[:num_tokens])


def test_moe_layer_applies_manager_hooks_to_inputs_and_outputs(monkeypatch):
    class ScaledManager(dispatch_module._HybridEPManager):
        def wrap_layer_input(self, hidden_states):
            return hidden_states * 2

        def finalize_layer_output(self, output):
            return output * 3

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    config = MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=2,
        moe_router_topk=1, moe_token_dispatcher_type="flex",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = MoELayer(config, object(), submodules=submodules)
    candidate = MoELayer(config, object(), submodules=submodules)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
    candidate.load_state_dict(reference.state_dict())
    candidate.token_dispatcher._comm_manager = ScaledManager(
        object(), 2, 2, 1, config,
    )
    x = torch.randn(1, 3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    output, logits = candidate(x)
    ref_output, ref_logits = reference(ref_x * 2)
    ref_output = ref_output * 3
    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(logits, ref_logits)
    output.sum().backward()
    ref_output.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize(
    "dispatcher_cls,dispatcher_type",
    [
        (dispatch_module.MoEAlltoAllTokenDispatcher, "alltoall"),
        (dispatch_module.MoEFlexTokenDispatcher, "flex"),
    ],
)
def test_six_stages_restore_token_order_and_gradients(dispatcher_cls, dispatcher_type):
    dispatcher = dispatcher_cls(
        2, [0, 1], MoEConfig(
            num_moe_experts=2,
            moe_router_topk=1,
            moe_token_dispatcher_type=dispatcher_type,
        ), object(),
    )
    assert not isinstance(dispatcher, dispatch_module._DispatchManager)
    x = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    probs = torch.tensor([[0.0, 0.5], [0.25, 0.0], [0.0, 0.75]], requires_grad=True)
    packed, packed_probs = dispatcher.dispatch_preprocess(
        x, probs.detach().bool(), probs, expert_indices=torch.tensor([[1], [0], [1]]),
    )
    if dispatcher_type == "alltoall":
        torch.testing.assert_close(packed, torch.tensor([[20.0], [10.0], [30.0]]))
    received, received_probs = dispatcher.token_dispatch(packed, packed_probs)
    torch.testing.assert_close(received, torch.tensor([[20.0], [10.0], [30.0]]))
    torch.testing.assert_close(received_probs, torch.tensor([.25, .5, .75]))
    local, counts, local_probs = dispatcher.dispatch_postprocess(received, received_probs)
    torch.testing.assert_close(counts, torch.tensor([1, 2]))
    torch.testing.assert_close(local_probs, received_probs)
    local = local * local_probs.unsqueeze(-1)
    returned = dispatcher.token_combine(dispatcher.combine_preprocess(local))
    output = dispatcher.combine_postprocess(returned)
    torch.testing.assert_close(output, torch.tensor([[5.0], [5.0], [22.5]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0.5], [0.25], [0.75]]))
    torch.testing.assert_close(probs.grad, torch.tensor([[0.0, 10.0], [20.0, 0.0], [0.0, 30.0]]))
    if isinstance(dispatcher, dispatch_module.MoEFlexTokenDispatcher):
        assert dispatcher._comm_manager.handle is None
    else:
        assert dispatcher.permutation_row_indices is None
        assert dispatcher.permutation_expert_indices is None

    other = torch.tensor([[40.0], [50.0]])
    other_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    other_packed, other_packed_probs = dispatcher.dispatch_preprocess(
        other, other_probs.bool(), other_probs, expert_indices=torch.tensor([[0], [1]]),
    )
    other_received, _ = dispatcher.token_dispatch(other_packed, other_packed_probs)
    other_returned = dispatcher.token_combine(dispatcher.combine_preprocess(other_received))
    torch.testing.assert_close(dispatcher.combine_postprocess(other_returned), other)


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
    packed, packed_probs = dispatcher.dispatch_preprocess(
        x, probs.detach().bool(), probs,
        expert_indices=torch.tensor([[0, 2], [1, 2], [0, 1]]),
    )
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
    packed, _ = dispatcher.dispatch_preprocess(
        x, routing_map, routing_map.float(),
        expert_indices=torch.tensor([[1, 3], [0, 1], [2, 3]]),
    )
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
@pytest.mark.parametrize("dispatcher_type", ["alltoall", "flex"])
def test_npu_dispatch_handles_empty_sources_and_input_probs(
    monkeypatch, num_tokens, on_input, policy, dispatcher_type,
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
        moe_token_dispatcher_type=dispatcher_type,
    )
    runtime = NpuRuntimeWithCpuKernels()
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=runtime, default_policy=policy)
    dispatcher_cls = (
        dispatch_module.MoEAlltoAllTokenDispatcher if dispatcher_type == "alltoall"
        else dispatch_module.MoEFlexTokenDispatcher
    )
    dispatcher = dispatcher_cls(2, [0, 1], config, object())
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16)[:num_tokens].requires_grad_()
    probs = torch.tensor([[0., .5], [.25, 0.], [0., .75]])[:num_tokens].requires_grad_()
    routing_map = probs.detach().bool() if on_input else torch.ones_like(probs, dtype=torch.bool)
    indices = routing_map.nonzero(as_tuple=True)[1].reshape(num_tokens, config.moe_router_topk)
    with monkeypatch.context() as cached_indices:
        cached_indices.setattr(
            torch, "nonzero",
            lambda *args, **kwargs: pytest.fail("Dispatcher must reuse cached expert indices."),
        )
        packed, packed_probs = dispatcher.dispatch_preprocess(
            x, routing_map, probs, expert_indices=indices,
        )
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
    assert bool(calls) == (
        dispatcher_type == "alltoall" and policy == OpPolicy.AUTO and num_tokens > 0
    )


@pytest.mark.parametrize("num_tokens", [0, 3])
def test_alltoall_counts_compact_assignments_including_unused_experts(num_tokens):
    dispatcher = dispatch_module.MoEAlltoAllTokenDispatcher(
        5, list(range(5)), MoEConfig(num_moe_experts=5, moe_router_topk=2), object(),
    )
    indices = torch.tensor([[3, 0], [1, 2], [2, 0]])[:num_tokens]
    counts = dispatcher.preprocess(indices)
    torch.testing.assert_close(
        counts, torch.tensor([2, 1, 2, 1, 0] if num_tokens else [0, 0, 0, 0, 0]),
    )
