"""Compact routing and explicit per-forward planning contracts."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist

from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.ops.moe import VirtualExpertPlan, VirtualExpertPlannerWorkspace
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(params=[OpPolicy.NATIVE, OpPolicy.CUSTOM])
def dispatcher(tmp_path, monkeypatch, request):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        # Selecting a complete native op must also select its native substeps,
        # even when the process policy excludes native implementations.
        overrides = {
            "plan_virtual_expert_routes": "torch_plan_virtual_expert_routes",
            "hybrid_ep_dispatch": "torch_hybrid_ep_dispatch",
            "hybrid_ep_combine": "torch_hybrid_ep_combine",
        }
        yield get_or_create_ops(
            device_runtime=CpuRuntime(), default_policy=request.param,
            overrides=overrides if request.param == OpPolicy.CUSTOM else None,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("frozen_tokens", [False, True])
def test_compact_dispatch_matches_dense_outputs_and_gradients(dispatcher, empty, frozen_tokens):
    ids = torch.tensor([[3, 0], [1, 2], [2, 0]])[:0 if empty else 3]
    x = torch.randn(3, 8)[:ids.shape[0], ::2].requires_grad_(not frozen_tokens)
    weights = torch.randn(3, 4)[:ids.shape[0], ::2].requires_grad_()
    ref_x = x.detach().clone().requires_grad_(not frozen_tokens)
    ref_weights = weights.detach().clone().requires_grad_()
    dense = torch.zeros(ids.shape[0], 4).scatter(1, ids, ref_weights)
    mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, ids, True)

    compact = dispatcher.hybrid_ep_dispatch(x, ids, weights, dist.group.WORLD, 4, 2)
    reference = dispatcher.hybrid_ep_dispatch(ref_x, mask, dense, dist.group.WORLD, 4, 2)
    for actual, expected in zip(compact[:3], reference[:3], strict=True):
        torch.testing.assert_close(actual, expected)
    output = dispatcher.hybrid_ep_combine(compact[0] * compact[1][:, None], compact[3])
    expected = dispatcher.hybrid_ep_combine(reference[0] * reference[1][:, None], reference[3])
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    if not frozen_tokens:
        torch.testing.assert_close(x.grad, ref_x.grad)


@pytest.mark.parametrize("deterministic", [False, True])
def test_combine_gradients_and_saved_storage(dispatcher, deterministic):
    tokens = torch.randn(2, 16)
    ids = torch.tensor([[0, 1], [0, 1]])
    _, _, _, handle = dispatcher.hybrid_ep_dispatch(
        tokens, ids, torch.ones_like(ids, dtype=tokens.dtype), dist.group.WORLD, 2, 2,
    )
    experts = torch.randn(4, 16, requires_grad=True)
    saved_bytes = []

    def pack(tensor):
        saved_bytes.append(tensor.untyped_storage().nbytes())
        return tensor

    previous_mode = torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(deterministic)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = dispatcher.hybrid_ep_combine(experts, handle)
        if not deterministic:
            assert all(size <= ids.nbytes for size in saved_bytes)
        torch.testing.assert_close(output, experts[:2] + experts[2:])
        grad = torch.randn_like(output)
        output.backward(grad)
        torch.testing.assert_close(experts.grad, grad.repeat(2, 1))
    finally:
        torch.set_deterministic_debug_mode(previous_mode)


@pytest.mark.parametrize("dense_routes", [False, True])
def test_dispatch_rejects_routes_with_missing_source_rows(dispatcher, dense_routes):
    x = torch.randn(3, 2)
    ids = torch.tensor([[0], [1]])
    weights = torch.ones(2, 1)
    routes = ids
    if dense_routes:
        routes = torch.zeros(2, 4, dtype=torch.bool).scatter_(1, ids, True)
        weights = torch.zeros(2, 4).scatter_(1, ids, weights)

    with pytest.raises(ValueError, match="num_tokens"):
        dispatcher.hybrid_ep_dispatch(x, routes, weights, dist.group.WORLD, 4, 1)


def test_compact_dispatch_skips_negative_slots_without_dropping_other_routes(dispatcher):
    x = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
    ids = torch.tensor([[-1, 2], [-1, -1], [2, 0]])
    gates = torch.tensor([[8., 4.], [5., 6.], [0., 7.]], requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        packed, probs, counts, handle = dispatcher.hybrid_ep_dispatch(x, ids, gates, dist.group.WORLD, 3, 2)
        output = dispatcher.hybrid_ep_combine(packed * probs[:, None], handle)
    # Native combine can use the exact source rows already used to pack the activations.
    assert "aten::scatter_" not in {event.key for event in profile.key_averages()}
    torch.testing.assert_close(counts, torch.tensor([1, 0, 2]))
    torch.testing.assert_close(packed.flatten(), torch.tensor([3., 1., 3.]))
    torch.testing.assert_close(probs, torch.tensor([7., 4., 0.]))
    torch.testing.assert_close(output.flatten(), torch.tensor([4., 0., 21.]))
    output.sum().backward()
    torch.testing.assert_close(gates.grad, torch.tensor([[0., 1.], [0., 0.], [3., 3.]]))
    torch.testing.assert_close(x.grad.flatten(), torch.tensor([4., 0., 7.]))


def test_dense_cached_int32_ids_preserve_int64_collective_counts(dispatcher):
    ids = torch.tensor([[0], [1]], dtype=torch.int32)
    mask = torch.eye(2, dtype=torch.bool)
    _, _, counts, _ = dispatcher.hybrid_ep_dispatch(
        torch.ones(2, 3), mask, mask.float(), dist.group.WORLD, 2, 1, expert_indices=ids,
    )
    torch.testing.assert_close(counts, torch.tensor([1, 1], dtype=torch.int64))


def test_planner_returns_explicit_state_and_dispatch_accepts_cloned_routes(dispatcher):
    ids = torch.tensor([[3, 0], [1, 2]])
    workspace = VirtualExpertPlannerWorkspace(
        num_experts=4, device=ids.device, group=dist.group.WORLD,
        runtime=dispatcher.device_runtime, peer_access=False,
    )
    plan = dispatcher.plan_virtual_expert_routes(ids, workspace=workspace)
    assert not isinstance(plan, tuple), "Planning must expose a per-forward context."
    assert not plan.has_global_replicas
    assert plan.num_local_replicas == 0
    x = torch.randn(2, 4)
    weights = torch.randn(2, 2)
    outputs = []
    for _ in range(2):
        packed, probs, counts, handle = dispatcher.hybrid_ep_dispatch(
            x, plan.virtual_experts.clone(), weights, dist.group.WORLD, 8, 2,
            dispatch_context=plan.dispatch_context,
        )
        outputs.append(dispatcher.hybrid_ep_combine(packed * probs[:, None], handle))
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[0], x * weights.sum(1, keepdim=True))


def test_wait_publishes_local_replica_count_with_opaque_context_once():
    routes = torch.tensor([[0]])
    copies = torch.tensor([[-1], [0]])
    weight_context = object()
    materialized = []

    def finish():
        materialized.append(True)
        return VirtualExpertPlan(
            routes, copies, None, weight_context, True, num_local_replicas=0,
        )

    plan = VirtualExpertPlan(routes, copies, None, None, None, _materialize=finish)
    assert plan.num_local_replicas is None
    assert plan.wait().num_local_replicas == 0
    assert plan.has_global_replicas and plan.weight_context is weight_context
    assert plan.wait() is plan
    assert len(materialized) == 1


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE, OpPolicy.CUSTOM])
@pytest.mark.parametrize("operation", ["hybrid_ep_dispatch", "hybrid_ep_combine"])
def test_hybrid_ep_provider_respects_npu_policy(monkeypatch, policy, operation):
    from torchturbo.ops import OpDispatcher, registry
    from torchturbo.runtime import TurboDevice
    from torchturbo.runtime.npu import NpuRuntime

    backend = ModuleType("torch_npu")

    def unused_kernel(*args, **kwargs):
        pytest.fail("Provider selection must not execute an NPU kernel.")

    backend.npu_moe_token_permute = unused_kernel
    backend.npu_moe_token_unpermute = unused_kernel
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry, default_policy=policy)
    selected = dispatcher._candidate_impl(operation, TurboDevice.NPU)
    prefix = "torch" if policy == OpPolicy.NATIVE else "npu"
    assert selected.op_spec().impl_name == f"{prefix}_{operation}"
