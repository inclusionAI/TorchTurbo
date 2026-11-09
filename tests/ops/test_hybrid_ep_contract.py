"""Compact routing and explicit per-forward planning contracts."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist

from torchturbo.ops import OpPolicy, get_or_create_ops
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


def test_planner_returns_explicit_state_and_dispatch_accepts_cloned_routes(dispatcher):
    ids = torch.tensor([[3, 0], [1, 2]])
    plan = dispatcher.plan_virtual_expert_routes(ids, 4, dist.group.WORLD)
    assert not isinstance(plan, tuple), "Planning must expose a per-forward context."
    assert not plan.has_global_replicas
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


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE, OpPolicy.CUSTOM])
@pytest.mark.parametrize("operation", ["hybrid_ep_dispatch", "hybrid_ep_combine"])
def test_shared_hybrid_ep_provider_respects_npu_policy(monkeypatch, policy, operation):
    from torchturbo.ops import OpDispatcher, registry
    from torchturbo.runtime import TurboDevice
    from torchturbo.runtime.npu import NpuRuntime

    backend = ModuleType("torch_npu")

    def unused_kernel(*args, **kwargs):
        pytest.fail("Provider selection must not execute an NPU kernel.")

    backend.npu_moe_token_permute = unused_kernel
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry, default_policy=policy)
    if policy == OpPolicy.CUSTOM:
        with pytest.raises(RuntimeError, match=f"No implementation.*{operation}.*policy=custom"):
            dispatcher._candidate_impl(operation, TurboDevice.NPU)
    else:
        selected = dispatcher._candidate_impl(operation, TurboDevice.NPU)
        assert selected.op_spec().impl_name == f"torch_{operation}"
