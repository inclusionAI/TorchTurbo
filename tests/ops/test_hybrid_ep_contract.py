"""Compact routing and explicit per-forward planning contracts."""

import importlib

import pytest
import torch
import torch.distributed as dist

from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture
def dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        yield get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)
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
def test_npu_compact_provider_respects_policy_and_explicit_selection(monkeypatch, policy):
    from torchturbo.ops import OpDispatcher, registry
    from torchturbo.ops.providers.npu.hybrid_ep import NpuHybridEPDispatch
    from torchturbo.runtime import TurboDevice
    from torchturbo.runtime.npu import NpuRuntime

    monkeypatch.setattr(NpuHybridEPDispatch, "available", lambda self: True)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry, default_policy=policy)
    expected = "torch_hybrid_ep_dispatch" if policy == OpPolicy.NATIVE else "npu_hybrid_ep_dispatch"
    selected = dispatcher._candidate_impl("hybrid_ep_dispatch", TurboDevice.NPU)
    assert selected.op_spec().impl_name == expected

    monkeypatch.setattr(NpuHybridEPDispatch, "available", lambda self: False)
    with pytest.raises(RuntimeError, match="npu_hybrid_ep_dispatch"):
        OpDispatcher(
            device_runtime=NpuRuntime(), registry=registry, default_policy=policy,
            overrides={"hybrid_ep_dispatch": "npu_hybrid_ep_dispatch"},
        )
