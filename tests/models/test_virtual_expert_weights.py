"""CPU replica-plan decisions used by ordered EP weight communication."""

import copy
import gc
import importlib
import time
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.token_dispatcher import _VirtualExpertHybridEPManager
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.ops.moe import VirtualExpertPlan as RoutingPlan
from torchturbo.ops.moe import VirtualExpertPlannerWorkspace
from torchturbo.ops.providers.native.virtual_expert_weight_exchange import (
    build_weight_exchange_context,
)
from torchturbo.runtime.cpu import CpuRuntime


@pytest.mark.parametrize("rank", range(3))
def test_empty_replica_plan_is_global(monkeypatch, rank):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.full((3, 2), -1), None,
    )
    assert not plan.has_global_replicas
    assert plan.send_indices.numel() == sum(plan.receive_splits) == 0


@pytest.mark.parametrize("rank", range(3))
def test_idle_rank_retains_global_replica_participation(monkeypatch, rank):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.tensor([[-1, -1], [0, -1], [-1, -1]]), None,
    )
    assert plan.has_global_replicas
    if rank == 2:
        assert plan.send_indices.numel() == sum(plan.receive_splits) == 0


@pytest.mark.parametrize("rank,send_indices,send_splits,receive_splits", [
    (0, [1, 0], [0, 2, 0], [0, 0, 1]),
    (1, [], [0, 0, 0], [2, 0, 0]),
    (2, [1], [1, 0, 0], [0, 0, 0]),
])
def test_owners_send_replicas_in_destination_slot_order(
    monkeypatch, rank, send_indices, send_splits, receive_splits,
):
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    plan = build_weight_exchange_context(
        torch.tensor([[5, -1], [1, 0], [-1, -1]]), None,
    )
    assert plan.send_indices.tolist() == send_indices
    assert plan.send_splits == send_splits
    assert plan.receive_splits == receive_splits


@pytest.mark.parametrize("rank_copies", [[4, 2], [-1, 2]])
def test_rejects_replica_slots_out_of_owner_order(monkeypatch, rank_copies):
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    with pytest.raises(AssertionError, match="prefix ordered by owner"):
        build_weight_exchange_context(
            torch.tensor([rank_copies, [-1, -1], [-1, -1]]), None,
        )


def _run_prefetched_weight_order(rank, init_file):
    import os
    import sys
    from datetime import timedelta

    from tests.models.test_moon_ep_distributed import _check_manual_weight_order

    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=3,
        timeout=timedelta(seconds=20),
    )
    try:
        get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)
        for copies in (
            None,
            torch.tensor([[-1, -1], [1, 0], [-1, -1]]),
            torch.full((3, 2), -1),
        ):
            for frozen in ((False, False), (True, False), (False, True), (True, True)):
                for unused in (False, True):
                    _check_manual_weight_order(
                        dist.group.WORLD, prefetched=True, copies=copies,
                        frozen=frozen, unused=unused,
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_native_prefetch_consumes_graph_free_replicas_without_forward_transport(tmp_path):
    # Bound mismatched collectives, with a third rank that has neither sends
    # nor receives in the sparse replica plan.
    context = mp.spawn(
        _run_prefetched_weight_order, args=(str(tmp_path / "prefetched-weight-order"),),
        nprocs=3, join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Prefetched weight workers did not complete within 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


@pytest.fixture
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    return get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def _replica_routes(routes, copies):
    context = build_weight_exchange_context(copies, None)
    return RoutingPlan(
        routes, copies, None, context, context.has_global_replicas,
        num_local_replicas=int((copies[0] >= 0).sum()),
    )


def _virtual_config():
    return MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=2,
        moe_router_topk=1, moe_token_dispatcher_type="flex",
        moe_virtual_expert_load_balance=True,
    )


@pytest.mark.parametrize("track_grad", [False, True])
def test_prefetched_weight_consumer_does_not_repack(cpu_dispatch, track_grad):
    context = build_weight_exchange_context(torch.tensor([[1, 0]]), None)
    weights = [torch.randn(2, 3, width, requires_grad=True) for width in (4, 5)]
    with torch.no_grad():
        prefetched = cpu_dispatch.virtual_expert_weight_exchange(
            *weights, None, None, context,
        )[:2]
    with torch.set_grad_enabled(track_grad), torch.profiler.profile() as profile:
        replicas = cpu_dispatch.virtual_expert_weight_exchange(
            *weights, None, None, context, prefetched=prefetched,
        )[:2]
    events = {event.key for event in profile.key_averages()}
    assert not events.intersection({"aten::index_select", "aten::cat"})
    for cached, replica in zip(prefetched, replicas):
        assert cached.data_ptr() == replica.data_ptr()
    if track_grad:
        (replicas[0][0].sum() + replicas[1][1].sum()).backward()
        for i, weight in enumerate(weights):
            expected = torch.zeros_like(weight)
            expected[1 - i] = 1
            torch.testing.assert_close(weight.grad, expected)


def test_layer_converts_replica_storage_after_inference(monkeypatch, cpu_dispatch):
    copies = torch.tensor([[0, 1]])

    def replicate(indices, *args, **kwargs):
        return _replica_routes(indices + 2, copies)

    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", replicate)
    layer = MoELayer(
        _virtual_config(), None, submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.inference_mode():
        for parameter in layer.parameters():
            parameter.fill_(0.25)
        layer(torch.ones(1, 2, 4))
    hook_count = len(layer.experts._forward_pre_hooks)
    reference = MoELayer(
        _virtual_config(), None, submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    ).bfloat16()
    reference.load_state_dict(layer.state_dict())
    layer.bfloat16()
    inputs = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    actual = layer(inputs)[0]
    expected = reference(inputs)[0]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for param, ref in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref.grad)
    assert len(layer.experts._forward_pre_hooks) == hook_count


@pytest.mark.parametrize("backward_order", [(1, 0), (0, 1), None])
def test_layers_share_replica_storage_and_restore_weights_for_backward(
    monkeypatch, cpu_dispatch, backward_order,
):
    config = _virtual_config()
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu

    def plan_routes(indices, *args, **kwargs):
        copies = torch.tensor([[1, 0]])
        return _replica_routes(3 - indices, copies)

    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", plan_routes)
    pushes = []
    dispatch = cpu_dispatch._dispatch

    def count_pushes(name, *args, **kwargs):
        if name == "virtual_expert_weight_exchange" and kwargs.get("prefetched") is None:
            pushes.append(args[0].data_ptr())
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", count_pushes)
    managers, references, inputs, reference_inputs, outputs, expected, executions = (
        [], [], [], [], [], [], [],
    )
    indices = torch.tensor([[1], [0]])
    for index in range(2):
        experts = GroupedMLP(2, False, config)
        with torch.no_grad():
            experts.weight1.copy_(torch.arange(24).reshape(8, 3) / 50 + 0.1 + index)
            experts.weight2.copy_(torch.arange(24).reshape(6, 4) / 40 + 0.2 + index)
        reference = copy.deepcopy(experts)
        manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
        manager.bind_experts(experts)
        x = (torch.arange(8).reshape(2, 4).float() / 10 + 1).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        tokens = manager.wrap_layer_input(x)
        manager.setup_metadata(indices, torch.ones(2, 1))
        execution = manager._plan
        tokens = manager.dispatch(tokens)
        result, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
        result = manager.finalize_layer_output(manager.combine(result))
        manager.release_routing_state()
        w1 = reference.weight1.view(2, 4, 3)[indices[:, 0]]
        w2 = reference.weight2.view(2, 3, 4)[indices[:, 0]]
        intermediate = torch.bmm(ref_x.unsqueeze(1), w1).relu()
        wanted = torch.bmm(intermediate, w2).squeeze(1)
        torch.testing.assert_close(result, wanted)
        managers.append(manager)
        references.append(reference)
        inputs.append(x)
        reference_inputs.append(ref_x)
        outputs.append(result)
        expected.append(wanted)
        executions.append(execution)

    # Two live autograd graphs share the same physical slots. Layer two has
    # overwritten layer one's weights, which its backward must restore.
    assert executions[0].slot_weights[0].data_ptr() == executions[1].slot_weights[0].data_ptr()
    assert executions[0].slot_weights[1].data_ptr() == executions[1].slot_weights[1].data_ptr()
    assert managers[0].storage is managers[1].storage
    assert len(pushes) == 2
    if backward_order is None:
        sum(value.square().sum() for value in outputs).backward()
    else:
        for index in backward_order:
            outputs[index].square().sum().backward()
    assert len(pushes) == 4
    for value in expected:
        value.square().sum().backward()
    for manager, reference, x, ref_x in zip(managers, references, inputs, reference_inputs):
        torch.testing.assert_close(x.grad, ref_x.grad)
        for actual, wanted in zip(manager._experts.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad)


def test_planning_prefetches_before_dispatch_and_output_owns_backward(monkeypatch, cpu_dispatch):
    config = _virtual_config()
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        experts.weight1.fill_(0.1)
        experts.weight2.fill_(0.3)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    copies = torch.tensor([[1, 0]])
    plan = _replica_routes(torch.tensor([[2], [3]]), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    x = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], requires_grad=True)
    wrapped = manager.wrap_layer_input(x)
    manager.setup_metadata(torch.tensor([[1], [0]]), torch.ones(2, 1))
    execution = manager._plan
    assert execution is not None and execution.slot_weights is not None
    assert manager.handle is None
    cache = execution.slot_weights
    tokens = manager.dispatch(wrapped)
    output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
    output = manager.combine(output)
    assert manager._plan is execution
    output = manager.finalize_layer_output(output)
    manager.release_routing_state()
    assert manager._plan is None
    torch.testing.assert_close(output, torch.full((2, 4), 0.9))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.36))
    assert all(parameter.grad is not None for parameter in experts.parameters())
    assert execution.slot_weights is cache
    assert manager._plan is None


def test_layer_runtime_resources_are_reused_across_token_counts(cpu_dispatch):
    layer = MoELayer(
        _virtual_config(), object(),
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.2)
    manager = layer.token_dispatcher._comm_manager
    first = layer(torch.ones(1, 2, 4, requires_grad=True))[0]
    planner = manager.planner
    assert planner is not None
    hooks = len(layer.experts._forward_pre_hooks)
    second = layer(torch.ones(1, 3, 4, requires_grad=True))[0]
    assert manager.planner is planner
    assert len(layer.experts._forward_pre_hooks) == hooks
    (first.sum() + second.sum()).backward()
    assert manager._plan is None


def test_backward_retains_runtime_after_layer_owner_is_released(cpu_dispatch):
    layer = MoELayer(
        _virtual_config(), object(),
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.2)
    x = torch.ones(1, 2, 4, requires_grad=True)
    output = layer(x)[0]
    del layer
    gc.collect()
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _check_finalize_rebuild(config, group, device):
    from torchturbo.models.modules.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer

    def make_layer():
        layer = MoELayer(
            config, group,
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        ).to(device)
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.fill_(0.2)
        inputs = torch.ones(1, 2, config.hidden_size, device=device, dtype=config.params_dtype, requires_grad=True)
        layer(inputs)[0].sum().backward()
        return layer

    layers = [make_layer(), make_layer()]
    managers = [layer.token_dispatcher._comm_manager for layer in layers]
    storage = managers[0].storage
    shared = managers[0]._shared_runtime
    planner = shared.planner
    slots = storage.slot_weights
    assert managers[1].storage is storage
    assert managers[1]._shared_runtime is shared
    expected_parameters = [p.detach().clone() for p in layers[0].parameters()]
    VirtualExpertLoadBalancer.finalize()
    VirtualExpertLoadBalancer.finalize()
    assert storage.weight_arena is None and storage.grad_arena is None
    assert all(slot.numel() == 0 for slot in slots)
    assert planner.gathered_counts is None
    assert shared.planner is None
    assert not VirtualExpertLoadBalancer.storages and not VirtualExpertLoadBalancer.runtimes
    for parameter, expected in zip(layers[0].parameters(), expected_parameters):
        torch.testing.assert_close(parameter, expected)
    replacement = make_layer()
    assert replacement.token_dispatcher._comm_manager.storage is not storage
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in replacement.parameters())
    VirtualExpertLoadBalancer.finalize()


def test_finalize_releases_shared_resources_before_rebuilding_layers(cpu_dispatch):
    _check_finalize_rebuild(_virtual_config(), None, torch.device("cpu"))


def test_storage_cleans_partial_allocation_on_failure(cpu_dispatch, monkeypatch):
    from torchturbo.models.modules.moe.virtual_expert_load_balancer import (
        _VirtualExpertConfig, _VirtualExpertStorage,
    )

    allocate = torch.empty
    calls = 0

    def fail_gradient_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("gradient allocation failed")
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_gradient_allocation)
    config = _VirtualExpertConfig(
        None, torch.device("cpu"), 2, (torch.Size((4, 3)), torch.Size((3, 4))),
        torch.float32, torch.float32,
    )
    storage = object.__new__(_VirtualExpertStorage)
    with pytest.raises(RuntimeError, match="gradient allocation failed"):
        storage.__init__(config)
    assert storage.weight_arena is None and storage.grad_arena is None
    assert storage.slot_weights == ()


def test_inference_workspace_can_run_training_planner(monkeypatch, cpu_dispatch):
    ep_group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    def gather(output, source, group):
        assert group is ep_group
        assert output.data_ptr() == workspace.gathered_counts.data_ptr()
        output.view(2, -1).copy_(source.expand(2, -1))

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    with torch.inference_mode():
        workspace = VirtualExpertPlannerWorkspace(
            num_experts=4, device=torch.device("cpu"), group=ep_group,
        )
        cpu_dispatch.plan_virtual_expert_routes(
            torch.zeros((3, 1), dtype=torch.int64), workspace=workspace,
        )
    plan = cpu_dispatch.plan_virtual_expert_routes(
        torch.full((5, 1), 3, dtype=torch.int64), workspace=workspace,
    )
    assert plan.wait() is plan
    assert plan.has_global_replicas
    assert plan.dispatch_context.input_splits == [5, 0]


def test_forward_captured_before_binding_consumes_replica_weights(monkeypatch, cpu_dispatch):
    config = _virtual_config()
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    forward = experts.forward
    with torch.no_grad():
        experts.weight1[:4].fill_(0.1)
        experts.weight1[4:].fill_(0.2)
        experts.weight2[:3].fill_(0.3)
        experts.weight2[3:].fill_(0.4)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    copies = torch.tensor([[1, 0]])
    plan = _replica_routes(torch.tensor([[2], [3]]), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    tokens = manager.wrap_layer_input(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]))
    manager.setup_metadata(torch.tensor([[1], [0]]), torch.ones(2, 1))
    tokens = manager.dispatch(tokens)
    try:
        output, _ = forward(tokens, manager.tokens_per_expert, manager.dispatched_probs)
        torch.testing.assert_close(output, torch.tensor([[2.4] * 4, [0.9] * 4]))
    finally:
        manager.release_routing_state()


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "compiled"])
def test_binding_uses_runtime_slots_without_registering_extra_parameters(
    monkeypatch, cpu_dispatch, compile_forward,
):
    config = _virtual_config()
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    native1, native2 = experts.native_weights()
    with torch.no_grad():
        native1[0].fill_(0.1)
        native1[1].fill_(0.2)
        native2[0].fill_(0.3)
        native2[1].fill_(0.4)
    owned = dict(experts.named_parameters())
    reference = copy.deepcopy(experts)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    routes = _replica_routes(torch.tensor([[0], [2]]), torch.tensor([[1, -1]]))
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: routes)
    inputs = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], requires_grad=True)
    probs = torch.ones(2, 1, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_probs = probs.detach().clone().requires_grad_()
    tokens = manager.wrap_layer_input(inputs)
    manager.setup_metadata(torch.tensor([[0], [1]]), probs)
    plan = manager._plan
    tokens = manager.dispatch(tokens)
    forward = torch.compile(experts.forward, backend="eager") if compile_forward else experts
    result, _ = forward(tokens, manager.tokens_per_expert, manager.dispatched_probs)
    result = manager.finalize_layer_output(manager.combine(result))
    manager.release_routing_state()

    expected, _ = reference(ref_inputs, torch.tensor([1, 1]), ref_probs.reshape(-1))
    torch.testing.assert_close(result, torch.tensor([[0.9] * 4, [2.4] * 4]))
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    assert all(slot.grad is None and not slot.requires_grad for slot in plan.slot_weights)
    assert dict(experts.named_parameters()).keys() == experts.state_dict().keys() == owned.keys()
    for name, parameter in experts.named_parameters():
        assert parameter is owned[name]
        torch.testing.assert_close(parameter, reference.get_parameter(name))
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad)


def test_bound_experts_follow_deepcopy_without_duplicate_parameters(cpu_dispatch):
    original = MoELayer(
        _virtual_config(), object(),
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    copied = copy.deepcopy(original)
    manager = copied.token_dispatcher._comm_manager
    assert manager._experts is copied.experts
    assert manager._experts is not original.experts
    assert dict(copied.named_parameters()).keys() == dict(original.named_parameters()).keys()
    assert copied.state_dict().keys() == original.state_dict().keys()
    with torch.no_grad():
        original.experts.weight1.fill_(2)
        copied.experts.weight1.fill_(7)
    assert torch.all(manager._experts.weight1 == 7)
    assert torch.all(original.experts.weight1 == 2)


class _TracingStream:
    def __init__(self, runtime, name):
        self.runtime = runtime
        self.name = name

    def wait_stream(self, stream):
        assert self is not stream
        self.runtime.trace.append("source_ready" if self.name == "weight" else f"{self.name}_source_ready")

    def wait_event(self, event):
        assert event.stream is not None, "Cannot wait on an event that was never recorded"
        self.runtime.trace.append({"weight": "wait", "grad": "grad_wait"}.get(event.stream, "plan_wait"))
        self.runtime.waited.append(event)


class _TracingEvent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.stream = None

    def record(self, stream):
        self.stream = stream.name
        self.runtime.trace.append({"weight": "ready", "grad": "grad_ready"}.get(stream.name, "plan_ready"))


class _TracingRuntime(CpuRuntime):
    def __init__(self):
        self.trace = []
        self.waited = []
        self.recorded = []
        self.created_streams = []
        self.active = _TracingStream(self, "main")

    @property
    def supports_async_streams(self):
        return True

    def current_stream(self):
        return self.active

    def new_stream(self, priority=0):
        stream = _TracingStream(self, ("planner", "weight", "weight", "grad")[len(self.created_streams) % 4])
        self.created_streams.append(stream)
        return stream

    def new_event(self):
        return _TracingEvent(self)

    @contextmanager
    def stream(self, stream=None):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous

    def record_stream(self, tensor, stream):
        self.recorded.append((tensor.data_ptr(), stream.name))


def test_compatible_layers_share_planner_and_streams(monkeypatch, cpu_dispatch):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    config = _virtual_config()
    group = object()
    managers = []
    for ep_group in (group, group, object()):
        manager = _VirtualExpertHybridEPManager(ep_group, 2, 2, config)
        manager.bind_experts(GroupedMLP(2, False, config))
        manager.wrap_layer_input(torch.ones(2, 4))
        managers.append(manager)
    first, second, other_group = managers
    assert first.planner is second.planner
    assert first.planner_stream is second.planner_stream
    assert first.weight_streams is second.weight_streams
    assert first.grad_stream is second.grad_stream
    assert first.planner is not other_group.planner
    assert first.planner_stream is not other_group.planner_stream


def test_weight_side_stream_avoids_the_current_weight_stream(monkeypatch, cpu_dispatch):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, _virtual_config())
    manager.wrap_layer_input(torch.ones(2, 4))
    first, second = manager.weight_streams
    for caller, expected in ((first, second), (second, first)):
        with runtime.stream(caller):
            with manager._on_side_stream(manager.weight_streams, None) as selected:
                assert selected is expected
                assert runtime.current_stream() is expected


def test_replay_joins_pending_push_before_replacing_execution(monkeypatch, cpu_dispatch):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    config = _virtual_config()
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(GroupedMLP(2, False, config))
    module = importlib.import_module("torchturbo.models.modules.moe.virtual_expert_load_balancer")
    copies = torch.tensor([[1, 0]])
    plan = _replica_routes(torch.tensor([[2], [3]]), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    inputs = torch.ones(2, 4)
    manager.wrap_layer_input(inputs)
    assert not runtime.waited
    manager.setup_metadata(torch.tensor([[1], [0]]), torch.ones(2, 1))
    manager._wait_weight_push()
    manager._start_weight_push(module.WeightDirection.BACKWARD)
    old_execution = manager._plan
    old_ready = old_execution.prefetch_done
    runtime.waited.clear()
    wait_event = runtime.active.wait_event

    def observe_wait(event):
        assert manager._plan is old_execution
        wait_event(event)

    monkeypatch.setattr(runtime.active, "wait_event", observe_wait)
    manager.wrap_layer_input(inputs)
    assert runtime.waited == [old_ready]
    assert not old_execution.push_in_flight
    assert manager._plan is not old_execution
    manager.release_routing_state()
    runtime.waited.clear()
    manager.wrap_layer_input(inputs)
    assert not runtime.waited, "An ordinary new forward must not add a replay wait"
    manager.release_routing_state()


def test_deferred_plan_push_precedes_metadata_wait_at_dispatch(monkeypatch, cpu_dispatch):
    config = _virtual_config()
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    experts = GroupedMLP(2, False, config)
    manager.bind_experts(experts)
    copies = torch.tensor([[1, -1]])
    routes = torch.tensor([[2], [2]])
    trace = []

    def materialize():
        trace.append("metadata_wait")
        return RoutingPlan(routes, copies, None, object(), True, num_local_replicas=1)

    plan = RoutingPlan(
        routes, copies, None, None, None, _materialize=materialize,
    )
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    dispatch = cpu_dispatch._dispatch

    def device_push(name, *args, **kwargs):
        if name == "virtual_expert_weight_exchange":
            assert args[4] is plan
            assert not torch.is_grad_enabled()
            trace.append("push")
            for slots in kwargs["out"]:
                slots[0].zero_()
            return *kwargs["out"], args[2], args[3]
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", device_push)
    tokens = manager.wrap_layer_input(torch.ones(2, 4))
    manager.setup_metadata(torch.tensor([[1], [1]]), torch.ones(2, 1))
    assert trace == ["push"]
    assert plan.experts_to_copy is copies
    tokens = manager.dispatch(tokens)
    assert trace == ["push", "metadata_wait"]
    assert plan.num_local_replicas == 1
    output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
    torch.testing.assert_close(output, torch.zeros_like(tokens))
    manager.release_routing_state()


@pytest.mark.parametrize("failure", ["router", "plan", "experts", "combine"])
def test_failed_forward_releases_execution_and_can_retry(monkeypatch, cpu_dispatch, failure):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    layer = MoELayer(
        _virtual_config(), object(),
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.2)

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    manager = layer.token_dispatcher._comm_manager
    module = importlib.import_module("torchturbo.models.modules.moe.virtual_expert_load_balancer")
    targets = {
        "router": (layer.router, "forward"),
        "plan": (module, "plan_virtual_expert_routes"),
        "experts": (layer.experts, "forward"),
        "combine": (manager, "combine"),
    }
    with monkeypatch.context() as broken:
        broken.setattr(*targets[failure], fail)
        with pytest.raises(RuntimeError, match="injected forward failure"):
            layer(torch.ones(1, 2, 4, requires_grad=True))
    assert manager._plan is None
    assert manager.token_probs is None and manager.handle is None
    output = layer(torch.ones(1, 3, 4, requires_grad=True))[0]
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert manager._plan is None


@pytest.mark.parametrize("consume", [False, True])
def test_weight_launch_dispatch_and_consume_order_preserves_replay_plan(
    monkeypatch, cpu_dispatch, consume,
):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    dispatch = cpu_dispatch._dispatch

    def traced_dispatch(name, *args, **kwargs):
        if name == "virtual_expert_weight_exchange":
            assert kwargs.get("prefetched") is None, "Expert compute must consume the prefetched slots directly."
            runtime.trace.append("push")
        elif name == "hybrid_ep_dispatch":
            runtime.trace.append("dispatch")
        elif name == "grouped_matmul":
            runtime.trace.append("gemm")
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", traced_dispatch)
    config = _virtual_config()
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(0.2)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    execution_plans = []
    for _ in range(2):
        runtime.trace.clear()
        # Self replicas keep real native transport and expert math on one CPU
        # rank; only stream/event primitives are simulated.
        copies = torch.tensor([[1, 0]])
        plan = _replica_routes(torch.tensor([[2], [3]]), copies)
        monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
        tokens = manager.wrap_layer_input(torch.ones(2, 4))
        manager.setup_metadata(torch.tensor([[1], [0]]), torch.ones(2, 1))
        execution = manager._plan
        cache, ready = execution.slot_weights, manager.prefetch_done
        assert cache is not None and ready is not None
        tokens = manager.dispatch(tokens)
        assert runtime.trace.index("source_ready") < runtime.trace.index("push")
        assert runtime.trace.index("push") < runtime.trace.index("ready") < runtime.trace.index("dispatch")
        assert "wait" not in runtime.trace
        assert (experts.weight1.data_ptr(), "weight") in runtime.recorded
        assert (experts.weight2.data_ptr(), "weight") in runtime.recorded
        if consume:
            output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
            assert torch.isfinite(output).all()
            assert runtime.trace.index("dispatch") < runtime.trace.index("wait")
            assert runtime.trace.index("wait") < runtime.trace.index("gemm")
            assert all((tensor.data_ptr(), "main") in runtime.recorded for tensor in cache)
        manager.release_routing_state()
        assert ready in runtime.waited
        assert manager.prefetch_done is None
        assert manager._plan is None
        assert execution.routing is plan
        assert execution.routing.experts_to_copy is copies
        assert execution.slot_weights is cache and execution.prefetch_done is ready
        assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in cache)
        metadata = manager._plan
        manager.release_routing_state()
        assert manager._plan is metadata
        assert execution.slot_weights is cache and execution.prefetch_done is ready
        execution_plans.append(execution)
    assert len(runtime.created_streams) == 4
    assert execution_plans[0].prefetch_done is not execution_plans[1].prefetch_done
    assert execution_plans[0].slot_weights is not execution_plans[1].slot_weights


def test_failed_weight_launch_orders_cleanup_on_its_prefetch_event(monkeypatch, cpu_dispatch):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    dispatch = cpu_dispatch._dispatch

    def fail_after_push(name, *args, **kwargs):
        result = dispatch(name, *args, **kwargs)
        if name == "virtual_expert_weight_exchange":
            runtime.trace.append("push")
            raise RuntimeError("weight launch failed after submission")
        return result

    monkeypatch.setattr(cpu_dispatch, "_dispatch", fail_after_push)
    config = _virtual_config()
    experts = GroupedMLP(2, False, config)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    copies = torch.tensor([[1, 0]])
    plan = _replica_routes(torch.tensor([[2], [3]]), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    manager.wrap_layer_input(torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="failed after submission"):
        manager.setup_metadata(torch.tensor([[1], [0]]), torch.ones(2, 1))
    execution = manager._plan
    ready = manager.prefetch_done
    assert ready is not None and execution.slot_weights is None
    assert runtime.trace[-3:] == ["plan_wait", "push", "ready"]
    manager.release_routing_state()
    assert "wait" in runtime.trace and ready in runtime.waited
    assert manager.prefetch_done is None
    assert manager._plan is None and execution.routing is plan
    assert execution.prefetch_done is ready


@pytest.mark.parametrize("shared_kind", ["shared_experts", "gated_shared_expert"])
def test_virtual_expert_layer_runs_shared_experts_before_joining_route_plan(
    monkeypatch, cpu_dispatch, shared_kind,
):
    runtime = _TracingRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    layer = MoELayer(
        _virtual_config(), object(),
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    if shared_kind == "shared_experts":
        layer.shared_experts = torch.nn.Linear(4, 4, bias=False)
        shared_module = layer.shared_experts
    else:
        layer.shared_expert = torch.nn.Linear(4, 4, bias=False)
        layer.shared_expert_gate = torch.nn.Linear(4, 1, bias=False)
        shared_module = layer.shared_expert
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters(), start=1):
            parameter.fill_(0.1 * index)
    copies = torch.tensor([[1, 0]])

    def materialize():
        assert runtime.current_stream() is runtime.active
        runtime.trace.append("metadata_wait")
        return _replica_routes(torch.tensor([[2], [3]]), copies)

    monkeypatch.setattr(
        cpu_dispatch, "plan_virtual_expert_routes",
        lambda *args, **kwargs: (
            runtime.trace.append(f"plan:{runtime.current_stream().name}")
            or RoutingPlan(
                torch.tensor([[2], [3]]), copies, None, None, None,
                _materialize=materialize,
            )
        ),
    )
    dispatch = cpu_dispatch._dispatch

    def traced_dispatch(name, *args, **kwargs):
        if name == "virtual_expert_weight_exchange" and kwargs.get("prefetched") is None:
            runtime.trace.append(f"push:{runtime.current_stream().name}")
            return *kwargs["out"], args[2], args[3]
        if name == "hybrid_ep_dispatch":
            runtime.trace.append(f"dispatch:{runtime.current_stream().name}")
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", traced_dispatch)
    shared_module.register_forward_pre_hook(
        lambda *_: runtime.trace.append(f"shared:{runtime.current_stream().name}")
    )
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], requires_grad=True)
    output, _ = layer(inputs)
    assert (
        runtime.trace.index("plan:planner")
        < runtime.trace.index("plan_ready")
        < runtime.trace.index("push:weight")
        < runtime.trace.index("shared:main")
        < runtime.trace.index("metadata_wait")
        < runtime.trace.index("dispatch:main")
    )
    output.sum().backward()
    assert shared_module.weight.grad is not None and torch.isfinite(shared_module.weight.grad).all()
