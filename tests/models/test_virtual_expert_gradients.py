"""Per-FC readiness, native parameter completion, and replica gradient transport."""

import copy
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.grouped_linear import grouped_linear
from torchturbo.models.modules.moe.token_dispatcher import _VirtualExpertHybridEPManager
from torchturbo.ops.providers.native.virtual_expert_weight_exchange import build_weight_exchange_context
from tests.models.test_virtual_expert_weights import _replica_routes, _virtual_config, cpu_dispatch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("backward_autocast", [False, True])
def test_autocast_linear_gradients_use_forward_operands(
    cpu_dispatch, dtype, frozen_input, backward_autocast,
):
    x = torch.tensor([[1.003, 0.125], [0.504, -0.251]], requires_grad=not frozen_input)
    weight = torch.tensor([
        [[0.503, 0.252], [0.127, 1.003]],
        [[1., 1.], [1., 1.]],
        [[0.251, 0.503], [1.003, 0.127]],
    ], requires_grad=True)
    counts = torch.tensor([1, 0, 1])
    reductions = []

    dy = torch.tensor([[0.503, 0.251], [-0.127, 1.003]], dtype=dtype)
    with torch.autocast("cpu", dtype=dtype):
        output = grouped_linear(x, counts, weight=weight, wgrad_store=reductions.append)
    backward_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    with torch.autocast("cpu", dtype=backward_dtype, enabled=backward_autocast):
        output.backward(dy)

    compute_x, compute_weight = x.detach().to(dtype), weight.detach().to(dtype)
    expected_output = torch.cat((compute_x[:1] @ compute_weight[0], compute_x[1:] @ compute_weight[2]))
    torch.testing.assert_close(output, expected_output)
    if not frozen_input:
        expected_dx = torch.cat((dy[:1] @ compute_weight[0].T, dy[1:] @ compute_weight[2].T))
        torch.testing.assert_close(x.grad, expected_dx.float())
    expected_dw = torch.zeros_like(weight)
    expected_dw[0] = compute_x[:1].float().T @ dy[:1].float()
    expected_dw[2] = compute_x[1:].float().T @ dy[1:].float()
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0], expected_dw)


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("external_counts", [False, True])
def test_grouped_linear_without_store_matches_matmul_backward(cpu_dispatch, count, train_weight, external_counts):
    x = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3).requires_grad_()
    weight = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).requires_grad_(train_weight)
    counts = torch.tensor([0, count])
    output = grouped_linear(x, counts, weight=weight, wgrad_counts=counts if external_counts else None)
    expected = x @ weight[1]
    torch.testing.assert_close(output, expected)
    gradients = (x, weight) if train_weight else (x,)
    actual = torch.autograd.grad(output.sum(), gradients)
    reference = torch.autograd.grad(expected.sum(), gradients)
    for grad, want in zip(actual, reference, strict=True):
        torch.testing.assert_close(grad, want)


def test_empty_external_wgrad_with_bf16_input_and_fp32_weight(cpu_dispatch):
    x = torch.empty(0, 3, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(2, 3, 2, requires_grad=True)
    stored = []
    y = grouped_linear(x, torch.zeros(2, dtype=torch.long), weight=weight, wgrad_store=stored.append)
    assert y.dtype == x.dtype
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.empty_like(x))
    torch.testing.assert_close(stored[0], torch.zeros_like(weight))


@pytest.mark.parametrize("routes", [[[0], [2]], [[0], [0]], [[2], [2]]])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_autocast_partitions_return_fp32_owner_gradients(
    monkeypatch, cpu_dispatch, routes, checkpointed,
):
    config = _virtual_config()
    config.hidden_size = config.moe_ffn_hidden_size = 1
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(1)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    if checkpointed:
        experts.forward = partial(checkpoint, experts.forward, use_reentrant=False)
    copies = torch.tensor([[0, -1]])
    plan = _replica_routes(torch.tensor(routes), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    x = torch.tensor([[1.003], [0.503]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = manager.wrap_layer_input(x)
        manager.setup_metadata(torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1))
        execution = manager._plan
        tokens = manager.dispatch(tokens)
        output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
        output = manager.finalize_layer_output(manager.combine(output))
        manager.release_routing_state()
    torch.testing.assert_close(output.float(), torch.tensor([[1.], [0.50390625]]))
    dy = torch.tensor([[1.], [-1.]])
    for _ in range(2):
        actual = torch.autograd.grad(output, (x, *experts.parameters()), dy, retain_graph=True)
        torch.testing.assert_close(actual[0], dy)
        for gradient in actual[1:]:
            torch.testing.assert_close(gradient, torch.tensor([[0.49609375], [0.]]))
        assert execution.grad_results == [None, None]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("routes", [[[0], [2]], [[0], [0]], [[2], [2]]])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_autocast_empty_partitions_preserve_swiglu_outputs_and_probability_gradients(
    monkeypatch, cpu_dispatch, dtype, routes, checkpointed,
):
    config = _virtual_config()
    config.hidden_size = config.moe_ffn_hidden_size = 1
    config.gated_linear_unit = True
    config.activation_func = torch.nn.functional.silu
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(1)
    reference = copy.deepcopy(experts)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    if checkpointed:
        experts.forward = partial(checkpoint, experts.forward, use_reentrant=False)
    copies = torch.tensor([[0, -1]])
    plan = _replica_routes(torch.tensor(routes), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    inputs = torch.ones(2, 1, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    probabilities = torch.full((2, 1), 0.50390625, requires_grad=True)
    ref_probabilities = probabilities.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=dtype):
        tokens = manager.wrap_layer_input(inputs)
        manager.setup_metadata(torch.zeros(2, 1, dtype=torch.int64), probabilities)
        tokens = manager.dispatch(tokens)
        output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
        output = manager.finalize_layer_output(manager.combine(output))
        manager.release_routing_state()
        expected, _ = reference(ref_inputs, torch.tensor([2, 0]), ref_probabilities.flatten())
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)
    torch.testing.assert_close(probabilities.grad, ref_probabilities.grad)


def test_bf16_partitions_preserve_fp32_weight_gradients(monkeypatch, cpu_dispatch):
    config = _virtual_config()
    config.hidden_size = config.moe_ffn_hidden_size = 1
    config.params_dtype = torch.bfloat16
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(1)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    copies = torch.tensor([[0, -1]])
    plan = _replica_routes(torch.tensor([[0], [0], [2], [2]]), copies)
    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", lambda *args, **kwargs: plan)
    reductions = []
    dispatch = cpu_dispatch._dispatch

    def record_gradients(name, *args, **kwargs):
        if name == "virtual_expert_grad_reduce":
            reductions.append((args[0].detach().clone(), args[1].detach().clone()))
            result = dispatch(name, *args, **kwargs)
            assert result.data_ptr() == args[0].data_ptr()
            return result
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", record_gradients)
    x = torch.ones(4, 1, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.tensor([0.625, -0.005615234375, -0.65625, 0.006256103515625], dtype=torch.bfloat16).view(4, 1)
    tokens = manager.wrap_layer_input(x)
    manager.setup_metadata(torch.zeros(4, 1, dtype=torch.int64), torch.ones(4, 1))
    execution = manager._plan
    tokens = manager.dispatch(tokens)
    output, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
    output = manager.finalize_layer_output(manager.combine(output))
    manager.release_routing_state()
    torch.testing.assert_close(output, x)
    expected = torch.tensor([[dy.float().sum().item()], [0]], dtype=torch.bfloat16)
    for _ in range(2):
        actual = torch.autograd.grad(output, (x, *experts.parameters()), dy, retain_graph=True)
        torch.testing.assert_close(actual[0], dy)
        for gradient in actual[1:]:
            torch.testing.assert_close(gradient, expected, atol=2e-3, rtol=3e-2)
        assert execution.grad_results == [None, None]
    assert manager.storage.grad_arena.dtype == torch.float32
    assert all(native.dtype == replica.dtype == torch.float32 for native, replica in reductions)


def test_single_fc_reduction_preserves_native_gradients_and_scratch_sections(cpu_dispatch):
    context = build_weight_exchange_context(torch.tensor([[1, 0]]), None)
    native = torch.arange(24).reshape(2, 4, 3).transpose(1, 2).float()
    replicas = torch.arange(24, 48).reshape(2, 4, 3).transpose(1, 2).float()
    scratch = torch.full((38,), -17.0)
    actual = cpu_dispatch.virtual_expert_grad_reduce(
        native, replicas, context, grad_buffer=scratch, arena_offset=7,
    )
    torch.testing.assert_close(actual, native + replicas.flip(0))
    torch.testing.assert_close(scratch[:7], torch.full((7,), -17.0))
    torch.testing.assert_close(scratch[31:], torch.full((7,), -17.0))


@pytest.mark.parametrize("copies", [[1, -1], [-1, -1]])
def test_single_fc_reuses_only_an_owned_gradient(cpu_dispatch, copies):
    context = build_weight_exchange_context(torch.tensor([copies]), None)
    count = sum(context.receive_splits)
    # The grouped wgrad result backs both disjoint views until native autograd takes ownership.
    grouped = torch.arange((2 + count) * 3, dtype=torch.float32).reshape(-1, 3)
    owner, replica = grouped[:2], grouped[2:]
    original = grouped.clone()
    expected = owner.clone()
    if count:
        expected[1] += replica[0]
    kwargs = {"grad_buffer": torch.empty_like(replica).reshape(-1)}
    public = cpu_dispatch.virtual_expert_grad_reduce(owner, replica, context, **kwargs)
    assert public.data_ptr() != owner.data_ptr()
    torch.testing.assert_close(grouped, original)
    torch.testing.assert_close(public, expected)
    owned = cpu_dispatch.virtual_expert_grad_reduce(owner, replica, context, allow_inplace=True, **kwargs)
    assert owned.data_ptr() == owner.data_ptr()
    torch.testing.assert_close(owned, expected)
    torch.testing.assert_close(replica, original[2:])


@pytest.mark.parametrize(("dtype", "large"), [(torch.float16, 2048), (torch.bfloat16, 256)])
@pytest.mark.parametrize("arena_dtype", [None, torch.float32])
def test_single_fc_accumulates_owner_and_peer_contributions_before_rounding(cpu_dispatch, dtype, large, arena_dtype):
    from dataclasses import replace

    context = build_weight_exchange_context(torch.tensor([[0, 1]]), None)
    context = replace(context, send_indices=torch.tensor([0, 0]))
    native = torch.tensor([large, 3], dtype=dtype)
    replicas = torch.tensor([1, -large], dtype=dtype)
    scratch = torch.empty(2, dtype=arena_dtype or dtype)
    actual = cpu_dispatch.virtual_expert_grad_reduce(native, replicas, context, grad_buffer=scratch)
    torch.testing.assert_close(actual, torch.tensor([1, 3], dtype=dtype))
    torch.testing.assert_close(native, torch.tensor([large, 3], dtype=dtype))


@pytest.mark.parametrize("frozen", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("mixed_native", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_fc2_reduction_precedes_fc1_backward_and_retained_gradients(
    monkeypatch, cpu_dispatch, frozen, frozen_input, mixed_native, checkpointed,
):
    if all(frozen) and frozen_input:
        pytest.skip("There is no differentiable output when every tensor is frozen")
    config = _virtual_config()
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.relu
    experts = GroupedMLP(2, False, config)
    with torch.no_grad():
        experts.weight1.copy_(torch.arange(24).reshape(8, 3) / 50 + 0.1)
        experts.weight2.copy_(torch.arange(24).reshape(6, 4) / 40 + 0.2)
    for weight, freeze in zip(experts.parameters(), frozen):
        weight.requires_grad_(not freeze)
    reference = copy.deepcopy(experts)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    manager.bind_experts(experts)
    if checkpointed:
        experts.forward = partial(checkpoint, experts.forward, use_reentrant=False)

    executions = []

    def plan(indices, *args, **kwargs):
        reverse = len(executions) % 2 == 0
        copies = torch.tensor([[1, 0]] if reverse else [[0, 1]])
        routes = 3 - indices if reverse else indices + 2
        if mixed_native:
            routes[0] = indices[0]
        return _replica_routes(routes, copies)

    monkeypatch.setattr(cpu_dispatch, "plan_virtual_expert_routes", plan)
    events = []
    dispatch = cpu_dispatch._dispatch

    def record_dispatch(name, *args, **kwargs):
        if name == "virtual_expert_grad_reduce":
            events.append("fc2_reduce" if args[0].shape[-1] == 4 else "fc1_reduce")
        return dispatch(name, *args, **kwargs)

    monkeypatch.setattr(cpu_dispatch, "_dispatch", record_dispatch)
    activate = experts.activation_func_with_probs

    def record_activation(*args):
        result = activate(*args)
        if result.requires_grad:
            result.register_hook(lambda grad: events.append("fc1_backward"))
        return result

    monkeypatch.setattr(experts, "activation_func_with_probs", record_activation)
    x = (torch.arange(8).reshape(2, 4).float() / 10 + 1).requires_grad_(not frozen_input)
    ref_x = x.detach().clone().requires_grad_(x.requires_grad)
    indices = torch.tensor([[1], [0]])
    outputs = []
    for _ in range(2):
        tokens = manager.wrap_layer_input(x)
        manager.setup_metadata(indices, torch.ones(2, 1))
        executions.append(manager._plan)
        tokens = manager.dispatch(tokens)
        result, _ = experts(tokens, manager.tokens_per_expert, manager.dispatched_probs)
        outputs.append(manager.finalize_layer_output(manager.combine(result)))
        manager.release_routing_state()
    w1 = reference.weight1.view(2, 4, 3)[indices[:, 0]]
    w2 = reference.weight2.view(2, 3, 4)[indices[:, 0]]
    expected = torch.bmm(torch.bmm(ref_x.unsqueeze(1), w1).relu(), w2).squeeze(1)
    for result in outputs:
        torch.testing.assert_close(result, expected)
    for targets in ("weights", "input", "all"):
        if targets == "weights":
            inputs, refs = list(experts.parameters()), list(reference.parameters())
        elif targets == "input":
            inputs, refs = [x], [ref_x]
        else:
            inputs, refs = [x, *experts.parameters()], [ref_x, *reference.parameters()]
        inputs = [value for value in inputs if value.requires_grad]
        refs = [value for value in refs if value.requires_grad]
        if not inputs:
            continue
        for active in ((0, 1), (0,), (1,), (0,)):
            events.clear()
            loss = sum(outputs[index].square().sum() for index in active)
            actual = torch.autograd.grad(loss, inputs, retain_graph=True)
            wanted = torch.autograd.grad(
                len(active) * expected.square().sum(), refs, retain_graph=True,
            )
            for grad, ref_grad in zip(actual, wanted):
                torch.testing.assert_close(grad, ref_grad)
            if not frozen[1]:
                assert "fc2_reduce" in events
                assert events.index("fc2_reduce") < events.index("fc1_backward")
            assert manager._plan is None
            for result in executions:
                assert result.grad_results == [None, None]
                assert not result.push_in_flight
