"""Native HCCL weight exchange and prefetch parity under torchrun EP2 or EP8."""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.providers.native.virtual_expert_weight_exchange import build_weight_exchange_context
from torchturbo.ops.registry import registry
from torchturbo.runtime import is_npu_available
from torchturbo.runtime.npu import NpuRuntime

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.fixture(scope="module")
def weight_env(tmp_path_factory):
    import torch_npu  # noqa: F401

    runtime = NpuRuntime()
    runtime.initialize()
    runtime.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = not dist.is_initialized()
    if owns_group:
        if "RANK" in os.environ:
            dist.init_process_group("hccl", timeout=timedelta(minutes=5))
        else:
            path = tmp_path_factory.mktemp("weight") / "rendezvous"
            dist.init_process_group(
                "hccl", init_method=f"file://{path}", rank=0, world_size=1,
                timeout=timedelta(minutes=5),
            )
    dispatcher = OpDispatcher(
        device_runtime=runtime, registry=registry, default_policy=OpPolicy.NATIVE,
    )
    yield dispatcher, dist.group.WORLD
    if owns_group:
        dist.destroy_process_group()


def _copies(size, case, generation=0):
    copies = torch.full((size, 3), -1, dtype=torch.int64)
    if case == "ring":
        for destination in range(size):
            owner = (destination + 1 + generation) % size
            copies[destination, :2] = torch.tensor([owner * 3, owner * 3 + 2])
    elif case == "multiple":
        copies[:, :2] = torch.tensor([0, 2])
    elif case == "sparse" and size > 1:
        copies[1, 0] = generation % 3
    return copies


def _matrix(rank, expert, rows, columns, dtype):
    values = torch.arange(rows * columns).reshape(rows, columns) % 17
    return (values + 8 * rank + 3 * expert).to(dtype)


def _weights(rank, dtype, strided, trainable):
    weights = []
    for rows, columns, requires_grad in ((5, 7, trainable[0]), (7, 4, trainable[1])):
        value = torch.stack([_matrix(rank, expert, rows, columns, dtype) for expert in range(3)])
        value = value.to("npu")
        if strided:
            storage = torch.empty((3, rows * 2, columns * 2), device="npu", dtype=dtype)
            storage[:, ::2, ::2] = value
            value = storage[:, ::2, ::2]
            assert not value.is_contiguous()
        weights.append(value.detach().requires_grad_(requires_grad))
    return weights


def _assert_equal_on_all_ranks(actual, wanted, group):
    error = None
    try:
        assert actual is not None, "Expected a materialized gradient, received None"
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    except AssertionError as exception:
        error = str(exception)
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failures = [(rank, message) for rank, message in enumerate(errors) if message is not None]
    assert not failures, f"Numerical parity failed across EP ranks: {failures}"


def _execute(dispatcher, group, copies, dtype, strided, trainable, used, cached=False):
    rank = dist.get_rank(group)
    weights = _weights(rank, dtype, strided, trainable)
    hidden = torch.ones((2, 7), device="npu", requires_grad=True)
    probs = torch.ones((2, 1), device="npu", requires_grad=True)
    context = build_weight_exchange_context(copies, group)
    prefetched = None
    if cached:
        with torch.no_grad():
            prefetched = dispatcher.virtual_expert_weight_exchange(
                *weights, hidden, probs, context,
            )[:2]
        assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in prefetched)
    result = dispatcher.virtual_expert_weight_exchange(
        *weights, hidden, probs, context, prefetched=prefetched,
    )
    if not any(trainable):
        assert result[2] is hidden and result[3] is probs
    expected = []
    for rows, columns in ((5, 7), (7, 4)):
        matrices = [_matrix(int(expert) // 3, int(expert) % 3, rows, columns, dtype)
                    for expert in copies[rank] if expert >= 0]
        expected.append(torch.stack(matrices) if matrices else torch.empty((0, rows, columns), dtype=dtype))
    for actual, wanted in zip(result[:2], expected):
        _assert_equal_on_all_ranks(actual.cpu(), wanted, group)
        assert actual.is_contiguous()
    coefficient = torch.arange(result[0].shape[0], device="npu") + rank + 1
    loss = 2 * result[2].sum() + 3 * result[3].sum()
    for index in used:
        loss = loss + (result[index] * coefficient[:, None, None]).sum()
    loss.backward()
    _assert_equal_on_all_ranks(hidden.grad, torch.full_like(hidden, 2), group)
    _assert_equal_on_all_ranks(probs.grad, torch.full_like(probs, 3), group)
    for index, weight in enumerate(weights):
        if not weight.requires_grad:
            assert weight.grad is None
            continue
        wanted = torch.zeros_like(weight)
        if index in used:
            for destination, row in enumerate(copies.tolist()):
                for slot, expert in enumerate(row):
                    if expert >= 0 and expert // 3 == rank:
                        wanted[expert % 3] += destination + slot + 1
        _assert_equal_on_all_ranks(weight.grad, wanted, group)
    if prefetched is not None:
        assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in prefetched)
    return [tensor.detach().cpu() for tensor in result[:2]], [weight.grad for weight in weights]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("case", ["ring", "multiple", "sparse", "empty"])
def test_weight_values_and_owner_gradients(weight_env, dtype, strided, case, cached):
    dispatcher, group = weight_env
    copies = _copies(dist.get_world_size(group), case)
    _execute(dispatcher, group, copies, dtype, strided, (True, True), (0, 1), cached=cached)


@pytest.mark.parametrize("trainable,used", [
    ((True, True), (0,)), ((True, True), (1,)),
    ((False, True), (0, 1)), ((True, False), (0, 1)), ((False, False), (0, 1)),
])
def test_prefetched_frozen_and_unused_outputs(weight_env, trainable, used):
    dispatcher, group = weight_env
    copies = _copies(dist.get_world_size(group), "sparse")
    _execute(dispatcher, group, copies, torch.float32, True, trainable, used, cached=True)


def test_two_plans_before_backward_and_repeated_stream_reuse(weight_env):
    dispatcher, group = weight_env
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    weights = _weights(rank, torch.float32, True, (True, True))
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    retained = []
    for generation in range(16):
        graphs = []
        wanted_grads = [torch.zeros_like(weight) for weight in weights]
        for offset in range(2):
            plan_id = generation * 2 + offset
            copies = _copies(size, "ring", plan_id)
            context = build_weight_exchange_context(copies, group)
            stream = streams[plan_id % 2]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream), torch.no_grad():
                cache = dispatcher.virtual_expert_weight_exchange(
                    *weights, torch.zeros(1, device="npu"), torch.zeros(1, device="npu"),
                    context,
                )[:2]
            torch.npu.current_stream().wait_stream(stream)
            hidden = torch.ones(1, device="npu")
            result = dispatcher.virtual_expert_weight_exchange(
                *weights, hidden, hidden, context, prefetched=cache,
            )
            coefficient = plan_id + rank + 1
            graphs.append(sum(value.sum() * coefficient for value in result))
            retained.append((cache, copies))
            for destination, row in enumerate(copies.tolist()):
                for expert in row:
                    if expert >= 0 and expert // 3 == rank:
                        for gradient in wanted_grads:
                            gradient[expert % 3] += plan_id + destination + 1
        for loss in reversed(graphs):
            loss.backward()
        for weight, wanted in zip(weights, wanted_grads):
            _assert_equal_on_all_ranks(weight.grad, wanted, group)
            weight.grad = None
    for cache, copies in retained:
        for value, (rows, columns) in zip(cache, ((5, 7), (7, 4))):
            expected = torch.stack([_matrix(int(expert) // 3, int(expert) % 3, rows, columns, torch.float32)
                                    for expert in copies[rank] if expert >= 0])
            torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)


def test_noncontiguous_ep_subgroups_with_one_empty_group(weight_env):
    dispatcher, world = weight_env
    size = dist.get_world_size(world)
    if size < 4 or size % 2:
        pytest.skip("Requires an even WORLD size of at least four")
    rank = dist.get_rank(world)
    groups = [dist.new_group(list(range(parity, size, 2))) for parity in range(2)]
    group = groups[rank % 2]
    try:
        for active in range(2):
            copies = _copies(size // 2, "multiple" if rank % 2 == active else "empty")
            _execute(dispatcher, group, copies, torch.float32, True, (True, True), (0, 1))
            dist.barrier(world)
    finally:
        dist.destroy_process_group(group)


def test_prefetched_consumer_does_not_exchange_again(weight_env, monkeypatch):
    from torchturbo.ops.providers.native import virtual_expert_weight_exchange as native

    dispatcher, group = weight_env
    context = build_weight_exchange_context(_copies(dist.get_world_size(group), "multiple"), group)
    weights = _weights(dist.get_rank(group), torch.float32, False, (True, True))
    hidden = torch.ones(1, device="npu")
    with torch.no_grad():
        cached = dispatcher.virtual_expert_weight_exchange(*weights, hidden, hidden, context)[:2]

    def unexpected_exchange(*args, **kwargs):
        raise AssertionError("Cached consumption must not repeat forward weight transport")

    with monkeypatch.context() as patch:
        patch.setattr(native, "ep_all_to_all", unexpected_exchange)
        result = dispatcher.virtual_expert_weight_exchange(
            *weights, hidden, hidden, context, prefetched=cached,
        )
    assert result[2].requires_grad and result[3].requires_grad
    assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in cached)
    sum(tensor.sum() for tensor in result).backward()


def test_replica_only_loss_materializes_input_probability_zero_gradients(weight_env):
    dispatcher, group = weight_env
    context = build_weight_exchange_context(_copies(dist.get_world_size(group), "sparse"), group)
    weights = _weights(dist.get_rank(group), torch.float32, True, (True, True))
    hidden = torch.ones((2, 7), device="npu", requires_grad=True)
    probs = torch.ones((2, 1), device="npu", requires_grad=True)
    result = dispatcher.virtual_expert_weight_exchange(*weights, hidden, probs, context)
    (result[0].sum() + result[1].sum()).backward()
    _assert_equal_on_all_ranks(hidden.grad, torch.zeros_like(hidden), group)
    _assert_equal_on_all_ranks(probs.grad, torch.zeros_like(probs), group)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-s"]))
