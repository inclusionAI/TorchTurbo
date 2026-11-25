"""Virtual expert output and gradient parity through real EP and FSDP collectives."""

import importlib
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime import DeviceRuntime
import torchturbo.runtime.runtime as runtime_module


def _check_moon(
    ep_group, *, empty="none", boundary="none", reentrant=False,
    freeze_inputs=False, freeze_experts=False, on_input=False, zero_probs=False,
    fsdp_mesh=None, mixed_precision=False, cast_inputs=False, route_mode="original",
    runtime: DeviceRuntime | None = None, backward_together=False, inference_warmup=False,
    padding="none",
):
    device = torch.device("cpu") if runtime is None else runtime.device
    ep_size = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)
    torch.manual_seed(172 + rank)
    config = MoEConfig(
        hidden_size=8, moe_ffn_hidden_size=4, num_moe_experts=ep_size * 2,
        expert_model_parallel_size=ep_size, moe_router_topk=1 if on_input else 2,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32", moe_apply_probs_on_input=on_input,
        moe_router_score_function="sigmoid" if on_input else "softmax",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = MoELayer(config, ep_group, submodules=submodules).to(device)
    candidate = MoELayer(
        replace(
            config,
            moe_token_dispatcher_type="flex",
            moe_virtual_expert_load_balance=True,
        ),
        ep_group,
        submodules=submodules,
    ).to(device)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
        # Positive inputs select only rank zero's experts; negative inputs move
        # the hotspot to another owner, changing placement between forwards.
        reference.router.weight.zero_()
        for expert in range(config.moe_router_topk):
            reference.router.weight[expert].fill_(0.2 / (expert + 1))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for name, value in candidate.state_dict().items():
        assert value.shape == reference.state_dict()[name].shape
    execution_plans = []
    expert_forward = candidate.experts.forward

    def observe_expert_plan(*args):
        execution = candidate.token_dispatcher._comm_manager._plan
        execution_plans.append(execution)
        if execution.routing.has_global_replicas:
            assert execution.slot_weights is not None
            assert all(
                tensor.grad_fn is None and not tensor.requires_grad
                for tensor in execution.slot_weights
            )
        return expert_forward(*args)

    # Install before the checkpoint wrapper so its captured function also
    # observes expert-only replay, which directly invokes that function.
    candidate.experts.forward = observe_expert_plan
    def uses_balanced(length):
        if route_mode == "balanced":
            return True
        if route_mode == "toggle":
            return length == 3 + rank
        if route_mode == "toggle_reverse":
            return length == 5 + rank
        if route_mode == "edp_mixed":
            return dist.get_rank(fsdp_mesh["edp"].get_group()) == 0
        return False

    if route_mode != "original":
        for layer in (reference, candidate):
            original_router = layer.router.forward
            def selected_router(values, *args, router=original_router, **kwargs):
                probs, routing_map, expert_indices, logits = router(values, *args, **kwargs)
                length = values.shape[1]
                if not uses_balanced(length):
                    return probs, routing_map, expert_indices, logits
                base = 3 if length == 3 + rank else 5
                start = (rank * base + rank * (rank - 1) // 2) * config.moe_router_topk
                occurrence = torch.arange(length * config.moe_router_topk, device=values.device) + start
                ids = (occurrence % ep_size) * 2 + (occurrence // ep_size) % 2
                ids = ids.reshape(length, config.moe_router_topk).sort(dim=1).values
                flat_logits = logits.reshape(-1, config.num_moe_experts)
                selected = flat_logits.float().softmax(dim=-1).gather(1, ids)
                selected = selected / selected.sum(dim=-1, keepdim=True)
                if routing_map.dtype != torch.bool:
                    return selected, ids, ids, logits
                probs = torch.zeros_like(flat_logits).scatter(1, ids, selected)
                mask = torch.zeros_like(flat_logits, dtype=torch.bool).scatter_(1, ids, True)
                return probs, mask, ids, logits
            layer.router.forward = selected_router

    if freeze_inputs:
        reference.router.requires_grad_(False)
        candidate.router.requires_grad_(False)
    if freeze_experts:
        reference.experts.requires_grad_(False)
        candidate.experts.requires_grad_(False)
    if zero_probs:
        for layer in (reference, candidate):
            original_router = layer.router.forward

            def masked_router(*args, router=original_router):
                probs, routing_map, expert_indices, logits = router(*args)
                first_column = expert_indices[:, :1]
                if routing_map.dtype != torch.bool:
                    first_column = torch.zeros_like(first_column)
                selected = probs.gather(1, first_column)
                zero_scores = selected - selected.detach()
                probs = probs.scatter(1, first_column, zero_scores)
                return probs, routing_map, expert_indices, logits

            layer.router.forward = masked_router

    if boundary != "none":
        backend = SimpleNamespace(
            config=EngineConfig.from_dict({
                "activation_checkpointing": {"enabled": True, "use_reentrant": reentrant},
            }),
            prefetch_manager=SimpleNamespace(
                suspend_expert_forward_prefetch=lambda model: nullcontext(),
            ),
        )
        key = "" if boundary == "whole" else "experts"
        plan = SimpleNamespace(modules={key: FSDPPlanSpec(activation_checkpoint=True)})
        FSDP2Backend._apply_activation_checkpointing(backend, candidate, plan)
    if fsdp_mesh is not None:
        policy = MixedPrecisionPolicy()
        if mixed_precision:
            policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
            )
            reference.experts.to(torch.bfloat16)
            if cast_inputs:
                def cast_reference_inputs(module, args):
                    tokens, counts, probs = args
                    return tokens.to(torch.bfloat16), counts, probs.to(torch.bfloat16)

                reference.experts.register_forward_pre_hook(cast_reference_inputs)
        fully_shard(candidate.experts, mesh=fsdp_mesh["edp"], mp_policy=policy)
        fully_shard(candidate, mesh=fsdp_mesh)

    lengths = [3 + rank, 5 + rank]
    if empty == "all" or (empty == "one" and rank == 0):
        lengths = [0, 0]
    elif empty == "tiny":
        lengths = [1, 1] if rank == 0 else [0, 0]
    dtype = torch.bfloat16 if mixed_precision else torch.float32
    inputs = [
        ((torch.rand(1, length, 8, dtype=dtype, device=device) + 1) * sign).requires_grad_(not freeze_inputs)
        for length, sign in zip(lengths, (1, -1), strict=True)
    ]
    ref_inputs = [x.detach().clone().requires_grad_(not freeze_inputs) for x in inputs]
    padding_masks = []
    for index, x in enumerate(inputs):
        mask = None
        if padding != "none":
            mask = (torch.arange(x.shape[1], device=device) + rank + index).remainder(2).bool()[None, :]
            if padding == "all" or (padding == "one" and rank == index % ep_size):
                mask.fill_(True)
        padding_masks.append(mask)
    if inference_warmup:
        with torch.inference_mode():
            candidate(inputs[0])
        execution_plans.clear()
    actual = []
    for index, x in enumerate(inputs):
        candidate.set_runtime_inputs(MoERuntimeInputs(padding_mask=padding_masks[index]))
        actual.append(candidate(x)[0])
        base_length = (3, 5)[index]
        source_lengths = [base_length + source for source in range(ep_size)]
        if empty == "one":
            source_lengths[0] = 0
        elif empty == "all":
            source_lengths = [0] * ep_size
        elif empty == "tiny":
            source_lengths = [1] + [0] * (ep_size - 1)
        total_routes = sum(source_lengths) * config.moe_router_topk
        target = total_routes // ep_size + (rank < total_routes % ep_size)
        counts = candidate.token_dispatcher._comm_manager.tokens_per_expert
        manager = candidate.token_dispatcher._comm_manager
        assert (
            manager.routing_map is None
            and manager.token_probs is None
            and manager.dispatched_probs is None
            and manager.handle is None
            and manager._plan is None
            and manager.prefetch_done is None
        )
        assert counts.numel() == 4
        assert counts.sum().item() == target
        if uses_balanced(x.shape[1]):
            routing = execution_plans[-1].routing
            assert not routing.has_global_replicas
            assert routing.num_local_replicas == 0
        if index == 0 and rank > 0 and target > 0 and not uses_balanced(x.shape[1]):
            assert counts[2:].sum().item() > 0
    original_plans = execution_plans.copy()
    assert len(original_plans) == 2 and original_plans[0] is not original_plans[1]
    original_caches = [execution.slot_weights for execution in original_plans]
    expected = []
    for x, mask in zip(ref_inputs, padding_masks, strict=True):
        output = reference(x)[0]
        expected.append(output if mask is None else output.masked_fill(mask.unsqueeze(-1), 0))
    context = (
        f"rank={rank}, EP={ep_size}, empty={empty}, boundary={boundary}, "
        f"reentrant={reentrant}, frozen_inputs={freeze_inputs}, frozen_experts={freeze_experts}, "
        f"on_input={on_input}, zero_probs={zero_probs}, FSDP={fsdp_mesh is not None}, "
        f"mode={route_mode}, padding={padding}"
    )
    tolerance = {"atol": 3e-3, "rtol": 3e-2} if mixed_precision else {"atol": 2e-6, "rtol": 2e-5}
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted, **tolerance, msg=context)

    # Exercise both separate backwards and two live passes in one graph task.
    losses = [result.square().sum() for result in reversed(actual)]
    if backward_together:
        losses = [sum(losses)]
    for loss in losses:
        loss.backward()
        manager = candidate.token_dispatcher._comm_manager
        assert (
            manager.routing_map is None
            and manager.token_probs is None
            and manager.dispatched_probs is None
            and manager.handle is None
            and manager._plan is None
            and manager.prefetch_done is None
        )
    if boundary == "experts":
        replay_plans = execution_plans[2:]
        assert len(replay_plans) == 2
        assert sorted(map(id, replay_plans)) == sorted(map(id, original_plans))
    for plan, cache in zip(original_plans, original_caches, strict=True):
        assert plan.slot_weights is cache
        if cache is not None:
            assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in cache)
    for result in reversed(expected):
        result.square().sum().backward()
    for result, wanted in zip(inputs, ref_inputs, strict=True):
        torch.testing.assert_close(result.grad, wanted.grad, **tolerance, msg=context)

    ref_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        gradient = parameter.grad
        wanted = ref_parameters[name].grad
        if parameter.requires_grad:
            assert gradient is not None and wanted is not None, (context, name)
        if fsdp_mesh is not None and parameter.requires_grad:
            assert gradient is not None and wanted is not None, (context, name)
            gradient = gradient.full_tensor()
            reduction_group = (
                fsdp_mesh["edp"].get_group() if name.startswith("experts.")
                else dist.group.WORLD
            )
            dist.all_reduce(wanted, group=reduction_group)
            wanted /= dist.get_world_size(reduction_group)
        torch.testing.assert_close(
            gradient, wanted, **tolerance, msg=f"{context}, {name}",
            check_dtype=not mixed_precision,
        )

    torch.optim.SGD(candidate.parameters(), lr=0.05).step()
    torch.optim.SGD(reference.parameters(), lr=0.05).step()
    for name, parameter in candidate.named_parameters():
        actual_weight = parameter.full_tensor() if fsdp_mesh is not None else parameter
        torch.testing.assert_close(
            actual_weight, ref_parameters[name], **tolerance, msg=f"{context}, updated {name}",
            check_dtype=not mixed_precision,
        )


def _check_shared_moon_layers(
    ep_group, *, runtime=None, fsdp_mesh=None, boundary="none", reentrant=False,
):
    """Compare two chained MoE layers whose shared replica slots overwrite each other."""
    device = torch.device("cpu") if runtime is None else runtime.device
    rank = dist.get_rank(ep_group)
    size = dist.get_world_size(ep_group)
    config = MoEConfig(
        hidden_size=8, moe_ffn_hidden_size=4, num_moe_experts=size * 2,
        expert_model_parallel_size=size, moe_router_topk=1,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    references, candidates = [], []
    for index in range(2):
        reference = MoELayer(config, ep_group, submodules=submodules).to(device)
        candidate = MoELayer(
            replace(config, moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True),
            ep_group, submodules=submodules,
        ).to(device)
        with torch.no_grad():
            reference.router.weight.zero_()
            reference.router.weight[0].fill_(0.2)
            reference.experts.weight1.fill_(0.1 * (index + 1) + 0.01 * rank)
            reference.experts.weight2.fill_(0.15 * (index + 1) + 0.01 * rank)
        candidate.load_state_dict(reference.state_dict())
        if boundary != "none":
            backend = SimpleNamespace(
                config=EngineConfig.from_dict({
                    "activation_checkpointing": {"enabled": True, "use_reentrant": reentrant},
                }),
                prefetch_manager=SimpleNamespace(
                    suspend_expert_forward_prefetch=lambda model: nullcontext(),
                ),
            )
            key = "" if boundary == "whole" else "experts"
            plan = SimpleNamespace(modules={key: FSDPPlanSpec(activation_checkpoint=True)})
            FSDP2Backend._apply_activation_checkpointing(backend, candidate, plan)
        if fsdp_mesh is not None:
            fully_shard(candidate.experts, mesh=fsdp_mesh["edp"])
            fully_shard(candidate, mesh=fsdp_mesh)
        references.append(reference)
        candidates.append(candidate)

    x = torch.ones((1, 3 + rank, 8), device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = x, ref_x
    for candidate, reference in zip(candidates, references):
        actual = candidate(actual)[0]
        expected = reference(expected)[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    first = candidates[0].token_dispatcher._comm_manager.storage
    second = candidates[1].token_dispatcher._comm_manager.storage
    assert first is second
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
    for candidate, reference in zip(candidates, references):
        wanted_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            gradient = parameter.grad
            wanted = wanted_parameters[name].grad
            assert gradient is not None and wanted is not None
            if fsdp_mesh is not None:
                gradient = gradient.full_tensor()
                group = fsdp_mesh["edp"].get_group() if name.startswith("experts.") else dist.group.WORLD
                dist.all_reduce(wanted, group=group)
                wanted /= dist.get_world_size(group)
            torch.testing.assert_close(gradient, wanted, atol=2e-6, rtol=2e-5)


def _check_moon_weight_memory(ep_group, mesh, routing):
    """Reshard native weights while all saved replicas alias one reusable arena."""
    ep_rank = dist.get_rank(ep_group)
    torch.manual_seed(172 + ep_rank)
    num_local_experts = 4 if routing == "multiple_replicas" else 2
    config = MoEConfig(
        hidden_size=16, moe_ffn_hidden_size=12, num_moe_experts=2 * num_local_experts,
        expert_model_parallel_size=2, moe_router_topk=1,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = MoELayer(config, ep_group, submodules=submodules)
    candidate = MoELayer(
        replace(
            config,
            moe_token_dispatcher_type="flex",
            moe_virtual_expert_load_balance=True,
        ),
        ep_group,
        submodules=submodules,
    )
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
        reference.router.weight.zero_()
        for expert in range(config.num_moe_experts):
            reference.router.weight[expert, expert] = 4.0
    candidate.load_state_dict(reference.state_dict())
    native_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in candidate.experts.parameters()
    )
    root_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp",))
    fully_shard(candidate.experts, mesh=mesh["edp"], reshard_after_forward=True)
    fully_shard(candidate, mesh=root_mesh)
    parameters = dict(candidate.named_parameters())
    inputs = []
    for forward in range(2):
        if routing == "balanced":
            expert_ids = [2 * ep_rank, 2 * ep_rank + 1]
        elif routing == "multiple_replicas":
            expert_ids = list(range(4 * forward, 4 * forward + 4))
        else:
            expert_ids = [2 * forward] * 2
        inputs.append(torch.eye(16)[expert_ids].unsqueeze(0).requires_grad_(True))
    reference_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    saved_weights = []

    def save_tensor(tensor):
        # Include FC1/FC2 matrices and weight-exchange packets (384 + 192
        # elements per expert); none share activation shapes in this fixture.
        if tuple(tensor.shape[-2:]) in ((16, 24), (12, 16)) or (
            tensor.ndim == 2 and tensor.shape[1] == 576
        ):
            saved_weights.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        actual = [candidate(value)[0] for value in inputs]
    expected = [reference(value)[0] for value in reference_inputs]
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted)

    assert saved_weights
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in saved_weights
    }
    # Native FSDP views have released their storage. Saved replica matrices
    # only retain the one layout arena, independent of the number of forwards.
    shared = candidate.token_dispatcher._comm_manager.storage
    assert shared.weight_arena.untyped_storage().nbytes() == native_bytes
    assert shared.grad_arena.untyped_storage().nbytes() == native_bytes
    live_storages = {ptr: size for ptr, size in storages.items() if size}
    assert set(live_storages) <= {shared.weight_arena.data_ptr()}
    assert sum(live_storages.values()) <= native_bytes
    if routing == "balanced":
        assert not live_storages
    assert sum(
        parameter.to_local().untyped_storage().nbytes()
        for parameter in candidate.experts.parameters()
    ) == native_bytes // 2

    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    for value, wanted in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(value.grad, wanted.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in parameters.items():
        assert candidate.get_parameter(name) is parameter
        assert parameter.grad is not None
        wanted = reference_parameters[name].grad
        group = (
            mesh["edp"].get_group() if name.startswith("experts.")
            else dist.group.WORLD
        )
        dist.all_reduce(wanted, group=group)
        wanted /= dist.get_world_size(group)
        torch.testing.assert_close(parameter.grad.full_tensor(), wanted)


def _check_manual_weight_order(
    group, *, prefetched=False, copies=None, frozen=(False, False), unused=False,
):
    from torchturbo.ops.providers.native.virtual_expert_weight_exchange import (
        build_weight_exchange_context,
    )
    from torchturbo.ops import ops

    size, rank = dist.get_world_size(group), dist.get_rank(group)
    # Like the planner, fill a slot prefix from one owner: two replicas on even
    # ranks and one on odd ranks, in descending expert order.
    if copies is None:
        copies = torch.tensor([
            [(r + 1) % size * 2 + 1, (r + 1) % size * 2] if r % 2 == 0
            else [(r + 1) % size * 2, -1]
            for r in range(size)
        ])
    context = build_weight_exchange_context(copies, group)
    weight1 = (torch.arange(32).float().reshape(2, 4, 4) + rank * 100).requires_grad_(not frozen[0])
    weight2 = (torch.arange(16).float().reshape(2, 2, 4) + rank * 200).requires_grad_(not frozen[1])
    x = torch.ones(1, 4, requires_grad=True)
    probs = torch.ones(1, requires_grad=True)
    if prefetched:
        from unittest.mock import patch

        with torch.no_grad():
            r1, r2, _, _ = ops.virtual_expert_weight_exchange(
                weight1, weight2, x, probs, context,
            )
        cache = (r1, r2)
        for _ in range(2):
            # A replay attaches to fresh parameter views and graph inputs, while
            # the original tuple stays graph-free and keeps the same storage.
            current1 = weight1.detach().clone().requires_grad_(not frozen[0])
            current2 = weight2.detach().clone().requires_grad_(not frozen[1])
            current_x = x.detach().clone().requires_grad_()
            current_probs = probs.detach().clone().requires_grad_()
            with patch(
                "torchturbo.ops.providers.native.virtual_expert_weight_exchange.ep_all_to_all",
                side_effect=AssertionError("consume repeated forward communication"),
            ):
                actual = ops.virtual_expert_weight_exchange(
                    current1.view_as(weight1), current2.view_as(weight2),
                    current_x, current_probs, context, prefetched=cache,
                )
            replica1, replica2, out_x, out_probs = actual
            assert all(t.grad_fn is None and not t.requires_grad for t in cache)
            assert replica1.data_ptr() == r1.data_ptr()
            assert replica2.data_ptr() == r2.data_ptr()
            _assert_manual_weight_values_and_gradients(
                copies, rank, current1, current2, current_x, current_probs,
                actual, unused=unused,
            )
            assert all(t.grad_fn is None and not t.requires_grad for t in cache)
        return
    replica1, replica2, out_x, out_probs = ops.virtual_expert_weight_exchange(
        weight1, weight2, x, probs, context,
    )
    _assert_manual_weight_values_and_gradients(
        copies, rank, weight1, weight2, x, probs,
        (replica1, replica2, out_x, out_probs), unused=unused,
    )

    # Frozen experts receive the same replicas without the autograd exchange.
    frozen1, frozen2, frozen_x, frozen_probs = ops.virtual_expert_weight_exchange(
        weight1.detach(), weight2.detach(), x, probs, context,
    )
    assert not frozen1.requires_grad and not frozen2.requires_grad
    assert frozen_x is x and frozen_probs is probs
    torch.testing.assert_close(frozen1, replica1.detach(), rtol=0, atol=0)
    torch.testing.assert_close(frozen2, replica2.detach(), rtol=0, atol=0)


def _assert_manual_weight_values_and_gradients(
    copies, rank, weight1, weight2, x, probs, actual, *, unused,
):
    replica1, replica2, out_x, out_probs = actual
    replicas = [expert for expert in copies[rank].tolist() if expert >= 0]
    assert replica1.shape[0] == replica2.shape[0] == len(replicas)
    assert replica1.is_contiguous() and replica2.is_contiguous()
    for slot, expert in enumerate(replicas):
        owner, local = divmod(expert, 2)
        expected1 = torch.arange(16).float().reshape(4, 4) + local * 16 + owner * 100
        expected2 = torch.arange(8).float().reshape(2, 4) + local * 8 + owner * 200
        torch.testing.assert_close(replica1[slot], expected1, rtol=0, atol=0)
        torch.testing.assert_close(replica2[slot], expected2, rtol=0, atol=0)
    coefficients = torch.arange(1, len(replicas) + 1).float() + rank * 10
    loss = out_x.sum() + out_probs.sum()
    if not unused:
        loss = (
            loss + (replica1 * coefficients[:, None, None]).sum()
            + (replica2 * (coefficients + 100)[:, None, None]).sum()
        )
    loss.backward()
    for weight, offset in ((weight1, 0), (weight2, 100)):
        if not weight.requires_grad or (unused and not (copies >= 0).any()):
            assert weight.grad is None
            continue
        expected = torch.zeros_like(weight)
        for destination in range(copies.shape[0]):
            for slot, expert in enumerate(copies[destination].tolist()):
                if not unused and expert >= 0 and expert // 2 == rank:
                    expected[expert % 2] += destination * 10 + slot + 1 + offset
        torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, torch.ones_like(probs), rtol=0, atol=0)

    if not weight1.requires_grad and not weight2.requires_grad:
        assert out_x is x and out_probs is probs


def _run_moon(rank, world_size, init_file, fsdp, weight_memory=None, padding=False):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        if fsdp:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
            group = mesh["ep"].get_group()
            if weight_memory is not None:
                _check_moon_weight_memory(group, mesh, weight_memory)
                return
            if padding:
                for boundary in ("none", "whole", "experts"):
                    modes = (False,) if boundary == "none" else (False, True)
                    for reentrant in modes:
                        for mixed_precision in (False, True):
                            for mask in ("alternating", "one", "all"):
                                _check_moon(
                                    group, fsdp_mesh=mesh, padding=mask,
                                    boundary=boundary, reentrant=reentrant,
                                    mixed_precision=mixed_precision, freeze_inputs=mixed_precision,
                                    backward_together=not reentrant,
                                )
                                dist.barrier()
                for reentrant in (False, True):
                    for empty in ("one", "all"):
                        _check_moon(
                            group, fsdp_mesh=mesh, padding="alternating", empty=empty,
                            boundary="whole", reentrant=reentrant,
                            freeze_inputs=True, mixed_precision=True,
                        )
                    _check_moon(
                        group, fsdp_mesh=mesh, padding="one", on_input=True,
                        boundary="experts", reentrant=reentrant, freeze_inputs=True,
                        mixed_precision=True, cast_inputs=True,
                    )
                    _check_moon(
                        group, fsdp_mesh=mesh, padding="alternating", freeze_experts=True,
                        boundary="whole", reentrant=reentrant,
                    )
                return
            for boundary in ("whole", "experts"):
                for reentrant in (False, True):
                    for empty in ("none", "one", "all"):
                        _check_moon(
                            group, fsdp_mesh=mesh, freeze_inputs=True,
                            boundary=boundary, reentrant=reentrant, empty=empty,
                        )
                        dist.barrier()
            _check_moon(group, fsdp_mesh=mesh, freeze_inputs=True, mixed_precision=True)
            _check_moon(
                group, fsdp_mesh=mesh, freeze_inputs=True,
                mixed_precision=True, cast_inputs=True,
            )
            for boundary, reentrant in (("none", False), ("experts", False), ("experts", True)):
                _check_moon(group, fsdp_mesh=mesh, boundary=boundary, reentrant=reentrant)
                dist.barrier()
        else:
            group = dist.group.WORLD
            for boundary in ("none", "whole", "experts"):
                modes = (False,) if boundary == "none" else (False, True)
                for reentrant in modes:
                    for empty in ("none", "one", "all"):
                        _check_moon(group, boundary=boundary, reentrant=reentrant, empty=empty)
                        dist.barrier()
            for options in (
                {"on_input": True}, {"zero_probs": True}, {"freeze_experts": True},
                {"empty": "tiny", "on_input": True},
                {"freeze_inputs": True, "boundary": "experts", "reentrant": True},
                {"inference_warmup": True},
            ):
                _check_moon(group, **options)
                dist.barrier()
            for reentrant in (False, True):
                _check_moon(
                    group, boundary="experts", reentrant=reentrant, backward_together=True,
                )
                dist.barrier()
        route_modes = ("balanced", "toggle", "toggle_reverse")
        if fsdp:
            route_modes += ("edp_mixed",)
        for route_mode in route_modes:
            for boundary in ("none", "whole", "experts"):
                for reentrant in ((False,) if boundary == "none" else (False, True)):
                    _check_moon(
                        group, fsdp_mesh=mesh if fsdp else None,
                        route_mode=route_mode, boundary=boundary, reentrant=reentrant,
                    )
                    dist.barrier()
        if fsdp:
            for route_mode in ("balanced", "toggle"):
                for reentrant in (False, True):
                    for cast_inputs in (False, True):
                        _check_moon(
                            group, fsdp_mesh=mesh, route_mode=route_mode,
                            boundary="experts", reentrant=reentrant,
                            mixed_precision=True, cast_inputs=cast_inputs,
                            freeze_inputs=True,
                        )
                        dist.barrier()
        for boundary, reentrant in (("none", False), ("whole", False), ("whole", True)):
            _check_shared_moon_layers(
                group, fsdp_mesh=mesh if fsdp else None,
                boundary=boundary, reentrant=reentrant,
            )
            dist.barrier()
        _check_manual_weight_order(group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn_moon(world_size, init_file, fsdp, weight_memory=None, padding=False):
    context = mp.spawn(
        _run_moon, args=(world_size, str(init_file), fsdp, weight_memory, padding),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Moon EP workers did not complete within 120 seconds")
    finally:
        # A mismatched collective can hang Gloo's teardown after its timeout.
        # Keep regressions bounded even when a worker cannot unwind normally.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_moon_runner_preserves_dispatcher_configuration_error_and_tears_down_group(tmp_path, monkeypatch):
    dispatcher_module = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(dispatcher_module, "_OPS", None)
    monkeypatch.setattr(runtime_module, "_SELECTED_RUNTIME", None)
    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.AUTO)
    assert not dist.is_initialized()
    try:
        with pytest.raises(RuntimeError, match="Ops dispatcher is already configured.*policy="):
            _run_moon(0, 1, tmp_path / "constructor-error", False)
        assert not dist.is_initialized()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_moon_outputs_gradients_and_checkpoint(ep_size, tmp_path):
    _spawn_moon(ep_size, tmp_path / "moon-ep", False)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_moon_fsdp_checkpoint_and_mixed_precision(tmp_path):
    _spawn_moon(4, tmp_path / "moon-fsdp", True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_moon_padding_fsdp_checkpoint_and_mixed_precision(tmp_path):
    _spawn_moon(4, tmp_path / "moon-padding-fsdp", True, padding=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("routing", ["balanced", "skewed", "multiple_replicas"])
def test_moon_fsdp_releases_native_weight_storage(routing, tmp_path):
    _spawn_moon(4, tmp_path / "moon-weight-memory", True, weight_memory=routing)


def test_moon_rejects_te_and_expert_lora():
    from torchturbo.engine.peft.lora import GroupedMLPLoRA
    with pytest.raises(ValueError, match="moe_virtual_expert_load_balance"):
        MoEConfig(
            hidden_size=8,
            moe_ffn_hidden_size=4,
            moe_token_dispatcher_type="alltoall",
            moe_virtual_expert_load_balance=True,
        )
    config = MoEConfig(
        hidden_size=8,
        moe_ffn_hidden_size=4,
        moe_token_dispatcher_type="flex",
        moe_virtual_expert_load_balance=True,
    )
    with pytest.raises(ValueError, match="Virtual-expert.*TE"):
        GroupedMLP(2, True, config)
    experts = GroupedMLP(2, False, config)
    with pytest.raises(ValueError, match="Virtual-expert.*LoRA"):
        GroupedMLPLoRA(experts, "default", r=2)
