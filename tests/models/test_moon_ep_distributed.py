"""Virtual expert output and gradient parity through real EP and FSDP collectives."""

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
from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.utils import RoutingResult
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module


def _check_moon(
    ep_group, *, empty="none", boundary="none", reentrant=False,
    freeze_inputs=False, freeze_experts=False, on_input=False, zero_probs=False,
    fsdp_mesh=None, mixed_precision=False, cast_inputs=False, route_mode="original",
):
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
        # Positive inputs select only rank zero's experts; negative inputs move
        # the hotspot to another owner, changing placement between forwards.
        reference.router.weight.zero_()
        for expert in range(config.moe_router_topk):
            reference.router.weight[expert].fill_(0.2 / (expert + 1))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for name, value in candidate.state_dict().items():
        assert value.shape == reference.state_dict()[name].shape
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
                routing, logits = router(values, *args, **kwargs)
                length = values.shape[1]
                if not uses_balanced(length):
                    return routing, logits
                base = 3 if length == 3 + rank else 5
                start = (rank * base + rank * (rank - 1) // 2) * config.moe_router_topk
                occurrence = torch.arange(length * config.moe_router_topk, device=values.device) + start
                ids = (occurrence % ep_size) * 2 + (occurrence // ep_size) % 2
                ids = ids.reshape(length, config.moe_router_topk).sort(dim=1).values
                flat_logits = logits.reshape(-1, config.num_moe_experts)
                selected = flat_logits.float().softmax(dim=-1).gather(1, ids)
                selected = selected / selected.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(flat_logits).scatter(1, ids, selected)
                mask = torch.zeros_like(flat_logits, dtype=torch.bool).scatter_(1, ids, True)
                return RoutingResult(probs, mask, ids), logits
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
                routing, logits = router(*args)
                first_expert = routing.expert_indices[:, :1]
                selected = routing.probs.gather(1, first_expert)
                zero_scores = selected - selected.detach()
                probs = routing.probs.scatter(1, first_expert, zero_scores)
                return replace(routing, probs=probs), logits

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
        ((torch.rand(1, length, 8, dtype=dtype) + 1) * sign).requires_grad_(not freeze_inputs)
        for length, sign in zip(lengths, (1, -1), strict=True)
    ]
    ref_inputs = [x.detach().clone().requires_grad_(not freeze_inputs) for x in inputs]
    actual = []
    for index, x in enumerate(inputs):
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
            and manager.topk_idx is None
            and manager.dispatched_probs is None
            and manager.handle is None
            and manager._plan.virtual_experts is None
        )
        assert counts.numel() == 4
        assert counts.sum().item() == target
        if uses_balanced(x.shape[1]):
            expert_plan = candidate.token_dispatcher.get_expert_kwargs()["expert_plan"]
            assert not expert_plan.has_global_replicas
            assert expert_plan.weight_context.send_indices.numel() == 0
            assert sum(expert_plan.weight_context.receive_splits) == 0
        if index == 0 and rank > 0 and target > 0 and not uses_balanced(x.shape[1]):
            assert counts[2:].sum().item() > 0
    expected = [reference(x)[0] for x in ref_inputs]
    context = (
        f"rank={rank}, EP={ep_size}, empty={empty}, boundary={boundary}, "
        f"reentrant={reentrant}, frozen_inputs={freeze_inputs}, frozen_experts={freeze_experts}, "
        f"on_input={on_input}, zero_probs={zero_probs}, FSDP={fsdp_mesh is not None}, mode={route_mode}"
    )
    tolerance = {"atol": 3e-3, "rtol": 3e-2} if mixed_precision else {"atol": 2e-6, "rtol": 2e-5}
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted, **tolerance, msg=context)

    # Backward in reverse order, retaining both different plans until used.
    for result in reversed(actual):
        result.square().sum().backward()
        manager = candidate.token_dispatcher._comm_manager
        assert (
            manager.routing_map is None
            and manager.token_probs is None
            and manager.topk_idx is None
            and manager.dispatched_probs is None
            and manager.handle is None
            and manager._plan.virtual_experts is None
        )
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


def _check_moon_weight_memory(ep_group, mesh, routing):
    """Reshard native weights while retaining only selected remote matrices."""
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
        if tuple(tensor.shape) in ((16, 24), (12, 16)) or (
            tensor.ndim == 2 and tensor.shape[1] == 576
        ):
            saved_weights.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        actual = [candidate(value)[0] for value in inputs]
    expected = [reference(value)[0] for value in reference_inputs]
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted)

    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in saved_weights
    }
    # Balanced routes need no copies. Moving the hotspot between owners leaves
    # one (or two of four) incoming replicas per rank across the two graphs.
    replica_budget = 0 if routing == "balanced" else native_bytes // 2
    assert sum(storages.values()) <= replica_budget, (
        f"Saved expert weights retain {sum(storages.values())} bytes after reshard; "
        f"selected replicas need at most {replica_budget} bytes ({routing=})."
    )
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


def _check_manual_weight_order(group):
    from torchturbo.ops.providers.native.virtual_expert_weight_exchange import (
        build_weight_exchange_context,
    )
    from torchturbo.ops import ops

    size, rank = dist.get_world_size(group), dist.get_rank(group)
    # Like the planner, fill a slot prefix from one owner: two replicas on even
    # ranks and one on odd ranks, in descending expert order.
    copies = torch.tensor([
        [(r + 1) % size * 2 + 1, (r + 1) % size * 2] if r % 2 == 0
        else [(r + 1) % size * 2, -1]
        for r in range(size)
    ])
    context = build_weight_exchange_context(copies, group)
    weight1 = (torch.arange(32).float().reshape(2, 4, 4) + rank * 100).requires_grad_()
    weight2 = (torch.arange(16).float().reshape(2, 2, 4) + rank * 200).requires_grad_()
    x = torch.ones(1, 4, requires_grad=True)
    probs = torch.ones(1, requires_grad=True)
    replica1, replica2, out_x, out_probs = ops.virtual_expert_weight_exchange(
        weight1, weight2, x, probs, context,
    )
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
    loss = (
        (replica1 * coefficients[:, None, None]).sum()
        + (replica2 * (coefficients + 100)[:, None, None]).sum()
        + out_x.sum() + out_probs.sum()
    )
    loss.backward()
    for weight, offset in ((weight1, 0), (weight2, 100)):
        expected = torch.zeros_like(weight)
        for destination in range(size):
            for slot, expert in enumerate(copies[destination].tolist()):
                if expert >= 0 and expert // 2 == rank:
                    expected[expert % 2] += destination * 10 + slot + 1 + offset
        torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, torch.ones_like(probs), rtol=0, atol=0)

    # Frozen experts receive the same replicas without the autograd exchange.
    frozen1, frozen2, frozen_x, frozen_probs = ops.virtual_expert_weight_exchange(
        weight1.detach(), weight2.detach(), x, probs, context,
    )
    assert not frozen1.requires_grad and not frozen2.requires_grad
    assert frozen_x is x and frozen_probs is probs
    torch.testing.assert_close(frozen1, replica1.detach(), rtol=0, atol=0)
    torch.testing.assert_close(frozen2, replica2.detach(), rtol=0, atol=0)


def _run_moon(rank, world_size, init_file, fsdp, weight_memory=None):
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
            ):
                _check_moon(group, **options)
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
        _check_manual_weight_order(group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn_moon(world_size, init_file, fsdp, weight_memory=None):
    context = mp.spawn(
        _run_moon, args=(world_size, str(init_file), fsdp, weight_memory),
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
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_moon_outputs_gradients_and_checkpoint(ep_size, tmp_path):
    _spawn_moon(ep_size, tmp_path / "moon-ep", False)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_moon_fsdp_checkpoint_and_mixed_precision(tmp_path):
    _spawn_moon(4, tmp_path / "moon-fsdp", True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("routing", ["balanced", "skewed", "multiple_replicas"])
def test_moon_fsdp_releases_native_weight_storage(routing, tmp_path):
    _spawn_moon(4, tmp_path / "moon-weight-memory", True, weight_memory=routing)


def test_moon_rejects_te_and_expert_lora():
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
