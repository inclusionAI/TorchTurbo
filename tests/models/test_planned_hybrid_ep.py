"""Planned dispatch reuses the real planner's counts without a second count all-gather."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.token_dispatcher import (
    MoEFlexTokenDispatcher, _VirtualExpertHybridEPManager,
)
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.ops.moe import VirtualExpertPlannerWorkspace
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module
from tests.models.moonep_reference_cases import MEGATRON_PLANNER_CASES


class _NoPerTokenRouteSort(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.sort.default, torch.ops.aten.sort.stable):
            assert args[0].ndim == 1, (
                "Packing only needs a global stable sort on flattened assignments."
            )
        return func(*args, **(kwargs or {}))


def _run_plain_flex_padding(rank, size, init_file):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=size,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime_module._SELECTED_RUNTIME = CpuRuntime()
        get_or_create_ops(device_runtime=runtime_module._SELECTED_RUNTIME, default_policy=OpPolicy.NATIVE)
        config = MoEConfig(
            num_moe_experts=4, expert_model_parallel_size=size,
            moe_router_topk=2, moe_token_dispatcher_type="flex",
        )
        dispatcher = MoEFlexTokenDispatcher(2, [rank * 2, rank * 2 + 1], config, dist.group.WORLD)
        for lengths, all_padding in [((0, 3), False), ((3, 5), False), ((2, 3), True)]:
            length = lengths[rank]
            ids = torch.arange(length * 2).view(length, 2) % 4
            padding = (torch.arange(length) % 2 == 0) | (rank == 0) | all_padding
            valid = ~padding
            x = torch.arange(length * 3).view(length, 3).float().requires_grad_()
            p = torch.full((length, 2), 0.25, requires_grad=True)
            h, probs = dispatcher.dispatch_preprocess(x, ids, p, padding_mask=padding)
            h, probs = dispatcher.token_dispatch(h, probs)
            h, counts, probs = dispatcher.dispatch_postprocess(h, probs)
            expected_counts = torch.bincount(ids[valid].flatten(), minlength=4)
            dist.all_reduce(expected_counts)
            torch.testing.assert_close(counts, expected_counts[rank * 2 : rank * 2 + 2])
            restored = dispatcher.combine_postprocess(
                dispatcher.token_combine(dispatcher.combine_preprocess(h * probs[:, None]))
            )
            torch.testing.assert_close(restored, x * 0.5 * valid[:, None])
            restored.sum().backward()
            torch.testing.assert_close(x.grad, (0.5 * valid[:, None]).expand_as(x))
            torch.testing.assert_close(p.grad, (x.detach().sum(1, keepdim=True) * valid[:, None]).expand_as(p))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_plain_flex_excludes_padding_across_uneven_and_empty_ranks(tmp_path):
    mp.spawn(_run_plain_flex_padding, args=(2, str(tmp_path / "flex-padding")), nprocs=2, join=True)


def _run_planned_dispatch(rank, size, init_file, policy, mask_padding):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=size,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        overrides = {
            "plan_virtual_expert_routes": "torch_plan_virtual_expert_routes",
            "hybrid_ep_dispatch": "torch_hybrid_ep_dispatch",
            "hybrid_ep_combine": "torch_hybrid_ep_combine",
        }
        dispatcher = get_or_create_ops(
            device_runtime=runtime, default_policy=policy,
            overrides=overrides if policy == OpPolicy.CUSTOM else None,
        )
        for topk in (1, 2, 4):
            for empty in (False, True):
                length = 0 if empty and rank == 0 else 3 + rank
                ids = torch.arange(topk).expand(length, -1).clone().flip(1)
                probs = torch.arange(length * topk).reshape(length, topk).float().requires_grad_()
                padding_mask = torch.arange(length).remainder(2).bool() if mask_padding else None
                x = torch.arange(length * 2).reshape(length, 2).float().requires_grad_()
                count_exchanges = []
                gather = dist.all_gather_into_tensor

                def counted_gather(*args, **kwargs):
                    count_exchanges.append(1)
                    return gather(*args, **kwargs)

                dist.all_gather_into_tensor = counted_gather
                try:
                    config = MoEConfig(
                        hidden_size=2, moe_ffn_hidden_size=2, num_moe_experts=4,
                        moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True,
                        moe_router_topk=topk, expert_model_parallel_size=size,
                    )
                    manager = _VirtualExpertHybridEPManager(dist.group.WORLD, 2, 4, config)
                    wrapped = manager.wrap_layer_input(x)
                    manager.setup_metadata(ids, probs, padding_mask=padding_mask)
                    plan = manager._plan.routing
                    runtime_ids = plan.virtual_experts.clone()
                    if rank == 0:
                        plan.virtual_experts = plan.virtual_experts.clone()
                    with _NoPerTokenRouteSort():
                        packed = manager.dispatch(wrapped)
                    assert len(count_exchanges) == 1
                    assert plan.virtual_experts is None
                    assert plan.dispatch_context is not None
                finally:
                    dist.all_gather_into_tensor = gather
                # Compare with the ordinary dense path using the mapped assignments.
                ref_x = x.detach().clone().requires_grad_()
                ref_probs = probs.detach().clone().requires_grad_()
                ref_scores = ref_probs if padding_mask is None else ref_probs.masked_fill(padding_mask[:, None], 0)
                dense = torch.zeros(length, 8).scatter(1, runtime_ids, ref_scores)
                mask = torch.zeros(length, 8, dtype=torch.bool).scatter_(1, runtime_ids.sort(1).values, True)
                ref_packed, ref_weights, ref_counts, ref_handle = dispatcher.hybrid_ep_dispatch(
                    ref_x, mask, dense, dist.group.WORLD, 4, topk,
                    expert_indices=runtime_ids.sort(1).values.to(torch.int32 if rank == 0 else torch.int64),
                )
                torch.testing.assert_close(packed, ref_packed)
                torch.testing.assert_close(manager.dispatched_probs, ref_weights)
                torch.testing.assert_close(manager.tokens_per_expert, ref_counts)
                output = manager.combine(packed * manager.dispatched_probs[:, None])
                output = manager.finalize_layer_output(output)
                manager.release_routing_state()
                expected = dispatcher.hybrid_ep_combine(ref_packed * ref_weights[:, None], ref_handle)
                torch.testing.assert_close(output, expected)
                output.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, ref_x.grad)
                torch.testing.assert_close(probs.grad, ref_probs.grad)
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("policy", [OpPolicy.NATIVE, OpPolicy.CUSTOM])
@pytest.mark.parametrize("mask_padding", [False, True])
def test_virtual_dispatch_uses_one_count_exchange_and_matches_dense_gradients(tmp_path, policy, mask_padding):
    mp.spawn(
        _run_planned_dispatch, args=(2, str(tmp_path / "planned-ep"), policy, mask_padding),
        nprocs=2, join=True,
    )


def _run_megatron_planner_cases(rank, size, init_file):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=size,
        timeout=timedelta(seconds=30),
    )
    try:
        dispatcher = get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)
        for case in MEGATRON_PLANNER_CASES:
            if len(case["top_indices"]) != size:
                continue
            num_experts = size * case["num_local_experts"]
            workspace = VirtualExpertPlannerWorkspace(
                num_experts=num_experts, device=torch.device("cpu"), group=dist.group.WORLD,
                runtime=dispatcher.device_runtime, peer_access=False,
            )
            for dtype in (torch.int32, torch.int64):
                indices = torch.tensor(case["top_indices"][rank], dtype=dtype).reshape(-1, case["topk"])
                plan = dispatcher.plan_virtual_expert_routes(indices, workspace=workspace)
                expected_ids = torch.tensor(case["runtime_ids"][rank], dtype=torch.int64).reshape_as(indices)
                torch.testing.assert_close(plan.virtual_experts.long(), expected_ids, msg=case["name"])
                torch.testing.assert_close(
                    plan.experts_to_copy.long(), torch.tensor(case["copies"]), msg=case["name"],
                )
                copies = case["copies"]
                assert plan.num_local_replicas == sum(expert >= 0 for expert in copies[rank])
                assert plan.has_global_replicas == any(expert >= 0 for row in copies for expert in row)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("ep_size", [2, 4])
def test_planner_matches_megatron_common_domain_cases(tmp_path, ep_size):
    mp.spawn(
        _run_megatron_planner_cases, args=(ep_size, str(tmp_path / "megatron-cases")),
        nprocs=ep_size, join=True,
    )
