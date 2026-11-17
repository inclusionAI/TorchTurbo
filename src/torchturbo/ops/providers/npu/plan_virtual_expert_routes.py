"""Ascend Triton virtual-expert route planning provider."""

from __future__ import annotations

from functools import cache, partial
from dataclasses import replace

import torch
import torch.distributed as dist

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import HybridEPMetadata, VirtualExpertPlan, VirtualExpertPlannerWorkspace
from ...registry import register_op
from ..native.virtual_expert_weight_exchange import build_weight_exchange_context


def _finish_plan(virtual_experts, copies, statistics, sorted_ids, route_order, group, ready, device_copies):
    if ready is not None:
        ready.synchronize()
    ep_size, num_local_experts = copies.shape
    sent = statistics[:ep_size]
    received = statistics[ep_size:].reshape(ep_size, 2 * num_local_experts)
    dispatch_context = HybridEPMetadata(
        received_counts=received,
        tokens_per_expert=received.sum(0),
        input_splits=sent.tolist(),
        output_splits=received.sum(1).tolist(),
        sorted_expert_indices=sorted_ids,
        route_order=route_order,
    )
    weight_context = build_weight_exchange_context(copies, group)
    weight_context = replace(weight_context, experts_to_copy=device_copies)
    return VirtualExpertPlan(
        virtual_experts, device_copies, dispatch_context, weight_context,
        weight_context.has_global_replicas, ep_group=group,
    )


@cache
def _standalone_workspace(num_experts, device, group):
    from torchturbo.runtime.npu import NpuRuntime

    # SHMEM rendezvous allocations live for the process lifetime. Retain the
    # workspace for the same lifetime instead of allocating on every call.
    return VirtualExpertPlannerWorkspace(
        num_experts=num_experts, device=device, group=group, runtime=NpuRuntime(),
    )


def _prepare_workspace(workspace, group):
    import torch_npu  # noqa: F401

    if workspace.ep_group is not group:
        raise ValueError("NPU planner workspace is bound to a different EP group.")
    if workspace.last_use is not None:
        torch.npu.current_stream().wait_event(workspace.last_use)


@register_op(
    op_name="plan_virtual_expert_routes",
    impl_name="npu_triton_plan_virtual_expert_routes",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuPlanVirtualExpertRoutes(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
            import triton  # noqa: F401
            from torchturbo.runtime.npu import NpuRuntime

            NpuRuntime.shmem_library()
        except (ImportError, OSError):
            return False
        return True

    def __call__(
        self,
        top_indices: torch.Tensor,
        num_experts: int,
        ep_group: dist.ProcessGroup,
        *,
        workspace: VirtualExpertPlannerWorkspace | None = None,
        async_finish: bool = False,
    ) -> VirtualExpertPlan:
        """Plan on device; optionally defer the native transport's host metadata wait."""
        ep_size = dist.get_world_size(ep_group)
        num_local_experts = num_experts // ep_size
        if ep_size == 1:
            copies = torch.full(
                (1, num_local_experts), -1, dtype=torch.int64, device=top_indices.device,
            )
            context = build_weight_exchange_context(copies, ep_group)
            return VirtualExpertPlan(
                top_indices.to(torch.int64), copies, None, context, False, ep_group=ep_group,
            )

        import torch_npu  # noqa: F401
        from ._kernels.virtual_expert_planner import launch_virtual_expert_planner, _sort_routes

        if workspace is None:
            workspace = _standalone_workspace(num_experts, top_indices.device, ep_group)
        _prepare_workspace(workspace, ep_group)
        virtual_experts, copies = launch_virtual_expert_planner(top_indices, workspace)
        statistics = workspace.field("statistics")
        sorted_ids, route_order = _sort_routes(virtual_experts)
        if not async_finish:
            copies_cpu, statistics_cpu = copies.cpu(), statistics.cpu()
            workspace.last_use = torch.npu.Event()
            workspace.last_use.record(torch.npu.current_stream())
            return _finish_plan(
                virtual_experts, copies_cpu, statistics_cpu, sorted_ids,
                route_order, ep_group, None, copies,
            )
        copies_cpu = torch.empty(copies.shape, dtype=copies.dtype, device="cpu", pin_memory=True)
        statistics_cpu = torch.empty(
            statistics.shape, dtype=statistics.dtype, device="cpu", pin_memory=True,
        )
        copies_cpu.copy_(copies, non_blocking=True)
        statistics_cpu.copy_(statistics, non_blocking=True)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream())
        # The next planner must also wait for the shared statistics' host copy.
        workspace.last_use = ready
        plan = VirtualExpertPlan(
            virtual_experts, copies, None, None, None, ep_group=ep_group,
        )
        plan._materialize = partial(
            _finish_plan, virtual_experts, copies_cpu, statistics_cpu,
            sorted_ids, route_order, ep_group, ready, copies,
        )
        return plan
