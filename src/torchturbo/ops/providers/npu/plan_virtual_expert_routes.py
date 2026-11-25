"""Ascend Triton virtual-expert route planning provider."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch

from torchturbo.runtime import TurboDevice, get_device_runtime

from ...base import OpImpl, OpImplKind
from ...moe import HybridEPMetadata, VirtualExpertPlan, VirtualExpertPlannerWorkspace
from ...registry import register_op
from ..native.virtual_expert_weight_exchange import build_weight_exchange_context


def _finish_plan(virtual_experts, copies, statistics, group, ready, device_copies):
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
    )
    weight_context = build_weight_exchange_context(copies, group)
    weight_context = replace(weight_context, experts_to_copy=device_copies)
    return VirtualExpertPlan(
        virtual_experts, device_copies, dispatch_context, weight_context,
        weight_context.has_global_replicas, ep_group=group,
        num_local_replicas=sum(weight_context.receive_splits),
    )


@register_op(
    op_name="plan_virtual_expert_routes",
    impl_name="npu_triton_plan_virtual_expert_routes",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuPlanVirtualExpertRoutes(OpImpl):
    requires_peer_memory = True

    def available(self) -> bool:
        runtime = get_device_runtime()
        if not runtime.shmem.available():
            return False
        try:
            from triton.language.extra.cann.extension import sort  # noqa: F401
        except (ImportError, OSError, AttributeError):
            return False
        return True

    def __call__(
        self,
        top_indices: torch.Tensor,
        *,
        workspace: VirtualExpertPlannerWorkspace,
        async_finish: bool = False,
    ) -> VirtualExpertPlan:
        """Plan on device; optionally defer the native transport's host metadata wait."""
        if workspace.last_use is not None:
            torch.npu.current_stream().wait_event(workspace.last_use)
        ep_size, num_experts = workspace.gathered_counts.shape
        ep_group = workspace.ep_group
        if ep_size == 1:
            copies = torch.full(
                (1, num_experts), -1, dtype=torch.int64, device=top_indices.device,
            )
            context = build_weight_exchange_context(copies, ep_group)
            return VirtualExpertPlan(
                top_indices.to(torch.int64), copies, None, context, False, ep_group=ep_group,
                num_local_replicas=0,
            )

        from ._kernels.virtual_expert_planner import launch_virtual_expert_planner

        virtual_experts, copies = launch_virtual_expert_planner(top_indices, workspace)
        statistics = workspace.field("statistics")
        if not async_finish:
            copies_cpu, statistics_cpu = copies.cpu(), statistics.cpu()
            workspace.last_use = torch.npu.Event()
            workspace.last_use.record(torch.npu.current_stream())
            return _finish_plan(
                virtual_experts, copies_cpu, statistics_cpu, ep_group, None, copies,
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
            ep_group, ready, copies,
        )
        return plan
