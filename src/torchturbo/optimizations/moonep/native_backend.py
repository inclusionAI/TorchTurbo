# SPDX-License-Identifier: Apache-2.0
"""Native TorchTurbo MoE dispatch and experts with official MoonEP placement."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import weakref

import torch
import torch.distributed as dist

from torchturbo.logging import get_logger
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.models.modules.moe.utils import RoutingResult
from torchturbo.optimizations.routed_experts import (
    RoutedExpertsBackend,
    check_shared_shmem_runtime,
)

from torchturbo.optimizations.moonep.official_transfer import ExpertTransfer
from hyper_multicore.moonep.grouped_experts import TransferredExperts
from torchturbo.optimizations.moonep.official_planner import PlanningScratch
from hyper_multicore.moonep.placement import has_transfer_work, placement_is_profitable

logger = get_logger(__name__)


class NativeMoonEPBackend(RoutedExpertsBackend):
    """Use official planning and weight transfers with the native MoE executor."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        token_bound: int,
        *,
        grad_reduce_dtype: str = "fp32",
    ) -> None:
        super().__init__(group, token_bound)
        self.group = group
        self.grad_reduce_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
            grad_reduce_dtype
        ]
        self.planner = None
        self.planner_scratch = None
        self._placements = []
        self.resources = None
        self._runtime_acquired = False
        self._dispatchers = weakref.WeakKeyDictionary()
        self._cube_cores = None

    def _movement_is_profitable(
        self, maximum: int, target: int, config: Any, *, training: bool = True
    ) -> bool:
        if self._cube_cores is None:
            device = torch.device("npu", torch.npu.current_device())
            cores = torch.tensor(
                int(torch.npu.get_device_properties(device).cube_core_num),
                dtype=torch.int64,
                device=device,
            )
            # Use one common conservative compute rate even on heterogeneous
            # groups; no rank may choose a different communication path.
            dist.all_reduce(cores, op=dist.ReduceOp.MAX, group=self.group)
            self._cube_cores = int(cores)
        local = config.num_moe_experts // dist.get_world_size(self.group)
        weight_elements = 3 * local * config.hidden_size * config.moe_ffn_hidden_size
        shard_bytes = weight_elements * torch.bfloat16.itemsize
        return placement_is_profitable(
            maximum,
            target,
            config.hidden_size,
            config.moe_ffn_hidden_size,
            self._cube_cores,
            shard_bytes,
            projection_flops=12 if training else 6,
            ep_size=dist.get_world_size(self.group),
            local_experts=local,
            gradient_bytes=(
                weight_elements * self.grad_reduce_dtype.itemsize if training else 0
            ),
        )

    def _use_native_dispatcher(self, experts: Any) -> bool:
        config = experts.config
        ranks = dist.get_world_size(self.group)
        local = config.num_moe_experts // ranks
        if ranks == 1 or not has_transfer_work(
            self.token_bound, config.moe_router_topk, local
        ):
            return True
        target = self.token_bound * config.moe_router_topk
        # Capacity supplies only a worst-case upper bound. Real large calls
        # still use their shared count agreement before moving any expert.
        return not self._movement_is_profitable(
            ranks * target, target, config, training=experts.training
        )

    def register_dispatcher(
        self, experts: Any, dispatcher: MoEAlltoAllTokenDispatcher
    ) -> None:
        """Reuse each layer's native dispatcher and its existing stream ownership."""
        self._dispatchers[experts] = {dispatcher.num_local_experts: dispatcher}

    @torch.inference_mode(False)
    @torch.no_grad()
    def _initialize(self, experts: Any, hidden: torch.Tensor) -> None:
        check_shared_shmem_runtime(require_moonep=True)
        from hyper_multicore import shmem
        from torchturbo.optimizations._expert_cleanup import register

        config = experts.config
        from torchturbo.optimizations.moonep.official_planner import OfficialPlanner

        if (
            experts.weight1.dtype != torch.bfloat16
            or experts.weight2.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Native MoonEP requires BF16 expert parameters at the FSDP forward boundary."
            )

        self.planner = OfficialPlanner(
            config.num_moe_experts,
            config.moe_router_topk,
            dist.get_world_size(self.group),
            dist.get_rank(self.group),
            int(torch.npu.get_device_properties(hidden.device).cube_core_num),
        )
        projections = (
            (config.hidden_size, config.moe_ffn_hidden_size * 2),
            (config.moe_ffn_hidden_size, config.hidden_size),
        )
        local = config.num_moe_experts // dist.get_world_size(self.group)
        _, symmetric_bytes = self.planner.workspace_sizes(self.token_bound)
        # The native executor needs only planner scratch and two shared weight
        # banks. It acquires the same ordered EP root as the fused executor.
        required = symmetric_bytes + sum(
            2 * local * rows * columns * torch.bfloat16.itemsize
            for rows, columns in projections
        )
        required += 512 * (len(projections) + 1)
        # CANN SHMEM maps physical 2 MiB pages, including for small expert banks.
        page_bytes = 2 * 1024 * 1024
        required = (required + page_bytes - 1) // page_bytes * page_bytes
        shmem.acquire(self.group, heap_size_bytes=required)
        self._runtime_acquired = True
        try:
            self.planner_scratch = PlanningScratch(
                self.planner, self.token_bound, hidden.device, allocator=shmem
            )
            self.resources = ExpertTransfer(
                self.group,
                config.num_moe_experts,
                projections,
                hidden.device,
                allocator=shmem,
                grad_reduce_dtype=self.grad_reduce_dtype,
            )
            self.resources.bind()
            shmem.barrier(blocking=False)
            register(self.close, self.group)
        except BaseException:
            self.close()
            raise

    def route(
        self, layer: Any, hidden_states: torch.Tensor, routing: RoutingResult
    ) -> torch.Tensor | None:
        """Select placement from the original router mask before entering expert hooks."""
        self._check_token_capacity(hidden_states.shape[0])
        check_shared_shmem_runtime(require_moonep=True)
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "Native MoonEP requires BF16 activations; configure mixed_precision: bf16."
            )
        experts = layer.experts
        routing_map, expert_indices = routing.routing_map, routing.expert_indices
        self._check_ep_geometry(experts.config, hidden_states.device)
        if self._use_native_dispatcher(experts):
            return None
        tokens, num_experts = routing_map.shape
        counts = routing_map.sum(dim=0, dtype=torch.int64)
        all_counts = counts.new_empty((dist.get_world_size(self.group), num_experts))
        dist.all_gather_into_tensor(all_counts.view(-1), counts, group=self.group)
        # Read the small count table once. Both the capability check and the
        # placement decision need host values before the next collective.
        rows = all_counts.tolist()
        source_slots = [sum(row) for row in rows]
        supported = source_slots[0] > 0 and all(
            n == source_slots[0] for n in source_slots
        )
        if not supported:
            logger.warning_rank0_once(
                "Official MoonEP planning requires equal nonzero routed-token counts across EP ranks; "
                "using native MoE for this call without padding or changing router selections."
            )
        rank_count = all_counts.shape[0]
        local = num_experts // rank_count
        totals = [sum(column) for column in zip(*rows)]
        max_load = max(sum(totals[i : i + local]) for i in range(0, num_experts, local))
        # Balanced placement requires no expert movement. Preserve the original
        # executor and its parameter/FSDP hooks in that case.
        move = (
            supported
            and has_transfer_work(tokens, experts.config.moe_router_topk, local)
            and self._movement_is_profitable(
                max_load,
                tokens * experts.config.moe_router_topk,
                experts.config,
                training=experts.training,
            )
        )
        if move:
            if self.resources is None:
                self._initialize(experts, hidden_states)
            import torch_npu

            if sum(source_slots) > 2**31 - 1:
                raise ValueError(
                    "Official MoonEP planner routed row count exceeds INT32"
                )

            # Reuse the official counts-keyed plan while restoring this call's
            # fresh token order through the existing NPU routing primitive.
            with torch.no_grad():
                _, source_inverse, _, _ = torch_npu.npu_moe_init_routing_v2(
                    hidden_states,
                    expert_indices.int(),
                    active_num=expert_indices.numel(),
                    expert_num=num_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, num_experts],
                    row_idx_type=0,
                )
            placement = self.planner(
                hidden_states,
                expert_indices,
                all_counts.to(torch.int32),
                tokens * experts.config.moe_router_topk,
                dist.get_rank(self.group),
                resources=self.planner_scratch.inputs(),
                source_rows=tuple(source_slots),
                source_counts=tuple(map(tuple, rows)),
                source_inverse=source_inverse,
            )
            if placement is not None:
                # A synchronous model call enters its own parameter hooks. Each
                # autograd context subsequently saves this owned plan's tensors.
                self._placements.append(placement)
                try:
                    return super().route(layer, hidden_states, routing)
                finally:
                    self._placements.pop()
        return layer.compute_native_experts(
            hidden_states,
            routing,
            global_token_counts=all_counts,
        )

    def _dispatcher(
        self, experts: Any, config: Any, width: int
    ) -> MoEAlltoAllTokenDispatcher:
        dispatchers = self._dispatchers[experts]
        if not experts.num_local_experts <= width <= 2 * experts.num_local_experts:
            raise ValueError(
                "MoonEP virtual width exceeds the fixed home-plus-borrowed expert bank."
            )
        if width not in dispatchers:
            start = dist.get_rank(self.group) * width
            dispatchers[width] = MoEAlltoAllTokenDispatcher(
                width,
                list(range(start, start + width)),
                config,
                self.group,
                dtoh_stream=dispatchers[experts.num_local_experts].dtoh_stream,
            )
        return dispatchers[width]

    def execute(
        self,
        experts: Any,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        selected_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a profitable placement inside the model-owned parameter hooks."""
        if hidden.dtype != torch.bfloat16 or hidden.device.type != "npu":
            raise ValueError("Official Ascend MoonEP requires BF16 NPU activations")
        if not self._placements:
            raise RuntimeError(
                "Native MoonEP execution requires its current shared placement"
            )
        placement = self._placements[-1]
        virtual = placement.virtual_ids.long()
        virtual_config = replace(
            experts.config, num_moe_experts=placement.counts_by_source.shape[1]
        )
        shape = (ids.shape[0], virtual_config.num_moe_experts)
        virtual_probs = selected_probs.new_zeros(shape).scatter(
            1, virtual, selected_probs
        )
        virtual_mask = torch.zeros_like(virtual_probs, dtype=torch.bool).scatter_(
            1, virtual, True
        )
        routing = RoutingResult(virtual_probs, virtual_mask, virtual)
        width = virtual_config.num_moe_experts // dist.get_world_size(self.group)
        dispatcher = self._dispatcher(experts, virtual_config, width)
        received, probabilities = dispatcher.dispatch_preprocess(
            hidden,
            routing,
            global_token_counts=placement.counts_by_source,
        )
        received, probabilities = dispatcher.token_dispatch(received, probabilities)
        received, counts, probabilities = dispatcher.dispatch_postprocess(
            received, probabilities
        )
        output = TransferredExperts.apply(
            received,
            experts.weight1,
            experts.weight2,
            counts.to(device=hidden.device, dtype=torch.int64),
            placement.copies,
            self.resources,
            experts.activation_func,
            probabilities,
        )
        output = dispatcher.combine_preprocess(output)
        output = dispatcher.token_combine(output)
        return dispatcher.combine_postprocess(output)

    def close(self) -> None:
        """Finalize the official root after all parameter and graph leases end."""
        if not self._runtime_acquired:
            return
        from hyper_multicore import shmem

        torch.npu.synchronize()
        shmem.host_barrier()
        if self.resources is not None:
            self.resources.close()
        if self.planner_scratch is not None:
            self.planner_scratch.close()
        shmem.release()
        self._runtime_acquired = False
        self.resources = None
        self.planner = None
        self.planner_scratch = None
