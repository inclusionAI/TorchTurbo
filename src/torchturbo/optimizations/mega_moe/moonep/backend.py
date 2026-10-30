"""Dynamic expert placement using one fused compute and SHMEM runtime."""

from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..integration import RoutedExpertsBackend


class MoonEPMegaMoeBackend(RoutedExpertsBackend):
    """Share execution scratch across MoE layers independently of checkpoint mode."""

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        token_bound: int,
        modes: tuple[tuple[str, float], ...],
        receive_capacity_factor: float = 1.3,
    ) -> None:
        super().__init__(ep_group, token_bound)
        self.modes = modes
        self.receive_capacity_factor = receive_capacity_factor
        self._effective_receive_factor = receive_capacity_factor
        self.resources = None
        self.plan_pool = None
        self.executors: dict[tuple[str, float], Any] = {}

    def _initialize(self, hidden: torch.Tensor, config: Any) -> None:
        from ..runtime import _automatic, shmem
        from ..runtime.resources import ExecutionResources

        from .plans import SharedMegaMoePlanPool
        from .transfer import SharedExpertTransfer

        ranks = dist.get_world_size(self.ep_group)
        local_experts = config.num_moe_experts // ranks
        slots = self.token_bound * config.moe_router_topk
        # A router selects distinct experts, limiting each source's local fan-in.
        routed_rows = ranks * self.token_bound * min(local_experts, config.moe_router_topk)
        reserved_rows = (routed_rows + 127) // 128 * 128
        payload_bytes = reserved_rows * config.hidden_size * hidden.element_size()
        # Small copy workloads do not amortize moving a complete expert matrix.
        if slots < 16 * 128 and payload_bytes <= 512 * 1024**2:
            self._effective_receive_factor = max(
                self.receive_capacity_factor, min(float(ranks), reserved_rows / slots)
            )
        for mode, limit in self.modes:
            self.executors[mode, limit] = ExecutionResources(
                hidden,
                local_num_tokens=self.token_bound,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_ffn_hidden_size,
                num_experts=2 * config.num_moe_experts,
                top_k=config.moe_router_topk,
                ep_size=ranks,
                ep_group=self.ep_group,
                initial_capacity_factor=self._effective_receive_factor,
                swiglu_limit=limit or None,
                swiglu_mode=mode,
            )
        layouts = [
            executor.symmetric_bytes
            for executor in self.executors.values()
        ]
        native_bytes = sum(layouts)
        transfer_bytes = (
            2 * local_experts * 3 * config.hidden_size * config.moe_ffn_hidden_size * 2
        )
        page = 2 * 1024**2
        heap_bytes = (
            (native_bytes + transfer_bytes + 64 * 1024**2 + page - 1) // page * page
        )
        shmem.acquire(root_group=self.ep_group, heap_size_bytes=heap_bytes)
        self.resources = SharedExpertTransfer(
            self.ep_group,
            local_experts,
            config.hidden_size,
            config.moe_ffn_hidden_size,
            reuse_weight_slots=True,
        )
        self.plan_pool = SharedMegaMoePlanPool(
            self.ep_group, self.token_bound, self.resources, self.executors
        )
        self.resources.execution_pools.append(self.plan_pool)
        _automatic.register(self.close, self.ep_group)
        for _, limit in self.modes:
            self.plan_pool.prepare_all(hidden, config, limit)

    def execute(
        self,
        experts: Any,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve router selections and parameter ownership while moving work."""
        from torchturbo.optimizations.mega_moe.runtime.moonep import load_allocator

        from .plans import bucket_extra_width
        from .routing import build_route

        if hidden_states.dtype != torch.bfloat16 or hidden_states.device.type != "npu":
            raise ValueError("MegaMoE + MoonEP requires BF16 NPU activations.")
        config = experts.config
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, config.hidden_size)
        tokens = flat.shape[0]
        if tokens > self.token_bound:
            raise ValueError("MoE input exceeds the configured token capacity.")
        indices = indices.reshape(tokens, config.moe_router_topk).int()
        probs = probs.reshape(tokens, config.moe_router_topk).float()
        if tokens < self.token_bound:
            extra = self.token_bound - tokens
            flat = F.pad(flat, (0, 0, 0, extra))
            dummy = torch.arange(
                tokens * config.moe_router_topk,
                self.token_bound * config.moe_router_topk,
                device=indices.device,
            ).reshape(extra, config.moe_router_topk)
            indices = torch.cat(
                (indices, dummy.remainder(config.num_moe_experts).int()), 0
            )
            probs = F.pad(probs, (0, 0, 0, extra))
        if dist.get_world_size(self.ep_group) == 1:
            import torch_npu
            from torchturbo.ops import ops

            from .local_routing import LocalRouting

            routed, mapping, counts = LocalRouting.apply(flat, indices, config.num_moe_experts)
            weight1 = experts.weight1.view(config.num_moe_experts, config.hidden_size, -1)
            weight2 = experts.weight2.view(config.num_moe_experts, -1, config.hidden_size)
            projected = ops.grouped_matmul(routed, weight1, counts, trans_b=False)
            activated = experts.activation_func_with_probs(projected, None)
            projected = ops.grouped_matmul(activated, weight2, counts, trans_b=False)
            output = torch_npu.npu_moe_token_unpermute(projected, mapping, probs=probs)
            if tokens < self.token_bound:
                output = output[:tokens]
            return output.reshape(shape)
        if self.resources is None:
            self._initialize(flat, config)
        planned = build_route(
            indices,
            config.num_moe_experts,
            self.ep_group,
            load_allocator(),
            minimum_imbalance=self._effective_receive_factor,
            extra_width=bucket_extra_width,
            hidden_states=flat,
        )
        ranks = dist.get_world_size(self.ep_group)
        local_experts = config.num_moe_experts // ranks
        virtual_config = config
        if planned.extra_experts:
            virtual_config = replace(
                config, num_moe_experts=ranks * (local_experts + planned.extra_experts)
            )
        output = self.plan_pool.execute(
            flat,
            planned.virtual_ids,
            probs,
            (experts.weight1, experts.weight2),
            virtual_config,
            experts.swiglu_limit or 0.0,
            planned.copies,
            planned.extra_experts,
            planned,
        )
        if tokens < self.token_bound:
            output = output[:tokens]
        return output.reshape(shape)

    def close(self) -> None:
        """Release shared scratch at an idle, collective training boundary."""
        if self.resources is None:
            return
        from torchturbo.optimizations.mega_moe.runtime import shmem

        self.resources.close()
        for executor in self.executors.values():
            executor.close()
        shmem.release()
        self.executors.clear()
        self.resources = None
        self.plan_pool = None
