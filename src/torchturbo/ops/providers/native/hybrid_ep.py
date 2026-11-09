"""Native fallback for HybridEP token dispatch and combine."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...base import OpImpl, OpImplKind
from ...moe import (
    HybridEPHandle, MoEPermutation, ep_all_to_all, get_expert_indices,
)
from ...ops import ops
from ...registry import register_op


@register_op(
    op_name="hybrid_ep_dispatch",
    impl_name="torch_hybrid_ep_dispatch",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchHybridEPDispatch(OpImpl):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        ep_group: dist.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        expert_indices: torch.Tensor | None = None,
        *,
        dispatch_context: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HybridEPHandle]:
        """Permute tokens, exchange across EP, and sort by local expert."""
        planned_metadata = dispatch_context
        ep_size = dist.get_world_size(ep_group)
        num_experts = ep_size * num_local_experts
        if routing_map.dtype == torch.bool:
            if expert_indices is None:
                if routing_map.shape[0] == 0:
                    expert_indices = routing_map.new_empty((0, router_topk), dtype=torch.long)
                else:
                    expert_indices = get_expert_indices(routing_map)
        else:
            if probs.shape != routing_map.shape:
                raise ValueError("Compact route IDs and probabilities must have the same shape.")
            if planned_metadata is None:
                expert_indices, order = routing_map.long().sort(dim=-1, stable=True)
            else:
                expert_indices = planned_metadata.sorted_expert_indices
                order = planned_metadata.route_order
            probs = probs.gather(1, order)
            routing_map = expert_indices

        if planned_metadata is not None:
            tokens_per_expert = planned_metadata.tokens_per_expert
            input_splits = planned_metadata.input_splits if ep_size > 1 else None
            output_splits = planned_metadata.output_splits if ep_size > 1 else None
            num_global_tokens_per_local_expert = planned_metadata.received_counts.to(
                device=hidden_states.device,
            )
        else:
            if routing_map.dtype == torch.bool:
                num_local_tokens_per_expert = routing_map.sum(dim=0, dtype=torch.long)
            else:
                num_local_tokens_per_expert = torch.zeros(
                    num_experts, device=routing_map.device, dtype=torch.long,
                ).scatter_add_(
                    0, expert_indices.flatten(), torch.ones_like(expert_indices).flatten(),
                )
        if planned_metadata is None and ep_size > 1:
            ep_rank = dist.get_rank(ep_group)
            global_tokens_per_expert = torch.empty(
                ep_size * num_experts,
                dtype=num_local_tokens_per_expert.dtype,
                device=num_local_tokens_per_expert.device,
            )
            dist.all_gather_into_tensor(
                global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group,
            )
            global_tokens_per_expert = global_tokens_per_expert.reshape(
                ep_size, num_experts,
            )
            local_start = ep_rank * num_local_experts
            num_global_tokens_per_local_expert = global_tokens_per_expert[
                :, local_start : local_start + num_local_experts
            ].contiguous()
            tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=0)
            input_splits_tensor = (
                num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)
            )
            output_splits_tensor = num_global_tokens_per_local_expert.sum(dim=1)
            host_metadata = torch.cat(
                (tokens_per_expert, input_splits_tensor, output_splits_tensor)
            ).to("cpu")
            tokens_per_expert = host_metadata[:num_local_experts]
            input_splits = host_metadata[
                num_local_experts : num_local_experts + ep_size
            ].tolist()
            output_splits = host_metadata[num_local_experts + ep_size :].tolist()
        elif planned_metadata is None:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.unsqueeze(0)
            tokens_per_expert = num_local_tokens_per_expert.to("cpu")
            input_splits = None
            output_splits = None

        permutation = self._source_permutation(
            hidden_states, routing_map, probs, expert_indices, num_experts,
        )
        dispatched_hidden = ep_all_to_all(
            ep_group, permutation.tokens, output_splits, input_splits,
        )
        dispatched_probs = ep_all_to_all(
            ep_group, permutation.probs, output_splits, input_splits,
        )
        restore_output_by_local_experts = None
        if ep_size > 1 and num_local_experts > 1:
            chunk_indices = torch.arange(
                ep_size * num_local_experts, device=hidden_states.device,
            ).reshape(ep_size, num_local_experts)
            sort_input_by_local_experts = chunk_indices.T.ravel()
            restore_output_by_local_experts = chunk_indices.reshape(
                num_local_experts, ep_size,
            ).T.ravel()
            dispatched_hidden, dispatched_probs = ops.moe_sort_chunks(
                dispatched_hidden,
                num_global_tokens_per_local_expert.ravel(),
                sort_input_by_local_experts,
                probs=dispatched_probs,
            )

        handle = HybridEPHandle(
            row_indices=permutation.row_indices,
            expert_indices=permutation.expert_indices,
            restore_shape=hidden_states.shape,
            ep_group=ep_group,
            num_local_experts=num_local_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=(
                num_global_tokens_per_local_expert
                if ep_size > 1 and num_local_experts > 1
                else None
            ),
            restore_output_by_local_experts=restore_output_by_local_experts,
        )
        return dispatched_hidden, dispatched_probs, tokens_per_expert, handle

    def _source_permutation(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor,
        probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int,
    ) -> MoEPermutation:
        """Normalize compact routes only for providers requiring a dense mask."""
        if routing_map.dtype != torch.bool:
            probs = probs.new_zeros((routing_map.shape[0], num_experts)).scatter(
                1, expert_indices, probs,
            )
            routing_map = torch.zeros(
                probs.shape, dtype=torch.bool, device=routing_map.device,
            ).scatter_(1, expert_indices, True)
        return ops.moe_permutation(
            hidden_states, routing_map, probs=probs,
            num_out_tokens=expert_indices.numel(), expert_indices=expert_indices,
        )


@register_op(
    op_name="hybrid_ep_combine",
    impl_name="torch_hybrid_ep_combine",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchHybridEPCombine(OpImpl):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        handle: HybridEPHandle,
    ) -> torch.Tensor:
        """Restore chunk order, return across EP, and unpermute to source rows."""
        if handle.restore_output_by_local_experts is not None:
            hidden_states, _ = ops.moe_sort_chunks(
                hidden_states,
                handle.num_global_tokens_per_local_expert.T.ravel(),
                handle.restore_output_by_local_experts,
            )
        combined = ep_all_to_all(
            handle.ep_group,
            hidden_states,
            handle.input_splits,
            handle.output_splits,
        )
        permutation = MoEPermutation(
            combined,
            None,
            handle.row_indices,
            handle.expert_indices,
        )
        return ops.moe_unpermutation(permutation, restore_shape=handle.restore_shape)
