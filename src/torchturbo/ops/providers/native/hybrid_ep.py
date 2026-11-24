"""Native fallback for HybridEP token dispatch and combine."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...base import OpImpl, OpImplKind
from ...moe import (
    HybridEPHandle, HybridEPMetadata, MoEPermutation, ep_all_to_all, get_expert_indices,
)
from ...registry import register_op
from .moe_permutation import TorchMoEMaskPermutation, TorchMoEMaskUnpermutation, TorchMoESortChunks


@register_op(
    op_name="hybrid_ep_dispatch",
    impl_name="torch_hybrid_ep_dispatch",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchHybridEPDispatch(OpImpl):
    _permute = TorchMoEMaskPermutation()
    _sort_chunks = TorchMoESortChunks()
    _chunk_metadata_on_device = False

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
        dispatch_context: HybridEPMetadata | None = None,
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
            expert_indices = expert_indices.long()
            probs = probs.gather(1, expert_indices)
        else:
            if probs.shape != routing_map.shape:
                raise ValueError("Compact route IDs and probabilities must have the same shape.")
            expert_indices = routing_map.long()

        if expert_indices.shape != (hidden_states.shape[0], router_topk):
            raise ValueError("Expert indices must have shape [num_tokens, router_topk].")

        if planned_metadata is not None:
            tokens_per_expert = planned_metadata.tokens_per_expert
            input_splits = planned_metadata.input_splits if ep_size > 1 else None
            output_splits = planned_metadata.output_splits if ep_size > 1 else None
            num_global_tokens_per_local_expert = planned_metadata.received_counts
        else:
            # Bucket zero counts padding (-1). Only real experts enter EP splits.
            ids = expert_indices.flatten()
            counts = ids.new_zeros(num_experts + 1)
            counts.scatter_add_(0, ids + 1, torch.ones_like(ids))
            num_local_tokens_per_expert = counts[1:]
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
            input_splits_tensor = (
                num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)
            )
            host_metadata = torch.cat(
                (input_splits_tensor, num_global_tokens_per_local_expert.flatten())
            ).to("cpu")
            input_splits = host_metadata[:ep_size].tolist()
            received_counts_cpu = host_metadata[ep_size:].view(
                ep_size, num_local_experts,
            )
            tokens_per_expert = received_counts_cpu.sum(dim=0)
            output_splits = received_counts_cpu.sum(dim=1).tolist()
            if not self._chunk_metadata_on_device:
                num_global_tokens_per_local_expert = received_counts_cpu
        elif planned_metadata is None:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.unsqueeze(0)
            tokens_per_expert = num_local_tokens_per_expert.to("cpu")
            input_splits = None
            output_splits = None

        num_assignments = expert_indices.numel()
        num_valid = num_assignments
        if planned_metadata is None:
            num_valid = sum(input_splits) if input_splits is not None else int(tokens_per_expert.sum())
        if num_valid != num_assignments:
            # A sentinel after the last real expert keeps padded slots out of EP.
            expert_indices = expert_indices.masked_fill(expert_indices == -1, num_experts)
        permutation = self._permute(
            hidden_states, expert_indices=expert_indices, probs=probs, num_out_tokens=num_valid,
        )
        dispatched_hidden = ep_all_to_all(
            ep_group, permutation.tokens, output_splits, input_splits,
        )
        dispatched_probs = ep_all_to_all(
            ep_group, permutation.probs, output_splits, input_splits,
        )
        restore_output_by_local_experts = None
        if ep_size > 1 and num_local_experts > 1:
            if self._chunk_metadata_on_device:
                num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(
                    hidden_states.device, non_blocking=True,
                )
            chunk_indices = torch.arange(
                ep_size * num_local_experts, device=num_global_tokens_per_local_expert.device,
            ).reshape(ep_size, num_local_experts)
            sort_input_by_local_experts = chunk_indices.T.ravel()
            restore_output_by_local_experts = chunk_indices.reshape(
                num_local_experts, ep_size,
            ).T.ravel()
            dispatched_hidden, dispatched_probs = self._sort_chunks(
                dispatched_hidden,
                num_global_tokens_per_local_expert.ravel(),
                sort_input_by_local_experts,
                probs=dispatched_probs,
            )

        handle = HybridEPHandle(
            mapping=permutation.mapping,
            restore_shape=hidden_states.shape,
            ep_group=ep_group,
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


@register_op(
    op_name="hybrid_ep_combine",
    impl_name="torch_hybrid_ep_combine",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchHybridEPCombine(OpImpl):
    _unpermute = TorchMoEMaskUnpermutation()
    _sort_chunks = TorchMoESortChunks()

    def __call__(
        self,
        hidden_states: torch.Tensor,
        handle: HybridEPHandle,
    ) -> torch.Tensor:
        """Restore chunk order, return across EP, and unpermute to source rows."""
        if handle.restore_output_by_local_experts is not None:
            hidden_states, _ = self._sort_chunks(
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
        permutation = MoEPermutation(combined, None, mapping=handle.mapping)
        return self._unpermute(permutation, handle.restore_shape)
