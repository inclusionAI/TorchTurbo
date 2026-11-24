# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# Modified by the TorchTurbo contributors.

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import (
    HybridEPHandle, HybridEPMetadata, MoEPermutation, MoEPermutationMap, get_expert_indices,
)
from torchturbo.runtime import get_device_runtime

from ..comm.comm import all_to_all, gather_from_sequence_parallel_region
from .moe_config import MoEConfig
from .utils import maybe_move_tensor_to_cpu
from .virtual_expert_load_balancer import VirtualExpertLoadBalancer


class MoETokenDispatcher(ABC):
    """Separate source layout, EP communication, and local expert layout."""

    def __init__(self, config: MoEConfig, ep_group: dist.ProcessGroup) -> None:
        self.config = config
        self.ep_size = config.expert_model_parallel_size or 1
        self.ep_group = ep_group

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the layer input unchanged."""
        return hidden_states

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
        """Return the layer output unchanged."""
        return output

    def release_routing_state(self) -> None:
        """Release per-forward routing tensors after combine or recompute early exit."""

    @abstractmethod
    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        *,
        expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group source tokens by destination expert and prepare counts."""
        raise NotImplementedError

    @abstractmethod
    def token_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Send source tokens to their expert ranks."""
        raise NotImplementedError

    @abstractmethod
    def dispatch_postprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group received tokens by local expert."""
        raise NotImplementedError

    @abstractmethod
    def combine_preprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Restore local expert outputs to the received chunk order."""
        raise NotImplementedError

    @abstractmethod
    def token_combine(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return expert outputs to their source ranks."""
        raise NotImplementedError

    @abstractmethod
    def combine_postprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sum expert contributions and restore the source-token shape."""
        raise NotImplementedError


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """Variable-size all-to-all dispatch without token dropping or TP."""

    def __init__(
        self, num_local_experts: int, local_expert_indices: list[int],
        config: MoEConfig, ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(config=config, ep_group=ep_group)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(local_expert_indices) == num_local_experts, "Invalid local expert indices"
        for i in range(len(local_expert_indices) - 1):
            assert local_expert_indices[i] == local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        self.device_runtime = get_device_runtime()
        index_device = self.device_runtime.device
        input_chunk_idxs = torch.arange(self.num_experts, device=index_device)
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Start DtoH before source permutation, then wait at the first host user.
        if self.ep_size > 1:
            self.sync_point = "before_ep_alltoall"
        else:
            self.sync_point = "before_finish"
        self.dtoh_stream = self.device_runtime.new_stream()

        self.hidden_shape: torch.Size | None = None
        self.hidden_shape_before_permute: torch.Size | None = None
        self.permutation_mapping: MoEPermutationMap | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self.num_global_tokens_per_local_expert: torch.Tensor | None = None
        self.input_splits: torch.Tensor | np.ndarray | None = None
        self.output_splits: torch.Tensor | np.ndarray | None = None

    def preprocess(
        self, routing_map: torch.Tensor | None = None, *, expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute device counts and split sizes without copying them to CPU."""
        # Source counts are ordered by destination rank, then expert.
        assignments = expert_indices if expert_indices is not None else routing_map
        if assignments is None:
            raise ValueError("Alltoall preprocessing requires a routing map or expert indices.")
        if assignments.dtype == torch.bool:
            local_counts = assignments.sum(dim=0)
        else:
            indices = assignments.flatten().long()
            local_counts = indices.new_zeros(self.num_experts).scatter_add_(
                0, indices, torch.ones_like(indices),
            )
        self.input_splits = None
        self.output_splits = None
        if self.ep_size > 1:
            self.input_splits = local_counts.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            global_counts = gather_from_sequence_parallel_region(
                local_counts, group=self.ep_group
            ).reshape(self.ep_size, self.num_experts)
            received_counts = global_counts[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            self.output_splits = received_counts.sum(axis=1)
            tokens_per_expert = received_counts.sum(dim=0)
        else:
            received_counts = local_counts.reshape(1, self.num_local_experts)
            tokens_per_expert = local_counts
        self.num_global_tokens_per_local_expert = received_counts
        return tokens_per_expert

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        *,
        expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group source tokens and probabilities for weighting inside the experts."""
        if expert_indices is None:
            expert_indices = get_expert_indices(routing_map).view(
                routing_map.shape[0], self.config.moe_router_topk,
            )
        self.hidden_shape = hidden_states.shape
        self.tokens_per_expert = self.preprocess(expert_indices)
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.hidden_shape_before_permute = hidden_states.shape
        self._maybe_dtoh_and_synchronize("before_permutation_1")
        permutation = ops.moe_permutation(
            hidden_states, routing_map, probs=probs, expert_indices=expert_indices,
        )
        # Combine needs only the permutation indices after expert communication.
        self.permutation_mapping = permutation.mapping
        self._maybe_dtoh_and_synchronize("before_ep_alltoall")
        return permutation.tokens, permutation.probs

    def token_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange expert inputs and routing probabilities."""
        hidden_states = all_to_all(
            self.ep_group, hidden_states, self.output_splits, self.input_splits,
        )
        probs = all_to_all(
            self.ep_group, probs, self.output_splits, self.input_splits,
        )
        return hidden_states, probs

    def dispatch_postprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transpose received chunks from source-rank order to expert order."""
        if self.num_local_experts > 1:
            hidden_states, probs = ops.moe_sort_chunks(
                hidden_states, self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=probs,
            )
        self._maybe_dtoh_and_synchronize("before_finish")
        return hidden_states, self.tokens_per_expert, probs

    def combine_preprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Transpose expert outputs back to source-rank chunk order."""
        if self.num_local_experts > 1:
            hidden_states, _ = ops.moe_sort_chunks(
                hidden_states, self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
            )
        return hidden_states

    def token_combine(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return chunks with the inverse of the dispatch split sizes."""
        return all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits,
        )

    def combine_postprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sum already weighted expert outputs and restore source-token order."""
        permutation = MoEPermutation(hidden_states, None, mapping=self.permutation_mapping)
        output = ops.moe_unpermutation(
            permutation, restore_shape=self.hidden_shape_before_permute,
        )
        output = output.view(self.hidden_shape)
        self.release_routing_state()
        return output

    def release_routing_state(self) -> None:
        self.permutation_mapping = None
        self.num_global_tokens_per_local_expert = None

    def _maybe_dtoh_and_synchronize(self, point: str) -> None:
        if point == "before_permutation_1":
            current_stream = self.device_runtime.current_stream()
            on_side_stream = (
                current_stream is not None
                and self.dtoh_stream is not None
                and current_stream != self.dtoh_stream
            )
            if on_side_stream:
                self.dtoh_stream.wait_stream(current_stream)
            with self.device_runtime.stream(self.dtoh_stream):
                self.tokens_per_expert = maybe_move_tensor_to_cpu(
                    self.tokens_per_expert, record_stream=on_side_stream,
                )
                self.input_splits = maybe_move_tensor_to_cpu(
                    self.input_splits, as_numpy=True, record_stream=on_side_stream,
                )
                self.output_splits = maybe_move_tensor_to_cpu(
                    self.output_splits, as_numpy=True, record_stream=on_side_stream,
                )
        if point == self.sync_point:
            self.device_runtime.synchronize_stream(self.dtoh_stream)


class _DispatchManager(ABC):
    """A manager class to handle dispatch and combine processes for MoEFlexTokenDispatcher."""

    @abstractmethod
    def setup_metadata(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        """Set up metadata of routing_map and probs."""
        raise NotImplementedError

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Permute hidden_states and dispatch according to the routing metadata."""
        raise NotImplementedError

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the permuted hidden states and probabilities by local experts."""
        raise NotImplementedError

    @abstractmethod
    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Restore the hidden states by local experts."""
        raise NotImplementedError

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine and unpermute the hidden states after expert processing."""
        raise NotImplementedError

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the layer input unchanged."""
        return hidden_states

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
        """Return the layer output unchanged."""
        return output

    def release_routing_state(self) -> None:
        """Release per-forward routing tensors after combine or recompute early exit."""


class _HybridEPManager(_DispatchManager):
    """Manager class for token distribution using HybridEP backend."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        config: MoEConfig,
        router_topk: int | None = None,
    ) -> None:
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.router_topk = router_topk if router_topk is not None else config.moe_router_topk
        self.config = config

        self.routing_map: torch.Tensor | None = None
        self.token_probs: torch.Tensor | None = None
        self.dispatched_probs: torch.Tensor | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self.handle: HybridEPHandle | None = None

    def setup_metadata(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        if padding_mask is not None:
            if padding_mask.numel() != routing_map.shape[0]:
                raise ValueError("Flex padding_mask must contain one entry per token.")
            padding = padding_mask.reshape(-1, 1).bool()
            routing_map = routing_map.masked_fill(padding, -1)
            probs = probs.masked_fill(padding, 0)
        self.routing_map = routing_map
        self.token_probs = probs

    def dispatch(
        self, hidden_states: torch.Tensor, *,
        dispatch_context: HybridEPMetadata | None = None,
    ) -> torch.Tensor:
        routing_map, self.routing_map = self.routing_map, None
        token_probs, self.token_probs = self.token_probs, None
        (
            dispatched_hidden,
            self.dispatched_probs,
            self.tokens_per_expert,
            self.handle,
        ) = ops.hybrid_ep_dispatch(
            hidden_states,
            routing_map,
            token_probs,
            self.group,
            self.num_local_experts,
            self.router_topk,
            dispatch_context=dispatch_context,
        )
        return dispatched_hidden

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = ops.hybrid_ep_combine(hidden_states, self.handle)
        # Layer finalization still needs the virtual-expert execution state.
        _HybridEPManager.release_routing_state(self)
        return hidden_states

    def release_routing_state(self) -> None:
        self.routing_map = None
        self.token_probs = None
        self.dispatched_probs = None
        self.handle = None

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert


class _VirtualExpertHybridEPManager(VirtualExpertLoadBalancer, _HybridEPManager):
    """Glue virtual-expert load balancing onto the HybridEP transport."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        config: MoEConfig,
        router_topk: int | None = None,
    ) -> None:
        router_topk = router_topk if router_topk is not None else config.moe_router_topk
        self.initialize_virtual_expert_load_balancer(
            group=group,
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            num_experts=num_experts,
            config=config,
        )
        if self.ep_size == 1:
            raise ValueError("Virtual expert load balancing requires EP >= 2.")
        super().__init__(
            group=group,
            num_local_experts=self.num_runtime_experts,
            num_experts=self.ep_size * self.num_runtime_experts,
            router_topk=router_topk,
            config=config,
        )

    def setup_metadata(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        """Mask batch-major compact probabilities and start the virtual-expert planner."""
        if (
            routing_map.dtype not in (torch.int16, torch.int32, torch.int64)
            or routing_map.ndim != 2
            or routing_map.shape[1] != self.router_topk
            or probs.shape != routing_map.shape
        ):
            raise ValueError(
                "MoonEP requires expert IDs and probabilities shaped [tokens, topk]."
            )
        if padding_mask is not None:
            if padding_mask.numel() != routing_map.shape[0]:
                raise ValueError("MoonEP padding_mask must contain one entry per token.")
            probs = probs.masked_fill(padding_mask.reshape(-1, 1).bool(), 0)
        top_indices = routing_map.long()
        self.token_probs = probs
        self.plan_dispatch(top_indices)

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, runtime_experts = self.prepare_virtual_expert_dispatch(
            hidden_states
        )
        super().setup_metadata(runtime_experts, self.token_probs)
        return super().dispatch(
            hidden_states, dispatch_context=self._plan.routing.dispatch_context,
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.prepare_virtual_expert_combine(hidden_states)
        return super().combine(hidden_states)

    def release_routing_state(self) -> None:
        super().release_routing_state()
        if self._plan is not None:
            self._plan.routing.virtual_experts = None
        self._release_virtual_experts()


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """Flexible token dispatcher delegating communication to _DispatchManager."""

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(config=config, ep_group=ep_group)
        assert config.num_moe_experts is not None
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.hidden_shape: torch.Size | None = None
        manager_cls = (
            _VirtualExpertHybridEPManager
            if self.config.moe_virtual_expert_load_balance
            else _HybridEPManager
        )
        self._comm_manager: _DispatchManager = manager_cls(
            group=ep_group,
            num_local_experts=num_local_experts,
            num_experts=config.num_moe_experts,
            router_topk=config.moe_router_topk,
            config=config,
        )

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Wrap the layer input through the communication manager."""
        return self._comm_manager.wrap_layer_input(hidden_states)

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
        """Finalize the layer output through the communication manager."""
        return self._comm_manager.finalize_layer_output(output)

    def release_routing_state(self) -> None:
        self._comm_manager.release_routing_state()

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        *,
        expert_indices: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        if routing_map.dtype == torch.bool:
            if expert_indices is None:
                expert_indices = get_expert_indices(routing_map).view(
                    routing_map.shape[0], self.config.moe_router_topk,
                )
            routing_map = expert_indices.long()
            probs = probs.gather(1, routing_map)
        self._comm_manager.setup_metadata(routing_map, probs, padding_mask=padding_mask)
        return hidden_states, self._comm_manager.token_probs

    def token_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dispatched_hidden = self._comm_manager.dispatch(hidden_states)
        return dispatched_hidden, self._comm_manager.dispatched_probs

    def dispatch_postprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        self._comm_manager.dispatched_probs = None
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)

    def token_combine(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._comm_manager.combine(hidden_states)

    def combine_postprocess(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states.view(self.hidden_shape)
