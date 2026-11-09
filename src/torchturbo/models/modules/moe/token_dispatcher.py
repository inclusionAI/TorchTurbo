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
from torchturbo.ops.moe import HybridEPHandle, MoEPermutation
from torchturbo.runtime import get_device_runtime

from ..comm.comm import all_to_all, gather_from_sequence_parallel_region
from .moe_config import MoEConfig
from .utils import RoutingResult, maybe_move_tensor_to_cpu
from .virtual_expert_load_balancer import VirtualExpertLoadBalancer


class MoETokenDispatcher(ABC):
    """Separate source layout, EP communication, and local expert layout."""

    def __init__(self, config: MoEConfig, ep_group: dist.ProcessGroup) -> None:
        self.config = config
        self.ep_size = config.expert_model_parallel_size or 1
        self.ep_group = ep_group

    def get_expert_kwargs(self) -> dict[str, object]:
        """Return per-forward keyword arguments for the expert module."""
        return {}

    def release_routing_state(self) -> None:
        """Release per-forward routing tensors after combine or recompute early exit."""

    @abstractmethod
    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing: RoutingResult,
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
        self.permutation_row_indices: torch.Tensor | None = None
        self.permutation_expert_indices: torch.Tensor | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self.num_global_tokens_per_local_expert: torch.Tensor | None = None
        self.input_splits: torch.Tensor | np.ndarray | None = None
        self.output_splits: torch.Tensor | np.ndarray | None = None

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Compute device counts and split sizes without copying them to CPU."""
        # Source counts are ordered by destination rank, then expert.
        local_counts = routing_map.sum(dim=0).long()
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
        self, hidden_states: torch.Tensor, routing: RoutingResult,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group source tokens and probabilities for weighting inside the experts."""
        probs = routing.probs
        routing_map = routing.routing_map
        self.hidden_shape = hidden_states.shape
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        self.tokens_per_expert = self.preprocess(routing_map)
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.hidden_shape_before_permute = hidden_states.shape
        self._maybe_dtoh_and_synchronize("before_permutation_1")
        permutation = ops.moe_permutation(
            hidden_states, routing_map, probs=probs,
            num_out_tokens=routing_map.size(0) * self.config.moe_router_topk,
            expert_indices=routing.expert_indices,
        )
        # Retain only inverse metadata across expert communication; the source
        # packed activations and weights need not stay alive until combine.
        self.permutation_row_indices = permutation.row_indices
        self.permutation_expert_indices = permutation.expert_indices
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
        permutation = MoEPermutation(
            hidden_states, None,
            self.permutation_row_indices, self.permutation_expert_indices,
        )
        output = ops.moe_unpermutation(
            permutation, restore_shape=self.hidden_shape_before_permute,
        )
        output = output.view(self.hidden_shape)
        self.release_routing_state()
        return output

    def release_routing_state(self) -> None:
        self.permutation_row_indices = None
        self.permutation_expert_indices = None
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
        expert_indices: torch.Tensor | None = None,
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
    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Get the number of tokens per expert."""
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

    def release_routing_state(self) -> None:
        """Release per-forward routing tensors after combine or recompute early exit."""


class _HybridEPManager(_DispatchManager):
    """Manager class for token distribution using HybridEP backend."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        router_topk: int,
        config: MoEConfig,
    ) -> None:
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.config = config

        self.routing_map: torch.Tensor | None = None
        self.token_probs: torch.Tensor | None = None
        self.topk_idx: torch.Tensor | None = None
        self.dispatched_probs: torch.Tensor | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self.handle: HybridEPHandle | None = None
        self.dispatch_context: object | None = None

    def setup_metadata(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        expert_indices: torch.Tensor | None = None,
    ) -> None:
        self.routing_map = routing_map
        self.token_probs = probs
        self.topk_idx = expert_indices

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_map, self.routing_map = self.routing_map, None
        token_probs, self.token_probs = self.token_probs, None
        topk_idx, self.topk_idx = self.topk_idx, None
        dispatch_context, self.dispatch_context = self.dispatch_context, None
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
            expert_indices=topk_idx,
            dispatch_context=dispatch_context,
        )
        return dispatched_hidden

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = ops.hybrid_ep_combine(hidden_states, self.handle)
        self.release_routing_state()
        return hidden_states

    def release_routing_state(self) -> None:
        self.routing_map = None
        self.token_probs = None
        self.topk_idx = None
        self.dispatched_probs = None
        self.handle = None
        self.dispatch_context = None

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
        router_topk: int,
        config: MoEConfig,
    ) -> None:
        self.initialize_virtual_expert_load_balancer(
            group=group,
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            num_experts=num_experts,
            config=config,
        )
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
        expert_indices: torch.Tensor | None = None,
    ) -> None:
        """Consume compact router metadata and start the virtual-expert planner."""
        if (
            routing_map.dtype not in (torch.int16, torch.int32, torch.int64)
            or routing_map.ndim != 2
            or routing_map.shape[1] != self.router_topk
            or probs.shape != routing_map.shape
        ):
            raise ValueError(
                "MoonEP requires expert IDs and probabilities shaped [tokens, topk]."
            )
        top_indices = routing_map.long()
        self.token_probs = probs
        self.plan_dispatch(top_indices)

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, runtime_experts = self.prepare_virtual_expert_dispatch(
            hidden_states
        )
        super().setup_metadata(runtime_experts, self.token_probs)
        self.dispatch_context = self._plan.dispatch_context
        return super().dispatch(hidden_states)

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.prepare_virtual_expert_combine(hidden_states)
        return super().combine(hidden_states)

    def release_routing_state(self) -> None:
        super().release_routing_state()
        if self._plan is not None:
            self._plan.virtual_experts = None


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

    def get_expert_kwargs(self) -> dict[str, object]:
        if isinstance(self._comm_manager, _VirtualExpertHybridEPManager):
            return {"expert_plan": self._comm_manager._plan}
        return {}

    def release_routing_state(self) -> None:
        self._comm_manager.release_routing_state()

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing: RoutingResult,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        if self.config.moe_virtual_expert_load_balance:
            self._comm_manager.setup_metadata(
                routing.expert_indices, routing.selected_probs(),
            )
        else:
            self._comm_manager.setup_metadata(
                routing.routing_map,
                routing.probs,
                expert_indices=routing.expert_indices,
            )
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
