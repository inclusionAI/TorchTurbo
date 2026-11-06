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
from torchturbo.ops.moe import MoEPermutation
from torchturbo.runtime import get_device_runtime

from ..comm.comm import all_to_all, gather_from_sequence_parallel_region
from .moe_config import MoEConfig
from .utils import RoutingResult, maybe_move_tensor_to_cpu


class MoETokenDispatcher(ABC):
    """Separate source layout, EP communication, and local expert layout."""

    def __init__(self, config: MoEConfig, ep_group: dist.ProcessGroup) -> None:
        self.config = config
        self.ep_size = config.expert_model_parallel_size or 1
        self.ep_group = ep_group

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
        self.permutation_row_indices = None
        self.permutation_expert_indices = None
        return output

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
