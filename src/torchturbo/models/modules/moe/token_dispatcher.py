  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from torchturbo.ops import ops
from torchturbo.runtime import TurboDevice, get_device_runtime

from .moe_config import MoEConfig

# from torch_turbo.distributed.parallel_state import get_expert_model_parallel_group
from ..comm.comm import all_to_all, gather_from_sequence_parallel_region
from .utils import (
    RoutingResult,
    permute,
    unpermute,
    sort_chunks_by_idxs,
    maybe_move_tensor_to_cpu,
)


class MoETokenDispatcher:
    """MoE Token Dispatcher"""

    def __init__(self, config: MoEConfig, ep_group) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config

        # HACK: not support tp yet, always 1
        self.tp_size = 1
        self.ep_size = config.expert_model_parallel_size or 1
        self._ep_group = ep_group

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        # return get_expert_model_parallel_group()
        return self._ep_group

    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, routing: RoutingResult
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            routing (RoutingResult): Local routing weights, mask, and expert IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted probs of each token produced by the router.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: torch.Tensor, bias: torch.Tensor = None
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            bias (torch.Tensor): The bias tensor.

        Returns:
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: MoEConfig,
        ep_group,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (MoEConfig): Configuration for the transformer model.
        """
        super().__init__(config=config, ep_group=ep_group)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cpu")
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.capacity = None

        # A device stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called at
        # `self.dtoh_point`. The synchronization happens at `self.sync_point`, which is
        # decided based on the MoE and parallel settings. Valid points are "before_permutation_1",
        # "before_ep_alltoall", "before_permutation_2", "before_finish", and "no_sync".
        self.sync_point = "no_sync"
        self.sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.dtoh_point = "before_permutation_1"
        self.device_runtime = get_device_runtime()
        self.dtoh_stream = self.device_runtime.new_stream()

    @property
    def deferred_probs_to_unpermutation(self) -> bool:
        # This currently reflects the availability of the fused NPU path; the
        # flow itself is not conceptually NPU-only.
        return self.device_runtime.turbo_device is TurboDevice.NPU

    def _token_permutation_with_deferred_probs(
        self,
        hidden_states: torch.Tensor,
        routing: RoutingResult,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = routing.probs
        routing_map = routing.routing_map
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
        ) = ops.moe_token_permutation(hidden_states, routing.expert_indices)

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", tokens_per_expert
        )
        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", tokens_per_expert
        )
        if self.num_local_experts > 1:
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=None,
                fused=self.config.moe_permute_fusion,
            )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_finish", tokens_per_expert
        )

        return global_input_tokens, tokens_per_expert, probs

    def _token_unpermutation_with_deferred_probs(
        self,
        hidden_states: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        if probs is None:
            raise RuntimeError("Deferred probs path requires probs for unpermutation.")

        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )
        output = ops.moe_token_unpermutation(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=probs.to(torch.bfloat16),
        )
        output = output.view(self.hidden_shape)

        self.probs = None
        self.routing_map = None

        return output, None

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.dtoh_stream` at `self.dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """

        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        if self.ep_size > 1 or self.tp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            # HACK: tp_rank always 0
            self.output_splits = num_global_tokens_per_rank[0]
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
                dim=(0, 1)
            )

            # A synchronization is needed before expert parallel AlltoAll communication
            # to get the `input_splits` and `output_splits` CPU values.
            self._maybe_update_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

            # A synchronization is needed before the returns
            # to get the `num_tokens_per_local_expert` CPU value.
            self._maybe_update_sync_point("before_finish")

        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = (
                num_global_tokens_per_local_expert.view(-1, self.num_local_experts)
            )
            if not self.config.moe_permute_fusion:
                # A synchronization is needed before permutation 2
                # to get the `num_global_tokens_per_local_expert` CPU value.
                self._maybe_update_sync_point("before_permutation_2")

        assert (
            self.sync_point_priority[self.dtoh_point]
            <= self.sync_point_priority[self.sync_point]
        ), "sync_point must be after dtoh_point."
        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        routing: RoutingResult,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            routing (RoutingResult): Local routing weights, mask, and expert IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted probs of each token produced by the router.
        """
        if self.deferred_probs_to_unpermutation:
            return self._token_permutation_with_deferred_probs(
                hidden_states, routing
            )

        # Preprocess: Get the metadata for communication, permutation and computation operations.
        probs = routing.probs
        routing_map = routing.routing_map
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        # Permutation 1: input to AlltoAll input
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
        )

        # Perform expert parallel AlltoAll communication
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", tokens_per_expert
        )
        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs, self.output_splits, self.input_splits
        )

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", tokens_per_expert
        )
        if self.num_local_experts > 1:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
                fused=self.config.moe_permute_fusion,
            )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_finish", tokens_per_expert
        )

        return global_input_tokens, tokens_per_expert, global_probs

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        if self.deferred_probs_to_unpermutation:
            return self._token_unpermutation_with_deferred_probs(
                hidden_states, bias=bias, probs=probs
            )

        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        return output, None

    def _maybe_update_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.sync_point_priority[point]
            < self.sync_point_priority[self.sync_point]
        ):
            self.sync_point = point

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if point == self.dtoh_point:
            # Move all possible GPU tensors to CPU at self.dtoh_point.
            current_stream = self.device_runtime.current_stream()
            on_side_stream = (
                current_stream is not None
                and self.dtoh_stream is not None
                and current_stream != self.dtoh_stream
            )
            if on_side_stream:
                self.dtoh_stream.wait_stream(current_stream)
            with self.device_runtime.stream(self.dtoh_stream):
                # TODO: use MemcpyBatchAsync instead.
                tokens_per_expert = maybe_move_tensor_to_cpu(
                    tokens_per_expert, record_stream=on_side_stream
                )
                self.input_splits = maybe_move_tensor_to_cpu(
                    self.input_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits = maybe_move_tensor_to_cpu(
                    self.output_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits_tp = maybe_move_tensor_to_cpu(
                    self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                )
                self.num_out_tokens = maybe_move_tensor_to_cpu(
                    self.num_out_tokens, record_stream=on_side_stream
                )
                if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                    self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                        self.num_global_tokens_per_local_expert,
                        record_stream=on_side_stream,
                    )

        if point == self.sync_point:
            # Synchronize with the dtoh stream at self.sync_point.
            self.device_runtime.synchronize_stream(self.dtoh_stream)

        return tokens_per_expert
