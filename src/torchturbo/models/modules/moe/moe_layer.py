  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional, Union
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .moe_config import MoEConfig
from .router import TopKRouter
from .token_dispatcher import MoETokenDispatcher, MoEAlltoAllTokenDispatcher
from .module import ModuleSpec, build_module
from torchturbo.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


@dataclass
class MoERuntimeInputs:
    padding_mask: torch.Tensor | None = None


class BaseMoELayer(torch.nn.Module, ABC):
    """Base class for a mixture of experts layer."""

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        layer_number: Optional[int] = None,
    ):
        super().__init__()

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)

        self.config = config
        self.expert_parallel_size = ep_size
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = (
            self.config.num_moe_experts // self.expert_parallel_size
        )
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)
        )
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number
        self._runtime_inputs = MoERuntimeInputs()

    def set_runtime_inputs(self, inputs: MoERuntimeInputs) -> None:
        self._runtime_inputs = inputs

    @abstractmethod
    def forward(self, hidden_states, input_ids=None):
        """Forward method for the MoE layer."""
        pass


class MoELayer(BaseMoELayer):
    """
    Mixture of experts Layer **currently only supports no token dropping**.
    """

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        submodules: Optional[MoESubmodules] = None,
        use_fused_permute_unpermute: Optional[bool] = False,
        use_te: Optional[bool] = False,
        layer_number: Optional[int] = None,
        sp_group=None,
    ):
        self.use_fused_permute_unpermute = use_fused_permute_unpermute
        self.use_te = use_te
        self.submodules = submodules
        super(MoELayer, self).__init__(
            config=config, ep_group=ep_group, layer_number=layer_number
        )

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            layer_number=self.layer_number,
            sp_group=sp_group,
        )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                ep_group=ep_group,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.use_te,
            self.config,
            layer_number=self.layer_number,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config
            )

    def forward(self, hidden_states: torch.Tensor, input_ids=None):
        # debug
        rank = torch.distributed.get_rank()

        # process MoE
        def custom_forward(hidden_states, input_ids=None):
            hidden_states_dtype = hidden_states.dtype
            routing, logits = self.router(
                hidden_states, input_ids, self._runtime_inputs.padding_mask
            )
            if self.config.moe_router_score_function == "softmax":
                routing = replace(
                    routing,
                    probs=routing.probs / routing.probs.sum(dim=-1, keepdim=True),
                )
            logits = logits.view(-1, logits.shape[-1])
            (dispatched_input, tokens_per_expert, permuted_probs) = (
                self.token_dispatcher.token_permutation(
                    hidden_states, routing
                )
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )

            if self.token_dispatcher.deferred_probs_to_unpermutation:
                probs_for_unpermutation = routing.selected_probs()
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias, probs=probs_for_unpermutation
                )
            else:
                expert_output = expert_output * permuted_probs.unsqueeze(-1)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias
                )
            output = output.to(hidden_states_dtype)

            if hasattr(self, "shared_expert") and self.shared_expert is not None:
                shared_expert_output = self.shared_expert(hidden_states)
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states))
                    * shared_expert_output
                )
                output = output + shared_expert_output

            if hasattr(self, "shared_experts") and self.shared_experts is not None:
                shared_expert_output = self.shared_experts(hidden_states)
                output = output + shared_expert_output

            if rank == 0 and int(os.environ.get('TORCH_TURBO_DEBUG', '0')):
                logger.debug_rank0(
                    "hidden_states.shape %s hidden_states %s",
                    hidden_states.shape, hidden_states,
                )
                logger.debug_rank0(
                    "dispatched_input.shape %s dispatched_input %s",
                    dispatched_input.shape, dispatched_input,
                )
                logger.debug_rank0(
                    "tokens_per_expert.shape %s tokens_per_expert %s",
                    tokens_per_expert.shape, tokens_per_expert,
                )
                logger.debug_rank0(
                    "permuted_probs.shape %s permuted_probs %s",
                    permuted_probs.shape, permuted_probs,
                )
                logger.debug_rank0(
                    "before unpermutation: expert_output.shape %s expert_output %s",
                    expert_output.shape, expert_output,
                )
                logger.debug_rank0(
                    "after unpermutation: expert_output.shape %s expert_output %s",
                    output.shape, output,
                )

            return output, mlp_bias, logits

        output, mlp_bias, logits = custom_forward(hidden_states, input_ids)
        return output, logits
