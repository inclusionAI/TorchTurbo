  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchturbo.ops.moe import MoEMegakernel, RouteFormat

from .moe_config import MoEConfig
from .router import TopKRouter
from .token_dispatcher import MoETokenDispatcher, MoEAlltoAllTokenDispatcher
from .module import ModuleSpec, build_module


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


@dataclass
class MoERuntimeInputs:
    padding_mask: torch.Tensor | None = None
    # False while recomputation replays a forward that already counted its tokens.
    accumulate_tokens_per_expert: bool = True


class BaseMoELayer(torch.nn.Module, ABC):
    """Base class for a mixture of experts layer."""

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)

        self.config = config
        self.ep_group = ep_group
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
        self.shared_expert = None
        self.shared_expert_gate = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number
        self._runtime_inputs = MoERuntimeInputs()

    def set_runtime_inputs(self, inputs: MoERuntimeInputs) -> None:
        self._runtime_inputs = inputs

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        use_te: bool = False,
        layer_number: Optional[int] = None,
        sp_group: dist.ProcessGroup | None = None,
    ) -> None:
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

        self._megakernel: MoEMegakernel | None = None

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

    def route(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        runtime = self._runtime_inputs
        router = self.router
        compact = getattr(self._megakernel, "route_format", None) is RouteFormat.TOPK
        # Instance-level router wrappers may still use the original four-argument signature.
        if compact and type(router) is TopKRouter and vars(router).keys().isdisjoint(("forward", "routing")):
            return router(hidden_states, input_ids, runtime.padding_mask,
                          runtime.accumulate_tokens_per_expert, compact=True)
        weights, mask, ids, logits = router(
            hidden_states, input_ids, runtime.padding_mask, runtime.accumulate_tokens_per_expert,
        )
        if not compact:
            return weights, mask, ids, logits
        if ids is None or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Top-k routing requires the custom router's original integer expert IDs.")
        rows = hidden_states.numel() // hidden_states.size(-1)
        if weights.shape != (rows, self.config.num_moe_experts) or ids.shape != (rows, self.config.moe_router_topk):
            raise ValueError("Expected custom router weights [tokens, experts] and expert IDs [tokens, topk].")
        return weights.gather(1, ids.long()), None, ids, logits

    def _forward_native_routed(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs, routing_map, expert_indices, logits = self.route(hidden_states, input_ids)
        dispatcher = self.token_dispatcher
        # Reuse stage variables so completed layouts are not kept alive here.
        expert_tokens, expert_probs = dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs, expert_indices=expert_indices,
        )
        expert_tokens, expert_probs = dispatcher.token_dispatch(expert_tokens, expert_probs)
        expert_tokens, tokens_per_expert, expert_probs = dispatcher.dispatch_postprocess(
            expert_tokens, expert_probs,
        )
        expert_tokens, bias = self.experts(expert_tokens, tokens_per_expert, expert_probs)
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        expert_tokens = dispatcher.combine_preprocess(expert_tokens)
        expert_tokens = dispatcher.token_combine(expert_tokens)
        expert_tokens = dispatcher.combine_postprocess(expert_tokens)
        output = expert_tokens.to(hidden_states.dtype)
        return output, logits

    def add_shared_experts(self, hidden_states: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Add this layer's shared expert contributions to the routed output."""
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            output = output + shared_expert_output
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._megakernel is None:
            output, logits = self._forward_native_routed(hidden_states, input_ids)
            output = self.add_shared_experts(hidden_states, output)
        else:
            output, logits = self._megakernel.forward_experts(self, hidden_states, input_ids)
        return output, logits.view(-1, logits.shape[-1])
