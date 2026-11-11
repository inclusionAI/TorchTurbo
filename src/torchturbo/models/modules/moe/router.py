from abc import ABC
from typing import Optional

import torch

from .moe_config import MoEConfig
from .utils import topk_softmax_with_capacity


class Router(ABC, torch.nn.Module):
    """Base Router class"""

    def __init__(self, config: MoEConfig) -> None:
        """Initialize the Router module."""
        super().__init__()
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.config.num_moe_experts, self.config.hidden_size),
                dtype=torch.float32,
            )
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == "cpu":
            self.weight.data = self.weight.data.to(device=input.device)
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == "fp64":
            router_dtype = torch.float64
        # HACK: avoid unexpected dtype conversion
        logits = torch.nn.functional.linear(
            input.to(router_dtype), self.weight.to(router_dtype)
        ).to(router_dtype)
        return logits


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: MoEConfig, layer_number: Optional[int] = None) -> None:
        """Initialize the zero token dropping router."""
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.layer_number = layer_number
        self.use_hash_routing = self._is_hash_routing_layer()

        self.enable_expert_bias = (
            self.config.moe_router_enable_expert_bias and not self.use_hash_routing
        )
        if self.enable_expert_bias:
            self.register_buffer(
                "expert_bias",
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
            )
        else:
            self.expert_bias = None

        if self.use_hash_routing:
            self.register_buffer(
                "tid2eid",
                torch.stack(
                    [
                        torch.randperm(self.config.num_moe_experts)[: self.topk]
                        for _ in range(self.config.vocab_size)
                    ]
                ).long(),
            )

    def _is_hash_routing_layer(self):
        return (
            self.layer_number is not None
            and self.layer_number < self.config.num_hash_layers
            and self.config.vocab_size > 0
        )

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, "expert_bias") and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def routing(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        input_ids = input_ids.view(-1) if input_ids is not None else None
        if self.use_hash_routing and input_ids is None:
            raise ValueError("Hash routing requires input_ids.")

        if self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                use_hash_routing=self.use_hash_routing,
                hash_table=getattr(self, "tid2eid", None),
                input_ids=input_ids,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                tokens_per_expert = routing_map.sum(dim=0).to(torch.float32)
                if not hasattr(self, "local_tokens_per_expert"):
                    self.register_buffer(
                        "local_tokens_per_expert",
                        torch.zeros_like(tokens_per_expert),
                        persistent=False,
                    )
                self.local_tokens_per_expert.add_(tokens_per_expert)

        return scores, routing_map

    def forward(self, input: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        logits = self.gating(input)

        scores, routing_map = self.routing(logits, input_ids)

        return scores, routing_map, logits
