  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import ABC
from dataclasses import replace
from typing import Optional

import torch
import torch.distributed as dist

from .moe_config import MoEConfig
from .moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
    mask_padding_tokens,
    z_loss_func,
)
from .utils import RoutingResult, topk_softmax_with_capacity


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

    def __init__(
        self,
        config: MoEConfig,
        layer_number: Optional[int] = None,
        sp_group=None,
    ) -> None:
        """Initialize the zero token dropping router."""
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.layer_number = layer_number
        self.sp_group = sp_group
        self._metrics = None
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

    def _token_aux_loss(
        self,
        scores: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        probability_sum = scores.sum(dim=0)
        sp_size = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        # FSDP averages replicated Router gradients across SP ranks.
        return (
            torch.sum(probability_sum * tokens_per_expert)
            * self.num_experts
            * sp_size
            / (self.topk * total_num_tokens * total_num_tokens)
        )

    def _apply_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map_for_aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        aux_loss_coeff = self.config.moe_aux_loss_coeff
        if aux_loss_coeff == 0.0:
            return probs

        tokens_per_expert = routing_map_for_aux_loss.sum(dim=0)
        sp_size = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        if sp_size > 1:
            dist.all_reduce(
                tokens_per_expert, op=dist.ReduceOp.SUM, group=self.sp_group
            )
        # A fully padded batch routes no tokens. The floor keeps its aux loss
        # and gradient at zero instead of 0 / 0, as in the z-loss path.
        total_num_tokens = (tokens_per_expert.sum() / self.topk).clamp_min(1)

        raw_aux_loss = self._token_aux_loss(
            scores_for_aux_loss,
            tokens_per_expert,
            total_num_tokens,
        )
        return self.attach_and_log_load_balancing_loss(
            probs,
            aux_loss_coeff,
            raw_aux_loss * aux_loss_coeff,
        )

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Attach the scaled AUX loss and record its metrics."""
        self._metrics = {
            **(self._metrics or {}),
            "aux_loss": (aux_loss / aux_loss_coeff).detach(),
            "aux_loss_scaled": aux_loss.detach(),
        }
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)

    def apply_z_loss(
        self,
        logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply router z-loss through the MoE loss autoscaler."""
        z_loss_coeff = self.config.moe_z_loss_coeff
        if (
            z_loss_coeff is None
            or z_loss_coeff == 0.0
            or not (self.training and torch.is_grad_enabled())
        ):
            return logits

        if padding_mask is not None:
            valid_mask = ~padding_mask.reshape(-1).bool()
            local_num_tokens = valid_mask.sum()
        else:
            local_num_tokens = torch.tensor(
                logits.shape[0], device=logits.device, dtype=torch.long
            )
        total_num_tokens = local_num_tokens.clone()

        sp_size = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        if sp_size > 1:
            dist.all_reduce(
                total_num_tokens,
                op=dist.ReduceOp.SUM,
                group=self.sp_group,
            )

        local_z_loss = z_loss_func(
            logits,
            z_loss_coeff,
            padding_mask=padding_mask,
        )
        # Convert the local mean to the global SP/FSDP-normalized contribution.
        global_token_denominator = torch.clamp(total_num_tokens, min=1)
        local_to_global_scale = sp_size * local_num_tokens / global_token_denominator
        z_loss = local_z_loss * local_to_global_scale
        self._metrics = {
            **(self._metrics or {}),
            "z_loss": (z_loss / z_loss_coeff).detach(),
            "z_loss_scaled": z_loss.detach(),
        }
        return MoEAuxLossAutoScaler.apply(logits, z_loss)

    def pop_metrics(self) -> dict[str, torch.Tensor]:
        metrics = self._metrics or {}
        self._metrics = None
        return metrics

    def routing(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        padding_mask: torch.Tensor | None = None,
    ) -> RoutingResult:
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            RoutingResult: Final weights, assignment mask, and sorted expert IDs.
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        input_ids = input_ids.view(-1) if input_ids is not None else None
        if self.use_hash_routing and input_ids is None:
            raise ValueError("Hash routing requires input_ids.")

        logits = self.apply_z_loss(logits, padding_mask=padding_mask)

        if self.routing_type == "none":
            # A naive top-k routing without load balancing
            routing = topk_softmax_with_capacity(
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

        if (
            self.training
            and torch.is_grad_enabled()
            and self.config.moe_aux_loss_coeff > 0.0
        ):
            aux_routing_map, aux_scores = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                padding_mask,
            )
            routing = replace(
                routing,
                probs=self._apply_aux_loss(routing.probs, aux_scores, aux_routing_map),
            )

        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                bias_routing_map = mask_padding_tokens(
                    routing.routing_map, padding_mask
                )
                tokens_per_expert = bias_routing_map.sum(dim=0).to(torch.float32)
                if not hasattr(self, "local_tokens_per_expert"):
                    self.register_buffer(
                        "local_tokens_per_expert",
                        torch.zeros_like(tokens_per_expert),
                        persistent=False,
                    )
                self.local_tokens_per_expert.add_(tokens_per_expert)

        return routing

    def forward(
        self,
        input: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[RoutingResult, torch.Tensor]:
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        logits = self.gating(input)

        routing = self.routing(logits, input_ids, padding_mask)

        return routing, logits
