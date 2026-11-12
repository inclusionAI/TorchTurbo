  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

import functools
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    return functools.partial(torch.nn.init.normal_, mean=0.0, std=sigma)


@dataclass
class MoEConfig:
    # model architecture
    hidden_size: int = 0
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    num_moe_experts: Optional[int] = None
    # expert parallel
    expert_model_parallel_size: int = 1
    # training
    params_dtype: torch.dtype = torch.float32
    # initilization
    perform_initialization: bool = True
    init_method: Optional[Callable] = None
    init_method_std: float = 0.02
    # MoE related
    """Shared expert total ffn hidden size."""
    moe_shared_expert_intermediate_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""
    moe_ffn_hidden_size: Optional[int] = None
    """The load balancing strategy for the router."""
    moe_router_load_balancing_type: str = "none"
    """Token-level auxiliary loss coefficient."""
    moe_aux_loss_coeff: float = 0.0
    """Z-loss coefficient for router logits; 0.0 disables."""
    moe_z_loss_coeff: float = 0.0
    """Number of experts to route to for each token."""
    moe_router_topk: int = 2
    """Number of groups to divide experts into for group-limited routing."""
    moe_router_num_groups: Optional[int] = None
    """Number of selected groups for group-limited routing."""
    moe_router_group_topk: Optional[int] = None
    """Apply softmax before top-k; otherwise apply it to the selected logits."""
    moe_router_pre_softmax: bool = False
    """Scaling factor applied after top-k selection and any probability normalization."""
    moe_router_topk_scaling_factor: Optional[float] = None
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""
    moe_router_score_function: str = "softmax"
    moe_router_dtype: Optional[str] = None
    moe_router_enable_expert_bias: bool = False
    """The type of token dispatcher to use."""
    moe_token_dispatcher_type: str = "alltoall"
    """Apply probabilities to expert inputs instead of activations before FC2."""
    moe_apply_probs_on_input: bool = False
    """Per-layer swiglu limit for clamping gate/up values in SwiGLU activation."""
    moe_swiglu_limit: Optional[List[float]] = None
    """Cap the activated gate, min(act(gate), limit), instead of the gate
    pre-activation when applying moe_swiglu_limit."""
    moe_swiglu_clamp_gate_after_activation: bool = False
    """Number of layers using token-id hash routing."""
    num_hash_layers: int = 0
    """Vocabulary size for token-id hash routing table."""
    vocab_size: int = 0

    def __post_init__(self):
        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)
        if (
            self.moe_router_enable_expert_bias
            and self.moe_router_score_function == "softmax"
            and not self.moe_router_pre_softmax
        ):
            raise ValueError(
                "Softmax expert bias requires moe_router_pre_softmax=True; "
                "post-top-k softmax would ignore the bias."
            )
