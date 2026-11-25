  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .grouped_linear import grouped_linear
from .moe_config import MoEConfig
from torchturbo.ops import ops
from torchturbo.logging import get_logger

if TYPE_CHECKING:
    from torchturbo.engine.optimizers import MuonParameterSpec
    from .virtual_expert_load_balancer import VirtualExpertLoadBalancer

logger = get_logger(__name__)

try:
    from transformer_engine.pytorch import GroupedLinear
except ImportError:
    GroupedLinear = None


def transformer_engine_is_available():
    """Check if transformer_engine is available."""
    return GroupedLinear is not None


class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        use_te: bool,
        config: MoEConfig,
        layer_number: int = None,
    ):
        super().__init__()
        if config.moe_virtual_expert_load_balance and use_te:
            raise ValueError(
                "Virtual-expert load balancing does not support TE experts; "
                "use fused-weight GroupedMLP."
            )
        self.config = config
        self._load_balancer: VirtualExpertLoadBalancer | None = None
        self.num_local_experts = num_local_experts
        self.layer_number = layer_number
        limit_list = getattr(config, "moe_swiglu_limit", None)
        if (
            limit_list is not None
            and layer_number is not None
            and 0 <= layer_number < len(limit_list)
        ):
            self.swiglu_limit = float(limit_list[layer_number])
        else:
            self.swiglu_limit = 0.0

        if self.swiglu_limit < 0:
            raise ValueError("moe_swiglu_limit must be non-negative.")

        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError(
                    "Activation function must be silu or gelu when using GroupedMLP."
                )

            def glu(x):
                if (
                    self.config.activation_func is F.silu
                    and self.swiglu_limit == 0.0
                ):
                    return ops.swiglu(x)
                return ops.gated_activation(
                    x,
                    activation_func=self.config.activation_func,
                    limit=self.swiglu_limit,
                    clamp_gate_after_activation=(
                        self.config.moe_swiglu_clamp_gate_after_activation
                    ),
                )

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        if config.gated_linear_unit:
            fc1_output_size = self.config.moe_ffn_hidden_size * 2
        else:
            fc1_output_size = self.config.moe_ffn_hidden_size
        fc1_output_size_per_partition = fc1_output_size

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = fc2_input_size

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.use_te = use_te
        if transformer_engine_is_available() and self.use_te:
            logger.debug_rank0("Using TE GroupedLinear.")
            self.fc1_linear = GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=self.config.hidden_size,
                out_features=fc1_output_size_per_partition,
                bias=False,  # bias not supported in original implementation
                params_dtype=config.params_dtype,
                device="cuda",
            )

            self.fc2_linear = GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=fc2_input_size_per_partition // self.num_local_experts,
                out_features=self.config.hidden_size,
                bias=False,  # bias not supported in original implementation
                params_dtype=config.params_dtype,
                device="cuda",
            )
        else:
            logger.debug_rank0("Using fused op grouped_matmul.")
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size * self.num_local_experts,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )

    def get_muon_parameter_specs(self) -> dict[str, "MuonParameterSpec"]:
        """Describe the independent matrices packed into fused expert storage."""
        from torch.distributed.tensor import Shard

        from torchturbo.engine.optimizers import (
            ExpertDownComputeView,
            ExpertUpComputeView,
            MuonParameterSpec,
        )

        if transformer_engine_is_available() and self.use_te:
            # GroupedLinear owns separate weights instead of fused expert tensors.
            return {}

        for name, parameter, matrix_rows in (
            ("weight1", self.weight1, self.config.hidden_size),
            ("weight2", self.weight2, self.config.moe_ffn_hidden_size),
        ):
            if parameter.shape[0] % matrix_rows:
                raise ValueError(
                    f"GroupedMLP parameter {name!r} has invalid shape "
                    f"{tuple(parameter.shape)}"
                )
        return {
            "weight1": MuonParameterSpec(
                compute_placement=Shard(0),
                compute_view=ExpertUpComputeView(
                    num_experts=self.weight1.shape[0] // self.config.hidden_size,
                    fused_gate_up=self.config.gated_linear_unit,
                ),
            ),
            "weight2": MuonParameterSpec(
                compute_placement=Shard(0),
                compute_view=ExpertDownComputeView(
                    num_experts=self.weight2.shape[0] // self.config.moe_ffn_hidden_size,
                ),
            ),
        }

    def activation_func_with_probs(
        self, hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> torch.Tensor:
        """Weight activated expert features before FC2, retaining probability precision."""
        hidden_states = self.activation_func(hidden_states)
        return (hidden_states * probs).to(hidden_states.dtype)

    def bind_virtual_experts(self, load_balancer: "VirtualExpertLoadBalancer") -> None:
        """Use this load balancer to bind weights for each expert call."""
        self._load_balancer = load_balancer

    def native_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """View the current unsharded FC parameters as expert matrices."""
        return (
            self.weight1.view(self.num_local_experts, self.config.hidden_size, -1),
            self.weight2.view(self.num_local_experts, -1, self.config.hidden_size),
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Apply routing probabilities inside each expert before its second linear."""
        hidden, counts, probs = permuted_local_hidden_states, tokens_per_expert, permuted_probs
        # CUDA grouped_gemm reads host splits, including during checkpoint replay.
        if isinstance(counts, torch.Tensor) and counts.is_cuda:
            counts = counts.cpu()
        manager = self._load_balancer
        if manager is None:
            if transformer_engine_is_available() and self.use_te:
                fc1, fc2 = self.fc1_linear, self.fc2_linear
                counts = counts.tolist()
            else:
                fc1, fc2 = (partial(grouped_linear, weight=w) for w in self.native_weights())
        # FSDP can reshard between passes. Bind the current unsharded views each time.
        with (
            nullcontext((hidden, counts, probs, fc1, fc2)) if manager is None else
            manager.bind_expert_weights(hidden, counts, probs, self.native_weights())
        ) as (hidden, counts, probs, fc1, fc2):
            probs = probs.unsqueeze(-1)
            if self.config.moe_apply_probs_on_input:
                assert self.config.moe_router_topk == 1, "`moe_apply_probs_on_input` requires top-k 1."
                hidden = (hidden * probs).to(hidden.dtype)
                probs = torch.ones_like(probs)
            intermediate = self.activation_func_with_probs(fc1(hidden, counts), probs)
            return fc2(intermediate, counts), None
