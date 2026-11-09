  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .moe_config import MoEConfig
from torchturbo.ops.moe import VirtualExpertPlan
from torchturbo.ops import ops
from torchturbo.logging import get_logger

if TYPE_CHECKING:
    from torchturbo.engine.optimizers import MuonParameterSpec

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

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        expert_plan: VirtualExpertPlan | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Apply routing probabilities inside each expert before its second linear."""
        if expert_plan is not None and not expert_plan.has_global_replicas:
            # Every EP rank can omit the empty weight collective together.
            tokens_per_expert = tokens_per_expert[:self.num_local_experts]
            expert_plan = None
        if not (transformer_engine_is_available() and self.use_te):
            w1 = self.weight1.view(
                self.num_local_experts, self.config.hidden_size, -1,
            )
            w2 = self.weight2.view(
                self.num_local_experts, -1, self.config.hidden_size,
            )
            if expert_plan is not None:
                # Run communication before any saved GEMM activations so
                # non-reentrant checkpoint early-stop cannot skip the exchange.
                replica1, replica2, permuted_local_hidden_states, permuted_probs = (
                    ops.virtual_expert_weight_exchange(
                        w1, w2, permuted_local_hidden_states, permuted_probs,
                        expert_plan.weight_context,
                    )
                )
        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(
                original_dtype
            )
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        activation_probs = permuted_probs.unsqueeze(-1)

        # `tokens_per_expert` is the per-expert token count used as the grouped-GEMM
        # `batch_sizes`. The grouped_gemm C++ kernel hard-requires it on CPU
        # (`assert batch_sizes.is_cpu()`). During normal forward the AlltoAll token
        # dispatcher moves it to CPU via an async side-stream + sync point; but under
        # activation-checkpoint recompute (driven by autograd in backward) that side
        # stream/sync timing is not reliably hit, so the GPU tensor can leak through
        # and trip the is_cpu() assert. Pin it to CPU here at the call site instead of
        # relying on the dispatcher's async DtoH, so forward and recompute behave the
        # same. (The TE branch already does the equivalent via .tolist() below.)
        if isinstance(tokens_per_expert, torch.Tensor) and tokens_per_expert.is_cuda:
            tokens_per_expert = tokens_per_expert.cpu()

        if transformer_engine_is_available() and self.use_te:
            # TE also handles empty inputs and keeps its expert weights in the graph.
            m_splits = tokens_per_expert.tolist()
            # First grouped linear: hidden_size -> fc1_output_size
            fc1_output = self.fc1_linear(permuted_local_hidden_states, m_splits)

            # Apply activation function
            intermediate_parallel = self.activation_func_with_probs(
                fc1_output, activation_probs
            )

            # Second grouped linear: fc2_input_size -> hidden_size
            fc2_output = self.fc2_linear(intermediate_parallel, m_splits)
        elif expert_plan is not None:
            native_counts = tokens_per_expert[:self.num_local_experts]
            num_native_tokens = int(native_counts.sum())
            native_output = self._forward_grouped(
                permuted_local_hidden_states[:num_native_tokens],
                w1, w2, native_counts, activation_probs[:num_native_tokens],
            )
            replica_counts = tokens_per_expert[
                self.num_local_experts:self.num_local_experts + replica1.shape[0]
            ]
            replica_output = self._forward_grouped(
                permuted_local_hidden_states[num_native_tokens:],
                replica1, replica2, replica_counts, activation_probs[num_native_tokens:],
            )
            # Preserve native FSDP storage; only selected replicas own new weights.
            fc2_output = torch.cat((native_output, replica_output), dim=0)
        else:
            fc2_output = self._forward_grouped(
                permuted_local_hidden_states, w1, w2,
                tokens_per_expert, activation_probs,
            )

        return fc2_output, None

    def _forward_grouped(
        self, hidden_states: torch.Tensor,
        weight1: torch.Tensor, weight2: torch.Tensor,
        tokens_per_expert: torch.Tensor, probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute native or compact replica experts through the shared ops."""
        if weight1.shape[0] == 0:
            # No replica slots means no replica tokens. Skip kernels with zero
            # experts; weight exchange already links these hidden states
            # to the weight-exchange backward.
            return hidden_states
        if hidden_states.nelement() != 0:
            fc1_output = ops.grouped_matmul(
                hidden_states, weight1, tokens_per_expert, trans_b=False
            )
            intermediate_parallel = self.activation_func_with_probs(
                fc1_output, probs
            )
            return ops.grouped_matmul(
                intermediate_parallel, weight2, tokens_per_expert, trans_b=False
            )

        assert torch.count_nonzero(tokens_per_expert) == 0
        # Preserve parameter gradients when no token is allocated to this group.
        weight1 = weight1.reshape(self.config.hidden_size, -1)
        weight2 = weight2.reshape(-1, self.config.hidden_size)
        hidden_states = torch.matmul(hidden_states, weight1)
        hidden_states = self.activation_func_with_probs(hidden_states, probs)
        return torch.matmul(hidden_states, weight2)
