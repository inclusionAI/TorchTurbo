"""Internal expert execution on the public parameter-construction interface."""
import torch
import torch.nn.functional as F
from ...ops import ops
from ..modules.moe.experts import GroupedMLP, transformer_engine_is_available


class DeepseekV41GroupedMLP(GroupedMLP):
    apply_probs_after_experts = True

    def activation_func_with_probs(self, hidden_states, probs):
        # The internal MoE layer applies probabilities after FC2.
        return self.activation_func(hidden_states)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        if self._execution_backend is not None:
            return self._execution_backend.execute(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
            ), None
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

        if permuted_local_hidden_states.nelement() != 0:

            if transformer_engine_is_available() and self.use_te:
                # Use TransformerEngine GroupedLinear if available, otherwise fall back to grouped_gemm
                m_splits = tokens_per_expert.tolist()
                # First grouped linear: hidden_size -> fc1_output_size
                fc1_output = self.fc1_linear(permuted_local_hidden_states, m_splits)

                # Apply activation function
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )

                # Second grouped linear: fc2_input_size -> hidden_size
                fc2_output = self.fc2_linear(intermediate_parallel, m_splits)
            else:
                # Fall back to the grouped_matmul op registry.
                # Reshape the weights for the grouped GEMMs.
                w1 = self.weight1.view(
                    self.num_local_experts, self.config.hidden_size, -1
                )
                w2 = self.weight2.view(
                    self.num_local_experts, -1, self.config.hidden_size
                )


                fc1_output = ops.grouped_matmul(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
                )
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = ops.grouped_matmul(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False
                )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
            fc2_output = torch.matmul(h, w2)

        return fc2_output, None
