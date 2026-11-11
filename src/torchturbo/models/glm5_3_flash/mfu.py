"""Theoretical text FLOPs for GLM-5.3-Flash (two FLOPs per multiply-add).

Counts KDA projections/convolution/recurrent state work, MLA sparse scores,
k-pool indexers, activated MoE experts, dense MLPs, mHC and the LM head.
Sparse scores use a selected-key upper bound, not dense SDPA implementation
work. Norms, activations, routing/top-k selection and communication are omitted.
"""

from __future__ import annotations

import torch

from .configuration_glm5_3_flash import Glm5NextConfig, Glm5NextTextConfig


class Glm5NextFlops:
    """Forward work for valid text tokens, including each packed segment once."""

    def __init__(self, config: Glm5NextConfig | Glm5NextTextConfig) -> None:
        self.config = config.text_config if isinstance(config, Glm5NextConfig) else config

    def components(self, seq_segments: list[int]) -> dict[str, int]:
        """Return a model FLOP breakdown for flattened per-row segment lengths."""
        c = self.config
        tokens = sum(seq_segments)
        h = c.hidden_size
        heads = c.num_attention_heads
        qdim, vdim = c.qk_head_dim, c.v_head_dim
        linear_heads, dim = c.linear_num_heads, c.linear_head_dim
        width = linear_heads * dim
        n_linear = c.layer_types.count("linear_attention")
        n_sparse = c.layer_types.count("indexed_attention")
        n_indexers = sum(
            attn == "indexed_attention" and indexer == "full"
            for attn, indexer in zip(c.layer_types, c.indexer_types)
        )
        n_moe = c.mlp_layer_types.count("sparse")
        n_dense = c.num_hidden_layers - n_moe

        # Q/K/V/output plus the forget gate, output gate and beta projection.
        kda_proj = 2 * tokens * (
            4 * h * width + 2 * (h * dim + dim * width) + h * linear_heads
        )
        kda_conv = 6 * tokens * width * c.linear_conv_kernel_dim
        # State decay (1), K/state read (2), rank-one update (2), Q read (2).
        kda_state = 7 * tokens * linear_heads * dim * dim

        mla_proj = 2 * tokens * (
            h * c.q_lora_rank + c.q_lora_rank * heads * qdim
            + h * c.kv_lora_rank + c.kv_lora_rank * heads * (qdim + vdim)
            + heads * vdim * h
        )
        selected_keys = c.index_topk + (
            c.index_kpool - 1 if c.index_kpool_always_select_tail else 0
        )
        sparse_pairs = sum(s * min(s, selected_keys) for s in seq_segments)
        mla_scores = 2 * heads * (qdim + vdim) * sparse_pairs

        # The indexer runs only on full-indexer layers. It is frozen in SFT.
        index_proj = 2 * tokens * (
            c.q_lora_rank * c.index_n_heads * c.index_head_dim
            + 2 * h * c.index_head_dim + h * c.index_n_heads
        )
        pool_pairs = sum(s * (s // c.index_kpool) for s in seq_segments)
        index_scores = 2 * c.index_n_heads * (c.index_head_dim + 1) * pool_pairs
        moe = (
            6 * tokens * h * c.moe_intermediate_size
            * (c.num_experts_per_tok + c.n_shared_experts)
            + 2 * tokens * h * c.n_routed_experts
        )
        hc = c.hc_mult
        # Two coefficient projections and two HC residual-mixing matmuls.
        mhc = 4 * tokens * (hc * h * (2 * hc + hc * hc) + hc * hc * h)
        return {
            "kda/projections": kda_proj * n_linear,
            "kda/convolution": kda_conv * n_linear,
            "kda/state": kda_state * n_linear,
            "mla/projections": mla_proj * n_sparse,
            "mla/sparse_scores": mla_scores * n_sparse,
            "indexer": (index_proj + index_scores) * n_indexers,
            "ffn/moe": moe * n_moe,
            "ffn/dense": 6 * tokens * h * c.intermediate_size * n_dense,
            "mhc": mhc * c.num_hidden_layers,
            "lm_head": 2 * tokens * h * c.vocab_size,
        }

    def forward_flops(self, inputs: dict[str, torch.Tensor]) -> int:
        """Estimate the full input before sequence-parallel slicing."""
        if any(inputs.get(key) is not None for key in ("pixel_values", "pixel_values_videos")):
            raise NotImplementedError("GLM-5.3-Flash MFU currently supports text-only inputs.")
        from .modeling_glm5_3_flash import sequence_segments

        source = inputs.get("input_ids")
        if source is None:
            source = inputs["inputs_embeds"]
        batch, length = source.shape[:2]
        segments = sequence_segments(
            inputs.get("attention_mask"), inputs.get("position_ids"), batch, length
        )
        lengths = [end - start for row in segments for start, end in row]
        return sum(self.components(lengths).values())
