"""DeepSeek-V4.1 forward-FLOP recipe (CSA2 sparse latent attention + MoE).

Counts the matmul shapes of ``modeling_deepseek_v41.py`` per layer type as the
full-input upper bound (the estimator divides the sum by ``sp_size``):

- latent attention: ``wq_a``/``wq_b`` (num_heads x head_dim), one shared
  ``wkv`` stream (``num_key_value_heads == 1``, used as both K and V), a
  grouped low-rank output projection (``wo_a`` + ``wo_b``), and sparse scores
  over the sliding window plus the indexer's top-k blocks (no O(S^2) term);
- CSA2: kv-source compressors (``wkv``/``wgate`` project every token before
  pooling) and lightning indexers (``wq_b``/``weights_proj`` plus the
  O(S * T) score einsum over the shared compressed KV);
- MoE FFN (top-k routed + shared expert + router), hyper-connection mixing,
  engram lookups, an optional MTP predictor, and the lm head.

Omitted as negligible (elementwise or tiny): rope rotations, RMS norms,
sinkhorn iterations, softmax/pooling reductions, engram gating, the attention
sink column. Approximations are flagged in the ``Component`` notes.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component, LLMFlops

__all__ = ["DeepseekV41Flops"]


class DeepseekV41Flops(LLMFlops):
    """DeepSeek-V4.1 theoretical forward FLOPs (CSA2 sparse-MLA + MoE + HC)."""

    def _parse_config(self) -> None:
        super()._parse_config()  # num_layers/hidden/heads/head_dim/vocab
        g = self._cfg

        self.q_lora_rank: int = int(g("q_lora_rank", default=self.hidden_size))
        self.o_lora_rank: int = int(g("o_lora_rank", default=self.hidden_size))
        self.o_groups: int = int(g("o_groups", default=1))
        self.window_size: int = int(g("sliding_window", default=0))

        # MoE FFN.
        self.num_experts: int = int(g("n_routed_experts", "num_experts", default=0))
        self.top_k: int = int(g("num_experts_per_tok", default=0))
        self.expert_intermediate_size: int = int(g("moe_intermediate_size", default=0))
        self.num_shared_experts: int = int(g("n_shared_experts", "num_shared_experts", default=0))

        # CSA2 layer layout (compress_ratios is per-layer, already normalized
        # by DeepseekV41TextConfig to num_hidden_layers entries).
        self.layer_ratios: list[int] = [int(r) for r in (g("compress_ratios", default=[]) or [])]
        self.kv_source_layer_ids: list[int] = list(g("kv_source_layer_ids", default=[]) or [])
        self.index_source_layer_ids: list[int] = list(g("index_source_layer_ids", default=[]) or [])
        self.index_n_heads: int = int(g("index_n_heads", default=0))
        self.index_head_dim: int = int(g("index_head_dim", default=0))
        self.index_topk: int = int(g("index_topk", default=0))

        # Engram: the n-gram lookup projects hashed context rows to the HC
        # stream stack (hidden * (hc_mult + 1) outputs).
        self.engram_layer_ids: list[int] = list(g("engram_layer_ids", default=[]) or [])
        engram_hash_cols = (int(g("engram_max_ngram_size", default=0)) - 1) * int(
            g("engram_n_heads", default=0)
        )
        self.engram_wkv_in: int = engram_hash_cols * int(g("engram_head_dim", default=0))
        hc_mult = int(g("hc_mult", default=1))
        self.engram_wkv_out: int = self.hidden_size * (hc_mult + 1)

        # HC mixing: two [hc_dim -> hc_streams] coefficient projections.
        self.hc_mix_hc: int = (2 + hc_mult) * hc_mult
        self.hc_dim: int = hc_mult * self.hidden_size

        self.mtp_layers: int = int(g("num_nextn_predict_layers", default=0))

    # -- per-layer costs ---------------------------------------------------- #

    def _num_blocks(self, ratio: int) -> int:
        """Complete compressed blocks in the input (ratio > 0 only)."""
        return self.S_eff // ratio if ratio > 0 else 0

    def _projections_flops(self) -> int:
        """wq_a + wq_b + shared wkv + grouped wo_a/wo_b, one layer."""
        bs = self.B * self.S_eff
        q_path = flops.linear_flops(bs, self.hidden_size, self.q_lora_rank) + flops.linear_flops(
            bs, self.q_lora_rank, self.num_heads * self.head_dim
        )
        kv_proj = flops.linear_flops(bs, self.hidden_size, self.head_dim)
        wo_a = flops.linear_flops(
            bs, self.num_heads * self.head_dim, self.o_groups * self.o_lora_rank
        )
        wo_b = flops.linear_flops(bs, self.o_groups * self.o_lora_rank, self.hidden_size)
        return q_path + kv_proj + wo_a + wo_b

    def _scores_flops_for_k(self, k: int) -> int:
        """qk + pv einsums over ``k`` selected keys (kv doubles as K and V)."""
        per = 2 * self.B * self.S_eff * self.num_heads * k * self.head_dim
        return 2 * per

    def _scores_flops_for_ratio(self, ratio: int) -> int:
        k = min(self.window_size, self.S_eff) + min(self.index_topk, self._num_blocks(ratio))
        return self._scores_flops_for_k(k)

    def _compressor_flops(self, layer_idx: int) -> int:
        """kv-source compressor: wkv + wgate project every token (ratio > 1)."""
        if self.layer_ratios[layer_idx] <= 1:
            return 0
        return 2 * flops.linear_flops(self.B * self.S_eff, self.hidden_size, self.head_dim)

    def _indexer_flops(self, layer_idx: int) -> int:
        """Lightning indexer: projections + the O(S * T) score einsum."""
        num_blocks = self._num_blocks(self.layer_ratios[layer_idx])
        if num_blocks <= 0:
            return 0
        bs = self.B * self.S_eff
        projections = flops.linear_flops(
            bs, self.q_lora_rank, self.index_n_heads * self.index_head_dim
        ) + flops.linear_flops(bs, self.hidden_size, self.index_n_heads)
        einsum = 2 * self.B * self.S_eff * self.index_n_heads * num_blocks * self.index_head_dim
        if layer_idx in self.kv_source_layer_ids:
            # owns_k: project the shared compressed KV into index keys.
            einsum += flops.linear_flops(self.B * num_blocks, self.head_dim, self.index_head_dim)
        return projections + einsum

    def _ffn_flops(self) -> int:
        if self.num_experts <= 0 or self.top_k <= 0:
            return 0
        shared_inter = self.expert_intermediate_size * self.num_shared_experts
        return flops.moe_flops(
            self.B,
            self.S_eff,
            self.hidden_size,
            self.expert_intermediate_size,
            self.num_experts,
            self.top_k,
            shared_expert_intermediate_size=shared_inter,
            use_gate=True,
        )

    def _engram_flops(self) -> int:
        """One engram layer: the wkv lookup projection (gather itself is free)."""
        return flops.linear_flops(self.B * self.S_eff, self.engram_wkv_in, self.engram_wkv_out)

    def _hc_flops(self) -> int:
        """Two coefficient projections per layer (attention + FFN sides)."""
        return 2 * flops.linear_flops(self.B * self.S_eff, self.hc_dim, self.hc_mix_hc)

    # -- breakdown ----------------------------------------------------------- #

    def components(self) -> list[Component]:
        n_layers = self.num_layers
        n_window = sum(1 for r in self.layer_ratios if r == 0)
        n_comp = n_layers - n_window
        n_kv_sources = sum(
            1 for i in self.kv_source_layer_ids if i < n_layers and self.layer_ratios[i] > 1
        )
        n_index_sources = sum(1 for i in self.index_source_layer_ids if i < n_layers)
        n_engram = sum(1 for i in self.engram_layer_ids if i < n_layers)

        proj_per = self._projections_flops()
        window_per = self._scores_flops_for_k(min(self.window_size, self.S_eff))
        comp_per = sum(
            self._scores_flops_for_ratio(r) for r in self.layer_ratios if r > 0
        )
        compressor = sum(
            self._compressor_flops(i) for i in self.kv_source_layer_ids if i < n_layers
        )
        indexer = sum(
            self._indexer_flops(i) for i in self.index_source_layer_ids if i < n_layers
        )
        ffn_per = self._ffn_flops()
        hc_per = self._hc_flops()

        comps: list[Component] = [
            Component(
                f"attention/proj ({n_layers} layers)",
                proj_per * n_layers,
                note=f"q_lora={self.q_lora_rank}, o groups={self.o_groups}x{self.o_lora_rank}; "
                f"per-layer fwd={proj_per:,}",
            ),
            Component(
                f"attention/scores-window ({n_window} layers)",
                window_per * n_window,
                note=f"k=window {self.window_size} (sink not counted); "
                f"per-layer fwd={window_per:,}",
            ),
            Component(
                f"attention/scores-topk ({n_comp} layers)",
                comp_per,
                note=f"k=window {self.window_size}+topk {self.index_topk} blocks; "
                f"summed per-layer fwd={comp_per:,}",
            ),
            Component(
                f"csa2/compressor ({n_kv_sources} kv sources)",
                compressor,
                note="wkv+wgate over every token; pooling itself is elementwise",
            ),
            Component(
                f"csa2/indexer ({n_index_sources} index sources)",
                indexer,
                note="wq_b/weights_proj + O(S*T) score einsum; top-k/relu not counted",
            ),
            Component(
                f"ffn/moe ({n_layers} layers)",
                ffn_per * n_layers,
                note=f"top_k={self.top_k} of {self.num_experts} experts + "
                f"{self.num_shared_experts} shared, inter={self.expert_intermediate_size}; "
                f"per-layer fwd={ffn_per:,}",
            ),
            Component(
                f"engram ({n_engram} layers)",
                self._engram_flops() * n_engram,
                note="wkv lookup projection only; embedding gather and gating not counted",
            ),
            Component(
                f"hc mixing ({n_layers} layers)",
                hc_per * n_layers,
                note="two coefficient projections per layer; sinkhorn/collapse not counted",
            ),
        ]

        comps.extend(self.mtp_components())

        lg = self.logits_component()
        comps.append(Component(lg.name, lg.fwd_flops, lg.note))
        return comps

    def mtp_components(self) -> list[Component]:
        if self.mtp_layers <= 0:
            return []
        # One MTP layer approximates to one compressed-attention layer + the
        # MoE FFN + the 2H->H fusion fc + its own lm_head pass.
        comp_ratio = next((r for r in self.layer_ratios if r > 0), 0)
        mtp_attn = self._projections_flops() + self._scores_flops_for_ratio(comp_ratio)
        mtp_fc = flops.linear_flops(self.B * self.S_eff, 2 * self.hidden_size, self.hidden_size)
        mtp_logits = flops.logits_flops(self.B, self.S_eff, self.hidden_size, self.vocab_size)
        per_layer = mtp_attn + self._ffn_flops() + mtp_fc + mtp_logits
        return [
            Component(
                f"mtp ({self.mtp_layers} layers, attn+ffn+fc+lm_head)",
                per_layer * self.mtp_layers,
                note="approx: one compressed-attn layer + FFN + fusion fc + lm_head per layer",
            )
        ]
