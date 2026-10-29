"""Theoretical forward-FLOP recipe for Qwen3.8-Flash-Next.

The decoder shares GatedDeltaNet and MoE with Qwen3.5, but its QSA indexer,
hyper-connections and PLE branch are real model work and must not disappear
from MFU accounting.  This recipe counts those Qwen3.8-specific components
while retaining the common Qwen3.5 definitions for GDN, MoE and vision.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component
from ..qwen3_5_moe.mfu import Qwen3_5MoeFlops

__all__ = ["Qwen4ExpFlops"]


class Qwen4ExpFlops(Qwen3_5MoeFlops):
    """Qwen3.8-Flash-Next forward FLOPs, including QSA, HC and PLE."""

    def _parse_config(self) -> None:
        super()._parse_config()
        # The vendored Qwen4 model intentionally ignores checkpoint MTP
        # weights and does not instantiate an MTP predictor.  Do not charge
        # those configured-but-unexecuted layers to the online MFU numerator.
        self.mtp_layers = 0
        g = self._cfg
        self.hc_count = int(g("hc_count", default=1))
        self.hc_lowrank = int(g("hc_lowrank", default=0))
        self.ple_layer_ids = list(g("ple_layer_ids", default=[]) or [])
        self.ple_embed_dim = int(g("ple_embed_dim", default=self.hidden_size))
        self.ple_conv_kernel_size = int(g("ple_conv_kernel_size", default=1))
        self.indexer_num_heads = int(g("indexer_n_heads", default=0) or 0)
        self.indexer_num_kv_heads = int(g("indexer_kv_heads", default=0) or 0)
        self.indexer_head_dim = int(g("indexer_head_dim", default=0) or 0)
        self.indexer_budget = int(g("indexer_budget", default=0) or 0)
        self.indexer_compress_ratio = int(g("indexer_compress_ratio", default=1) or 1)

    @property
    def qsa_enabled(self) -> bool:
        return bool(
            self.indexer_num_heads
            and self.indexer_num_kv_heads
            and self.indexer_head_dim
            and self.indexer_budget
        )

    def _full_attention_flops_for_seg(self, seg: int) -> int:
        if not self.qsa_enabled:
            return super()._full_attention_flops_for_seg(seg)

        rows = self.B * seg
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        projections = (
            flops.linear_flops(rows, self.hidden_size, 2 * q_width)
            + flops.linear_flops(rows, self.hidden_size, kv_width)
            + flops.linear_flops(rows, self.hidden_size, kv_width)
            + flops.linear_flops(rows, q_width, self.hidden_size)
        )
        # The incomplete causal block contributes at most ratio-1 extra rows.
        selected_width = min(seg, self.indexer_budget + self.indexer_compress_ratio - 1)
        sparse_qk_pv = 4 * rows * selected_width * q_width
        return projections + sparse_qk_pv

    def _indexer_flops_for_seg(self, seg: int) -> int:
        if not self.qsa_enabled:
            return 0
        rows = self.B * seg
        projection_width = (
            self.indexer_num_heads + self.indexer_num_kv_heads
        ) * self.indexer_head_dim
        projection = flops.linear_flops(rows, self.hidden_size, projection_width)

        # Query i scores every complete causal compression block.  This is the
        # work the current exact top-k implementation performs before selecting
        # ``indexer_budget / ratio`` blocks.
        ratio = self.indexer_compress_ratio
        complete_groups, remainder = divmod(seg, ratio)
        scored_blocks = ratio * complete_groups * (
            complete_groups - 1
        ) // 2 + complete_groups * (remainder + 1)
        scores = (
            2 * self.B * scored_blocks * self.indexer_num_heads * self.indexer_head_dim
        )
        pooled_key_rows = seg // ratio
        pooling = self.B * pooled_key_rows * ratio * self.indexer_head_dim
        return projection + scores + pooling

    def _hyper_connection_flops(self) -> int:
        if self.hc_count <= 1 or self.hc_lowrank <= 0:
            return 0
        rows = self.B * self.S_eff
        hc_hidden = self.hc_count * self.hidden_size
        mixer = flops.linear_flops(rows, hc_hidden, self.hc_lowrank)
        mixer += flops.linear_flops(rows, self.hc_lowrank, hc_hidden)
        injection = flops.linear_flops(rows, hc_hidden, self.hc_count)
        # Attention and MLP each own a mixer+injection; the final model mixer
        # has no block-injection projection.
        return self.num_layers * 2 * (mixer + injection) + mixer

    def _ple_flops(self) -> int:
        if not self.ple_layer_ids:
            return 0
        rows = self.B * self.S_eff
        hc_hidden = self.hc_count * self.hidden_size
        per_layer = flops.linear_flops(rows, self.ple_embed_dim, hc_hidden)
        per_layer += flops.linear_flops(rows, self.ple_embed_dim, self.hidden_size)
        per_layer += 2 * rows * hc_hidden  # key/query dot product for the gate
        per_layer += flops.depthwise_conv1d_flops(
            self.B,
            self.S_eff,
            hc_hidden,
            self.ple_conv_kernel_size,
        )
        return len(self.ple_layer_ids) * per_layer

    def components(self) -> list[Component]:
        components = super().components()
        additions: list[Component] = []
        if self.qsa_enabled and self.n_full:
            indexer = sum(self._indexer_flops_for_seg(seg) for seg in self.seq_segments)
            additions.append(
                Component(
                    f"qsa/indexer ({self.n_full} layers)",
                    indexer * self.n_full,
                    note=(
                        f"heads={self.indexer_num_heads}, dim={self.indexer_head_dim}, "
                        f"ratio={self.indexer_compress_ratio}, budget={self.indexer_budget}"
                    ),
                )
            )
        hyper = self._hyper_connection_flops()
        if hyper:
            additions.append(
                Component(
                    f"hyper-connection ({self.num_layers} layers + final mixer)",
                    hyper,
                    note=f"streams={self.hc_count}, lowrank={self.hc_lowrank}",
                )
            )
        ple = self._ple_flops()
        if ple:
            additions.append(
                Component(
                    f"ple ({len(self.ple_layer_ids)} layers)",
                    ple,
                    note=(
                        f"embed={self.ple_embed_dim}, streams={self.hc_count}, "
                        f"conv_kernel={self.ple_conv_kernel_size}; lookup=0 FLOPs"
                    ),
                )
            )

        logits_index = next(
            (
                index
                for index, item in enumerate(components)
                if item.name.startswith("logits")
            ),
            len(components),
        )
        components[logits_index:logits_index] = additions
        return components
