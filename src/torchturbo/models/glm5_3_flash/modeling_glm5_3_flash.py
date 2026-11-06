# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for TorchTurbo under the Apache License, Version 2.0.
"""Training implementation of GLM-5.3-Flash, independent of HF model code.

Parameter names follow the original checkpoint (including unfused convolutions
and mHC parameters). Only stable transformers config/output/base classes are used.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F, LayerNorm
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)

from .configuration_glm5_3_flash import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


class Glm5NextTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-06) -> None:
        """
        Glm5NextTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm5NextTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextTextTopkRouter(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32)
        )
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(
                -1, self.num_group, self.num_experts // self.num_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf")
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return (router_logits, topk_weights, topk_indices)


class Glm5NextTextUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-06):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )


class Glm5NextTextHyperConnection(nn.Module):
    """
    A module to implement manifold-constrained Hyper-Connections (mHC) (Xie et al., 2026) which strengthens the
    conventional residual connections between adjacent Transformer blocks.

    When using mHC, each token is projected onto `hc_mult` streams, so the shape of decoder layer inputs changes from
    [batch_size, sequence_length, hidden_size] to [batch_size, sequence_length, hc_mult, hidden_size].
    To keep the same input shape for attention or MLP blocks, the streams are collapsed into one upon entering a block,
    and expanded back into `hc_mult` streams upon exiting. There is also a weighted residual connection between the
    input and output streams.
    The weights used for collapsing (pre), expanding (post) and mixing (comb) are computed from the `hc_mult` input
    streams through a learned projection (plus Sinkhorn-Knopp algorithm for the comb weight).

    The diagram below shows the flow of the mHC streams (B = batch_size, S = seq_length, N = hc_mult, D = hidden_size):

                                                  ┌───────────────────┐
                             N input streams  ────│ FLATTEN + PROJECT |────> (pre, post, comb) weights
                              [B, S, N, D]        └───────────────────┘      ([B, S, N],  [B, S, N],  [B, S, N, N])
                                  │ │ │
               ╭──────────────────┴─┼─┼──────────────────╮
               │ ╭──────────────────┴─┼────────────────╮ │
               │ │ ╭──────────────────┴──────────────╮ │ │
               │ │ │                                 │ │ │
        ┌─────────────────┐                          │ │ │
        │ COLLAPSE (pre)  │                          │ │ │
        └─────────────────┘                          │ │ │
                 │                                   │ │ │
            Block input                              │ │ │
             [B, S, D]                               │ │ │
                 │                                   │ │ │
        ┌─────────────────┐                     ┌─────────────┐
        │   ATTN or MLP   │                     │  MIX (comb) │
        └─────────────────┘                     └─────────────┘
                 │                                   │ │ │
           Block output                              │ │ │
             [B, S, D]                               │ │ │
                 │                                   │ │ │
        ┌─────────────────┐                          │ │ │
        │  EXPAND (post)  │                          │ │ │
        └─────────────────┘                          │ │ │
               │ │ │                                 │ │ │
        N expanded streams                    N residual streams
           [B, S, N, D]                          [B, S, N, D]
               │ │ │                                 │ │ │
               │ │ ╰───────────┌─────────┐───────────╯ │ │
               │ ╰─────────────│   ADD   │─────────────╯ │
               ╰───────────────└─────────┘───────────────╯
                                  │ │ │
                                  ▼ ▼ ▼
                            N output streams
                              [B, S, N, D]
    """

    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(
            torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size)
        )
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the weights used to mix in the `hc_mult` streams with the input and output of the next layer, which can
        be an attention or a MLP layer. This is done through three weights:

        - pre: used to collapse the `hc_mult` input streams into one, creating an input tensor for the next layer
        - post: used to expand the output of the next layer back into `hc_mult` streams
        - comb: used to mix the `hc_mult` input streams with the `hc_mult` output streams

        All weights are returned except "pre", which is consumed here.
        """
        (batch_size, seq_len) = hidden_streams.shape[:2]
        hc = self.hc_mult
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        flattened = F.linear(flattened, self.fn.float())
        (pre_w, post_w, comb_w) = flattened.split([hc, hc, hc * hc], dim=-1)
        (pre_b, post_b, comb_b) = self.base.split([hc, hc, hc * hc])
        (pre_scale, post_scale, comb_scale) = self.scale.unbind(0)
        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
        comb_b = comb_b.view(hc, hc)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (
            (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        )
        return (post, comb, collapsed)


class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class Glm5NextTextForgetGate(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.safe_gate_lower_bound = config.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(
            hidden_shape
        )
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        return -decay_rate * g_softplus


class Glm5NextTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-06, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-06):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def recurrent_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    (query, key, value, g, beta) = [
        x.to(torch.float32) for x in (query, key, value, g, beta)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    (batch_size, sequence_length, num_heads, k_head_dim) = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    core_attn_out = torch.zeros(
        batch_size,
        sequence_length,
        num_heads,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]
        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)
    return (
        core_attn_out.to(initial_dtype),
        last_recurrent_state if output_final_state else None,
    )


class Glm5NextTextIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer with k-pool compression for GLM-5.3-Flash.

    The indexer uses lightweight projections (`wq_b`, `wk`) separate from the main MLA
    attention path. It scores compressed k-pool candidates, expands selected pools back
    into raw cache indices, and optionally appends the current incomplete tail pool.

    **Cache strategy**: the indexer state cache lives on the per-layer `DynamicIndexedLayer`
    (or the `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-06)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** (-0.5)
        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, self.hidden_size)
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
        past_key_values: Cache | None,
    ) -> torch.LongTensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA) based on grouping pools (and tails).

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            attention_mask: Local boolean padding mask of shape `[B, S]`.
            past_key_values: Cache object containing the indexer state cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]` (or `[B, S, 2*topk - 1]` with tail).
            The eager / SDPA paths turn these into an additive sparse mask.
        """
        (batch_size, seq_len) = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)
        kv_len = seq_len
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            packed_states = past_key_values.update_indexer(
                packed_states, self.layer_idx
            )
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()
        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(
            valid_keys=valid_keys, q_length=seq_len, current_length=current_length
        )
        (pool_keys, pool_indices, pool_valid) = self.get_pooled_states(
            packed_states=packed_states
        )
        scores = torch.matmul(
            q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1)
        )
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype)
        ).float() * self.n_heads ** (-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(
            dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1)
        )
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(
            ~valid_candidates, torch.finfo(index_scores.dtype).min
        )
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
        )
        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(
                topk_indices, visible_tokens, valid_keys
            )
            output_width += self.index_kpool - 1
        topk_indices = F.pad(
            topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1
        )
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)

    def get_visible_tokens(
        self, valid_keys: torch.BoolTensor, q_length: int, current_length: int
    ) -> torch.BoolTensor:
        """
        Check whether a token is visible for Q based on
            - Causality
            - Padding status
                - Valid keys which is a (cached) variation of the attention mask
        """
        device = valid_keys.device
        kv_positions = torch.arange(valid_keys.shape[-1], device=device)
        q_positions = current_length - q_length + torch.arange(q_length, device=device)
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]
        return causal & valid_keys[:, None, :]

    def get_pooled_states(
        self, packed_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        (Re-)Build compressed k-pool candidates from the packed state.

        Pooling starts at the first real token, not raw slot 0. This is the part
        that makes:

            [P, P, A, B, C, D, ...]

        behave like:

            [A, B, C, D, ...]

        for k-pool grouping.
        """
        (keys, gate_scores, valid_keys) = torch.split(
            packed_states, [self.head_dim, self.head_dim, 1], dim=-1
        )
        valid_keys = valid_keys.bool().squeeze(-1)
        (batch_size, seq_len) = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device
        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)
        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]
        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)
        logits = (
            grouped_gate_scores.float()
            + self.index_kpool_compress_ape.float()[None, None]
        )
        logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)
        keep = pool_valid.any(0)
        return (pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep])

    def append_visible_tail(
        self,
        topk_indices: torch.Tensor,
        token_visible: torch.BoolTensor,
        key_valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Append the current incomplete pool as raw token indices.
        So we if we have a pool size of 4:
            - [P P A B C D E F]
            - [A B C D] (selected pool)
            - [E F] (appended tail)
        """
        if (max_tail_width := (self.index_kpool - 1)) == 0:
            return topk_indices
        (batch_size, _, kv_length) = token_visible.shape
        device = token_visible.device
        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch_size,), kv_length, dtype=torch.long, device=device),
        )
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(self.index_kpool)
        tail_offsets = torch.arange(max_tail_width, device=device)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (
            tail_offsets[None, None, :] < tail_count[..., None]
        ) & tail_indices.lt(kv_length)
        kv_idx = tail_indices.clamp(0, kv_length - 1)
        tail_visible = token_visible.gather(dim=-1, index=kv_idx)
        tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)
        return torch.cat([topk_indices, tail_indices], dim=-1)


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config, bias: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_state):
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        swiglu_limit: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-06) -> None:
        """
        Glm5NextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    (q, k) = (q.float(), k.float())
    (cos, sin) = (cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float())
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return (q_embed, k_embed)


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


def sequence_segments(mask, position_ids, batch, length):
    """Contiguous valid sequences; reset convolution, KDA state and pools here.

    Both neat-packing labels and reset position_ids (with no mask) are accepted.
    Compute the small piece of metadata once per text forward, before SP slicing.
    """
    if mask is not None and (mask.ndim != 2 or tuple(mask.shape) != (batch, length)):
        raise ValueError("GLM-5.3-Flash requires a 2D padding/packing attention_mask")
    if position_ids is not None:
        if (
            position_ids.ndim != 2
            or position_ids.shape[-1] != length
            or position_ids.shape[0] not in (1, batch)
        ):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
    masks = (
        mask.detach().cpu().tolist()
        if mask is not None
        else [[1] * length for _ in range(batch)]
    )
    positions = (
        position_ids.detach().cpu().tolist() if position_ids is not None else None
    )
    result = []
    for b, row in enumerate(masks):
        pos = (
            positions[b if len(positions) > 1 else 0] if positions is not None else None
        )
        segments, start = [], None
        for i, label in enumerate(row):
            boundary = i > 0 and (
                label != row[i - 1] or (pos is not None and pos[i] <= pos[i - 1])
            )
            if start is not None and (not label or boundary):
                segments.append((start, i))
                start = None
            if label and start is None:
                start = i
        if start is not None:
            segments.append((start, length))
        result.append(segments)
    return result


def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    if not gate_logits:
        return None
    device = gate_logits[0].device
    counts = torch.zeros(num_experts, device=device)
    probability_sum = torch.zeros(num_experts, device=device)
    total = 0
    for logits in gate_logits:
        probabilities = logits.float().softmax(-1)
        selected = probabilities.topk(top_k, dim=-1).indices
        valid = (
            attention_mask.reshape(-1).ne(0).to(device=device, dtype=torch.float32)
            if attention_mask is not None
            else torch.ones(logits.shape[0], device=device)
        )
        counts = counts.scatter_add(
            0, selected.flatten(), valid.repeat_interleave(top_k)
        )
        probability_sum = probability_sum + (probabilities * valid[:, None]).sum(0)
        total = total + valid.sum()
    return (counts * probability_sum).sum() * num_experts / total.clamp_min(1).square()


class _SequenceAttention(nn.Module):
    """Instance-scoped SP state, with a real reduction of gathered gradients."""

    sp_group = None
    sp_size = 1
    sp_rank = 0

    def gather(self, hidden_states):
        if self.sp_size == 1:
            return hidden_states
        from ..modules.attention.ulysses import all_gather_sp_with_grad_reduce

        return all_gather_sp_with_grad_reduce(
            hidden_states, 1, self.sp_group, self.sp_size
        )

    def local_output(self, output):
        if self.sp_size == 1:
            return output
        return output.chunk(self.sp_size, dim=1)[self.sp_rank].contiguous()


class Glm5NextTextLinearAttention(_SequenceAttention):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_heads, self.head_dim = config.linear_num_heads, config.linear_head_dim
        if config.hidden_act != "silu":
            raise ValueError("GLM-5.3-Flash KDA convolution requires silu activation")
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, self.qkv_dim, bias=False))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(
                self,
                name,
                nn.Conv1d(
                    self.qkv_dim,
                    self.qkv_dim,
                    self.conv_kernel_size,
                    groups=self.qkv_dim,
                    bias=False,
                    padding=self.conv_kernel_size - 1,
                ),
            )
        # Keep original checkpoint names rather than HF's forget_gate.* aliases.
        gate = Glm5NextTextForgetGate(config)
        self.f_a_proj, self.f_b_proj = gate.f_a_proj, gate.f_b_proj
        self.A_log, self.dt_bias = gate.A_log, gate.dt_bias
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextTextRMSNormGated(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, config.hidden_size, bias=False)

    def _segment(self, hidden):
        b, t = hidden.shape[:2]
        shape = (b, t, self.num_heads, self.head_dim)
        qkv = []
        for name in ("q", "k", "v"):
            x = getattr(self, name + "_proj")(hidden).transpose(1, 2)
            conv = getattr(self, name + "_conv1d")
            x = F.conv1d(
                x.to(conv.weight.dtype),
                conv.weight,
                padding=self.conv_kernel_size - 1,
                groups=self.qkv_dim,
            )[..., :t]
            qkv.append(F.silu(x).transpose(1, 2).reshape(shape))
        forget = self.f_b_proj(self.f_a_proj(hidden)).float() + self.dt_bias.float()
        rate = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
        forget = forget.view(shape)
        lower = self.config.linear_lower_bound
        g = (
            lower * torch.sigmoid(rate * forget)
            if lower is not None
            else -rate * F.softplus(forget)
        )
        beta = self.b_proj(hidden).sigmoid()
        backend = getattr(self.config, "kda_backend", "auto")
        if backend == "fla" or (backend == "auto" and hidden.device.type == "cuda"):
            try:
                from fla.ops.kda import chunk_kda
            except ImportError:
                if backend == "fla":
                    raise ImportError(
                        "kda_backend='fla' requires fla.ops.kda.chunk_kda"
                    ) from None
            else:
                out, _ = chunk_kda(
                    q=qkv[0],
                    k=qkv[1],
                    v=qkv[2],
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                return self._project(out, hidden, shape)
        out, _ = recurrent_kimi_delta_attention(
            *qkv,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return self._project(out, hidden, shape)

    def _project(self, out, hidden, shape):
        gate = self.g_b_proj(self.g_a_proj(hidden)).view(shape)
        return self.o_proj(self.o_norm(out, gate).flatten(2))

    def forward(
        self,
        hidden_states,
        segments=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        hidden_states = self.gather(hidden_states)
        b, t, d = hidden_states.shape
        segments = segments or sequence_segments(attention_mask, position_ids, b, t)
        rows = []
        for row, runs in enumerate(segments):
            parts, cursor = [], 0
            for start, end in runs:
                parts.append(hidden_states[row : row + 1, cursor:start] * 0)
                parts.append(self._segment(hidden_states[row : row + 1, start:end]))
                cursor = end
            parts.append(hidden_states[row : row + 1, cursor:] * 0)
            rows.append(torch.cat(parts, dim=1))
        return self.local_output(torch.cat(rows, dim=0))


class Glm5NextTextAttention(_SequenceAttention):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.qk_head_dim, self.v_head_dim = config.qk_head_dim, config.v_head_dim
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = Glm5NextTextRMSNorm(
            config.q_lora_rank, config.rms_norm_eps
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = Glm5NextTextRMSNorm(
            config.kv_lora_rank, config.rms_norm_eps
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = self.qk_head_dim**-0.5
        self.indexer = (
            None
            if config.indexer_types[layer_idx] == "shared"
            else Glm5NextTextIndexer(config, layer_idx)
        )
        # Like the supplied HF forward, top-k is non-differentiable. The indexer
        # is frozen; a separate indexer-distillation objective is not implied.
        if self.indexer is not None:
            self.indexer.requires_grad_(False)

    def forward(self, hidden_states, segments=None, prev_topk_indices=None, **kwargs):
        hidden_states = self.gather(hidden_states)
        b, t = hidden_states.shape[:2]
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = (
            self.q_b_proj(q_resid)
            .view(b, t, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        kv = (
            self.kv_b_proj(latent)
            .view(b, t, self.num_heads, self.qk_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k, v = kv.split([self.qk_head_dim, self.v_head_dim], dim=-1)
        if self.indexer is None:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require preceding full-layer indices"
                )
            indices = prev_topk_indices
        else:
            width = self.config.index_topk + (
                self.config.index_kpool - 1
                if self.config.index_kpool_always_select_tail
                else 0
            )
            indices = torch.full(
                (b, t, width), -1, device=hidden_states.device, dtype=torch.int32
            )
            runs = segments or sequence_segments(
                kwargs.get("attention_mask"), kwargs.get("position_ids"), b, t
            )
            with torch.no_grad():
                for row, row_runs in enumerate(runs):
                    for start, end in row_runs:
                        selected = self.indexer(
                            hidden_states[row : row + 1, start:end],
                            q_resid[row : row + 1, start:end],
                            torch.ones(
                                (1, end - start),
                                dtype=torch.bool,
                                device=hidden_states.device,
                            ),
                            None,
                        )
                        indices[row : row + 1, start:end] = torch.where(
                            selected >= 0, selected + start, selected
                        )
        counts = torch.zeros((b, t, t), dtype=torch.int32, device=hidden_states.device)
        counts.scatter_add_(
            -1, indices.long().clamp(0, t - 1), (indices >= 0).to(torch.int32)
        )
        mask = counts.ne(0).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        out = self.o_proj(out.transpose(1, 2).reshape(b, t, -1))
        return self.local_output(out), indices


class Glm5NextTextMoE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.gate = Glm5NextTextTopkRouter(config)
        self.experts = nn.ModuleList(
            [
                Glm5NextTextMLP(config, config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = (
            Glm5NextTextMLP(
                config, config.moe_intermediate_size * config.n_shared_experts
            )
            if config.n_shared_experts
            else None
        )

    def forward(self, hidden_states):
        logits, weights, indices = self.gate(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == expert_id)
            if tokens.numel():
                value = expert(flat[tokens]) * weights[tokens, slots, None]
                out = out.index_add(0, tokens, value.to(out.dtype))
        out = out.view_as(hidden_states)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out, logits


class Glm5NextTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        cls = (
            Glm5NextTextLinearAttention
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention
        )
        self.self_attn = cls(config, layer_idx)
        self.mlp = (
            Glm5NextTextMoE(config, layer_idx)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Glm5NextTextMLP(config)
        )
        self.input_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        for kind in ("attn", "ffn"):
            hc = Glm5NextTextHyperConnection(config)
            for part in ("fn", "base", "scale"):
                self.register_parameter(f"hc_{kind}_{part}", getattr(hc, part))
        self.hc_mult, self.hc_eps = config.hc_mult, config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_norm_eps = config.rms_norm_eps

    def _pre(self, hidden, kind):
        flat = hidden.flatten(2).float()
        flat = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.hc_norm_eps
        )
        mixes = F.linear(flat, getattr(self, f"hc_{kind}_fn").float())
        n, eps = self.hc_mult, self.hc_eps
        a, z, r = mixes.split([n, n, n * n], dim=-1)
        ab, zb, rb = getattr(self, f"hc_{kind}_base").float().split([n, n, n * n])
        sa, sz, sr = getattr(self, f"hc_{kind}_scale").float().unbind()
        pre, post = torch.sigmoid(a * sa + ab) + eps, 2 * torch.sigmoid(z * sz + zb)
        comb = torch.softmax((r * sr + rb).view(*r.shape[:-1], n, n), dim=-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        collapsed = (pre.unsqueeze(-1) * hidden).sum(2).to(hidden.dtype)
        return collapsed, post.to(hidden.dtype), comb.to(hidden.dtype)

    def forward(self, hidden_states, segments, prev_topk_indices=None):
        collapsed, post, comb = self._pre(hidden_states, "attn")
        x = self.input_layernorm(collapsed)
        if self.block_type == "linear_attention":
            x = self.self_attn(x, segments=segments)
            topk = prev_topk_indices  # preserve across intervening KDA layers
        else:
            x, topk = self.self_attn(
                x, segments=segments, prev_topk_indices=prev_topk_indices
            )
        hidden_states = (
            post.unsqueeze(-1) * x.unsqueeze(2) + comb.transpose(-1, -2) @ hidden_states
        )
        collapsed, post, comb = self._pre(hidden_states, "ffn")
        x = self.mlp(self.post_attention_layernorm(collapsed))
        logits = None
        if isinstance(x, tuple):
            x, logits = x
        hidden_states = (
            post.unsqueeze(-1) * x.unsqueeze(2) + comb.transpose(-1, -2) @ hidden_states
        )
        return hidden_states, topk, logits


class Glm5NextPreTrainedModel(PreTrainedModel):
    config_class = Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Glm5NextTextDecoderLayer", "Glm5NextVisionBlock"]

    def __init__(self, config):
        # Training launchers may request FA2 globally. Sparse per-query masks
        # require this implementation's SDPA path rather than standard FA2.
        for cfg in (
            config,
            getattr(config, "text_config", None),
            getattr(config, "vision_config", None),
        ):
            if cfg is not None:
                cfg._attn_implementation = "sdpa"
        super().__init__(config)

    def _init_weights(self, module):
        config = getattr(self.config, "text_config", self.config)
        std = config.initializer_range
        if isinstance(
            module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)
        ):
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(
            module,
            (
                Glm5NextTextRMSNorm,
                Glm5NextRMSNorm,
                Glm5NextTextRMSNormGated,
                nn.LayerNorm,
            ),
        ):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Glm5NextTextTopkRouter):
            nn.init.normal_(module.weight, std=std)
            nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Glm5NextTextIndexer):
            nn.init.zeros_(module.index_kpool_compress_ape)
            nn.init.ones_(module.index_kpool_compress_gate)
        elif isinstance(module, Glm5NextTextDecoderLayer):
            for kind in ("attn", "ffn"):
                nn.init.normal_(getattr(module, f"hc_{kind}_fn"), std=0.02)
                nn.init.zeros_(getattr(module, f"hc_{kind}_base"))
                nn.init.ones_(getattr(module, f"hc_{kind}_scale"))
        elif isinstance(module, Glm5NextTextLinearAttention):
            if module.config.linear_lower_bound is not None:
                nn.init.zeros_(module.A_log)
            else:
                nn.init.uniform_(module.A_log, 1, 16)
                module.A_log.data.log_()
            nn.init.uniform_(module.dt_bias, math.log(1e-3), math.log(1e-1))
            dt = module.dt_bias.data.exp().clamp_min(1e-4)
            module.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))


class Glm5NextTextModel(Glm5NextPreTrainedModel):
    config_class = Glm5NextTextConfig
    sp_size, sp_rank = 1, 0

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextTextDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_head = Glm5NextTextHyperHead()
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        past_key_values=None,
        output_hidden_states=False,
        output_router_logits=False,
        return_dict=True,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "GLM-5.3-Flash training forward requires use_cache=False and no KV cache"
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        b, t = hidden.shape[:2]
        segments = sequence_segments(attention_mask, position_ids, b, t)
        if self.sp_size > 1:
            if t % self.sp_size:
                raise ValueError(
                    "Pad sequence length to a multiple of sp_size before GLM-5.3-Flash forward"
                )
            hidden = hidden.chunk(self.sp_size, dim=1)[self.sp_rank].contiguous()
        hidden = (
            hidden.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        )
        states, routers, topk = [], [], None
        for layer in self.layers:
            if output_hidden_states:
                states.append(hidden)
            if self.gradient_checkpointing and self.training:

                def call(x, indices, layer=layer):
                    return layer(x, segments, indices)

                hidden, topk, logits = checkpoint(
                    call, hidden, topk, use_reentrant=False
                )
            else:
                hidden, topk, logits = layer(hidden, segments, topk)
            if output_router_logits and logits is not None:
                routers.append(logits)
        hidden = self.norm(self.hc_head(hidden))
        if output_hidden_states:
            states.append(hidden)
        result = MoeModelOutputWithPast(
            last_hidden_state=hidden,
            hidden_states=tuple(states) if output_hidden_states else None,
            router_logits=tuple(routers) if output_router_logits else None,
        )
        return result if return_dict else result.to_tuple()


class Glm5NextVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rope_parameters["rope_type"] != "axial":
            raise ValueError("GLM-5.3-Flash vision uses axial RoPE")
        self.config = config

    def forward(self, hidden, positions):
        # Recompute the tiny buffer on the input device: also works after to_empty.
        spatial_dim = self.config.hidden_size // self.config.num_heads // 2
        inv = 1 / self.config.rope_parameters["rope_theta"] ** (
            torch.arange(0, spatial_dim, 2, device=hidden.device).float() / spatial_dim
        )
        freq = positions.float().unsqueeze(-1) * inv
        freq = torch.cat([freq[:, 0], freq[:, 1]], dim=-1).repeat(1, 2)
        return freq.cos(), freq.sin()


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        dim = config.hidden_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=config.attention_bias)
        self.proj = nn.Linear(dim, dim, bias=config.attention_bias)
        self.q_norm = Glm5NextRMSNorm(dim // config.num_heads, config.rms_norm_eps)
        self.k_norm = Glm5NextRMSNorm(dim // config.num_heads, config.rms_norm_eps)

    def forward(self, hidden, lengths, position_embeddings):
        q, k, v = (
            self.qkv(hidden).view(hidden.shape[0], 3, self.num_heads, -1).unbind(1)
        )
        q, k = apply_rotary_pos_emb_vision(
            self.q_norm(q), self.k_norm(k), *position_embeddings
        )
        q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        outputs = [
            F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
            for qi, ki, vi in zip(
                q.split(lengths, 2), k.split(lengths, 2), v.split(lengths, 2)
            )
        ]
        out = torch.cat(outputs, dim=2).squeeze(0).transpose(0, 1).reshape_as(hidden)
        return self.proj(out)


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config, bias=config.attention_bias)

    def forward(self, hidden, lengths, position_embeddings):
        hidden = hidden + self.attn(self.norm1(hidden), lengths, position_embeddings)
        return hidden + self.mlp(self.norm2(hidden))


class Glm5NextVisionModel(Glm5NextPreTrainedModel):
    config_class = Glm5NextVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [Glm5NextVisionBlock(config) for _ in range(config.depth)]
        )
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm5NextVisionPatchMerger(
            config.out_hidden_size,
            config.projection_intermediate_size,
            config.hidden_act,
            config.swiglu_limit,
        )
        self.post_layernorm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, hidden_states, grid_thw, return_dict=True, **kwargs):
        if grid_thw is None:
            raise ValueError("grid_thw is required for image/video patches")
        hidden = self.patch_embed(hidden_states)
        positions, lengths = [], []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.detach().cpu().tolist():
            if h % merge or w % merge:
                raise ValueError(
                    "Image grid height/width must be divisible by spatial_merge_size"
                )
            for _ in range(t):
                rh = torch.arange(h, device=hidden.device).view(h, 1).expand(h, w)
                rw = torch.arange(w, device=hidden.device).view(1, w).expand(h, w)
                reorder = lambda x: (
                    x.reshape(h // merge, merge, w // merge, merge)
                    .permute(0, 2, 1, 3)
                    .flatten()
                )
                positions.append(torch.stack([reorder(rh), reorder(rw)], dim=-1))
                lengths.append(h * w)
        if sum(lengths) != hidden.shape[0]:
            raise ValueError("Pixel patch count does not match grid_thw")
        rope = self.rotary_pos_emb(hidden, torch.cat(positions))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, lengths, rope, use_reentrant=False)
            else:
                hidden = block(hidden, lengths, rope)
        hidden = self.post_layernorm(hidden)
        hidden = hidden.view(-1, merge, merge, hidden.shape[-1]).permute(0, 3, 1, 2)
        hidden = self.downsample(hidden).flatten(1)
        result = BaseModelOutputWithPooling(
            last_hidden_state=hidden, pooler_output=self.merger(hidden)
        )
        return result if return_dict else result.to_tuple()


class Glm5NextModel(Glm5NextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = Glm5NextTextModel(config.text_config)
        self.visual = Glm5NextVisionModel(config.vision_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        result = self.visual(pixel_values, image_grid_thw)
        sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        result.pooler_output = result.pooler_output.split(sizes)
        return result

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_placeholder_mask(
        self, input_ids, inputs_embeds, image_features=None, video_features=None
    ):
        if input_ids is None:
            embedding = self.get_input_embeddings()

            def match(token):
                ids = torch.tensor(token, device=inputs_embeds.device)
                return (inputs_embeds == embedding(ids)).all(-1)
        else:

            def match(token):
                return input_ids == token

        in_video = match(self.config.video_start_token_id).cumsum(-1) > match(
            self.config.video_end_token_id
        ).cumsum(-1)
        # Checkpoint processor uses image placeholders inside video spans.
        # Also accept the dedicated video_token_id for processors that emit it.
        image_mask = match(self.config.image_token_id) & ~in_video
        video_mask = (match(self.config.image_token_id) & in_video) | match(
            self.config.video_token_id
        )
        for mask, features, modality in (
            (image_mask, image_features, "image"),
            (video_mask, video_features, "video"),
        ):
            if features is not None and mask.sum().item() != features.shape[0]:
                raise ValueError(
                    f"{modality} placeholder count does not match feature count"
                )
        return image_mask.unsqueeze(-1), video_mask.unsqueeze(-1)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_encoder_outputs=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if mm_encoder_outputs is not None and (
            pixel_values is not None or pixel_values_videos is not None
        ):
            raise ValueError("Pass pixel inputs or mm_encoder_outputs, not both")
        encoded = dict(mm_encoder_outputs or {})
        if pixel_values is not None:
            encoded["image"] = self.get_image_features(pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            encoded["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
        for modality in ("image", "video"):
            if encoded.get(modality) is not None:
                features = encoded[modality].pooler_output
                features = (
                    torch.cat(features)
                    if isinstance(features, (tuple, list))
                    else features
                )
                features = features.to(inputs_embeds)
                masks = self.get_placeholder_mask(
                    input_ids, inputs_embeds, **{modality + "_features": features}
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    masks[modality == "video"], features
                )
        return self.language_model(inputs_embeds=inputs_embeds, **kwargs)


class Glm5NextForConditionalGeneration(Glm5NextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = True

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm5NextModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        return self.model.get_video_features(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        logits_to_keep=0,
        output_router_logits=None,
        output_hidden_states=False,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        output_router_logits = (
            self.config.text_config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        selected = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, selected])
        loss = None
        if labels is not None:
            if self.model.language_model.sp_size > 1:
                raise ValueError(
                    "With SP, compute loss through TorchTurboEngine's distributed CE"
                )
            if logits.shape[1] != labels.shape[1]:
                raise ValueError(
                    "Training labels require full logits (logits_to_keep=0)"
                )
            targets = labels[:, 1:].clone()
            # Do not train transitions across packed samples or through padding.
            segments = sequence_segments(
                attention_mask, position_ids, labels.shape[0], labels.shape[1]
            )
            valid = torch.zeros_like(labels, dtype=torch.bool)
            for row, runs in enumerate(segments):
                for start, end in runs:
                    valid[row, start + 1 : end] = True
            targets.masked_fill_(~valid[:, 1:], -100)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            denominator = kwargs.get("num_items_in_batch")
            loss = loss / (
                targets.ne(-100).sum().clamp_min(1)
                if denominator is None
                else denominator
            )
        aux_loss = None
        if output_router_logits:
            local_mask = attention_mask
            language = self.model.language_model
            if local_mask is not None and language.sp_size > 1:
                local_mask = local_mask.chunk(language.sp_size, dim=1)[language.sp_rank]
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.n_routed_experts,
                self.config.text_config.num_experts_per_tok,
                local_mask,
            )
            if loss is not None and aux_loss is not None:
                loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss
        result = MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            router_logits=outputs.router_logits,
            aux_loss=aux_loss,
        )
        return result if return_dict else result.to_tuple()
