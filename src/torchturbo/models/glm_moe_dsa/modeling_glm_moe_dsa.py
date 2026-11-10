# coding=utf-8
# Copyright 2026 the HuggingFace Team.
# Apache 2.0

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_glm_moe_dsa import GlmMoeDsaConfig


class GlmMoeDsaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    x1 = q[..., 0::2]
    x2 = q[..., 1::2]
    t = torch.cat((x1, x2), dim=-1)

    t_embed = (t * cos) + (rotate_half(t) * sin)
    return t_embed.to(q.dtype)


class GlmMoeDsaIndexer(nn.Module):
    """
    Dynamic Sparse Attention (DSA) indexer.
    """

    def __init__(self, config: "GlmMoeDsaConfig", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        self._cached_keys = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        use_cache: bool = False,
    ) -> torch.LongTensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_pos_emb(
            k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2
        ).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        scores = (
            torch.einsum("bshd,btd->bsht", q.float(), k_cached.float())
            * self.softmax_scale
        )
        scores = F.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        topk_indices = index_scores.topk(topk, dim=-1).indices
        return topk_indices


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                config.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        # index share: `"shared"` 层不跑 indexer，复用上一个 full 层的 top-k。
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        prev_topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)

        query_states = query_states.view(
            batch_size, seq_length, -1, self.qk_head_dim
        ).transpose(1, 2)
        q_nope, q_pe = torch.split(
            query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(
            batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value_states = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # DSA: full 层自己跑 indexer 产出 top-k；shared 层复用上一个 full 层的 top-k。
        if self.indexer is not None:
            indexer_mask = (
                attention_mask[:, 0, :, :]
                if attention_mask is not None and attention_mask.dim() == 4
                else attention_mask.unsqueeze(1) if attention_mask is not None else None
            )
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                use_cache=use_cache,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = index_mask.unsqueeze(1)

        if attention_mask is not None:
            causal_mask = attention_mask[..., :total_len]
            combined_mask = index_mask + causal_mask
        else:
            combined_mask = index_mask

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value, topk_indices


class GlmMoeDsaMLP(nn.Module):
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

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GlmMoeDsaTopkRouter(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size))
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros((self.n_routed_experts), dtype=torch.float32),
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        return router_logits


class GlmMoeDsaMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = GlmMoeDsaTopkRouter(config)
        self.shared_experts = GlmMoeDsaMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                GlmMoeDsaMLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                )
                for _ in range(self.n_routed_experts)
            ]
        )

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(
            hidden_states, topk_indices=topk_indices, topk_weights=topk_weights
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                dummy_input = hidden_states.new_empty((0, hidden_states.shape[-1]))
                dummy_output = expert(dummy_input)
                zero_contribution = (dummy_output.sum() * 0).to(
                    final_hidden_states.dtype
                )
                final_hidden_states = final_hidden_states + zero_contribution

        return final_hidden_states.type(hidden_states.dtype)


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = GlmMoeDsaMoE(config)
        else:
            self.mlp = GlmMoeDsaMLP(config)

        self.input_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (topk_indices,)  # 始终置于元组末尾，供层循环接力
        return outputs


class GlmMoeDsaPreTrainedModel(PreTrainedModel):
    config_class = GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GlmMoeDsaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GlmMoeDsaTopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()

    # WARNING: the following code is old stype of gradient checkpoint which should not use
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, GlmMoeDsaModel):
    #         module.gradient_checkpointing = value


class GlmMoeDsaRotaryEmbedding(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, device=None):
        super().__init__()
        self.config = config
        self.attention_scaling = 1.0

    def _precompute_inv_freq(self, device):
        config = self.config
        rope_theta = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.qk_rope_head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        if not hasattr(self, "inv_freq"):
            self.register_buffer(
                "inv_freq", self._precompute_inv_freq(x.device), persistent=False
            )
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._sp_size = 1

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        past_key_values_length = 0
        if len(past_key_values) > 0 and past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if self._use_flash_attention_2:
            # NPU indexers consume global segment IDs even when hidden states
            # have already been sliced for sequence parallelism. The ids are
            # derived inline from the cu boundaries (1..n per packed sample,
            # 0 beyond the last boundary), mirroring
            # torchturbo.utils.flash_attn_utils.cu_seqlens_to_segment_ids
            # while keeping this file torchturbo-free.
            if attention_mask is None and cu_seq_lens is not None:
                total = seq_length * self._sp_size + past_key_values_length
                token_positions = torch.arange(total, device=cu_seq_lens.device)
                segment_ids = torch.bucketize(token_positions, cu_seq_lens[1:], right=True) + 1
                attention_mask = segment_ids.masked_fill(
                    token_positions >= cu_seq_lens[-1], 0
                ).unsqueeze(0)
            elif attention_mask is not None and torch.all(attention_mask == 1):
                attention_mask = None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        # index share: top-k 在层间接力传递，full 层产出，shared 层复用。
        topk_indices = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx]

            if self.gradient_checkpointing and self.training:
                use_cache = False
                past_key_value = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions,
                    position_embeddings,
                    topk_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    prev_topk_indices=topk_indices,
                )

            hidden_states = layer_outputs[0]
            topk_indices = layer_outputs[-1]

            output_offset = 1
            if output_attentions:
                all_self_attns += (layer_outputs[output_offset],)
                output_offset += 1

            if use_cache:
                next_decoder_cache += (layer_outputs[output_offset],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_decoder_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:].clone()

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens_q: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seq_lens=cu_seq_lens_q,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
