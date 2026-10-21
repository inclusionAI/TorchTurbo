# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch DeepSeek V4 model for transformers library.

Uses PreAttention/InnerAttention/PostAttention structure for compilation support.
"""

import math
from typing import Optional
from weakref import WeakValueDictionary

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_deepseek_v4 import DeepseekV4Config

logger = logging.get_logger(__name__)



def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings using complex freqs_cis format."""
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(original_dtype)


def hadamard_transform_ref(x, hadamard_mat, scale=1.0):
    """Eager implementation of the Hadamard transform for indexer rotation."""
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard_mat)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard_mat: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard_mat, scale=hidden_size ** -0.5)


def generate_hadamard_matrix(dim: int) -> torch.Tensor:
    """Generate Hadamard matrix using Sylvester construction.

    Args:
        dim: Size of the Hadamard matrix (must be a power of 2)

    Returns:
        torch.Tensor: Hadamard matrix of shape (dim, dim)
    """
    # Sylvester construction: H_2n = [[H_n, H_n], [H_n, -H_n]]
    log_dim = int(math.log2(dim))
    H = torch.tensor([[1.0]], dtype=torch.float32)
    for _ in range(log_dim):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H

class DeepseekV4RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV4Compressor(nn.Module):
    """KV Cache Compressor via learned gated pooling."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int, rotate: bool = False):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate

        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        """Transform tensor for overlapping compression."""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """Forward pass for compression during prefill."""
        _, seqlen, _ = x.size()
        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype

        if seqlen < ratio:
            return None

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        freqs_cis = freqs_cis[:cutoff:ratio]

        if remainder > 0:
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))

        # Apply RoPE
        kv_rot = apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)
        kv = torch.cat([kv[..., :-self.rope_head_dim], kv_rot], dim=-1)
        return kv

    def init_weights(self, _init_std: float):
        for linear in [self.wkv, self.wgate]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.ape, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)


class DeepseekV4Indexer(nn.Module):
    """Indexer for Dynamic Sparse Attention with Hadamard rotation."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        # Indexer has its own compressor with rotate=True for learned index scoring
        self.compressor = DeepseekV4Compressor(config, compress_ratio, self.head_dim, rotate=True)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        offset: int,
    ):
        """Forward pass to compute Q/K for indexing."""
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        freq_cis = freqs_cis[:seqlen]

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q = q.clone()
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freq_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = rotate_activation(q, hadamard_mat)

        k = self.compressor(x, freqs_cis)
        if k is None:
            return None, None, None
        k = rotate_activation(k, hadamard_mat)

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        return q, k, weights

    def init_weights(self, init_std: float):
        for linear in [self.wq_b, self.weights_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        self.compressor.init_weights(init_std)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """AutoScaler that triggers backward pass and scales grad for DSA indexer loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=loss.device)
        scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_grad = torch.ones_like(loss) * scale
        return grad_output, scaled_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


class DSAIndexerLoss(nn.Module):
    """Compute DSA indexer loss for training compression indices."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        selected_main_attn_dist: torch.Tensor,
        index_score: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        valid = topk_indices >= 0
        index_score = index_score.float().masked_fill(
            ~valid, torch.finfo(torch.float32).min
        )
        index_score = F.softmax(index_score, dim=-1).masked_fill(~valid, 0)
        selected_main_attn_dist = selected_main_attn_dist.masked_fill(~valid, 0)
        selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)

        loss = F.kl_div(
            (index_score + 1e-10).log(),
            selected_main_attn_dist,
            reduction="none",
        ).sum()
        num_queries = valid.any(dim=-1).sum()
        return loss / num_queries.clamp_min(1)


class GetAttnScores(nn.Module):
    """Aggregate main attention scores for indexer loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_attn_head_per_group: int,
        attn_scale: float,
    ) -> torch.Tensor:
        if num_attn_head_per_group > 1:
            key = key.repeat_interleave(num_attn_head_per_group, dim=1)

        num_head_q = query.shape[1]
        num_head_k = key.shape[1]
        if num_head_q != num_head_k and num_head_k != 1:
            raise NotImplementedError(
                f"Only support num_head_q == num_head_k or num_head_k == 1. "
                f"Current {num_head_q=}, {num_head_k=}."
            )

        attn = (query @ key.transpose(-1, -2)) * attn_scale

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attn.masked_fill_(attention_mask, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask, 0)
        attn = attn.sum(dim=1)
        return attn


class LiLoss(nn.Module):
    """Indexer loss computation wrapper."""

    def __init__(
        self, n_heads: int, softmax_scale: float, compress_ratio: int
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        self.compress_ratio = compress_ratio
        self.get_attn_scores = GetAttnScores()
        self.compute_dsa_indexer_loss = DSAIndexerLoss()

    def forward(
        self,
        q: torch.Tensor,
        kv_compress: torch.Tensor,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        compress_topk_idxs: torch.Tensor,
        index_score: torch.Tensor | None,
        attention_masks: torch.Tensor | None,
        offset: int,
    ) -> torch.Tensor:
        compress_topk_idxs = torch.where(
            compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs - offset
        )
        safe_indices = compress_topk_idxs.clamp_min(0)
        if attention_masks is None:
            query_positions = torch.arange(q.size(1), device=q.device)
            block_ends = (
                torch.arange(kv_compress.size(1), device=q.device) + 1
            ) * self.compress_ratio
            attention_masks = block_ends > (query_positions[:, None] + 1)
            attention_masks = attention_masks.unsqueeze(0)

        main_attn_dist = self.get_attn_scores(
            q.transpose(1, 2).detach(),
            kv_compress.unsqueeze(1).detach(),
            attention_masks,
            self.n_heads,
            self.softmax_scale,
        )

        selected_main_attn_dist = torch.gather(
            main_attn_dist, dim=-1, index=safe_indices
        )

        if index_score is None:
            # Fused index selection supplies indices without differentiable scores.
            batch_indices = torch.arange(q.size(0), device=q.device)[:, None, None]
            selected_keys = k_indexer[batch_indices, safe_indices]
            index_score = torch.einsum(
                "bshd,bskd->bshk", q_indexer, selected_keys
            )
            index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

        loss = self.compute_dsa_indexer_loss(
            selected_main_attn_dist,
            index_score,
            compress_topk_idxs,
        )
        return loss


class ComputeIndexerLoss(nn.Module):
    """Wrapper for computing indexer loss."""

    def __init__(
        self,
        n_heads: int,
        softmax_scale: float,
        compress_ratio: int,
    ) -> None:
        super().__init__()
        self.li_loss = LiLoss(n_heads, softmax_scale, compress_ratio)

    def forward(
        self,
        compress_topk_idxs: torch.Tensor,
        offset: int,
        q: torch.Tensor,
        kv_compress: torch.Tensor,
        attention_masks: torch.Tensor | None,
        index_score: torch.Tensor | None,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.li_loss(
            q,
            kv_compress,
            q_indexer,
            k_indexer,
            weights,
            compress_topk_idxs,
            index_score,
            attention_masks,
            offset,
        )


class GetWindowTopkIdxs(nn.Module):
    """Get sliding window top-k indices."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, window_size: int, bsz: int, seqlen: int, device):
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        window_topk = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        window_topk = torch.where(window_topk > base, -1, window_topk)
        return window_topk.unsqueeze(0).expand(bsz, -1, -1)


class GetCompressTopkIdxs(nn.Module):
    """Get compression top-k indices."""

    def __init__(self, ratio: int = 1) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor, offset: int):
        bsz, seqlen = x.shape[0], x.shape[1]
        matrix = torch.arange(seqlen // self.ratio, device=x.device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, 1 + seqlen, device=x.device).unsqueeze(1) // self.ratio
        compress_topk = torch.where(mask, -1, matrix + offset)
        return compress_topk.unsqueeze(0).expand(bsz, -1, -1)


class LiCompute(nn.Module):
    """Compute learned compression top-k indices with Hadamard rotation."""

    def __init__(self, ratio, index_topk) -> None:
        super().__init__()
        self.ratio = ratio
        self.index_topk = index_topk

    def forward(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        seqlen: int,
        offset: int,
    ):
        end_pos = seqlen
        # Compute scores with Hadamard rotation
        index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        device = index_score.device
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        mask = (
            torch.arange(seqlen // self.ratio, device=device).unsqueeze(0)
            >= (base + 1) // self.ratio
        )
        index_score = index_score + torch.where(mask, torch.finfo(q_indexer.dtype).min, 0)

        index_score, topk_idxs = index_score.topk(
            min(self.index_topk, end_pos // self.ratio), dim=-1
        )

        mask = topk_idxs >= (base + 1) // self.ratio
        compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        return compress_topk_idxs, index_score


class PreAttentionOps(nn.Module):
    """Pre-attention operations (no parameters) - logic only."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < config.num_hidden_layers else 1

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        # Parameters passed from parent
        wq_a, q_norm, wq_b, wkv, kv_norm,
        compressor, indexer,
    ):
        rd = self.rope_head_dim
        seq_len = x.size(1)
        freq_cis = freqs_cis[:seq_len]

        # Q projection with low-rank
        qr = q_norm(wq_a(x))
        q = wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freq_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV projection
        kv = wkv(x)
        kv = kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = apply_rotary_emb(kv_rope, freq_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        offset = kv.size(1)

        # Indexer for learned compression (only when compress_ratio == 4)
        q_indexer = k_indexer = weights = None
        if self.compress_ratio > 1 and indexer is not None:
            q_indexer, k_indexer, weights = indexer(
                x.detach(), qr.detach(), freqs_cis, hadamard_mat, offset
            )

        # Compression
        kv_compress = None
        if self.compress_ratio > 1 and compressor is not None:
            kv_compress = compressor(x, freqs_cis)

        return q, kv, kv_compress, q_indexer, k_indexer, weights, offset


class SparseAttention(nn.Module):
    """Sparse attention computation with sliding window and compression."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < config.num_hidden_layers else 1
        self.softmax_scale = config.head_dim ** -0.5

        self.get_window_topk_idxs = GetWindowTopkIdxs()
        self.get_compress_topk_idxs = GetCompressTopkIdxs(self.compress_ratio)

    def forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attn_sink: torch.Tensor,
        kv_compress: Optional[torch.Tensor] = None,
        compress_topk_idxs: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _, _ = query_states.size()

        # Get window attention indices
        topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, seqlen, query_states.device)

        if self.compress_ratio > 1 and kv_compress is not None:
            offset = kv_states.size(1)
            if compress_topk_idxs is None:
                compress_topk_idxs = self.get_compress_topk_idxs(query_states, offset)
            topk_idxs = torch.cat([topk_idxs.to(kv_states.device), compress_topk_idxs.to(kv_states.device)], dim=-1)

        topk_idxs = topk_idxs.int()

        # Concatenate KV and compressed KV
        if self.compress_ratio > 1 and kv_compress is not None:
            kv_states = torch.cat([kv_states, kv_compress], dim=1)

        # Attention computation
        query_states = query_states.transpose(1, 2)
        kv_states = kv_states.unsqueeze(1)
        attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * self.softmax_scale

        # Apply top-k mask
        topk_idxs = topk_idxs.to(query_states.device)
        index_mask = torch.full(
            (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),  # +1 for attention sink
            fill_value=torch.finfo(attn_weights.dtype).min,
            dtype=attn_weights.dtype,
            device=query_states.device,
        )

        valid_mask = topk_idxs >= 0  # [bsz, seqlen, k]
        if valid_mask.any():
            b_indices, s_indices, k_indices = torch.where(valid_mask)
            valid_topk = topk_idxs[b_indices, s_indices, k_indices]
            index_mask[b_indices, 0, s_indices, valid_topk] = 0

        attn_weights = attn_weights + index_mask[..., :-1]

        # Add attention sink
        sinks = attn_sink.reshape(1, -1, 1, 1).expand(query_states.shape[0], -1, query_states.shape[2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        probs = F.softmax(combined_logits.float(), dim=-1, dtype=attn_weights.dtype)
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores, kv_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output


class InnerAttentionOps(nn.Module):
    """Inner attention operations (no parameters) - logic only."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < config.num_hidden_layers else 1

        self.sparse_attn = SparseAttention(config, layer_idx)
        if self.compress_ratio == 4:
            self.li_compute = LiCompute(self.compress_ratio, config.index_topk)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_compress: Optional[torch.Tensor],
        q_indexer: Optional[torch.Tensor],
        k_indexer: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        seqlen: int,
        offset: int,
        attn_sink: torch.Tensor,
    ):
        compress_topk_idxs = None
        index_score = None

        # Compute learned compression indices
        if (
            self.compress_ratio == 4
            and q_indexer is not None
            and k_indexer is not None
            and weights is not None
            and hasattr(self, "li_compute")
        ):
            compress_topk_idxs, index_score = self.li_compute(
                q_indexer, k_indexer, weights, seqlen, offset
            )

        # Sparse attention
        o = self.sparse_attn(q, kv, attn_sink, kv_compress, compress_topk_idxs)
        return o, compress_topk_idxs, index_score


class PostAttentionOps(nn.Module):
    """Post-attention operations (no parameters) - logic only."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.rope_head_dim = config.qk_rope_head_dim
        self.head_dim = config.head_dim

    def forward(
        self,
        o: torch.Tensor,
        freqs_cis: torch.Tensor,
        bsz: int,
        seqlen: int,
        n_local_groups: int,
        wo_a, wo_b,
    ):
        freqs_cis = freqs_cis[:o.size(1)]
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = apply_rotary_emb(o_rope, freqs_cis, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a_weight = wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
        return wo_b(o.reshape(bsz, seqlen, -1))


_FREQS_CIS_CACHE: WeakValueDictionary[tuple, torch.Tensor] = WeakValueDictionary()


def get_shared_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Share read-only RoPE tables with matching parameters and device."""
    device = torch.empty(0, device=device).device
    key = (dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device)
    cached = _FREQS_CIS_CACHE.get(key)
    if cached is None:
        # Build on the target device so meta construction and materialization
        # never share a meta table with a real-device model.
        with torch.device(device):
            cached = DeepseekV4Attention._precompute_freqs_cis(
                dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
            )
        _FREQS_CIS_CACHE[key] = cached
    return cached


class DeepseekV4Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with pre/inner/post structure.

    All parameters are defined directly in this module to maintain checkpoint compatibility.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < config.num_hidden_layers else 1

        # Pre-attention parameters
        self.q_lora_rank = config.q_lora_rank
        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.eps)

        # Post-attention parameters
        self.o_lora_rank = config.o_lora_rank
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias=False)

        # Attention sink parameter
        self.attn_sink = nn.Parameter(torch.empty(config.num_attention_heads, dtype=torch.float32))

        # Compression and indexer (when needed)
        if self.compress_ratio == 4:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            self.indexer = DeepseekV4Indexer(config, self.compress_ratio)
        elif self.compress_ratio > 1:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)

        # Operation modules (no parameters, just logic)
        self.pre_attention_ops = PreAttentionOps(config, layer_idx)
        self.inner_attention_ops = InnerAttentionOps(config, layer_idx)
        self.post_attention_ops = PostAttentionOps(config)

        # Precompute freqs_cis based on compress_ratio
        if self.compress_ratio > 1:
            original_seq_len, rope_theta = config.original_max_position_embeddings, config.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, config.rope_theta

        freqs_cis = get_shared_freqs_cis(
            self.rope_head_dim,
            config.max_position_embeddings,
            original_seq_len,
            rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Hadamard matrix for indexer rotation
        if self.compress_ratio == 4:
            hadamard_dim = 2 ** math.ceil(math.log2(config.index_head_dim))
            hadamard_mat = generate_hadamard_matrix(hadamard_dim)
            self.register_buffer("hadamard_mat", hadamard_mat, persistent=False)
        else:
            self.hadamard_mat = None

    @staticmethod
    def _find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    @classmethod
    def _find_correction_range(cls, low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val), 0, 1)

    @classmethod
    def _precompute_freqs_cis(cls, dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
        """Precomputes complex exponentials for rotary embeddings with YaRN scaling."""
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        if original_seq_len > 0:
            low, high = cls._find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
            smooth = 1 - cls._linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ):
        """Forward pass with pre/inner/post structure.

        ``cu_seq_lens`` carries packed boundaries for the sequence-parallel
        patch; the base reference path is not packed-aware and ignores it.
        """
        bsz, seqlen, _ = hidden_states.size()
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        hadamard_mat = self.hadamard_mat if self.hadamard_mat is not None else None

        # Get compressor and indexer if they exist
        compressor = getattr(self, "compressor", None)
        indexer = getattr(self, "indexer", None)

        # Pre-attention: projections
        q, kv, kv_compress, q_indexer, k_indexer, weights, offset = self.pre_attention_ops(
            hidden_states, freqs_cis, hadamard_mat,
            self.wq_a, self.q_norm, self.wq_b, self.wkv, self.kv_norm,
            compressor, indexer,
        )

        n_local_groups = self.n_groups // (self.n_heads // q.shape[2])

        # Inner attention: sparse attention computation
        o, compress_topk_idxs, index_score = self.inner_attention_ops(
            q, kv, kv_compress, q_indexer, k_indexer, weights, seqlen, offset, self.attn_sink
        )

        # Post-attention: output projection
        attn_output = self.post_attention_ops(
            o, freqs_cis, bsz, seqlen, n_local_groups, self.wo_a, self.wo_b
        )

        if use_cache:
            present_key_value = kv
        else:
            present_key_value = None

        return (
            attn_output,
            None,  # attn_weights
            present_key_value,
            compress_topk_idxs,
            offset,
            q,
            kv_compress,
            None,  # The reference loss uses the implicit compressed causal mask.
            index_score,
            q_indexer,
            k_indexer,
            weights,
        )

    def init_weights(self, init_std: float):
        # Init pre-attention params
        for linear in [self.wq_a, self.wq_b, self.wkv]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.q_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.kv_norm.weight, mean=1, std=0.02)

        # Init post-attention params
        for linear in [self.wo_a, self.wo_b]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        # Init attention sink
        nn.init.trunc_normal_(self.attn_sink, mean=0.0, std=init_std)

        # Init compressor and indexer
        if self.compress_ratio == 4:
            self.indexer.init_weights(init_std)
            self.compressor.init_weights(init_std)
        elif self.compress_ratio > 1:
            self.compressor.init_weights(init_std)


class DeepseekV4Gate(nn.Module):
    """MoE gating with hash-based routing support."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.route_scale
        self.n_routed_experts = config.n_routed_experts
        self.hash = layer_idx < config.num_hash_layers

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))

        if self.hash:
            # Hash-based routing: token_id -> expert_id mapping
            self.register_buffer(
                "tid2eid",
                torch.stack(
                    [
                        torch.randperm(config.n_routed_experts)[
                            : config.num_experts_per_tok
                        ]
                        for _ in range(config.vocab_size)
                    ]
                ).long(),
                persistent=True,
            )
            self.bias = None
        else:
            self.register_buffer(
                'bias', torch.zeros(config.n_routed_experts, dtype=torch.float32),
                persistent=True,
            )

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.LongTensor] = None):
        """Forward pass for gating."""
        scores = F.linear(hidden_states.float(), self.weight.float())

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()

        original_scores = scores

        if self.bias is not None:
            scores = scores + self.bias

        if self.hash and input_ids is not None:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]

        weights = original_scores.gather(1, indices)

        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        weights = weights * self.route_scale

        return weights, indices

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, mean=0.0, std=init_std)


class DeepseekV4Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.dim = config.hidden_size
        self.inter_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit

        self.w1 = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.w2 = nn.Linear(self.inter_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.inter_dim, bias=False)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """Forward pass for expert."""
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()

        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)

        x = F.silu(gate) * up

        if weights is not None:
            x = x * weights

        return self.w2(x.to(dtype))


class DeepseekV4MoE(nn.Module):
    """Mixture-of-Experts with hash-based routing."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok

        self.gate = DeepseekV4Gate(config, layer_idx)

        self.experts = nn.ModuleList([
            DeepseekV4Expert(config)
            for _ in range(self.n_routed_experts)
        ])

        assert config.n_shared_experts == 1
        self.shared_experts = DeepseekV4Expert(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.LongTensor] = None):
        """Forward pass for MoE."""
        shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.dim)

        if input_ids is not None:
            input_ids_flat = input_ids.flatten()
        else:
            input_ids_flat = None

        weights, indices = self.gate(hidden_states, input_ids_flat)

        y = torch.zeros_like(hidden_states, dtype=torch.float32)

        for i in range(self.n_routed_experts):
            mask = indices == i
            if mask.any():
                idx, top = torch.where(mask)
                expert = self.experts[i]
                expert_weights = weights[idx, top].unsqueeze(-1)
                y[idx] += expert(hidden_states[idx], expert_weights)

        y += self.shared_experts(hidden_states)

        return y.type_as(hidden_states).view(shape)

    def init_weights(self, init_std: float):
        self.gate.init_weights(init_std)
        for expert in self.experts:
            for linear in [expert.w1, expert.w2, expert.w3]:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for linear in [self.shared_experts.w1, self.shared_experts.w2, self.shared_experts.w3]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class DeepseekV4HCSplitSinkhorn(nn.Module):
    """HC Split Sinkhorn for hyper-connections."""

    def __init__(self, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

    def forward(self, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Forward pass for HC split sinkhorn."""
        hc_mult = self.hc_mult
        eps = self.hc_eps

        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = (
            F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0))
            + eps
        )
        post = 2 * F.sigmoid(
            post * hc_scale[1]
            + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(
            hc_mult, hc_mult
        ).unsqueeze(0).unsqueeze(0)

        comb = comb.softmax(-1) + eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)

        for _ in range(self.hc_sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + eps)

        return pre, post, comb


class DeepseekV4Head(nn.Module):
    """Output head: performs HC aggregation, normalization, and output projection.

    Similar to ParallelHead in model.py, but simplified for single-GPU inference.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dim = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        # Output projection weight (not a Linear layer, just weight for F.linear)
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ):
        """Forward pass: HC aggregation -> norm -> output projection.

        Args:
            x: Input tensor of shape (batch, seq_len, hc_mult, dim)
            hc_fn: HC head projection matrix (hc_mult, hc_mult * dim)
            hc_scale: HC head scale (1,)
            hc_base: HC head base (hc_mult,)
            norm: RMSNorm module

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        # HC aggregation
        x = self._hc_head(x, hc_fn, hc_scale, hc_base)
        # Normalization
        x = norm(x)
        # Output projection
        logits = F.linear(x.float(), self.weight)
        return logits

    def _hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Aggregate hc copies -> 1."""
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


class HcPre(nn.Module):
    """HC pre-processing: reduce hc copies -> 1.

    Kept as separate module for easy fusion operator substitution.
    """

    def __init__(self, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float, norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_split_sinkhorn = DeepseekV4HCSplitSinkhorn(hc_mult, hc_sinkhorn_iters, hc_eps)

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = self.hc_split_sinkhorn(mixes, hc_scale, hc_base)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb


class HcPost(nn.Module):
    """HC post-processing: expand 1 -> hc copies.

    Kept as separate module for easy fusion operator substitution.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


class DeepseekV4DecoderLayer(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        # Use rms_norm_eps for norm, hc_eps is separate
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters

        # Named 'attn' and 'ffn' to match model.py checkpoint format
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)

        # Named 'attn_norm' and 'ffn_norm' to match model.py checkpoint format
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)

        # Hyper-Connection parameters
        hc_mult = self.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # HC pre/post modules (for easy fusion operator substitution)
        self.hc_pre = HcPre(self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps, self.norm_eps)
        self.hc_post = HcPost()

        # Indexer loss computation
        compress_ratio = config.compress_ratios[layer_idx]
        if layer_idx < config.num_hidden_layers and compress_ratio > 1:
            self.cal_index_loss = ComputeIndexerLoss(
                config.num_attention_heads,
                config.head_dim ** -0.5,
                compress_ratio,
            )
        else:
            self.cal_index_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ):
        """Forward pass with training logic."""
        # Attention with HC
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.attn_norm(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cu_seq_lens=cu_seq_lens,
        )

        attn_output = attn_outputs[0]
        (
            compress_topk_idxs,
            offset,
            q,
            kv_compress,
            compressed_attention_mask,
            index_score,
            q_indexer,
            k_indexer,
            weights,
        ) = attn_outputs[3:]

        # Compute indexer loss for training
        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff
        if self.cal_index_loss is not None and self.training and indexer_loss_coeff > 0.0:
            if compress_topk_idxs is not None:
                loss = self.cal_index_loss(
                    compress_topk_idxs,
                    offset,
                    q,
                    kv_compress,
                    compressed_attention_mask,
                    index_score,
                    q_indexer,
                    k_indexer,
                    weights,
                )
                # Apply loss scaler for proper gradient scaling
                attn_output = DSAIndexerLossAutoScaler.apply(
                    attn_output, loss * indexer_loss_coeff
                )

        hidden_states = self.hc_post(attn_output, residual, post, comb)

        # MLP with HC
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, input_ids)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        return outputs

    def init_weights(self, init_std: float):
        for norm in [self.attn_norm, self.ffn_norm]:
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        self.attn.init_weights(init_std)
        self.ffn.init_weights(init_std)
        for param in [self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale,
                      self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale]:
            if param is not None:
                nn.init.trunc_normal_(param, mean=0.0, std=0.02)


class DeepseekV4PreTrainedModel(PreTrainedModel):
    """Base class for DeepSeek V4 models."""

    config_class = DeepseekV4Config
    base_model_prefix = ""  # Deepseek HF weights don't have 'model.' prefix
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["DeepseekV4DecoderLayer"]
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
        elif isinstance(module, DeepseekV4RMSNorm):
            module.weight.data.fill_(1.0)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    """DeepSeek V4 for Causal Language Modeling.

    Note: This class directly inherits all modules from DeepseekV4Model to match
    the checkpoint structure (no 'model.' prefix in weights).
    """

    _tied_weights_keys = ["head.weight"]

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        # Directly define modules here (not nested under self.model) to match checkpoint
        # Named 'embed' to match model.py checkpoint format
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head (performs HC aggregation + norm + output projection)
        self.head = DeepseekV4Head(config)

        # HC head parameters (independent params, passed to head.forward)
        hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        # new_embeddings should be compatible with head.weight
        self.head.weight = new_embeddings.weight if hasattr(new_embeddings, 'weight') else new_embeddings

    def set_decoder(self, decoder):
        # Copy attributes from decoder to self
        self.embed = decoder.embed
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.head = decoder.head
        self.hc_head_fn = decoder.hc_head_fn
        self.hc_head_base = decoder.hc_head_base
        self.hc_head_scale = decoder.hc_head_scale

    def get_decoder(self):
        # Return a shallow wrapper that mimics DeepseekV4Model interface
        return self

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, DeepseekV4DecoderLayer):
            module.init_weights(std)
        elif isinstance(module, DeepseekV4Head):
            module.init_weights(std)
        elif module is self:
            # hc_head_* are bare parameters on the root module, so apply()
            # never reaches them through a child module. Without this branch
            # they stay torch.empty and forward yields NaN logits.
            self.init_hc_head_weights(std)
        else:
            super()._init_weights(module)

    def init_hc_head_weights(self, init_std: float):
        """Initialize HC head parameters."""
        nn.init.trunc_normal_(self.hc_head_fn, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.hc_head_base, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.hc_head_scale, mean=0.0, std=init_std)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
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

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens_q: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare inputs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length + past_key_values_length),
        #         dtype=torch.bool,
        #         device=inputs_embeds.device,
        #     )

        # 4D causal mask
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask,
        #     (batch_size, seq_length),
        #     inputs_embeds,
        #     past_key_values_length,
        # )

        # Expand to hc copies for Hyper-Connections
        hidden_states = inputs_embeds.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx]

            if self.gradient_checkpointing and self.training:
                use_cache = False
                past_key_value = None
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions,
                    input_ids if idx < self.config.num_hash_layers else None,
                    cu_seq_lens_q,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    input_ids=input_ids if idx < self.config.num_hash_layers else None,
                    cu_seq_lens=cu_seq_lens_q,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

        # Apply head: HC aggregation -> norm -> output projection -> logits
        logits = self.head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm,
        )

        if output_hidden_states:
            all_hidden_states += (hidden_states.mean(dim=2),)

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
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_self_attns,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
