"""Standard MTP and DSpark modules for DeepSeek-V4."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from torchturbo.models.modules.attention.ulysses import (
    all_gather_sp_with_grad_reduce,
)
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

from ..modules.moe.moe_layer import BaseMoELayer, MoERuntimeInputs
from .configuration_deepseek_v4 import DeepseekV4Config
from .modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4RMSNorm,
)
from .packing_utils import apply_rotary_emb

# Draft rows per chunk while the DSpark loss holds full-vocabulary tensors.
_DSPARK_LOSS_CHUNK_SIZE = 64


class DeepseekV4MTPDecoderLayer(DeepseekV4DecoderLayer):
    """One standard DeepSeek-V4 next-token prediction layer."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hnorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def init_weights(self, init_std: float):
        super().init_weights(init_std)
        nn.init.trunc_normal_(self.e_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.h_proj.weight, mean=0.0, std=init_std)
        for param in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            nn.init.trunc_normal_(param, mean=0.0, std=init_std)

    def forward(
        self,
        backbone_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        shared_head: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.h_proj(self.hnorm(backbone_hidden_states))
        hidden_states = hidden_states + self.e_proj(
            self.enorm(inputs_embeds)
        ).unsqueeze(-2)

        hidden_states = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_ids=input_ids,
            cu_seq_lens=cu_seq_lens,
        )[0]
        # NPU F.linear saves its compute-dtype HC weights. Materialize that
        # dtype before the shared head so checkpoint forward and recompute see
        # identical saved-tensor metadata.
        hc_dtype = hidden_states.dtype
        return hidden_states, shared_head(
            hidden_states,
            self.hc_head_fn.to(hc_dtype),
            self.hc_head_scale.to(hc_dtype),
            self.hc_head_base.to(hc_dtype),
            self.norm,
        )


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """Attention from DSpark draft blocks to target context and their block.

    Each block attends to the ``sliding_window`` target positions before its
    anchor and, bidirectionally, to every position of the same block.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        context_anchors: torch.LongTensor,
        context_mask: torch.Tensor,
        block_mask: torch.Tensor,
        query_position_ids: torch.LongTensor,
        context_position_ids: torch.LongTensor,
    ):
        """Attend from draft blocks to their context window and block.

        Args:
            hidden_states: Draft states ``[batch, num_blocks * block_size, hidden]``.
            context_hidden_states: Projected target states
                ``[batch, context_length, hidden]``.
            context_anchors: Anchor index into the context ``[batch, num_blocks]``.
            context_mask: Visible context slots ``[batch, num_blocks, window]``.
                Slot ``k`` holds context index ``anchor - 1 - k``.
            block_mask: Blocks that hold a sampled anchor ``[batch, num_blocks]``.
            query_position_ids: RoPE positions of the draft states.
            context_position_ids: RoPE positions of the context states.
        """
        rd = self.rope_head_dim
        batch_size, draft_length, _ = hidden_states.shape
        num_blocks = context_anchors.size(1)
        block_size = draft_length // num_blocks
        window = context_mask.size(-1)

        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = apply_rotary_emb(
            q_rope, self.freqs_cis, position_ids=query_position_ids
        )
        q = torch.cat([q_nope, q_rope], dim=-1)

        context_kv = self.kv_norm(self.wkv(context_hidden_states))
        draft_kv = self.kv_norm(self.wkv(hidden_states))
        context_nope, context_rope = torch.split(context_kv, [self.head_dim - rd, rd], dim=-1)
        draft_nope, draft_rope = torch.split(draft_kv, [self.head_dim - rd, rd], dim=-1)
        context_rope = apply_rotary_emb(
            context_rope, self.freqs_cis, position_ids=context_position_ids
        )
        draft_rope = apply_rotary_emb(
            draft_rope, self.freqs_cis, position_ids=query_position_ids
        )
        context_kv = torch.cat([context_nope, context_rope], dim=-1)
        draft_kv = torch.cat([draft_nope, draft_rope], dim=-1)

        # Gather each block's window from the context instead of attending to
        # (or materializing gradients for) the whole context per block.
        context_indices = context_anchors.unsqueeze(-1) - 1 - torch.arange(
            window, device=context_anchors.device
        )
        window_kv = context_kv.gather(
            1,
            context_indices.view(batch_size, -1, 1).expand(-1, -1, self.head_dim),
        ).view(batch_size, num_blocks, window, self.head_dim)
        block_kv = draft_kv.view(batch_size, num_blocks, block_size, self.head_dim)
        kv = torch.cat([window_kv, block_kv], dim=2).unsqueeze(2)
        visible = torch.cat(
            [context_mask, block_mask.unsqueeze(-1).expand(-1, -1, block_size)],
            dim=-1,
        )

        q = q.view(
            batch_size, num_blocks, block_size, self.n_heads, self.head_dim
        )
        attn_weights = torch.matmul(
            q, kv.transpose(3, 4)
        ).mul_(self.config.head_dim ** -0.5)
        attn_weights.masked_fill_(
            ~visible[:, :, None, None, :],
            torch.finfo(attn_weights.dtype).min,
        )

        combined_logits = torch.cat(
            [attn_weights, self.attn_sink.reshape(1, 1, 1, -1, 1).expand_as(
                attn_weights[..., :1]
            )],
            dim=-1,
        )
        combined_logits -= combined_logits.max(
            dim=-1, keepdim=True
        ).values
        probs = F.softmax(combined_logits.float(), dim=-1, dtype=attn_weights.dtype)
        o = torch.matmul(probs[..., :-1], kv)
        o = o.reshape(batch_size, draft_length, self.n_heads, self.head_dim)

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = apply_rotary_emb(
            o_rope,
            self.freqs_cis,
            position_ids=query_position_ids,
            inverse=True,
        )
        o = torch.cat([o_nope, o_rope], dim=-1)
        o = o.view(batch_size, draft_length, self.n_groups, -1)
        wo_a_weight = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
        return self.wo_b(o.reshape(batch_size, draft_length, -1))


class DeepseekV4DSparkMarkovHead(nn.Module):
    """Low-rank first-order transition head used by DSpark."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.markov_w1 = nn.Embedding(
            config.vocab_size, config.dspark_markov_rank
        )
        self.markov_w2 = nn.Linear(
            config.dspark_markov_rank, config.vocab_size, bias=False
        )

    def forward(self, previous_token_ids: torch.LongTensor):
        markov_embed = self.markov_w1(previous_token_ids)
        return self.markov_w2(markov_embed), markov_embed


class DeepseekV4DSparkConfidenceHead(nn.Module):
    """Predict conditional target-acceptance logits for DSpark positions."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.proj = nn.Linear(
            config.hidden_size + config.dspark_markov_rank, 1, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed: torch.Tensor,
    ):
        features = torch.cat([hidden_states, markov_embed], dim=-1)
        confidence = self.proj(features.to(hidden_states.dtype))
        return confidence.float().squeeze(-1)


class DeepseekV4DSparkDecoderLayer(DeepseekV4DecoderLayer):
    """One DSpark draft stage stored in the checkpoint's mtp.* namespace."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn = DeepseekV4DSparkAttention(config, layer_idx)
        self.stage_idx = layer_idx - config.num_hidden_layers
        self.dspark_num_layers = config.dspark_num_layers

        if self.stage_idx == 0:
            self.main_proj = nn.Linear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                bias=False,
            )
            self.main_norm = DeepseekV4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        if self.stage_idx == self.dspark_num_layers - 1:
            self.norm = DeepseekV4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.markov_head = DeepseekV4DSparkMarkovHead(config)
            self.confidence_head = DeepseekV4DSparkConfidenceHead(config)
            hc_mult = config.hc_mult
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def init_weights(self, init_std: float):
        super().init_weights(init_std)
        if self.stage_idx == 0:
            nn.init.trunc_normal_(self.main_proj.weight, mean=0.0, std=init_std)
        if self.stage_idx == self.dspark_num_layers - 1:
            nn.init.trunc_normal_(
                self.markov_head.markov_w1.weight, mean=0.0, std=init_std
            )
            nn.init.trunc_normal_(
                self.markov_head.markov_w2.weight, mean=0.0, std=init_std
            )
            nn.init.trunc_normal_(
                self.confidence_head.proj.weight, mean=0.0, std=init_std
            )
            for param in [
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ]:
                nn.init.trunc_normal_(param, mean=0.0, std=init_std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: Optional[torch.Tensor],
        main_hidden_states: Optional[torch.Tensor],
        context_anchors: torch.LongTensor,
        context_mask: torch.Tensor,
        block_mask: torch.Tensor,
        query_position_ids: torch.LongTensor,
        context_position_ids: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        previous_token_ids: Optional[torch.LongTensor] = None,
        shared_head: Optional[nn.Module] = None,
    ):
        if self.stage_idx == 0:
            context_hidden_states = self.main_norm(
                self.main_proj(main_hidden_states)
            )

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.attn_norm(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            context_hidden_states=context_hidden_states,
            context_anchors=context_anchors,
            context_mask=context_mask,
            block_mask=block_mask,
            query_position_ids=query_position_ids,
            context_position_ids=context_position_ids,
        )
        hidden_states = self.hc_post(attn_output, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, input_ids)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        if self.stage_idx == self.dspark_num_layers - 1:
            hc_dtype = hidden_states.dtype
            hc_head_fn = self.hc_head_fn.to(hc_dtype)
            hc_head_scale = self.hc_head_scale.to(hc_dtype)
            hc_head_base = self.hc_head_base.to(hc_dtype)
            aggregated_hidden = shared_head._hc_head(
                hidden_states,
                hc_head_fn,
                hc_head_scale,
                hc_head_base,
            )
            confidence_hidden = self.norm(aggregated_hidden)
            logits = shared_head(
                hidden_states,
                hc_head_fn,
                hc_head_scale,
                hc_head_base,
                self.norm,
            )
            markov_bias, markov_embed = self.markov_head(previous_token_ids)
            # Linear does not save its output for backward, so adding in place
            # avoids another full-vocabulary fp32 buffer.
            logits.add_(markov_bias)
            return (
                logits,
                self.confidence_head(confidence_hidden, markov_embed),
            )
        if self.stage_idx == 0:
            return hidden_states, context_hidden_states
        return hidden_states


@dataclass
class DeepseekV4MTPCausalLMOutputWithPast(CausalLMOutputWithPast):
    mtp_logits: Optional[list[torch.FloatTensor]] = None
    dspark_loss_fn: Optional[
        Callable[[torch.LongTensor], torch.FloatTensor]
    ] = None


class DeepseekV4MultiTokenPredictor(nn.ModuleList):
    """Build the standard-MTP or DSpark layers requested by ``config``.

    ``nn.ModuleList`` is subclassed so checkpoint names remain ``mtp.0`` instead
    of adding a predictor-owned ``layers`` level. DSpark training metrics of
    the latest micro-batch are exposed through :meth:`pop_metrics`.
    """

    def __init__(self, config: DeepseekV4Config):
        # main_proj consumes the target features in forward (ascending) order.
        target_layer_ids = list(config.dspark_target_layer_ids)
        if config.dspark_block_size > 0 and (
            not target_layer_ids
            or target_layer_ids != sorted(set(target_layer_ids))
            or target_layer_ids[0] < 0
            or target_layer_ids[-1] >= config.num_hidden_layers
        ):
            raise ValueError(
                "dspark_target_layer_ids must be strictly increasing backbone layer "
                f"indices below num_hidden_layers={config.num_hidden_layers}, "
                f"got {target_layer_ids}"
            )
        layer_cls = (
            DeepseekV4DSparkDecoderLayer
            if config.dspark_block_size > 0
            else DeepseekV4MTPDecoderLayer
        )
        if config.dspark_block_size > 0:
            num_layers = config.dspark_num_layers
            if num_layers <= 0:
                raise ValueError(
                    "Cannot infer DeepSeek-V4 DSpark layer count: "
                    "model.safetensors.index.json has no mtp.<stage> keys and "
                    "config.dspark_num_layers is unset."
                )
            config.dspark_num_layers = num_layers
        else:
            num_layers = config.num_nextn_predict_layers
        super().__init__(
            layer_cls(config, config.num_hidden_layers + stage_idx)
            for stage_idx in range(num_layers)
        )
        for layer in self:
            layer.init_weights(config.initializer_range)
        self._metrics: Optional[dict[str, torch.Tensor]] = None

    def pop_metrics(self) -> dict[str, torch.Tensor]:
        metrics = self._metrics or {}
        self._metrics = None
        return metrics


class _DSparkDraftLoss(torch.autograd.Function):
    """Weighted draft cross entropy plus draft-to-target L1 over row chunks.

    The logits gradient is computed chunk by chunk during forward, so no
    full-vocabulary probabilities are kept for backward. Also returns the
    raw sums and the per-row accept rate ``1 - TV(draft, target)``, all
    without gradient.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.LongTensor,
        weights: torch.Tensor,
        target_logits_fn: Callable[[int, int], torch.Tensor],
        ce_alpha: float,
        l1_alpha: float,
    ):
        num_rows = logits.size(0)
        grad_logits = torch.empty_like(logits) if ctx.needs_input_grad[0] else None
        ce_sum = logits.new_zeros((), dtype=torch.float32)
        l1_sum = logits.new_zeros((), dtype=torch.float32)
        accept_rate = logits.new_empty(num_rows, dtype=torch.float32)
        for start in range(0, num_rows, _DSPARK_LOSS_CHUNK_SIZE):
            end = min(start + _DSPARK_LOSS_CHUNK_SIZE, num_rows)
            chunk_targets = targets[start:end]
            chunk_weights = weights[start:end]
            log_probs = torch.log_softmax(logits[start:end].float(), dim=-1)
            probs = log_probs.exp()
            target_probs = torch.softmax(target_logits_fn(start, end).float(), dim=-1)
            diff = probs - target_probs
            l1 = diff.abs().sum(dim=-1)
            ce = -log_probs.gather(1, chunk_targets.unsqueeze(1)).squeeze(1)
            ce_sum += (ce * chunk_weights).sum()
            l1_sum += (l1 * chunk_weights).sum()
            accept_rate[start:end] = (1.0 - 0.5 * l1).clamp(0.0, 1.0)

            if grad_logits is not None:
                # d sum|p - q| / dz = p * (s - <p, s>) with s = sign(p - q).
                sign = diff.sign_()
                grad = probs * (sign - (probs * sign).sum(dim=-1, keepdim=True))
                grad.mul_(l1_alpha)
                # d(-log p[t]) / dz = p - onehot(t).
                grad.add_(probs, alpha=ce_alpha)
                rows = torch.arange(end - start, device=logits.device)
                grad[rows, chunk_targets] -= ce_alpha
                grad_logits[start:end] = grad * chunk_weights.unsqueeze(1)

        ctx.grad_logits = grad_logits
        ctx.mark_non_differentiable(ce_sum, l1_sum, accept_rate)
        loss = ce_alpha * ce_sum + l1_alpha * l1_sum
        return loss, ce_sum, l1_sum, accept_rate

    @staticmethod
    def backward(ctx, grad_loss, _grad_ce_sum, _grad_l1_sum, _grad_accept_rate):
        grad_logits = ctx.grad_logits
        ctx.grad_logits = None
        grad_logits.mul_(grad_loss.to(grad_logits.dtype))
        return grad_logits, None, None, None, None, None


def _gather_tokens(tensor: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
    """Gather ``tensor[b, index[b, ...]]`` from a ``[batch, sequence]`` tensor."""
    return tensor.gather(1, index.flatten(1)).view_as(index)


class DeepseekV4MTPForwarder:
    """Run standard MTP or DSpark training around the DeepSeek-V4 backbone.

    Holds the sequence-parallel context: ``input_ids`` and ``position_ids``
    arrive as this rank's shard, while labels and packed boundaries cover the
    global sequence.
    """

    def __init__(
        self,
        sp_group: Optional[dist.ProcessGroup] = None,
        sp_size: int = 1,
        sp_rank: int = 0,
    ):
        self.sp_group = sp_group
        self.sp_size = sp_size
        self.sp_rank = sp_rank

    def forward(
        self,
        model,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cu_seq_lens_q: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mtp = model._modules.get("mtp")
        run_mtp = model.training and mtp is not None and input_ids is not None
        use_dspark = run_mtp and model.config.dspark_block_size > 0
        if not run_mtp or (use_dspark and not model.config.dspark_enabled):
            return DeepseekV4ForCausalLM.forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cu_seq_lens_q=cu_seq_lens_q,
                **kwargs,
            )

        if position_ids is None:
            local_length = input_ids.size(1)
            position_ids = torch.arange(
                self.sp_rank * local_length,
                (self.sp_rank + 1) * local_length,
                device=input_ids.device,
            ).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size(0), -1)

        target_layer_ids = model.config.dspark_target_layer_ids if use_dspark else ()
        target_hidden_states = []
        final_hidden_states = []
        handles = []
        for idx, layer in enumerate(model.layers):
            if idx in target_layer_ids:
                # DSpark trains through target features averaged over HC copies.
                handles.append(layer.register_forward_hook(
                    lambda _module, _args, output:
                        target_hidden_states.append(output[0].mean(dim=2))
                ))
            if idx == len(model.layers) - 1:
                handles.append(layer.register_forward_hook(
                    lambda _module, _args, output:
                        final_hidden_states.append(output[0])
                ))
        try:
            outputs = DeepseekV4ForCausalLM.forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cu_seq_lens_q=cu_seq_lens_q,
                **kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        hidden_states = final_hidden_states[0]
        mtp_logits = None
        dspark_loss_fn = None
        if use_dspark:
            main_hidden_states = torch.cat(target_hidden_states, dim=-1)
            # Reference distributions are fixed targets for the draft loss.
            backbone_hidden_states = hidden_states.detach()

            def dspark_loss_fn(labels):
                return self._dspark_loss(
                    model,
                    input_ids,
                    position_ids,
                    cu_seq_lens_q,
                    labels,
                    main_hidden_states,
                    backbone_hidden_states,
                )
        else:
            mtp_logits = self._mtp_logits(
                model,
                hidden_states,
                input_ids,
                attention_mask,
                position_ids,
                cu_seq_lens_q,
            )

        return DeepseekV4MTPCausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mtp_logits=mtp_logits,
            dspark_loss_fn=dspark_loss_fn,
        )

    def _mtp_logits(
        self,
        model,
        hidden_states,
        input_ids,
        attention_mask,
        position_ids,
        cu_seq_lens,
    ):
        """Run the standard MTP stages; stage ``k`` predicts token ``t + k + 1``."""
        local_length = input_ids.size(-1)
        global_input_ids = all_gather_sp_with_grad_reduce(
            input_ids, dim=1, sp_group=self.sp_group, sp_size=self.sp_size
        )
        if cu_seq_lens is None:
            segment_ids = torch.ones_like(global_input_ids)
        else:
            segment_ids = cu_seqlens_to_segment_ids(
                cu_seq_lens, global_input_ids.size(-1)
            ).to(input_ids.device)
        mtp_start = self.sp_rank * local_length
        local_segment_ids = segment_ids[..., mtp_start:mtp_start + local_length]
        mtp_logits = []
        for layer_idx, mtp_layer in enumerate(model.mtp, start=1):
            start = mtp_start + layer_idx
            mtp_input_ids = global_input_ids[..., start:start + local_length]
            mtp_input_ids = F.pad(
                mtp_input_ids,
                (0, local_length - mtp_input_ids.size(-1)),
                value=model.padding_idx,
            )
            shifted_segment_ids = segment_ids[..., start:start + local_length]
            shifted_segment_ids = F.pad(
                shifted_segment_ids,
                (0, local_length - shifted_segment_ids.size(-1)),
                value=0,
            )
            # MTP inputs must stay within their original sample. Zero segment
            # IDs exclude both existing padding and the newly padded tail.
            padding_mask = (shifted_segment_ids == 0) | (
                shifted_segment_ids != local_segment_ids
            )
            runtime_inputs = MoERuntimeInputs(padding_mask=padding_mask)
            for module in mtp_layer.modules():
                if isinstance(module, BaseMoELayer):
                    module.set_runtime_inputs(runtime_inputs)
            embedding_hidden_states = model.embed(mtp_input_ids)
            hidden_states, mtp_logits_i = mtp_layer(
                hidden_states,
                embedding_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=mtp_input_ids,
                shared_head=model.head,
                cu_seq_lens=cu_seq_lens,
            )
            mtp_logits.append(mtp_logits_i)
        return mtp_logits

    def _dspark_loss(
        self,
        model,
        input_ids,
        position_ids,
        cu_seq_lens,
        labels,
        main_hidden_states,
        backbone_hidden_states,
    ):
        """Return the DSpark loss of the global batch as a weighted mean.

        ``labels`` and ``cu_seq_lens`` describe the global sequence; the other
        tensors hold this rank's SP shard. Draft position ``k`` of a block
        predicts the label at ``anchor + 1 + k``.
        """
        config = model.config
        device = input_ids.device
        batch_size, local_length = input_ids.shape
        block_size = config.dspark_block_size
        window = config.sliding_window
        if cu_seq_lens is None:
            segment_ids = torch.ones_like(labels)
        else:
            segment_ids = cu_seqlens_to_segment_ids(
                cu_seq_lens, labels.size(1)
            ).to(device)

        global_anchors, local_anchors, block_mask = self._sample_anchors(
            labels, segment_ids, config.dspark_num_anchors, local_length
        )
        anchor_segments = _gather_tokens(segment_ids, global_anchors).unsqueeze(-1)

        # Supervision stops at the first ignored label or sample boundary.
        label_indices = global_anchors.unsqueeze(-1) + torch.arange(
            1, block_size + 1, device=device
        )
        safe_label_indices = label_indices.clamp(max=labels.size(1) - 1)
        block_labels = _gather_tokens(labels, safe_label_indices)
        valid_labels = (block_labels != -100) & (label_indices < labels.size(1))
        valid_labels &= _gather_tokens(segment_ids, safe_label_indices) == anchor_segments
        valid_labels &= block_mask.unsqueeze(-1)
        eval_mask = valid_labels.to(torch.int32).cumprod(dim=-1).bool()
        target_ids = block_labels.masked_fill(~eval_mask, 0)

        # Context slot k holds the target state at anchor - 1 - k.
        context_indices = global_anchors.unsqueeze(-1) - 1 - torch.arange(
            window, device=device
        )
        context_mask = (context_indices >= 0) & block_mask.unsqueeze(-1)
        context_mask &= (
            _gather_tokens(segment_ids, context_indices.clamp(min=0))
            == anchor_segments
        )

        # Prepend the previous window so anchors near a shard start see the
        # same context as the target model's sliding window.
        main_hidden_states = torch.cat(
            [self._sp_halo(main_hidden_states, window, previous=True), main_hidden_states],
            dim=1,
        )
        context_position_ids = torch.cat(
            [self._sp_halo(position_ids, window, previous=True), position_ids],
            dim=1,
        )
        context_anchors = local_anchors + window

        # Blocks use consecutive positions from the anchor, as in decoding, even
        # past a sample end. Positions beyond the RoPE table are never
        # supervised, so clamping them is safe.
        anchor_positions = _gather_tokens(position_ids, local_anchors)
        query_position_ids = anchor_positions.unsqueeze(-1) + torch.arange(
            block_size, device=device
        )
        query_position_ids = query_position_ids.clamp(
            max=config.max_position_embeddings - 1
        )

        anchor_token_ids = _gather_tokens(input_ids, local_anchors)
        previous_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        draft_input_ids = torch.full_like(
            previous_token_ids, config.dspark_noise_token_id
        )
        draft_input_ids[..., 0] = anchor_token_ids
        draft_input_ids = draft_input_ids.view(batch_size, -1)

        padding_mask = ~block_mask.repeat_interleave(block_size, dim=1)
        # Placeholder blocks still run for collective participation, but are
        # excluded from router losses and expert-bias statistics.
        draft_runtime_inputs = MoERuntimeInputs(padding_mask=padding_mask)
        for module in model.mtp.modules():
            if isinstance(module, BaseMoELayer):
                module.set_runtime_inputs(draft_runtime_inputs)

        hidden_states = model.embed(draft_input_ids)
        # Reentrant checkpointing needs a grad input even with frozen embeddings.
        hidden_states.requires_grad_(True)
        hidden_states = hidden_states.unsqueeze(2).repeat(
            1, 1, config.hc_mult, 1
        )

        context_hidden_states = None
        last_stage_idx = len(model.mtp) - 1
        for stage_idx, dspark_layer in enumerate(model.mtp):
            layer_kwargs = {}
            if stage_idx == last_stage_idx:
                layer_kwargs["previous_token_ids"] = previous_token_ids.view(
                    batch_size, -1
                )
                layer_kwargs["shared_head"] = model._modules["head"]

            # Pass both context paths positionally for reentrant checkpointing.
            layer_outputs = dspark_layer(
                hidden_states,
                context_hidden_states,
                main_hidden_states if stage_idx == 0 else None,
                context_anchors=context_anchors,
                context_mask=context_mask,
                block_mask=block_mask,
                query_position_ids=query_position_ids.view(batch_size, -1),
                context_position_ids=context_position_ids,
                input_ids=draft_input_ids,
                **layer_kwargs,
            )
            if stage_idx == last_stage_idx:
                dspark_logits, confidence_logits = layer_outputs
            elif stage_idx == 0:
                hidden_states, context_hidden_states = layer_outputs
            else:
                hidden_states = layer_outputs

        return self._dspark_objective(
            model,
            dspark_logits,
            confidence_logits,
            target_ids,
            eval_mask,
            local_anchors,
            backbone_hidden_states,
        )

    def _sample_anchors(self, labels, segment_ids, num_anchors, local_length):
        """Sample anchors shared by the SP group and keep this rank's anchors.

        Returns global anchor positions, local anchor positions and the mask of
        blocks that hold an anchor, each shaped ``[batch, num_blocks]``.
        """
        sequence_length = labels.size(1)
        if sequence_length < 2:
            # No row position can hold an anchor. Keep one masked block so every
            # rank still runs the draft layers and their collectives.
            empty_anchors = labels.new_zeros(labels.size(0), 1)
            return empty_anchors, empty_anchors, empty_anchors.bool()

        # An anchor needs a supervised next token in the same sample.
        candidates = (labels[:, :-1] != -100) & (labels[:, 1:] != -100)
        candidates &= segment_ids[:, :-1] != 0
        candidates &= segment_ids[:, :-1] == segment_ids[:, 1:]

        num_slots = min(num_anchors, candidates.size(1))
        scores = torch.rand(candidates.shape, device=labels.device)
        scores = scores.masked_fill(~candidates, 2.0)
        anchors = scores.topk(num_slots, dim=1, largest=False).indices
        has_anchor = (
            torch.arange(num_slots, device=labels.device)
            < candidates.sum(dim=1, keepdim=True)
        )
        # Empty slots hold sequence_length, which sorts after every candidate.
        anchors = anchors.masked_fill(~has_anchor, sequence_length).sort(dim=1).values
        if self.sp_size > 1:
            # Every SP rank trains on the anchors sampled by SP rank 0.
            source_rank = dist.get_global_rank(self.sp_group, 0)
            dist.broadcast(anchors, source_rank, group=self.sp_group)

        # Keep DSpark block capacity uniform across SP ranks. Padding blocks
        # are masked out of the objective below.
        is_local = anchors < sequence_length
        is_local &= anchors.div(local_length, rounding_mode="floor") == self.sp_rank
        local_blocks = is_local.sum(dim=1).max()
        if self.sp_size > 1:
            dist.all_reduce(
                local_blocks,
                op=dist.ReduceOp.MAX,
                group=self.sp_group,
            )
        num_blocks = max(1, int(local_blocks.item()))
        # Move this rank's anchors to the front while keeping them sorted.
        slots = torch.arange(num_slots, device=labels.device).expand_as(anchors)
        order = slots.masked_fill(~is_local, num_slots).sort(dim=1).values
        order = order[:, :num_blocks]
        block_mask = order < num_slots
        global_anchors = anchors.gather(1, order.clamp(max=num_slots - 1))
        global_anchors = global_anchors.masked_fill(~block_mask, 0)
        local_anchors = global_anchors - self.sp_rank * local_length
        local_anchors = local_anchors.masked_fill(~block_mask, 0)
        return global_anchors, local_anchors, block_mask

    def _sp_halo(self, tensor, count, previous):
        """Return ``count`` positions next to this rank's shard along dim 1.

        ``previous`` selects the positions before the shard, otherwise the ones
        after it. Positions outside the global sequence are zero.
        """
        if self.sp_size == 1:
            return tensor.new_zeros(tensor.size(0), count, *tensor.shape[2:])

        # Each rank contributes its boundary tokens. A halo longer than a shard
        # receives whole neighbouring shards, so the stitched result is exact.
        edge = min(count, tensor.size(1))
        piece = tensor[:, -edge:] if previous else tensor[:, :edge]
        pieces = all_gather_sp_with_grad_reduce(
            piece, dim=1, sp_group=self.sp_group, sp_size=self.sp_size
        )
        if previous:
            end = self.sp_rank * edge
            context = pieces[:, max(0, end - count):end]
            padding = (count - context.size(1), 0)
        else:
            start = (self.sp_rank + 1) * edge
            context = pieces[:, start:start + count]
            padding = (0, count - context.size(1))
        # Even empty boundary slices retain the collective's backward edge so
        # every rank participates in reducing context gradients.
        return F.pad(context, (0, 0) * (tensor.ndim - 2) + padding)

    def _dspark_objective(
        self,
        model,
        dspark_logits,
        confidence_logits,
        target_ids,
        eval_mask,
        local_anchors,
        backbone_hidden_states,
    ):
        """Combine the draft CE, distribution-match and confidence losses."""
        config = model.config
        device = eval_mask.device
        batch_size, _, block_size = eval_mask.shape
        weights = eval_mask.to(torch.float32)
        if config.dspark_loss_decay_gamma > 0.0:
            positions = torch.arange(block_size, device=device, dtype=torch.float32)
            weights = weights * torch.exp(-positions / config.dspark_loss_decay_gamma)

        # Target distributions at the draft positions, which may extend into
        # the next shard.
        backbone_hidden_states = torch.cat(
            [
                backbone_hidden_states,
                self._sp_halo(backbone_hidden_states, block_size, previous=False),
            ],
            dim=1,
        )
        target_rows = local_anchors.unsqueeze(-1) + torch.arange(block_size, device=device)
        batch_offsets = torch.arange(batch_size, device=device).view(-1, 1, 1)
        target_rows = (target_rows + batch_offsets * backbone_hidden_states.size(1)).flatten()
        flat_backbone = backbone_hidden_states.flatten(0, 1)
        head = model._modules["head"]

        def target_logits(start: int, end: int) -> torch.Tensor:
            target_hidden = flat_backbone[target_rows[start:end]].unsqueeze(0)
            return head(
                target_hidden,
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.norm,
            )[0]

        draft_loss, ce_sum, l1_sum, accept_rate = _DSparkDraftLoss.apply(
            dspark_logits.view(-1, config.vocab_size),
            target_ids.flatten(),
            weights.flatten(),
            target_logits,
            config.dspark_ce_loss_alpha,
            config.dspark_l1_loss_alpha,
        )
        accept_rate = accept_rate.view_as(weights)
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits.float().view_as(weights),
            accept_rate,
            reduction="none",
        )
        confidence_sum = (confidence_loss * weights).sum()
        loss_sum = draft_loss + config.dspark_confidence_loss_alpha * confidence_sum

        stats = torch.stack([
            weights.sum(),
            ce_sum,
            l1_sum,
            confidence_sum.detach(),
            (accept_rate * eval_mask).sum(),
            eval_mask.sum().to(torch.float32),
        ])
        if self.sp_size > 1:
            dist.all_reduce(stats, group=self.sp_group)
            loss_sum = torch.distributed.nn.all_reduce(loss_sum, group=self.sp_group)
        weight_sum = stats[0].clamp_min(1.0)
        loss = loss_sum / weight_sum
        model.mtp._metrics = {
            "dspark/loss": loss.detach(),
            "dspark/ce_loss": stats[1] / weight_sum,
            "dspark/l1_loss": stats[2] / weight_sum,
            "dspark/confidence_loss": stats[3] / weight_sum,
            "dspark/accept_rate": stats[4] / stats[5].clamp_min(1.0),
        }
        return loss
