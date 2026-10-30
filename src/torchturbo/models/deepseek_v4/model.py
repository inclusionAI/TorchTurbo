import json
import math
import os
import re
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.models.modules.attention.ulysses import all_gather_sp_with_grad_reduce
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

from . import packing_utils as pu
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.models.modules.rope import get_shared_rope_table
from torchturbo.runtime import TurboDevice, get_device_runtime
from torchturbo.runtime import is_npu_available  # noqa: F401 - module-level availability probe

from ..base import TurboModel
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .configuration_deepseek_v4 import DeepseekV4Config
from .mtp import DeepseekV4MTPForwarder, DeepseekV4MultiTokenPredictor
from .modeling_deepseek_v4 import DeepseekV4ForCausalLM


class DeepseekV4WeightConverter(BaseWeightConverter):
    def get_expert_gate_name(self):
        return "w1"

    def get_expert_up_name(self):
        return "w3"

    def get_expert_down_name(self):
        return "w2"

    def get_hf_expert_mlp_name(self):
        return "ffn"

    def get_hf_gate_bias_name(self):
        return "gate.bias"

    def get_hf_gate_hash_name(self):
        return "gate.tid2eid"

    def get_router_hash_name(self):
        return "router.tid2eid"

    def is_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return True
        if self.get_router_hash_name() in name:
            return True
        return False

    def convert_persist_buffer(self, name):
        if self.get_router_hash_name() in name:
            return name.replace(self.get_router_hash_name(), self.get_hf_gate_hash_name())
        return super().convert_persist_buffer(name)


@register_model(
    "deepseek_v4", config_cls=DeepseekV4Config, auto_class=AutoModelForCausalLM
)
class TurboDeepseekV4ForCausalLM(TurboModel, DeepseekV4ForCausalLM):
    config_class = DeepseekV4Config
    base_model_prefix = ""
    _forward_mtp = DeepseekV4MTPForwarder()

    def __init__(self, config: DeepseekV4Config):
        self._pre_init_patches(config)
        if config.dspark_block_size > 0:
            config.dspark_num_layers = self._resolve_dspark_num_layers(config)
        super().__init__(config)
        self.mtp = (
            DeepseekV4MultiTokenPredictor(config)
            if config.num_nextn_predict_layers > 0 or config.dspark_block_size > 0
            else None
        )
        self._post_init_patches()

    @staticmethod
    def _resolve_dspark_num_layers(config: DeepseekV4Config) -> int:
        """Resolve the DSpark layer count from checkpoint keys."""
        model_path = getattr(config, "_name_or_path", "")
        index_path = Path(model_path) / "model.safetensors.index.json"
        if model_path and index_path.is_file():
            with index_path.open() as index_file:
                weight_map = json.load(index_file)["weight_map"]
            stages = {
                int(match.group(1))
                for name in weight_map
                if (match := re.fullmatch(r"mtp\.(\d+)\..+", name))
            }
            if stages:
                return max(stages) + 1

        return config.dspark_num_layers

    @classmethod
    @contextmanager
    def init_context(cls, config: DeepseekV4Config, engine_config):
        dspark = engine_config.dspark
        if not engine_config.mtp.enabled:
            config.dspark_block_size = 0
        config.dspark_enabled = dspark.enabled
        config.dspark_num_anchors = dspark.num_anchors
        config.dspark_loss_decay_gamma = dspark.loss_decay_gamma
        config.dspark_ce_loss_alpha = dspark.ce_loss_alpha
        config.dspark_l1_loss_alpha = dspark.l1_loss_alpha
        config.dspark_confidence_loss_alpha = dspark.confidence_loss_alpha

        with super().init_context(config, engine_config):
            yield

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context, model_path):
        return DeepseekV4WeightConverter(self, parallel_context, model_path)

    def get_norm(self) -> torch.nn.Module:
        return self.norm

    def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
        if optimizer_name == "muon":
            from .optimizer_recipe import get_muon_recipe

            return get_muon_recipe(self)
        return super().get_optimizer_recipe(optimizer_name)

    def get_lm_head(self):
        return self.head

    @wraps(DeepseekV4ForCausalLM.forward)
    def forward(self, *args, **kwargs):
        lm_head = self._modules["head"]
        noop_lm_head = self.head
        if noop_lm_head is lm_head:
            return self._forward_mtp.forward(self, *args, **kwargs)

        def head_forward(hidden_states, hc_fn, hc_scale, hc_base, norm):
            hidden_states = lm_head._hc_head(
                hidden_states,
                hc_fn,
                hc_scale,
                hc_base,
            )
            return noop_lm_head(norm(hidden_states).float())

        object.__setattr__(self, "head", head_forward)
        try:
            return self._forward_mtp.forward(self, *args, **kwargs)
        finally:
            object.__setattr__(self, "head", noop_lm_head)

    @staticmethod
    def _pre_init_patches(config):
        from . import modeling_deepseek_v4 as deepseek_v4_model
        from torchturbo.ops import ops

        origin_moe_forward = getattr(
            MoELayer, "_torchturbo_deepseek_v4_origin_forward", MoELayer.forward
        )
        MoELayer._torchturbo_deepseek_v4_origin_forward = origin_moe_forward

        def deepseek_moe_forward(self, hidden_states: torch.Tensor, input_ids=None):
            output, _ = origin_moe_forward(self, hidden_states, input_ids)
            return output

        MoELayer.forward = deepseek_moe_forward

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        deepseek_v4_model.DeepseekV4RMSNorm.forward = rms_norm_forward
        config._attn_implementation = "eager"

    def _post_init_patches(self):
        self._use_flash_attention_2 = True
        self.config._attn_implementation = "flash_attention_2"

    def on_materialized(self):
        """Rebuild runtime buffers cleared by FSDP2 meta-device loading."""
        from . import modeling_deepseek_v4 as dsv4

        for layer in (*self.layers, *(self.mtp or [])):
            attn = layer.attn
            if attn.compress_ratio > 1:
                original_seq_len = attn.config.original_max_position_embeddings
                rope_theta = attn.config.compress_rope_theta
            else:
                original_seq_len = 0
                rope_theta = attn.config.rope_theta

            freqs_cis = get_shared_rope_table(
                dsv4.DeepseekV4Attention._precompute_freqs_cis,
                (
                    attn.rope_head_dim,
                    attn.config.max_position_embeddings,
                    original_seq_len,
                    rope_theta,
                    attn.config.rope_factor,
                    attn.config.beta_fast,
                    attn.config.beta_slow,
                ),
                device=attn.attn_sink.device,
                dtype=torch.complex64,
            )
            attn.register_buffer("freqs_cis", freqs_cis, persistent=False)

            if attn.compress_ratio == 4:
                hadamard_dim = 2 ** math.ceil(math.log2(attn.config.index_head_dim))
                hadamard_mat = dsv4.generate_hadamard_matrix(hadamard_dim).to(
                    attn.attn_sink.device
                )
                attn.register_buffer("hadamard_mat", hadamard_mat, persistent=False)
            else:
                attn.hadamard_mat = None

    def on_parallelized(self, parallel_context):
        from . import modeling_deepseek_v4 as dsv4

        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        assert sp_size >= 1, "must be"

        def indexer_loss_forward(
            self,
            selected_main_attn_dist: torch.Tensor,
            index_score: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            loss = dsv4.DSAIndexerLoss.forward(
                self, selected_main_attn_dist, index_score, topk_indices
            )
            if sp_size > 1:
                local_count = (topk_indices >= 0).any(dim=-1).sum()
                global_count = local_count.clone()
                torch.distributed.all_reduce(global_count, group=sp_group)
                # Undo the local mean and account for FSDP gradient averaging.
                loss = loss * local_count.clamp_min(1) * sp_size
                loss = loss / global_count.clamp_min(1)
            return loss

        for layer in self.layers:
            if layer.cal_index_loss is not None:
                loss_fn = layer.cal_index_loss.li_loss.compute_dsa_indexer_loss
                loss_fn.forward = MethodType(indexer_loss_forward, loss_fn)
        self.patch_deepseek_v4_attention_for_sp(sp_group, sp_size, sp_rank)
        self._forward_mtp = DeepseekV4MTPForwarder(
            sp_group=sp_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
        )

    @classmethod
    def patch_deepseek_v4_attention_for_sp(cls, sp_group, sp_size, sp_rank):
        from . import modeling_deepseek_v4 as dsv4

        def inner_attention_forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            kv_compress,
            q_indexer,
            k_indexer,
            weights,
            seqlen: int,
            offset: int,
            attn_sink: torch.Tensor,
            q_global_positions: torch.LongTensor,
            segment_ids: torch.Tensor,
        ):
            compress_topk_idxs = None
            index_score = None

            if (
                self.compress_ratio == 4
                and q_indexer is not None
                and k_indexer is not None
                and weights is not None
                and k_indexer.size(1) > 0
            ):
                compress_topk_idxs, index_score = self.li_compute(
                    q_indexer,
                    k_indexer,
                    weights,
                    seqlen,
                    offset,
                    q_global_positions=q_global_positions,
                    segment_ids=segment_ids,
                )

            o = self.sparse_attn(
                q,
                kv,
                attn_sink,
                q_global_positions=q_global_positions,
                segment_ids=segment_ids,
                kv_compress=kv_compress,
                compress_topk_idxs=compress_topk_idxs,
            )
            return o, compress_topk_idxs, index_score

        def compressor_forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            segment_ids: torch.Tensor,
            position_ids: torch.LongTensor = None,
        ):
            ratio, overlap = self.compress_ratio, self.overlap
            dtype = x.dtype
            x = x.float()
            kv = self.wkv(x)
            score = self.wgate(x)
            if sp_size > 1:
                kv = all_gather_sp_with_grad_reduce(kv, dim=1, sp_group=sp_group, sp_size=sp_size)
                score = all_gather_sp_with_grad_reduce(score, dim=1, sp_group=sp_group, sp_size=sp_size)
                position_ids = all_gather_sp_with_grad_reduce(position_ids, dim=1, sp_group=sp_group, sp_size=sp_size)

            kv, compress_positions = pu.compress_packed_kv(
                kv,
                score,
                self.ape,
                segment_ids[:, : kv.size(1)],
                position_ids[:, : kv.size(1)],
                ratio,
                self.head_dim,
                overlap,
            )
            if kv is None:
                return None

            kv = self.norm(kv.to(dtype))
            kv_rot = pu.apply_rotary_emb(
                kv[..., -self.rope_head_dim:],
                freqs_cis,
                position_ids=compress_positions,
            )
            return torch.cat([kv[..., :-self.rope_head_dim], kv_rot], dim=-1)

        def indexer_forward(
            self,
            x: torch.Tensor,
            qr: torch.Tensor,
            freqs_cis: torch.Tensor,
            hadamard_mat: torch.Tensor,
            offset: int,
            position_ids: torch.LongTensor,
            segment_ids: torch.Tensor,
        ):
            bsz, seqlen, _ = x.size()
            rd = self.rope_head_dim

            q = self.wq_b(qr)
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
            q = q.clone()
            q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
            q_rope = pu.apply_rotary_emb(q_rope, freqs_cis, position_ids=position_ids)
            q = torch.cat([q_nope, q_rope], dim=-1)
            q = dsv4.rotate_activation(q, hadamard_mat)

            k = self.compressor(x, freqs_cis, segment_ids, position_ids)
            if k is None:
                return None, None, None
            k = dsv4.rotate_activation(k, hadamard_mat)

            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

            return q, k, weights

        def window_topk_forward(
            self,
            window_size: int,
            bsz: int,
            seqlen: int,
            device,
            segment_ids: torch.Tensor,
        ):
            base = torch.arange(
                seqlen * sp_rank, seqlen * (sp_rank + 1), device=device
            ).unsqueeze(1)
            global_seqlen = seqlen * sp_size
            window_topk = (base - window_size + 1).clamp(0) + torch.arange(
                min(global_seqlen, window_size), device=device
            )
            window_topk = torch.where(window_topk > base, -1, window_topk)
            window_topk = window_topk.unsqueeze(0).expand(bsz, -1, -1)
            window_token_indices = window_topk.clamp_min(0).reshape(bsz, -1)
            batch_index = torch.arange(bsz, device=device).unsqueeze(1)
            window_ids = segment_ids[batch_index, window_token_indices].view_as(
                window_topk
            )
            query_ids = segment_ids[
                :, sp_rank * seqlen : (sp_rank + 1) * seqlen
            ].unsqueeze(-1)
            mask = (window_ids == query_ids) & (query_ids > 0)
            return torch.where(mask, window_topk, -1)

        def compress_topk_forward(
            self,
            x: torch.Tensor,
            offset: int,
            q_global_positions: torch.LongTensor,
            segment_ids: torch.Tensor,
        ):
            seqlen = x.shape[1]
            local_segment_ids = segment_ids[
                :, sp_rank * seqlen : (sp_rank + 1) * seqlen
            ]
            global_positions = torch.arange(
                segment_ids.size(1), device=segment_ids.device
            ).unsqueeze(0).expand_as(segment_ids)
            compress_kv_ids, compress_block_pos = (
                pu.packed_compress_kv_ids_and_positions(
                    segment_ids,
                    self.ratio,
                    global_positions,
                )
            )
            mask = pu.packed_compress_mask(
                local_segment_ids,
                compress_kv_ids.unsqueeze(1),
                compress_block_pos.unsqueeze(1),
                self.ratio,
                q_global_positions,
            )
            matrix = torch.arange(compress_kv_ids.size(1), device=x.device)
            matrix = matrix.view(1, 1, -1).expand(
                local_segment_ids.size(0), seqlen, -1
            )
            return torch.where(mask, -1, matrix + offset)

        def li_compute_forward(
            self,
            q_indexer: torch.Tensor,
            k_indexer: torch.Tensor,
            weights: torch.Tensor,
            seqlen: int,
            offset: int,
            q_global_positions: torch.LongTensor,
            segment_ids: torch.Tensor,
        ):
            index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

            T_kv = k_indexer.shape[1]
            local_segment_ids = segment_ids[
                :, sp_rank * seqlen : (sp_rank + 1) * seqlen
            ]
            global_positions = torch.arange(
                segment_ids.size(1), device=segment_ids.device
            ).unsqueeze(0).expand_as(segment_ids)
            compress_kv_ids, compress_block_pos = (
                pu.packed_compress_kv_ids_and_positions(
                    segment_ids,
                    self.ratio,
                    global_positions,
                )
            )
            if compress_kv_ids.size(1) * sp_size == T_kv:
                compress_kv_ids = compress_kv_ids.repeat_interleave(sp_size, dim=1)
                compress_block_pos = compress_block_pos.repeat_interleave(
                    sp_size, dim=1
                )
            mask = pu.packed_compress_mask(
                local_segment_ids,
                compress_kv_ids.unsqueeze(1),
                compress_block_pos.unsqueeze(1),
                self.ratio,
                q_global_positions,
            )
            index_score = index_score + torch.where(
                mask, torch.finfo(q_indexer.dtype).min, 0
            )

            index_score, topk_idxs = index_score.topk(
                min(self.index_topk, T_kv), dim=-1
            )
            flat_topk = topk_idxs.reshape(local_segment_ids.size(0), -1)
            batch_index = torch.arange(
                segment_ids.size(0), device=segment_ids.device
            ).unsqueeze(1)
            topk_ids = compress_kv_ids[batch_index, flat_topk].view_as(topk_idxs)
            topk_pos = compress_block_pos[batch_index, flat_topk].view_as(topk_idxs)
            mask = pu.packed_compress_mask(
                local_segment_ids,
                topk_ids,
                topk_pos,
                self.ratio,
                q_global_positions,
            )
            compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
            return compress_topk_idxs, index_score

        def pre_attention_forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            hadamard_mat: torch.Tensor,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor,
            indexer,
            position_ids: torch.LongTensor,
            segment_ids: torch.Tensor,
        ):
            rd = self.rope_head_dim

            qr = q_norm(wq_a(x))
            q = wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
            q_rope = pu.apply_rotary_emb(q_rope, freqs_cis, position_ids=position_ids)
            q = torch.cat([q_nope, q_rope], dim=-1)

            kv = wkv(x)
            kv = kv_norm(kv)
            kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
            kv_rope = pu.apply_rotary_emb(kv_rope, freqs_cis, position_ids=position_ids)
            kv = torch.cat([kv_nope, kv_rope], dim=-1)
            if sp_size > 1:
                kv = all_gather_sp_with_grad_reduce(
                    kv, dim=1, sp_group=sp_group, sp_size=sp_size
                )

            offset = kv.size(1)

            q_indexer = k_indexer = weights = None
            if self.compress_ratio > 1 and indexer is not None:
                q_indexer, k_indexer, weights = indexer(
                    x.detach(),
                    qr.detach(),
                    freqs_cis,
                    hadamard_mat,
                    offset,
                    position_ids,
                    segment_ids,
                )

            kv_compress = None
            if self.compress_ratio > 1 and compressor is not None:
                kv_compress = compressor(x, freqs_cis, segment_ids, position_ids)

            return q, kv, kv_compress, q_indexer, k_indexer, weights, offset

        def post_attention_forward(
            self,
            o: torch.Tensor,
            freqs_cis: torch.Tensor,
            bsz: int,
            seqlen: int,
            n_local_groups: int,
            wo_a,
            wo_b,
            position_ids: torch.LongTensor,
        ):
            rd = self.rope_head_dim
            o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
            o_rope = pu.apply_rotary_emb(
                o_rope,
                freqs_cis,
                inverse=True,
                position_ids=position_ids,
            )
            o = torch.cat([o_nope, o_rope], dim=-1)

            o = o.view(bsz, seqlen, n_local_groups, -1)
            wo_a_weight = wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
            return wo_b(o.reshape(bsz, seqlen, -1))

        def sparse_attention_forward(
            self,
            query_states: torch.Tensor,
            kv_states: torch.Tensor,
            attn_sink: torch.Tensor,
            q_global_positions: torch.LongTensor,
            segment_ids: torch.Tensor,
            kv_compress: Optional[torch.Tensor] = None,
            compress_topk_idxs: Optional[torch.Tensor] = None,
        ):
            bsz, seqlen, _, _ = query_states.size()

            topk_idxs = self.get_window_topk_idxs(
                self.window_size,
                bsz,
                seqlen,
                query_states.device,
                segment_ids=segment_ids,
            )

            if self.compress_ratio > 1 and kv_compress is not None:
                offset = kv_states.size(1)
                if compress_topk_idxs is None:
                    compress_topk_idxs = self.get_compress_topk_idxs(
                        query_states,
                        offset,
                        q_global_positions=q_global_positions,
                        segment_ids=segment_ids,
                    )
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

            topk_idxs = topk_idxs.int()

            if self.compress_ratio > 1 and kv_compress is not None:
                kv_states = torch.cat([kv_states, kv_compress], dim=1)

            query_states = query_states.transpose(1, 2)
            kv_states = kv_states.unsqueeze(1)
            attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * self.softmax_scale

            index_mask = torch.full(
                (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
                fill_value=torch.finfo(attn_weights.dtype).min,
                dtype=attn_weights.dtype,
                device=query_states.device,
            )

            valid_mask = topk_idxs >= 0
            if valid_mask.any():
                b_indices, s_indices, k_indices = torch.where(valid_mask)
                valid_topk = topk_idxs[b_indices, s_indices, k_indices]
                index_mask[b_indices, 0, s_indices, valid_topk] = 0

            attn_weights = attn_weights + index_mask[..., :-1]

            sinks = attn_sink.reshape(1, -1, 1, 1).expand(query_states.shape[0], -1, query_states.shape[2], -1)
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

            probs = F.softmax(combined_logits.float(), dim=-1, dtype=attn_weights.dtype)
            scores = probs[..., :-1]
            attn_output = torch.matmul(scores, kv_states)
            attn_output = attn_output.transpose(1, 2).contiguous()

            return attn_output

        def attention_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_value=None,
            use_cache: bool = False,
            output_attentions: bool = False,
            cu_seq_lens: torch.Tensor = None,
        ):
            bsz, seqlen, _ = hidden_states.size()
            if position_ids is not None and position_ids.size(0) == 1 and bsz > 1:
                position_ids = position_ids.expand(bsz, -1)
            # Segment ids drive every packed operator (compression blocks,
            # window/compress top-k, sparse attention). Packed batches carry
            # global cu boundaries; attention_mask is never set (the engine
            # rejects it and DPO passes None), so batches without cu are
            # treated as one causal sequence per row.
            if cu_seq_lens is not None:
                segment_ids = cu_seqlens_to_segment_ids(
                    cu_seq_lens, seqlen * sp_size
                ).to(hidden_states.device)
            else:
                segment_ids = torch.ones(
                    (bsz, seqlen * sp_size),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            freqs_cis = self.freqs_cis.to(hidden_states.device)
            hadamard_mat = self.hadamard_mat if self.hadamard_mat is not None else None

            compressor = getattr(self, "compressor", None)
            indexer = getattr(self, "indexer", None)

            q, kv, kv_compress, q_indexer, k_indexer, weights, offset = self.pre_attention_ops(
                hidden_states, freqs_cis, hadamard_mat,
                self.wq_a, self.q_norm, self.wq_b, self.wkv, self.kv_norm,
                compressor, indexer, position_ids, segment_ids,
            )

            n_local_groups = self.n_groups // (self.n_heads // q.shape[2])

            # Causal order follows physical token offsets, not RoPE coordinates.
            q_global_positions = torch.arange(
                sp_rank * seqlen,
                (sp_rank + 1) * seqlen,
                device=hidden_states.device,
            ).unsqueeze(0).expand(bsz, -1)
            o, compress_topk_idxs, index_score = self.inner_attention_ops(
                q,
                kv,
                kv_compress,
                q_indexer,
                k_indexer,
                weights,
                seqlen,
                offset,
                self.attn_sink,
                q_global_positions=q_global_positions,
                segment_ids=segment_ids,
            )

            compressed_attention_mask = None
            if (
                self.training
                and self.config.dsa_indexer_loss_coeff > 0
                and compress_topk_idxs is not None
            ):
                global_positions = torch.arange(
                    segment_ids.size(1), device=hidden_states.device
                ).unsqueeze(0).expand_as(segment_ids)
                block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
                    segment_ids, self.compress_ratio, global_positions
                )
                local_segment_ids = segment_ids[
                    :, sp_rank * seqlen : (sp_rank + 1) * seqlen
                ]
                compressed_attention_mask = pu.packed_compress_mask(
                    local_segment_ids,
                    block_ids.unsqueeze(1),
                    block_positions.unsqueeze(1),
                    self.compress_ratio,
                    q_global_positions,
                )

            attn_output = self.post_attention_ops(
                o,
                freqs_cis,
                bsz,
                seqlen,
                n_local_groups,
                self.wo_a,
                self.wo_b,
                position_ids,
            )

            if use_cache:
                present_key_value = kv
            else:
                present_key_value = None

            return (
                attn_output,
                None,
                present_key_value,
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

        dsv4.DeepseekV4Compressor.forward = compressor_forward
        dsv4.DeepseekV4Indexer.forward = indexer_forward
        dsv4.GetWindowTopkIdxs.forward = window_topk_forward
        dsv4.GetCompressTopkIdxs.forward = compress_topk_forward
        dsv4.LiCompute.forward = li_compute_forward
        dsv4.InnerAttentionOps.forward = inner_attention_forward
        dsv4.PreAttentionOps.forward = pre_attention_forward
        dsv4.PostAttentionOps.forward = post_attention_forward
        dsv4.SparseAttention.forward = sparse_attention_forward
        dsv4.DeepseekV4Attention.forward = attention_forward
        dsv4.DSAIndexerLossAutoScaler = MoEAuxLossAutoScaler
        if (
            get_device_runtime().turbo_device is TurboDevice.NPU
            and os.environ.get("TORCHTURBO_ENABLE_NPU_SFA_LI", "1") == "1"
        ):
            from .npu import patch_deepseek_v4_npu_sfa_li

            patch_deepseek_v4_npu_sfa_li(sp_rank, sp_group, sp_size)
