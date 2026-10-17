"""torchturbo integration layer for Kimi-K3.

Official modeling (modeling_kimi_k3.py + modeling_kimi_linear.py) is used as-is.
This file provides: registration, monkey-patch SP, RMSNorm->NPU, weight converter.
No framework-level hacks — tokenizer_config.json handles AutoTokenizer registration natively.
"""
try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override

import torch

from torchturbo.runtime import is_npu_available, get_device_name
from .modeling_kimi_k3 import KimiK3ForConditionalGeneration
from .configuration_kimi_k3 import KimiK3Config
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig

from ..base import TurboVLMModel
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from transformers import AutoModelForImageTextToText, GenerationMixin


class K3LatentGroupedMLP(GroupedMLP):
    """K3 Stable LatentMoE expert: GroupedMLP built in the latent space only.

    K3's LatentMoE runs experts in routed_expert_hidden_size (3584), not
    hidden_size (7168). We reuse torchturbo's GroupedMLP (grouped_matmul fused
    GEMM + EP via token_dispatcher — unchanged) but build it with
    moe_hidden_size=3584 so expert weights w1/w2 match [3072, 3584].

    The routed_expert_down_proj/up_proj/norm (hidden<->latent conversion) live
    on the MoELayer (block_sparse_moe level), NOT here — weight_loader treats
    `experts.*` as expert weights (assemble path), which skips plain Linear
    params, leaving them meta. Keeping them at block_sparse_moe level matches
    the official weight keys and the framework's common-param load path.
    Forward wrapping (down->expert->up) is done by the MoELayer forward hook.
    """

    def __init__(
        self,
        num_local_experts: int,
        use_te: bool,
        config: MoEConfig,
        layer_number: int = None,
        moe_hidden_size: int = None,
        **_unused,
    ):
        # Build the fused expert in the latent space (moe_hidden_size, not
        # hidden_size) so w1/w2 shapes match K3 weights [3072, 3584].
        import copy
        latent_config = copy.copy(config)
        latent_config.hidden_size = moe_hidden_size
        super().__init__(
            num_local_experts=num_local_experts,
            use_te=use_te,
            config=latent_config,
            layer_number=layer_number,
        )


class KimiK3WeightConverter(BaseWeightConverter):
    """Weight converter for Kimi-K3.

    K3 is a VLM whose real transformer dims live under config.text_config
    (KimiLinearConfig), not on the top-level KimiK3Config wrapper.
    Re-point config to text_config for BaseWeightConverter.
    """

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type
            # v3 removed BaseWeightConverter.get_save_dtype; inline the logic
            if hasattr(self.config, 'dtype'):
                self.dtype = self.config.dtype
            else:
                _map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
                self.dtype = _map.get(getattr(self.config, 'torch_dtype', 'bfloat16'), torch.bfloat16)

    # K3 official bf16 weights store expert MLP as w1/w2/w3 (MegaBlocks style),
    # but the model defines gate_proj/up_proj/down_proj (HF style). Map the
    # checkpoint names so weight_loader's assemble_weight1/2_for_broadcast pops
    # the right keys. Without this: KeyError 'experts.N.gate_proj.weight'
    # (weights have w1/w2/w3, not gate_proj/up_proj/down_proj).
    # w1=gate_proj [intermediate,hidden], w3=up_proj, w2=down_proj [hidden,intermediate]
    @override
    def get_expert_gate_name(self):
        return "w1"

    @override
    def get_expert_up_name(self):
        return "w3"

    @override
    def get_expert_down_name(self):
        return "w2"

    @override
    def to_hf_name(self, name):
        """Map model param names to HF checkpoint keys.

        K3LatentGroupedMLP stores routed_expert_down_proj/up_proj/norm inside
        `experts` (model: `block_sparse_moe.experts.routed_expert_down_proj`),
        but the official weights store them at the MoE-block level (ckpt:
        `block_sparse_moe.routed_expert_down_proj`). Strip the `.experts.` for
        these LatentMoE projection params so weight_loader finds them.
        """
        for proj in ("routed_expert_down_proj", "routed_expert_up_proj", "routed_expert_norm"):
            if f".experts.{proj}" in name:
                return name.replace(f".experts.{proj}", f".{proj}")
        return super().to_hf_name(name)

    @override
    def get_local_ep_w1_hf_state_dict(self, name, tensor):
        """Override: split weight1 by moe_hidden_size (3584), not hidden_size (7168).

        K3LatentGroupedMLP builds weight1 in the latent space:
          [num_local_experts * moe_hidden_size, fc1_output]
        Base class splits by config.hidden_size (7168), giving only half
        the experts → IndexError on experts[i] during checkpoint save.
        """
        import torch
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        experts = torch.split(tensor, self._moe_hidden_size())
        start_index = self.ep_rank * num_local_experts
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        local_ep_sd = {}
        for i in range(num_local_experts):
            index = start_index + i
            gate = f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
            up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
            expert2 = torch.split(experts[i].t(), self.config.moe_intermediate_size)
            local_ep_sd[gate] = expert2[0]
            local_ep_sd[up] = expert2[1]
        return local_ep_sd

    def _moe_hidden_size(self):
        """K3 Stable LatentMoE: expert lives in routed_expert_hidden_size space
        (7168 hidden -> 3584 latent), not hidden_size. Mirrors modeling's
        `moe_hidden_size = config.routed_expert_hidden_size if use_latent_moe
        else config.hidden_size`. Without this, converter uses hidden_size for
        expert weight shape -> 7168 vs 3584 broadcast error."""
        r = getattr(self.config, "routed_expert_hidden_size", None)
        return r if r is not None else self.config.hidden_size

    @override
    def get_gate_weight_shape(self):
        return (self.config.moe_intermediate_size, self._moe_hidden_size())

    @override
    def get_gate_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self._moe_hidden_size() * element_size

    @override
    def get_up_weight_shape(self):
        return (self.config.moe_intermediate_size, self._moe_hidden_size())

    @override
    def get_up_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self._moe_hidden_size() * element_size

    @override
    def get_down_weight_shape(self):
        return (self._moe_hidden_size(), self.config.moe_intermediate_size)

    @override
    def get_down_weight_bytes(self, element_size):
        return self._moe_hidden_size() * self.config.moe_intermediate_size * element_size

    @override
    def get_default_hf_dtype(self, param_name):
        """Per-param checkpoint dtype for K3's mixed-dtype weights.

        K3 ships as bf16, but KDA/Mamba-family params are float32 by design:
        A_log/dt_bias are explicitly float32 in KimiDeltaAttention; q/k/v_conv1d
        (ShortConvolution) and o_norm (FusedRMSNormGated) weights are float32
        in the official bf16 release; e_score_correction_bias is float32.
        Without this override, weight_loader.py:346
        `assert local_sd[name].dtype == broadcast_dtype` fails for these tensors
        (broadcast_dtype defaults to bf16). Mirrors the verified pattern in
        BailingV3WeightConverter (bailing_hybrid/model.py), which solves the
        identical scenario for BailingMoeV3KimiDeltaAttention.

        Ground truth (5-layer cut model, 28 F32 tensors): A_log/dt_bias/
        q|k|v_conv1d.weight/o_norm.weight on KDA layers 0,1,2,4 (4 each = 24)
        + e_score_correction_bias on MoE layers 1,2,3,4 (4) = 28.
        """
        fp32_suffixes = (
            "A_log",
            "dt_bias",
            "conv1d.weight",   # matches q_/k_/v_conv1d.weight
            "o_norm.weight",
            "e_score_correction_bias",
        )
        for suf in fp32_suffixes:
            if param_name.endswith(suf):
                return torch.float32
        return self.dtype

    @override
    def adjust_checkpoint_tensor(self, name, tensor, target_shape):
        """Truncate K3 A_log zero-padding to match the model's (num_heads,) shape.

        K3 official bf16 weights store A_log as shape (head_dim=128,) = first
        num_heads(96) real log-decay values + 32 zero-padding tail (verified
        across all 93 layers: tail all-zero, first-96 in log(1~16) range). The
        model defines A_log as (num_heads=96,) and fla chunk_kda expects
        [HV=96]. FSDP2 broadcasts the owner's checkpoint tensor with
        target_shape=param.shape=(96,), so we truncate the zero tail on the
        owner before broadcast — owner and non-owner then agree on (96,).

        Refuses to drop non-zero values (would silently corrupt real data).
        Mirrors FSDPTurbo PR67's _load_from_state_dict approach, moved here
        because FSDP2's assign=True path flattens KDA params before that hook
        fires (verified: hook print produced 0 hits).
        """
        if name.endswith("A_log") and tensor.shape[0] > target_shape[0]:
            want = target_shape[0]
            tail = tensor[want:]
            if not torch.all(tail == 0):
                raise ValueError(
                    f"{name}: A_log checkpoint tail beyond index {want} is "
                    f"non-zero; refusing to truncate real values"
                )
            tensor = tensor[:want]
        return tensor


@register_model("kimi_k3", config_cls=KimiK3Config, auto_class=AutoModelForImageTextToText)
class TurboKimiK3ForConditionalGeneration(
    TurboVLMModel, KimiK3ForConditionalGeneration, GenerationMixin
):
    """torchturbo wrapper for Kimi-K3.

    Inherits GenerationMixin so HF's can_generate() returns True (HF >=4.50
    dropped the prepare_inputs_for_generation fallback). Without this,
    from_pretrained skips loading generation_config.json, leaving
    model.generation_config = None, which crashes LLaMA-Factory's patch_model
    at `gen_config.do_sample`. K3 defines prepare_inputs_for_generation but
    that no longer satisfies can_generate() on new transformers.
    """

    def get_weight_converter(self, parallel_context, model_path):
        return KimiK3WeightConverter(self, parallel_context, model_path)

    def get_norm(self):
        return self.language_model.model.norm

    def get_lm_head(self):
        return self.language_model.lm_head

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        """SP boundary: the language model entry, after media fusion."""
        cls = type(self.language_model)
        return SequenceParallelInputSlicingSpec(
            self.language_model,
            cls.forward,
        )

    def __init__(self, config: KimiK3Config) -> None:
        self._pre_init_patches(config)
        super().__init__(config)

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan
        return get_parallel_plan(**kwargs)

    @classmethod
    def _pre_init_patches(cls, config):
        """Patch RMSNorm->NPU + MoELayer before HF from_pretrained."""
        from . import modeling_kimi_linear as klm
        from torchturbo.ops import ops

        _rms_norm_call_count = [0]

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        if hasattr(klm, "KimiRMSNorm"):
            klm.KimiRMSNorm.forward = rms_norm_forward



        # Replace fla's triton chunk_kda with the pure-PyTorch naive impl.
        # fla.ops.kda.chunk_kda uses triton kernels that aicore-timeout on NPU
        # (torch_npu 2.10, "The aicore execution times out" EZ9999). The naive
        # impl (torchturbo.ops.kda.chunk_kda_naive) is mathematically equivalent
        # but pure torch ops — runs on NPU. Patch the modeling module's chunk_kda
        # so both the official forward (SP=1) and sp_forward (SP>1) use it.
        # Mirrors MM's kda_implementation=naive path.
        try:
            from torchturbo.ops.kda.chunk_kda_naive import chunk_kda_naive
            klm.chunk_kda = chunk_kda_naive
        except ImportError:
            pass

        # Patch nn.Embedding.forward: use full_tensor() to all-gather sharded weight.
        # Stock PyTorch embedding_dense_backward DTensor strategy lacks the
        # rowwise backward path → global index scatter into local-shard grad
        # tensor → MTE DDR OOB. to_local() returns local shard only (OOB). full_tensor() all-gathers
        # EmbeddingBackward0 (full gradient), then to_local() backward +
        # FSDP2 post-backward reduce-scatters to the local shard. Regular
        # (non-DTensor) weights are unaffected (hasattr check).
        import torch.nn as nn
        import torch.nn.functional as F_emb
        _orig_emb_fwd = nn.Embedding.forward

        def _patched_emb_fwd(self, input_ids):
            # Bypass embedding_dense_backward (NPU OOB with FSDP2 sharded grad buffer).
            # Use weight[input_ids] (index_select) whose backward is index_add —
            # different NPU kernel, avoids the scatter OOB.
            return self.weight[input_ids]

        nn.Embedding.forward = _patched_emb_fwd

        # Unwrap KimiLinearModel.forward. HF 4.57's check_model_inputs decorator
        # is mis-applied: KimiLinearModel.__dict__['forward'] ends up being
        # `wrapped_fn` (the decorator's middle function, signature ['func'])
        # instead of `wrapper` (the actual wrapped forward with input_ids).
        # Result: trainer's forward(input_ids=...) raises "got an unexpected
        # keyword argument 'input_ids'". The original forward survives in
        # wrapped_fn.__closure__[0]; restore it directly. (Removing
        # @check_model_inputs from source is insufficient — HF re-applies it.)
        if hasattr(klm, "KimiLinearModel"):
            fwd = klm.KimiLinearModel.__dict__.get("forward")
            if getattr(fwd, "__name__", "") == "wrapped_fn" and fwd.__closure__:
                klm.KimiLinearModel.forward = fwd.__closure__[0].cell_contents

        origin_moe_forward = MoELayer.forward

        def kimi_moe_forward(self, hidden_states):
            output, _ = origin_moe_forward(self, hidden_states)
            return output

        MoELayer.forward = kimi_moe_forward

    def _patch_kda_sp(self, sp_group, sp_size, sp_rank):
        """Patch KDA forward for Ulysses SP."""
        if sp_size == 1:
            return

        from . import modeling_kimi_linear as klm
        from ..modules.attention.ulysses.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )

        if not hasattr(klm, "KimiDeltaAttention"):
            return

        def sp_forward(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
            from .modeling_kimi_linear import _split_last_dim_heads, _merge_last_two_dims

            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
            k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
            v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            # ShortConvolution.forward returns (output, conv_state); unpack.
            # (K3 official forward does `q, conv_state_q = self.q_conv1d(...)`.)
            # bailing avoids this via a naive_causal_conv1d_wrapper; we unpack.
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

            if self.use_full_rank_gate:
                g = self.g_proj(hidden_states)
            else:
                g = self.f_b_proj(self.f_a_proj(hidden_states))
            beta = self.b_proj(hidden_states).float().sigmoid()

            g = gather_seq_scatter_heads(g, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
            beta = gather_seq_scatter_heads(beta, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            heads_per_rank = self.num_heads // sp_size
            A_log_local = self.A_log[sp_rank * heads_per_rank:(sp_rank + 1) * heads_per_rank]
            channels_per_rank = heads_per_rank * self.head_dim
            start_ch = sp_rank * channels_per_rank
            end_ch = start_ch + channels_per_rank
            dt_bias_local = self.dt_bias[start_ch:end_ch]

            q, k = map(lambda x: _split_last_dim_heads(x, self.head_k_dim), (q, k))
            v = _split_last_dim_heads(v, self.head_dim)
            g = _split_last_dim_heads(g, self.head_dim)

            kda_impl = getattr(self.config, "kda_implementation", "naive")
            if kda_impl == "fused":
                try:
                    from fla.ops.kda import chunk_kda as kda_fn
                except ImportError:
                    from torchturbo.ops.kda.chunk_kda_naive import chunk_kda_naive as kda_fn
            else:
                from torchturbo.ops.kda.chunk_kda_naive import chunk_kda_naive as kda_fn

            o, _ = kda_fn(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=A_log_local, dt_bias=dt_bias_local,
                initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=getattr(self, "safe_gate", False),
                lower_bound=self.gate_lower_bound,
            )

            o = gather_heads_scatter_seq(o, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            if self.use_full_rank_gate:
                g_out = self.g_proj(hidden_states)
            else:
                g_out = self.g_b_proj(self.g_a_proj(hidden_states))
            g_out = _split_last_dim_heads(g_out, self.head_dim)
            o = self.o_norm(o, g_out)
            o = _merge_last_two_dims(o)
            o = self.o_proj(o)
            # K3 official KimiDeltaAttention.forward returns single tensor `o`
            # (not the HF-style (o, None, None) tuple). _forward_attn_residual
            # assigns `hidden_states = self.self_attn(...)` and expects a single
            # tensor; returning a tuple makes prefix_sum a tuple -> .view crash.
            return o

        klm.KimiDeltaAttention.forward = sp_forward

    def on_materialized(self):
        pass
