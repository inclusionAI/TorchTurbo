# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for TorchTurbo from the user-provided GLM-5.3-Flash implementation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""GLM-5.3-Flash configs compatible with transformers 4.x and 5.x."""

from copy import deepcopy
from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"
    attribute_map = {"num_local_experts": "n_routed_experts"}
    _defaults = {
        "vocab_size": 154880,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 45,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": 288,
        "output_router_logits": False,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "qk_nope_head_dim": 256,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "max_position_embeddings": 1048576,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "pad_token_id": 154820,
        "bos_token_id": None,
        "eos_token_id": None,
        "tie_word_embeddings": False,
        "mlp_layer_types": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "head_dim": 0,
        "layer_types": None,
        "indexer_types": None,
        "swiglu_limit": 10.0,
        "linear_head_dim": 128,
        "linear_num_heads": 64,
        "linear_conv_kernel_dim": 4,
        "linear_lower_bound": -5.0,
        "hc_mult": 4,
        "hc_eps": 1e-06,
        "hc_sinkhorn_iters": 20,
        "router_aux_loss_coef": 0.001,
        "index_kpool": 16,
        "index_kpool_always_select_tail": True,
    }

    def __init__(self, **kwargs):
        values = deepcopy(self._defaults)
        values.update(kwargs)
        linear = values.get("linear_attn_config") or {}
        for old, new in {
            "head_dim": "linear_head_dim",
            "num_heads": "linear_num_heads",
            "short_conv_kernel_size": "linear_conv_kernel_dim",
            "gate_lower_bound": "linear_lower_bound",
        }.items():
            if old in linear:
                values[new] = linear[old]
        if (
            linear
            and linear.get("safe_gate", True)
            and values["linear_lower_bound"] is None
        ):
            values["linear_lower_bound"] = -5.0
        n = values["num_hidden_layers"]
        if values["mlp_layer_types"] is None:
            dense = min(n, values.get("first_k_dense_replace", 3))
            values["mlp_layer_types"] = ["dense"] * dense + ["sparse"] * (n - dense)
        if values["layer_types"] is None:
            full = linear.get("full_attn_layers", list(range(3, n, 4)))
            values["layer_types"] = [
                "indexed_attention" if i in full else "linear_attention"
                for i in range(n)
            ]
        values["layer_types"] = [
            "indexed_attention"
            if t in ("full_attention", "deepseek_sparse_attention")
            else t
            for t in values["layer_types"]
        ]
        if values["indexer_types"] is None:
            pattern = values.get("index_topk_pattern")
            if pattern is not None:
                values["indexer_types"] = [
                    {"F": "full", "S": "shared"}.get(t, t) for t in pattern
                ]
            else:
                freq = max(values.get("index_topk_freq", 1), 1)
                offset = values.get("index_skip_topk_offset", 2)
                values["indexer_types"] = [
                    "full" if max(i - offset + 1, 0) % freq == 0 else "shared"
                    for i in range(n)
                ]
        for field, allowed in (
            ("layer_types", {"linear_attention", "indexed_attention"}),
            ("mlp_layer_types", {"dense", "sparse"}),
            ("indexer_types", {"full", "shared"}),
        ):
            if len(values[field]) != n or not set(values[field]) <= allowed:
                raise ValueError(
                    f"Invalid {field}: expected {n} entries from {allowed}"
                )
        if values["num_key_value_heads"] is None:
            values["num_key_value_heads"] = values["num_attention_heads"]
        if values["num_attention_heads"] != values["num_key_value_heads"]:
            raise ValueError("GLM-5.3-Flash MLA requires equal query and KV head counts")
        if values["q_lora_rank"] is None or values["qk_rope_head_dim"] != 0:
            raise ValueError(
                "GLM-5.3-Flash uses LoRA queries and NoPE (qk_rope_head_dim=0)"
            )
        pool = values["index_kpool"]
        if pool < 1 or values["index_topk"] < pool or values["index_topk"] % pool:
            raise ValueError("index_topk must be positive and divisible by index_kpool")
        experts, groups = values["n_routed_experts"], values["n_group"]
        if groups < 1 or experts % groups or experts // groups < 2:
            raise ValueError("Each routing group must contain at least two experts")
        if (
            not 0 < values["topk_group"] <= groups
            or not 0
            < values["num_experts_per_tok"]
            <= experts // groups * values["topk_group"]
        ):
            raise ValueError("Invalid MoE top-k/group configuration")
        seen_full = False
        for t, indexer in zip(values["layer_types"], values["indexer_types"]):
            if t == "indexed_attention":
                if indexer == "shared" and not seen_full:
                    raise ValueError(
                        "A shared indexer requires a preceding full DSA layer"
                    )
                seen_full |= indexer == "full"
        if values["hc_mult"] < 1 or values["hc_sinkhorn_iters"] < 1:
            raise ValueError("hc_mult and hc_sinkhorn_iters must be positive")
        values["head_dim"] = values["qk_rope_head_dim"]
        values["qk_head_dim"] = values["qk_nope_head_dim"] + values["qk_rope_head_dim"]
        values.setdefault("scoring_func", "sigmoid")
        values.setdefault("kda_backend", "auto")
        if values["kda_backend"] not in ("auto", "torch", "fla"):
            raise ValueError("kda_backend must be auto, torch or fla")
        if values["scoring_func"] != "sigmoid" or not values["norm_topk_prob"]:
            raise ValueError("GLM-5.3-Flash requires normalized sigmoid top-k routing")
        if (
            values.get("mhc", True) is False
            or values.get("index_kpool_compress", True) is False
        ):
            raise ValueError(
                "Only mHC and compressed k-pool GLM-5.3-Flash checkpoints are supported"
            )
        for key, value in values.items():
            setattr(self, key, value)
        super().__init__(**values)


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "num_heads"}
    _defaults = {
        "depth": 24,
        "hidden_size": 1024,
        "hidden_act": "silu",
        "attention_bias": True,
        "attention_dropout": 0.0,
        "num_heads": 16,
        "in_channels": 3,
        "image_size": 336,
        "patch_size": 14,
        "rms_norm_eps": 1e-05,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 1536,
        "intermediate_size": 4096,
        "initializer_range": 0.02,
        "rope_parameters": None,
        "projection_intermediate_size": 10240,
        "swiglu_limit": 10.0,
    }

    def __init__(self, **kwargs):
        values = deepcopy(self._defaults)
        values.update(kwargs)
        values["rope_parameters"] = values["rope_parameters"] or {
            "rope_type": "axial",
            "rope_theta": 10000.0,
        }
        if (
            values["hidden_size"] % values["num_heads"]
            or values["hidden_size"] // values["num_heads"] % 4
        ):
            raise ValueError("Vision head dimension must be divisible by four")
        for key, value in values.items():
            setattr(self, key, value)
        super().__init__(**values)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    keys_to_ignore_at_inference = ["past_key_values"]
    is_composition = True
    sub_configs = {
        "text_config": Glm5NextTextConfig,
        "vision_config": Glm5NextVisionConfig,
    }
    _defaults = {
        "text_config": None,
        "vision_config": None,
        "image_token_id": 154854,
        "video_token_id": 154855,
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
        "tie_word_embeddings": False,
    }

    def __init__(self, **kwargs):
        values = deepcopy(self._defaults)
        values.update(kwargs)
        for key, cls in self.sub_configs.items():
            value = values[key]
            if value is None:
                value = cls(**kwargs) if key == "text_config" else cls()
            elif isinstance(value, dict):
                value = cls(**value)
            elif not isinstance(value, cls):
                value = cls(**value.to_dict())
            values[key] = value
        for key, value in values.items():
            setattr(self, key, value)
        super().__init__(**values)
