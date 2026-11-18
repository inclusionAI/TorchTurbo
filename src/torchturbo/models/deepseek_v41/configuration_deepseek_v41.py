"""DeepSeek V4.1 configuration for transformers library.

Nested config following the qwen3_5 pattern: a top-level ``DeepseekV41Config``
holding ``text_config`` and ``vision_config`` sub-configs. Field names match the
released checkpoint's ``config.json`` keys exactly; ``rope_scaling`` is
flattened into the plain attributes used by the modeling code.
"""

from transformers import AutoConfig, PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    r"""Configuration for the DeepSeek V4.1 text backbone.

    The CSA2 fields control cross-layer KV/index sharing: layers in
    ``kv_source_layer_ids`` own a compressor that publishes the shared
    compressed KV for their layer group, layers in ``index_source_layer_ids``
    run the lightning indexer (Full/Reindex modes), and every other
    ``compress_ratios[i] > 0`` layer reuses the published indices (Reuse mode).
    """

    model_type = "deepseek_v41_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=5120,
        moe_intermediate_size=2304,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        hidden_act="silu",
        swiglu_limit=10.0,
        rms_norm_eps=1e-20,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=False,
        max_position_embeddings=1048576,
        rope_theta=10000.0,
        rope_scaling=None,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        gate_temp=1.0,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        sliding_window=128,
        compress_ratios=None,
        compress_rope_theta=160000.0,
        kv_source_layer_ids=None,
        index_source_layer_ids=None,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_source_layer_id=-1,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        indexer_kl_loss_coeff=0.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=None,
        engram_num_embeddings=None,
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
        num_nextn_predict_layers=0,
        **kwargs,
    ):
        # Flatten the YaRN dict into the plain attributes consumed by the
        # modeling rope tables; must happen before PretrainedConfig.__init__
        # sees the key.
        rope_scaling = rope_scaling or {}
        rope_factor = rope_scaling.get("factor", 1.0)
        original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 0
        )
        beta_fast = rope_scaling.get("beta_fast", 32)
        beta_slow = rope_scaling.get("beta_slow", 1)
        kwargs.pop("rope_scaling", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.gate_temp = gate_temp
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        self.sliding_window = sliding_window
        self.compress_ratios = compress_ratios or [0] * num_hidden_layers
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = kv_source_layer_ids or []
        self.index_source_layer_ids = index_source_layer_ids or []
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        # Coefficient for the indexer KL auxiliary loss (DSA indexer
        # training); 0 disables it. Only consumed on the fused NPU KL
        # path, which is active whenever the NPU backend is available.
        self.indexer_kl_loss_coeff = indexer_kl_loss_coeff

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.engram_layer_ids = engram_layer_ids or []
        self.engram_num_embeddings = engram_num_embeddings or []
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size

        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    """Configuration for the DeepSeek ViT vision tower and aligner."""

    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_hidden_layers=32,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=2816,
        patch_size=14,
        rope_theta=10000.0,
        downsample_ratio=3,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    r"""Top-level configuration for DeepSeek V4.1 (multimodal).

    Holds the nested ``text_config`` / ``vision_config`` sub-configs and the
    multimodal token ids. ``num_nextn_predict_layers`` is delegated to the
    text config so the engine's MTP stripping logic sees it.
    """

    model_type = "deepseek_v41"
    sub_configs = {
        "text_config": DeepseekV41TextConfig,
        "vision_config": DeepseekV41VisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=129264,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def vision_enabled(self) -> bool:
        return self.vision_config.num_hidden_layers > 0

    @property
    def use_cache(self):
        return self.text_config.use_cache

    @use_cache.setter
    def use_cache(self, value):
        self.text_config.use_cache = value

    @property
    def num_nextn_predict_layers(self):
        return self.text_config.num_nextn_predict_layers

    @num_nextn_predict_layers.setter
    def num_nextn_predict_layers(self, value):
        self.text_config.num_nextn_predict_layers = value


AutoConfig.register("deepseek_v41", DeepseekV41Config, exist_ok=True)
AutoConfig.register("deepseek_v41_text", DeepseekV41TextConfig, exist_ok=True)


def text_config_of(config: DeepseekV41Config | DeepseekV41TextConfig) -> DeepseekV41TextConfig:
    """Return text settings from either supported v4.1 configuration type."""
    return config.text_config if isinstance(config, DeepseekV41Config) else config
