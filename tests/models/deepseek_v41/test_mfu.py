"""CPU tests for the DeepSeek-V4.1 forward-FLOP recipe (pure arithmetic)."""

import pytest

from torchturbo.engine.metrics.flops import ComputeProfile
from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import (
    DeepseekV41TextConfig,
)
from torchturbo.models.deepseek_v41.mfu import DeepseekV41Flops


def _small_config() -> DeepseekV41TextConfig:
    # Tiny but architecturally complete: window + compressed + ratio-1 reuse
    # layers, one kv/index source, engram layers, MoE with a shared expert.
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        sliding_window=4,
        compress_ratios=[0, 2, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        engram_layer_ids=[0, 2],
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_head_dim=4,
    )


def _recipe(config: DeepseekV41TextConfig, batch: int = 2, seq: int = 32):
    profile = ComputeProfile(batch_size=batch, seq_length=seq)
    return DeepseekV41Flops(config, profile)


def _by_prefix(components) -> dict[str, int]:
    return {c.name.split(" (")[0]: c.fwd_flops for c in components}


# B=2, S=32 -> bs=64. Per-layer projection path:
#   wq_a 2*64*16*4   + wq_b 2*64*4*32  + wkv 2*64*16*8
# + wo_a 2*64*32*8   + wo_b 2*64*8*16  = 90112
PROJ_PER_LAYER = 90112
# window k=4: 2 * (2*2*32*4*4*8) = 32768
WINDOW_PER_LAYER = 32768
# ratio 2 -> blocks 16 -> k=4+2=6; ratio 1 -> blocks 32 -> k=6.
# 2 * (2*2*32*4*6*8) = 49152 per compressed layer.
TOPK_PER_LAYER = 49152
# wkv + wgate over every token: 2 * 2*64*16*8 = 32768
COMPRESSOR = 32768
# wq_b 4096 + weights_proj 4096 + einsum 2*2*32*2*16*4 = 16384
# + owns_k 2*32*8*4 = 2048
INDEXER = 26624
# routed (1 expert) 49152 + shared 49152 + router 2*64*16*2 = 4096
MOE_PER_LAYER = 102400
# wkv (2-1)*2*4=8 -> 16*3=48: 2*64*8*48 = 49152 per engram layer
ENGRAM_PER_LAYER = 49152
# hc_dim 32 -> 8 outputs, two projections: 2 * 2*64*32*8 = 65536
HC_PER_LAYER = 65536
LOGITS = 2 * 64 * 16 * 32


def test_components_match_hand_computed_totals():
    recipe = _recipe(_small_config())
    got = _by_prefix(recipe.components())

    assert got == {
        "attention/proj": PROJ_PER_LAYER * 3,
        "attention/scores-window": WINDOW_PER_LAYER * 1,
        "attention/scores-topk": TOPK_PER_LAYER * 2,
        "csa2/compressor": COMPRESSOR,
        "csa2/indexer": INDEXER,
        "ffn/moe": MOE_PER_LAYER * 3,
        "engram": ENGRAM_PER_LAYER * 2,
        "hc mixing": HC_PER_LAYER * 3,
        "logits": LOGITS,
    }
    total = sum(got.values())
    assert total == 270336 + 32768 + 98304 + 32768 + 26624 + 307200 + 98304 + 196608 + 65536


def test_window_only_config_has_no_topk_scores():
    config = _small_config()
    config.compress_ratios = [0, 0, 0]
    config.kv_source_layer_ids = []
    config.index_source_layer_ids = []
    got = _by_prefix(_recipe(config).components())

    assert got["attention/scores-window"] == WINDOW_PER_LAYER * 3
    assert got["attention/scores-topk"] == 0
    assert got["csa2/compressor"] == 0
    assert got["csa2/indexer"] == 0


def test_topk_capped_by_available_blocks():
    # Only 8 tokens -> 4 blocks at ratio 2 -> k = min(2, 4) + min(4, 8) = 6.
    recipe = _recipe(_small_config(), batch=1, seq=8)
    got = _by_prefix(recipe.components())
    # window part k=4: 2*(2*1*8*4*4*8)=4096; topk part k=2: 2*(2*1*8*4*2*8)=2048.
    assert got["attention/scores-window"] == 4096
    assert got["attention/scores-topk"] == (4096 + 2048) * 2


def test_mtp_layers_counted():
    config = _small_config()
    config.num_nextn_predict_layers = 1
    got = _by_prefix(_recipe(config).components())

    # proj + scores (first compressed ratio: k=6) + moe + fc 2*64*32*16 + logits
    expected = PROJ_PER_LAYER + TOPK_PER_LAYER + MOE_PER_LAYER + 65536 + LOGITS
    assert got["mtp"] == expected


def test_flash_sized_config_per_token_magnitude():
    # Realistic Flash dims; per-token forward FLOPs should track ~2x the
    # activated parameter count (attn proj + top-6 MoE + engram + lm head).
    config = DeepseekV41TextConfig(
        vocab_size=129280,
        hidden_size=5120,
        moe_intermediate_size=2304,
        num_hidden_layers=40,
        num_attention_heads=64,
        head_dim=512,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        sliding_window=128,
        compress_ratios=[0] + [2] * 39,
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        engram_layer_ids=[0, 5, 10],
    )
    recipe = _recipe(config, batch=1, seq=4096)
    total = sum(c.fwd_flops for c in recipe.components())
    per_token = total / 4096
    assert 45e9 < per_token < 70e9


def test_get_flops_class_registered():
    pytest.importorskip("torch")
    from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM

    model = object.__new__(TurboDeepseekV41ForCausalLM)
    assert model.get_flops_class() is DeepseekV41Flops
