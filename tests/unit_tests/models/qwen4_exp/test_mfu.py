"""Qwen3.8-specific theoretical FLOP accounting."""

from torchturbo.engine.metrics.flops import ComputeProfile
from torchturbo.models.qwen4_exp.mfu import Qwen4ExpFlops

from .test_model import _small_config


def _components(sequence_length: int = 16):
    recipe = Qwen4ExpFlops(
        _small_config(),
        ComputeProfile(batch_size=1, seq_length=sequence_length),
    )
    return {component.name: component for component in recipe.components()}


def test_qwen4_components_include_architecture_specific_work() -> None:
    components = _components()
    names = " | ".join(components)

    assert "attention/full" in names
    assert "attention/linear" in names
    assert "ffn/moe" in names
    assert "qsa/indexer" in names
    assert "hyper-connection" in names
    assert "ple" in names
    assert "mtp" not in names
    assert "logits" in names
    assert all(component.fwd_flops > 0 for component in components.values())


def test_qsa_attention_scales_with_budget_after_projection_cost() -> None:
    config = _small_config()
    profile = ComputeProfile(batch_size=1, seq_length=32)
    small = Qwen4ExpFlops(config, profile)
    small_flops = small._full_attention_flops_for_seg(32)

    config.text_config.indexer_budget = 16
    large = Qwen4ExpFlops(config, profile)
    large_flops = large._full_attention_flops_for_seg(32)

    assert large_flops > small_flops


def test_indexer_scoring_uses_all_visible_compressed_blocks() -> None:
    recipe = Qwen4ExpFlops(_small_config(), ComputeProfile(batch_size=1, seq_length=16))

    # Four-token blocks produce cumulative visible-block counts
    # 0+0+0+1 + 1+1+1+2 + 2+2+2+3 + 3+3+3+4 = 28.
    expected_pairs = 28
    projection = 2 * 16 * 64 * ((2 + 1) * 16)
    scores = 2 * expected_pairs * 2 * 16
    pooling = (16 // 4) * 4 * 16

    assert recipe._indexer_flops_for_seg(16) == projection + scores + pooling
