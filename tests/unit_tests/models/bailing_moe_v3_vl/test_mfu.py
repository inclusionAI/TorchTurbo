"""FLOP recipe coverage for BailingMoE V3 VL (text hybrid decoder + vision)."""

from torchturbo.engine.metrics import op_flops as flops
from torchturbo.engine.metrics.flops import ComputeProfile
from torchturbo.models.bailing_moe_v3_vl.mfu import BailingMoeV3VLFlops
from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
    BailingMoeV3VLForConditionalGeneration,
)

from .test_model import make_config


def _profile(grid=None):
    return ComputeProfile(batch_size=1, seq_length=8, image_grid_thw=grid)


def _vision_flops(config, profile) -> tuple[int, str]:
    vision = [
        c for c in BailingMoeV3VLFlops(config, profile).components()
        if c.name.startswith("vision")
    ]
    assert len(vision) == 1
    return vision[0].fwd_flops, vision[0].note


def test_components_cover_text_decoder_and_logits():
    recipe = BailingMoeV3VLFlops(make_config(hybrid=True), _profile())
    names = " | ".join(c.name for c in recipe.components())

    assert "attention/mla" in names
    assert "attention/kda" in names
    assert "ffn/dense" in names
    assert "ffn/moe" in names
    assert "logits" in names


def test_vision_component_uses_real_grids():
    flops, note = _vision_flops(make_config(hybrid=True), _profile([(1, 4, 4)]))

    assert flops > 0
    # make_config has disable_merger_proj=True and no deepstack indexes.
    assert "include_merger=False" in note
    assert "deepstack mergers=0" in note


def test_without_images_there_is_no_vision_component():
    recipe = BailingMoeV3VLFlops(make_config(), _profile())

    assert not [
        c for c in recipe.components() if c.name.startswith("vision")
    ]


def test_merger_projections_are_counted_when_enabled():
    config = make_config()
    config.vision_config.disable_merger_proj = False
    config.vision_config.deepstack_visual_indexes = [0]

    disabled, _ = _vision_flops(make_config(), _profile([(1, 4, 4)]))
    enabled, note = _vision_flops(config, _profile([(1, 4, 4)]))

    # Merger fc1/fc2 for the main pass plus one deepstack pass.
    assert enabled > disabled
    assert "include_merger=True" in note
    assert "deepstack mergers=1" in note


def test_projector_uses_the_model_input_width():
    config = make_config()
    grid = [(1, 4, 4)]
    projector = BailingMoeV3VLForConditionalGeneration(config).linear_proj
    hidden = config.text_config.hidden_size
    encoder, merged = flops.vision_encoder_flops(
        grid, config.vision_config, include_merger=False
    )
    expected = (
        encoder
        + flops.linear_flops(merged, projector[0].in_features, hidden)
        + flops.linear_flops(merged, hidden, hidden)
    )

    actual, _ = _vision_flops(config, _profile(grid))

    assert actual == expected


def test_video_frames_are_separate_attention_sequences():
    config = make_config()
    per_video = ComputeProfile(batch_size=1, seq_length=8, video_grid_thw=[(2, 4, 4)])
    per_frame = ComputeProfile(
        batch_size=1, seq_length=8, video_grid_thw=[(1, 4, 4), (1, 4, 4)]
    )

    video, _ = _vision_flops(config, per_video)
    frames, _ = _vision_flops(config, per_frame)
    images, _ = _vision_flops(config, _profile([(1, 4, 4), (1, 4, 4)]))

    assert video == frames == images
