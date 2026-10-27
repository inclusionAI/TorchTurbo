"""Numerical and lifecycle contracts for Bailing V3 VL integration."""

import copy
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchturbo.engine.config import EngineConfig, ParameterGroup
from torchturbo.models.bailing_moe_v3_vl.configuration_bailing_moe_v3_vl import (
    BailingMoeV3Config,
    BailingMoeV3VLConfig,
)
from torchturbo.models.bailing_moe_v3_vl.model import (
    TurboBailingMoeV3VLForConditionalGeneration,
)
from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
    BailingMoeV3KimiDeltaAttention,
    _prepare_mtp_input_ids,
)
from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
from torchturbo.models.modules.vision.io import VisionBatch


def make_config(*, hybrid=False, mtp=0):
    config = BailingMoeV3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=8,
            v_head_dim=8,
            kv_lora_rank=8,
            num_experts=4,
            num_shared_experts=1,
            moe_intermediate_size=8,
            moe_shared_expert_intermediate_size=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            layer_group_size=2 if hybrid else 1,
            first_k_dense_replace=1,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
            expert_swiglu_limit_list=[0.0, 4.0, 4.0],
            share_expert_swiglu_limit_list=[0.0, 5.0, 5.0],
            num_nextn_predict_layers=mtp,
            mtp_loss_scaling_factor=0.1,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=24,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=16,
            disable_merger_proj=True,
            deepstack_visual_indexes=[],
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        mrope_section=[2, 1, 1],
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    return config


def make_model(**kwargs):
    torch.manual_seed(42)
    model = TurboBailingMoeV3VLForConditionalGeneration(make_config(**kwargs))
    for module in model.modules():
        if isinstance(module, BailingMoeV3KimiDeltaAttention):
            nn.init.zeros_(module.dt_bias)
    return model.eval()


@pytest.fixture
def single_rank_group(tmp_path):
    torch.distributed.init_process_group(
        "gloo",
        init_method=(tmp_path / "vision_balance_group").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("argument_style", ["positional", "mixed", "keyword"])
def test_vision_balance_io_preserves_call_arguments(argument_style):
    tower = make_model().model.visual
    io = BailingVisionBalanceIO()
    pixels = torch.randn(20, 24)
    grid = torch.tensor([[1, 2, 2], [2, 2, 4]])
    metadata = {"attention_mask": None}
    kwargs = {"metadata": metadata}
    extra_argument = object()
    if argument_style == "positional":
        args = (pixels, grid, extra_argument)
    elif argument_style == "mixed":
        args = (pixels,)
        kwargs["grid_thw"] = grid
    else:
        args = ()
        kwargs.update(hidden_states=pixels, grid_thw=grid)

    original_kwargs = kwargs.copy()
    batch = io.read_inputs(tower, args, kwargs)
    assert batch.pixels is pixels
    assert batch.grid_thw is grid
    assert batch.output_lengths == {"features": [1, 4]}

    replacement = VisionBatch(
        pixels=torch.cat((pixels[4:], pixels[:4])),
        grid_thw=grid.flip(0),
        output_lengths={"features": [4, 1]},
    )
    new_args, new_kwargs = io.write_inputs(args, kwargs, replacement)
    assert len(new_args) == len(args)
    assert new_kwargs.keys() == kwargs.keys()
    assert new_kwargs["metadata"] is metadata
    assert all(kwargs[name] is value for name, value in original_kwargs.items())
    if argument_style == "positional":
        assert new_args == (replacement.pixels, replacement.grid_thw, extra_argument)
        assert args == (pixels, grid, extra_argument)
    elif argument_style == "mixed":
        assert new_args[0] is replacement.pixels
        assert new_kwargs["grid_thw"] is replacement.grid_thw
        assert args[0] is pixels
    else:
        assert new_kwargs["hidden_states"] is replacement.pixels
        assert new_kwargs["grid_thw"] is replacement.grid_thw


def test_vision_balance_io_dummy_runs_one_merged_token():
    tower = make_model().model.visual.double()
    io = BailingVisionBalanceIO()
    batch = io.make_dummy(tower)

    assert batch.pixels.shape == (4, 24)
    assert batch.pixels.dtype == torch.float64
    assert batch.pixels.device == next(tower.parameters()).device
    assert torch.count_nonzero(batch.pixels) == 0
    assert batch.grid_thw.tolist() == [[1, 2, 2]]
    assert batch.grid_thw.dtype == torch.long
    assert batch.grid_thw.device == batch.pixels.device
    assert batch.output_lengths == {"features": [1]}
    args, kwargs = io.write_inputs((), {}, batch)
    with torch.no_grad():
        features = tower(*args, **kwargs)
    assert features.shape == (1, 64)
    assert io.read_outputs(features)["features"] is features


@pytest.mark.parametrize("explicit_none", [False, True])
def test_vision_balance_io_requires_media_inputs(explicit_none):
    # Media-less batches reach the tower with a framework-provided dummy
    # image, so read_inputs must not invent media for missing inputs.
    tower = make_model().model.visual
    io = BailingVisionBalanceIO()
    args = (None, None) if explicit_none else ()

    with pytest.raises(ValueError, match="hidden_states and grid_thw together"):
        io.read_inputs(tower, args, {})


@pytest.mark.parametrize("missing", ["hidden_states", "grid_thw"])
def test_vision_balance_io_rejects_missing_input_pair(missing):
    tower = make_model().model.visual
    kwargs = {
        "hidden_states": torch.randn(4, 24),
        "grid_thw": torch.tensor([[1, 2, 2]]),
    }
    kwargs[missing] = None

    with pytest.raises(ValueError, match="hidden_states and grid_thw together"):
        BailingVisionBalanceIO().read_inputs(tower, (), kwargs)


def test_vision_balance_io_restores_tensor_and_rejects_structured_output():
    io = BailingVisionBalanceIO()
    output = torch.randn(1, 64)
    restored = torch.randn(2, 64)
    original_batch = VisionBatch(
        pixels=torch.randn(8, 24),
        grid_thw=torch.tensor([[1, 2, 4]]),
        output_lengths={"features": [2]},
    )

    assert io.read_outputs(output)["features"] is output
    assert io.write_outputs(output, {"features": restored}, original_batch) is restored
    with pytest.raises(TypeError, match="tensor tower output"):
        io.read_outputs((output, [torch.randn(1, 64)]))


def test_default_plan_declares_disabled_bailing_vision_balance_io():
    plan = TurboBailingMoeV3VLForConditionalGeneration.get_default_parallel_plan()
    spec = plan.plan["model.visual"]

    assert spec.standalone_world_shard
    assert spec.vision_balance.mode is None
    assert isinstance(spec.vision_balance.io, BailingVisionBalanceIO)


@pytest.mark.parametrize("mode", ["patch", "patch_sq"])
@pytest.mark.parametrize(
    "checkpoint_mode, frozen", [(None, False), (False, False), (True, False), (None, True)]
)
def test_vision_balance_registration_preserves_forward_and_backward(
    monkeypatch, single_rank_group, mode, checkpoint_mode, frozen
):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import ActivationCheckpointingConfig
    from torchturbo.engine.model_fqn import build_model_fqn_map
    from torchturbo.models.modules.vision import register_vision_balance
    from torchturbo.models.modules.vision.balance import _VisionBalanceState
    from torchturbo.parallel import (
        FSDPParallelPlan,
        FSDPPlanSpec,
        VisionBalanceMode,
        VisionBalanceSpec,
    )

    tower = make_model().model.visual.train().requires_grad_(not frozen)
    reference = copy.deepcopy(tower)
    if checkpoint_mode is not None:
        backend = FSDP2Backend.__new__(FSDP2Backend)
        backend.logger = logging.getLogger(__name__)
        backend.config = EngineConfig(
            activation_checkpointing=ActivationCheckpointingConfig(
                enabled=True, use_reentrant=checkpoint_mode
            )
        )
        backend.prefetch_manager = SimpleNamespace(
            suspend_expert_forward_prefetch=lambda model: nullcontext()
        )
        plan = FSDPParallelPlan(
            plan={"blocks.*": FSDPPlanSpec(activation_checkpoint=True)}
        )
        resolved = backend._compile_parallel_plan(plan, build_model_fqn_map(tower))
        backend._apply_activation_checkpointing(tower, resolved)

    events = []
    block_calls = []
    original_build_plan = _VisionBalanceState._build_plan
    original_post_forward = _VisionBalanceState._post_forward

    def build_plan(state, module, batch):
        events.append("balance.pre")
        return original_build_plan(state, module, batch)

    def post_forward(state, module, args, output):
        events.append("balance.post")
        return original_post_forward(state, module, args, output)

    monkeypatch.setattr(_VisionBalanceState, "_build_plan", build_plan)
    monkeypatch.setattr(_VisionBalanceState, "_post_forward", post_forward)
    # These markers occupy the same hook positions as FSDP. The tower and
    # collectives execute on CPU; this test does not shard its parameters.
    tower.register_forward_pre_hook(
        lambda module, args: events.append("fsdp.pre"), prepend=True
    )
    tower.register_forward_hook(
        lambda module, args, output: events.append("fsdp.post")
    )
    tower.blocks[0].attn.qkv.register_forward_pre_hook(
        lambda module, args: block_calls.append(None)
    )
    register_vision_balance(
        tower,
        spec=VisionBalanceSpec(mode=VisionBalanceMode(mode), io=BailingVisionBalanceIO()),
        sp_size=1,
        sp_rank=0,
    )
    assert tower.state_dict().keys() == reference.state_dict().keys()

    pixels = torch.randn(20, 24)
    grid = torch.tensor([[1, 2, 2], [2, 2, 4]])
    expected = reference(pixels, grid_thw=grid)
    if not frozen:
        expected.square().sum().backward()

    for _ in range(2):
        events.clear()
        block_calls.clear()
        tower.zero_grad(set_to_none=True)
        actual = tower(pixels, grid_thw=grid)
        torch.testing.assert_close(actual, expected)
        if not frozen:
            actual.square().sum().backward()
            for parameter, reference_parameter in zip(
                tower.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        assert events == ["fsdp.pre", "balance.pre", "fsdp.post", "balance.post"]
        assert len(block_calls) == (2 if checkpoint_mode is not None else 1)

    events.clear()
    tower.eval()
    reference.eval()
    with torch.no_grad():
        actual = tower(pixels, grid_thw=grid)
        torch.testing.assert_close(actual, reference(pixels, grid_thw=grid))
    assert events == ["fsdp.pre", "fsdp.post", "balance.post"]


@pytest.mark.parametrize("hybrid", [False, True])
def test_image_forward_backward(hybrid):
    model = make_model(hybrid=hybrid).train()
    ids = torch.tensor([[2, 62, 60, 63, 3, 4]])
    result = model(
        input_ids=ids,
        pixel_values=torch.randn(4, 24),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        labels=ids,
    )
    assert result.logits.shape == (1, 6, 64)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for parameter in [
        model.linear_proj[0].weight,
        model.model.visual.patch_embed.proj.weight,
        model.model.layers[0].attention.q_proj.weight,
    ]:
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_mixed_media_uses_one_visual_forward_and_preserves_logits(hybrid):
    model = make_model(hybrid=hybrid)
    ids = torch.tensor(
        [[2, 62, 61, 61, 63, 3, 62, 60, 63, 4, 62, 61, 61, 63, 5]]
    )
    image_pixels = torch.randn(4, 24)
    video_pixels = torch.randn(16, 24)
    image_grid = torch.tensor([[1, 2, 2]])
    video_grid = torch.tensor([[2, 2, 4]])
    with torch.no_grad():
        image_features = torch.cat(model.get_image_features(image_pixels, image_grid))
        video_features = torch.cat(model.get_video_features(video_pixels, video_grid))
        inputs_embeds = model.get_input_embeddings()(ids)
        for token_id, features in [(60, image_features), (61, video_features)]:
            mask = ids.eq(token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features)
        positions, _ = model.model.get_rope_index(
            ids, image_grid_thw=image_grid, video_grid_thw=video_grid
        )
        expected = model(inputs_embeds=inputs_embeds, position_ids=positions).logits
        with (
            patch.object(
                model.model.visual, "forward", wraps=model.model.visual.forward
            ) as visual,
            patch.object(
                model.linear_proj, "forward", wraps=model.linear_proj.forward
            ) as projector,
        ):
            result = model(
                input_ids=ids,
                pixel_values=image_pixels,
                image_grid_thw=image_grid,
                pixel_values_videos=video_pixels,
                video_grid_thw=video_grid,
            )
    assert visual.call_count == 1
    assert projector.call_count == 1
    torch.testing.assert_close(result.logits, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("training", [False, True])
def test_text_forward_does_not_synthesize_media(training):
    model = make_model().train(training)
    ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    with (
        patch.object(
            model.model.visual, "forward", wraps=model.model.visual.forward
        ) as visual,
        patch.object(
            model.linear_proj, "forward", wraps=model.linear_proj.forward
        ) as projector,
    ):
        result = model(input_ids=ids, labels=ids)
    assert visual.call_count == 0
    assert projector.call_count == 0
    torch.testing.assert_close(result.logits, expected, atol=0, rtol=0)
    if training:
        result.loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(("model.visual.", "linear_proj.")):
                assert parameter.grad is None, name
        assert model.model.layers[0].attention.q_proj.weight.grad.norm() > 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_text_generation_skips_visual_forward(hybrid):
    model = make_model(hybrid=hybrid)

    with (
        patch.object(
            model.model.visual, "forward", wraps=model.model.visual.forward
        ) as visual,
        patch.object(
            model.linear_proj, "forward", wraps=model.linear_proj.forward
        ) as projector,
    ):
        first = model(input_ids=torch.tensor([[2, 3, 4]]), use_cache=True)
        assert visual.call_count == 0
        assert projector.call_count == 0

        model(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            past_key_values=first.past_key_values,
            use_cache=True,
        )

    assert visual.call_count == 0
    assert projector.call_count == 0


def test_mtp_input_ids_remain_available_with_inputs_embeds():
    model = make_model(mtp=1)
    ids = torch.tensor([[2, 3, 4, 5]])
    inputs_embeds = model.get_input_embeddings()(ids)
    with patch.object(
        model.model,
        "compute_3d_position_ids",
        wraps=model.model.compute_3d_position_ids,
    ) as compute_positions:
        result = model(inputs_embeds=inputs_embeds, mtp_input_ids=ids, labels=ids)

    torch.testing.assert_close(compute_positions.call_args.kwargs["input_ids"], ids)
    assert result.mtp_logits[0].shape == result.logits.shape


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("tail_padding", [0, 2])
@pytest.mark.parametrize("lf_positions", [False, True])
def test_packed_images_match_independent_samples(
    hybrid, tail_padding, lf_positions
):
    model = make_model(hybrid=hybrid)
    first = torch.tensor([[2, 62, 60, 63, 3]])
    second = torch.tensor([[4, 62, 60, 60, 63, 5]])
    pixels = torch.randn(12, 24)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    separate = [
        model(
            input_ids=first, pixel_values=pixels[:4], image_grid_thw=grids[:1]
        ).logits,
        model(
            input_ids=second, pixel_values=pixels[4:], image_grid_thw=grids[1:]
        ).logits,
    ]
    ids = torch.cat([first, second], -1)
    valid_length = ids.shape[1]
    cu = torch.tensor([0, 5, valid_length], dtype=torch.int32)
    if tail_padding:
        ids = nn.functional.pad(ids, (0, tail_padding))
    positions = None
    if lf_positions:
        segment_ids = torch.tensor(
            [[1] * 5 + [2] * (valid_length - 5) + [0] * tail_padding]
        )
        vision_positions, _ = model.model.get_rope_index(
            ids, segment_ids=segment_ids, image_grid_thw=grids
        )
        positions = torch.cat((vision_positions[:1], vision_positions))
    result = model(
        input_ids=ids,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=6,
        max_length_k=6,
        position_ids=positions,
        pixel_values=pixels,
        image_grid_thw=grids,
    )
    torch.testing.assert_close(
        result.logits[:, :valid_length],
        torch.cat(separate, 1),
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("sp_size", [1, 4])
def test_prepared_swift_packed_batch_matches_independent_samples(hybrid, sp_size):
    """Swift packs text positions; the forward derives per-sample mRoPE."""
    from torchturbo.models.modules.attention.ulysses import ulysses_pad_inputs

    model = make_model(hybrid=hybrid)
    text = torch.tensor([[2, 3, 4]])
    image = torch.tensor([[4, 62, 60, 60, 63, 5]])
    pixels = torch.randn(8, 24)
    grids = torch.tensor([[1, 2, 4]])
    separate = torch.cat(
        (
            model(input_ids=text).logits,
            model(input_ids=image, pixel_values=pixels, image_grid_thw=grids).logits,
        ),
        dim=1,
    )
    cu = torch.tensor([0, 3, 9], dtype=torch.int32)
    inputs = model.on_prepare_inputs(
        {
            "input_ids": torch.cat((text, image), dim=1),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4, 5]]),
            "cu_seq_lens_q": cu,
            "cu_seq_lens_k": cu,
            "max_length_q": 6,
            "max_length_k": 6,
            "pixel_values": pixels,
            "image_grid_thw": grids,
        }
    )
    # The SP alignment tail stays outside every packed sample.
    inputs = ulysses_pad_inputs(
        inputs, model.get_sequence_parallel_input_slicing_spec(), sp_size
    )

    result = model(**inputs)

    torch.testing.assert_close(
        result.logits[:, : separate.size(1)], separate, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_batched_images_and_broadcast_positions(hybrid):
    model = make_model(hybrid=hybrid)
    ids = torch.tensor([[2, 62, 60, 63, 3, 0], [4, 62, 60, 60, 63, 5]])
    pixels = torch.randn(12, 24)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    result = model(
        input_ids=ids,
        attention_mask=ids.ne(0),
        position_ids=torch.arange(6)[None],
        pixel_values=pixels,
        image_grid_thw=grids,
    )
    for row, length, pixel_slice in [(0, 5, slice(0, 4)), (1, 6, slice(4, 12))]:
        expected = model(
            input_ids=ids[row : row + 1, :length],
            pixel_values=pixels[pixel_slice],
            image_grid_thw=grids[row : row + 1],
        ).logits
        torch.testing.assert_close(
            result.logits[row : row + 1, :length], expected, atol=2e-6, rtol=2e-5
        )


def test_position_rows_and_interleaved_video():
    model = make_model()
    ids = torch.tensor([[2, 62, 61, 63, 3, 62, 61, 63, 4]])
    grids = torch.tensor([[2, 2, 2]])
    pixels = torch.randn(8, 24)
    positions, _ = model.model.get_rope_index(ids, video_grid_thw=grids)
    expanded, _ = model.model.get_rope_index(
        ids, video_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]])
    )
    torch.testing.assert_close(positions, expanded)
    inputs = dict(input_ids=ids, pixel_values_videos=pixels, video_grid_thw=grids)
    expected = model(**inputs).logits
    for provided in [
        torch.arange(9)[None],
        positions,
        torch.cat([torch.arange(9)[None, None], positions]),
    ]:
        torch.testing.assert_close(
            model(**inputs, position_ids=provided).logits, expected
        )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("media", ["image", "video"])
def test_generation_without_mm_token_types_matches_forward(hybrid, media):
    model = make_model(hybrid=hybrid)
    if media == "image":
        ids = torch.tensor([[2, 62, 60, 60, 63, 3]])
        media_inputs = dict(
            pixel_values=torch.randn(8, 24),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )
    else:
        ids = torch.tensor([[2, 62, 61, 61, 63, 3, 62, 61, 61, 63, 4]])
        media_inputs = dict(
            pixel_values_videos=torch.randn(16, 24),
            video_grid_thw=torch.tensor([[2, 2, 4]]),
        )
    inputs = dict(
        input_ids=ids, attention_mask=torch.ones_like(ids), **media_inputs
    )
    with torch.no_grad():
        expected_logits = model(**inputs).logits[:, -1]
    generated = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    torch.testing.assert_close(
        generated.logits[0], expected_logits, atol=2e-6, rtol=2e-5
    )


def test_mtp_outputs_and_shift_boundaries():
    model = make_model(mtp=1)
    ids = torch.tensor([[2, 3, 4, 5]])
    result = model(input_ids=ids, labels=ids)
    assert len(result.mtp_logits) == 1
    assert result.mtp_logits[0].shape == result.logits.shape
    result.loss.backward()
    assert model.model.layers[2].eh_proj.weight.grad.norm() > 0
    shifted = _prepare_mtp_input_ids(ids, torch.tensor([[1, 1, 2, 2]]), 1, 0)
    assert shifted[0].tolist() == [[3, 0, 5, 0]]


@pytest.mark.parametrize("hybrid", [False, True])
def test_packed_mtp_matches_independent_samples(hybrid):
    model = make_model(hybrid=hybrid, mtp=1)
    first = torch.tensor([[2, 62, 60, 63, 3, 7]])
    second = torch.tensor([[4, 62, 60, 60, 63, 5, 9]])
    pixels = torch.randn(12, 24)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    with torch.no_grad():
        separate = [
            model(input_ids=first, pixel_values=pixels[:4], image_grid_thw=grids[:1]),
            model(input_ids=second, pixel_values=pixels[4:], image_grid_thw=grids[1:]),
        ]
        ids = torch.cat((first, second), dim=1)
        cu = torch.tensor([0, first.shape[1], ids.shape[1]], dtype=torch.int32)
        packed = model(
            input_ids=ids,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=second.shape[1],
            max_length_k=second.shape[1],
            pixel_values=pixels,
            image_grid_thw=grids,
        )

    torch.testing.assert_close(
        packed.logits,
        torch.cat([result.logits for result in separate], 1),
        atol=2e-6,
        rtol=2e-5,
    )
    # MTP inputs stop at each sample boundary, so every position matches.
    torch.testing.assert_close(
        packed.mtp_logits[0],
        torch.cat([result.mtp_logits[0] for result in separate], 1),
        atol=2e-6,
        rtol=2e-5,
    )


def test_mtp_disabled_by_engine_and_parameter_groups():
    config = make_config(mtp=1)
    with TurboBailingMoeV3VLForConditionalGeneration.init_context(
        config, EngineConfig()
    ):
        assert config.text_config.num_nextn_predict_layers == 0
    model = make_model()
    groups = model.get_parameter_groups()
    names = []
    for group in [
        ParameterGroup.VISION_TOWER,
        ParameterGroup.MULTI_MODAL_PROJECTOR,
        ParameterGroup.LANGUAGE_MODEL,
    ]:
        group_names = [name for name, _ in groups[group]]
        assert group_names
        names.extend(group_names)
    assert len(names) == len(set(names))
    assert set(names) == set(dict(model.named_parameters()))


def test_meta_rotary_materialization_and_safetensors_roundtrip(tmp_path):
    model = make_model()
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    restored = TurboBailingMoeV3VLForConditionalGeneration.from_pretrained(
        tmp_path, attn_implementation="eager"
    )
    ids = torch.tensor([[2, 3, 4]])
    torch.testing.assert_close(
        restored(input_ids=ids).logits, model(input_ids=ids).logits
    )
    with torch.device("meta"):
        meta_model = TurboBailingMoeV3VLForConditionalGeneration(make_config())
    meta_model.on_materialized()
    for name, buffer in meta_model.named_buffers():
        assert not buffer.is_meta, name


def test_config_roundtrip_preserves_llm_config_alias(tmp_path):
    config = make_config(mtp=1)
    config.save_pretrained(tmp_path)

    with (tmp_path / "config.json").open() as config_file:
        serialized = json.load(config_file)
    assert serialized["llm_config"] == serialized["text_config"]

    restored = BailingMoeV3VLConfig.from_pretrained(tmp_path)
    assert isinstance(restored.llm_config, BailingMoeV3Config)
    assert restored.llm_config is restored.text_config

    restored.num_nextn_predict_layers = 0
    assert restored.llm_config.num_nextn_predict_layers == 0


def test_config_accepts_llm_config_as_text_config():
    text_config = make_config().text_config.to_dict()
    config = BailingMoeV3VLConfig(
        llm_config=text_config,
        mrope_section=[2, 1, 1],
    )

    assert config.llm_config is config.text_config
    assert config.text_config.hidden_size == 16


@pytest.mark.parametrize("override_name", ["text_config", "llm_config"])
def test_config_from_pretrained_keeps_text_config_alias(tmp_path, override_name):
    make_config().save_pretrained(tmp_path)

    config = BailingMoeV3VLConfig.from_pretrained(
        tmp_path,
        **{override_name: {"num_nextn_predict_layers": 2}},
    )

    assert config.llm_config is config.text_config
    assert config.text_config.num_nextn_predict_layers == 2


def test_config_from_dict_prefers_text_config_override():
    config, unused_kwargs = BailingMoeV3VLConfig.from_dict(
        make_config().to_dict(),
        llm_config={"num_nextn_predict_layers": 1},
        text_config={"num_nextn_predict_layers": 2},
        unused_option=True,
        return_unused_kwargs=True,
    )

    assert config.llm_config is config.text_config
    assert config.text_config.num_nextn_predict_layers == 2
    assert unused_kwargs == {"unused_option": True}


def test_chunked_loss_uses_main_and_mtp_hidden_states():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    model = make_model(mtp=1).train()
    backend = FSDP2Backend.__new__(FSDP2Backend)
    head = model.lm_head
    outputs, hidden = backend._forward_for_lm_loss(
        model, {"input_ids": torch.tensor([[2, 3, 4, 5]])}, use_chunked_loss=True
    )
    assert outputs.logits is None and outputs.mtp_logits is None
    assert len(hidden) == 2
    assert all(value.shape == (1, 4, 16) for value in hidden)
    assert model.lm_head is head


def test_sp_slices_after_media_fusion_and_global_mtp_shift(monkeypatch):
    import torchturbo.models.bailing_moe_v3_vl.model as wrapper
    import torchturbo.models.modules.attention.ulysses.ulysses as ulysses
    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        BailingMoeV3VLModel,
        MoeV3ModelOutputWithPast,
    )

    model = make_model(mtp=1)
    ids = torch.tensor([[2, 62, 60, 63, 3, 4]])
    pixels = torch.randn(4, 24)
    grid = torch.tensor([[1, 2, 2]])
    expected_image = model.get_image_features(pixels, grid)[0]
    captured = {}

    def capture(module, **kwargs):
        captured.update(kwargs)
        return MoeV3ModelOutputWithPast(
            last_hidden_state=kwargs["inputs_embeds"], mtp_hidden_states=()
        )

    monkeypatch.setattr(
        ulysses,
        "slice_input_tensor",
        lambda tensor, dim, **kwargs: tensor.narrow(dim, 0, 3),
    )
    monkeypatch.setattr(model, "register_sp_attention_interface", lambda *args: None)
    monkeypatch.setattr(wrapper, "patch_kda_forward", lambda *args: None)
    monkeypatch.setattr(wrapper, "patch_rms_norm", lambda *args: None)
    monkeypatch.setattr(BailingMoeV3VLModel, "_forward_text", capture)
    model.config.text_config._attn_implementation = "flash_attention_2"
    sp_info = {"sp_group": None, "sp_size": 2, "sp_rank": 0}
    ulysses.patch_forward_for_sp_input_slicing(
        model.get_sequence_parallel_input_slicing_spec(), **sp_info
    )
    model.on_parallelized({"sp": sp_info})
    output = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
    assert output.logits.shape == (1, 3, 64)
    # No mask is rebuilt; global cu boundaries ride through to the layers.
    assert "attention_mask" not in captured
    assert captured["position_ids"].shape == (3, 1, 3)
    assert captured["text_position_ids"].tolist() == [[0, 1, 2]]
    assert captured["mtp_shifted_input_ids"][0].tolist() == [[62, 60, 63]]
    torch.testing.assert_close(captured["inputs_embeds"][0, 2:3], expected_image)


def test_expert_converter_and_layer_limits(tmp_path, monkeypatch):
    import torchturbo.models.bailing_hybrid.parallel_plan as shared
    from torchturbo.models.bailing_moe_v3_vl.parallel_plan import _replace_vl_moe
    from torchturbo.models.modules.moe.moe_layer import MoESubmodules

    model = make_model()
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    context = {"ep": {"ep_size": 1, "ep_rank": 0, "ep_group": None}}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    converter = model.get_weight_converter(context, str(tmp_path))
    assert converter.config is model.config.text_config
    assert (
        converter.to_hf_name("model.layers.1.mlp.router.expert_bias")
        == "model.layers.1.mlp.gate.expert_bias"
    )
    assert converter.to_hf_name("linear_proj.0.weight") == "linear_proj.0.weight"
    original = model.model.layers[1].mlp
    captured = {}

    def replace(**kwargs):
        captured.update(kwargs)
        return nn.Module()

    monkeypatch.setattr(shared, "MoELayer", replace)
    args = (
        model.config,
        {"sp_group": None},
        context["ep"],
        MoESubmodules(),
        EngineConfig(),
    )
    dense = model.model.layers[0].mlp
    assert _replace_vl_moe(dense, *args) is dense
    replacement = _replace_vl_moe(original, *args)
    assert captured["layer_number"] == 1
    assert captured["config"].moe_swiglu_limit == [0.0, 4.0, 4.0]
    assert captured["config"].moe_swiglu_clamp_gate_after_activation
    assert replacement.shared_experts is original.shared_experts


def test_grouped_experts_cap_activated_gate_like_checkpoint_mlp(monkeypatch):
    import importlib

    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        BailingMoeV3MLP,
    )
    from torchturbo.models.modules.moe.config_converter import get_base_moe_config
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_config import MoEConfig
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime

    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())
    torch.manual_seed(0)
    text_config = make_config().text_config
    limit = 4.0
    references = [
        BailingMoeV3MLP(
            text_config, text_config.moe_intermediate_size, swiglu_limit=limit
        )
        for _ in range(2)
    ]
    # Large inputs push many gate activations beyond the limit.
    hidden_states = torch.randn(6, text_config.hidden_size) * 8
    tokens_per_expert = torch.tensor([4, 2])
    with torch.no_grad():
        expected = torch.cat(
            (references[0](hidden_states[:4]), references[1](hidden_states[4:]))
        )
        gates = references[0].act_fn(references[0].gate_proj(hidden_states[:4]))
    assert (gates > limit).any()

    outputs = {}
    for clamp_gate_after_activation in (True, False):
        moe_config = MoEConfig(
            **get_base_moe_config(
                text_config,
                ep_size=1,
                num_moe_experts=2,
                moe_swiglu_limit=[limit],
                moe_swiglu_clamp_gate_after_activation=clamp_gate_after_activation,
            )
        )
        experts = GroupedMLP(
            num_local_experts=2, use_te=False, config=moe_config, layer_number=0
        )
        with torch.no_grad():
            experts.weight1.copy_(
                torch.cat(
                    [
                        torch.cat((mlp.gate_proj.weight.T, mlp.up_proj.weight.T), 1)
                        for mlp in references
                    ]
                )
            )
            experts.weight2.copy_(
                torch.cat([mlp.down_proj.weight.T for mlp in references])
            )
            outputs[clamp_gate_after_activation], _ = experts(
                hidden_states, tokens_per_expert, torch.ones(6)
            )

    torch.testing.assert_close(outputs[True], expected)
    assert not torch.allclose(outputs[False], expected)


@pytest.mark.parametrize(
    "text_positions, cu",
    [
        ([7, 8, 9, 10], None),
        ([2, 2, 2, 3], [0, 4]),
        ([7, 8, 0, 1], [0, 4]),
        ([2, 2, 2, 3], [0, 2, 4]),
    ],
)
def test_four_position_rows_preserve_caller_text_positions(
    monkeypatch, text_positions, cu
):
    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        BailingMoeV3VLModel,
        MoeV3ModelOutputWithPast,
    )

    model = make_model()
    captured = {}

    def capture(module, **kwargs):
        captured.update(kwargs)
        return MoeV3ModelOutputWithPast(
            last_hidden_state=kwargs["inputs_embeds"], mtp_hidden_states=()
        )

    monkeypatch.setattr(BailingMoeV3VLModel, "_forward_text", capture)
    text_positions = torch.tensor([text_positions])
    vision_positions = torch.tensor([[[2, 2, 2, 3]], [[2, 2, 3, 3]], [[2, 3, 2, 3]]])
    positions = torch.cat((text_positions.unsqueeze(0), vision_positions))
    inputs = dict(input_ids=torch.tensor([[2, 3, 4, 5]]), position_ids=positions)
    if cu is not None:
        cu = torch.tensor(cu, dtype=torch.int32)
        inputs.update(
            cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=2, max_length_k=2
        )
    model(**inputs)
    torch.testing.assert_close(captured["text_position_ids"], text_positions)
    torch.testing.assert_close(captured["position_ids"], vision_positions)
    # Boundaries ride with cu; no mask is rebuilt on the packed path.
    assert "attention_mask" not in captured
    if cu is None:
        assert "cu_seq_lens_q" not in captured
    else:
        assert captured["cu_seq_lens_q"] is cu


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("four_rows", [False, True])
def test_cu_boundaries_define_packing_not_positions(hybrid, four_rows):
    """Position resets alone no longer split samples; cu boundaries do."""
    model = make_model(hybrid=hybrid)
    first = torch.tensor([[2, 3]])
    second = torch.tensor([[4, 5]])
    separate = torch.cat(
        (model(input_ids=first).logits, model(input_ids=second).logits), dim=1
    )
    ids = torch.cat((first, second), dim=1)
    positions = torch.tensor([[0, 1, 0, 1]])
    if four_rows:
        positions = positions.unsqueeze(0).expand(4, -1, -1)

    # Without cu the reset positions describe one sequence: the samples
    # attend across the boundary and the logits differ.
    joined = model(input_ids=ids, position_ids=positions)
    assert not torch.allclose(joined.logits, separate, atol=2e-6, rtol=2e-5)

    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed = model(
        input_ids=ids,
        position_ids=positions,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=2,
        max_length_k=2,
    )
    torch.testing.assert_close(packed.logits, separate, atol=2e-6, rtol=2e-5)


def test_rl_packed_micro_batch_preserves_kda_boundaries(monkeypatch):
    from torchturbo.engine import TorchTurboEngine
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    model = make_model(hybrid=True)
    first = torch.tensor([[2, 3]])
    second = torch.tensor([[4, 5, 6]])
    separate = torch.cat(
        (model(input_ids=first).logits, model(input_ids=second).logits), dim=1
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.model = model
    engine.backend = FSDP2Backend.__new__(FSDP2Backend)
    engine.backend.get_sp_info = lambda: {"sp_size": 1}
    monkeypatch.setattr(engine, "get_sp_info", lambda: {"sp_size": 1})
    inputs = engine.prepare_packed_micro_batch(
        {"input_ids": torch.cat((first, second), dim=1).squeeze(0), "cu_seqlens": cu},
        [2, 3],
    )

    packed = model(**inputs)

    torch.testing.assert_close(packed.logits, separate, atol=2e-6, rtol=2e-5)
    assert inputs["cu_seq_lens_q"] is cu
    packed.logits.sum().backward()
    assert torch.isfinite(model.model.layers[0].attention.q_proj.weight.grad).all()


def test_default_training_initialization_supports_vision_flash_attention(monkeypatch):
    from packaging.version import Version
    from transformers import __version__ as transformers_version

    import transformers.modeling_utils as hf_modeling

    # Keep HF's model capability checks active; only replace kernel availability
    # and loading because construction does not execute an attention kernel.
    if Version(transformers_version) >= Version("5.4.0"):
        monkeypatch.setitem(
            hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
            "general_availability_check",
            lambda: True,
        )
    else:
        monkeypatch.setattr(hf_modeling, "is_flash_attn_2_available", lambda: True)
    loaded_backends = []
    monkeypatch.setattr(
        hf_modeling, "lazy_import_flash_attention", loaded_backends.append
    )
    config = make_config(hybrid=True)
    with (
        TurboBailingMoeV3VLForConditionalGeneration.init_context(config, EngineConfig()),
        torch.device("meta"),
    ):
        model = TurboBailingMoeV3VLForConditionalGeneration(config)

    assert model.model.visual.config._attn_implementation == "flash_attention_2"
    assert model.model.config._attn_implementation == "flash_attention_2"
    assert loaded_backends and set(loaded_backends) == {"flash_attention_2"}


def _segment_attention(module, query, key, value, attention_mask, **kwargs):
    """Fake HF flash interface: per-segment SDPA on [B, H, S, D] inputs."""
    assert attention_mask is None
    lengths = kwargs["cu_seq_lens_q"].diff().tolist()
    outputs = [
        nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=kwargs["is_causal"], scale=kwargs["scaling"]
        )
        for q, k, v in zip(
            *(torch.split(tensor, lengths, dim=2) for tensor in (query, key, value))
        )
    ]
    return torch.cat(outputs, dim=2).transpose(1, 2), None


@pytest.mark.parametrize("num_padding_tokens", [0, 2])
def test_mtp_attention_preserves_explicit_boundaries(monkeypatch, num_padding_tokens):
    from transformers.modeling_flash_attention_utils import (
        prepare_fa_kwargs_from_position_ids,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = make_model(mtp=1)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    num_tokens = ids.shape[1]
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    metadata = dict(
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=num_tokens,
        max_length_k=num_tokens,
    )
    expected = model(input_ids=ids, position_ids=positions, **metadata)
    calls = []

    def flash_interface(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs.copy())
        if kwargs.get("cu_seq_lens_q") is None:
            lengths, max_lengths = prepare_fa_kwargs_from_position_ids(
                kwargs["position_ids"]
            )
            kwargs.update(
                cu_seq_lens_q=lengths[0],
                cu_seq_lens_k=lengths[1],
                max_length_q=max_lengths[0],
                max_length_k=max_lengths[1],
            )
        valid_tokens = int(kwargs["cu_seq_lens_q"][-1])
        output, weights = _segment_attention(
            module,
            query[:, :, :valid_tokens],
            key[:, :, :valid_tokens],
            value[:, :, :valid_tokens],
            attention_mask,
            is_causal=module.is_causal,
            **kwargs,
        )
        output = nn.functional.pad(
            output, (0, 0, 0, 0, 0, query.shape[2] - valid_tokens)
        )
        return output, weights

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "flash_attention_2", flash_interface)
    model.config.text_config._attn_implementation = "flash_attention_2"
    actual = model(
        input_ids=nn.functional.pad(ids, (0, num_padding_tokens)),
        position_ids=nn.functional.pad(positions, (0, num_padding_tokens)),
        **metadata,
    )

    torch.testing.assert_close(
        actual.logits[:, :num_tokens], expected.logits, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        actual.mtp_logits[0][:, :num_tokens],
        expected.mtp_logits[0],
        atol=2e-6,
        rtol=2e-5,
    )
    assert len(calls) == model.config.text_config.num_hidden_layers + 1
    for call in calls:
        assert call["cu_seq_lens_q"] is cu
        assert call["cu_seq_lens_k"] is cu
        assert call["max_length_q"] == num_tokens
        assert call["max_length_k"] == num_tokens


@pytest.mark.parametrize("attn_implementation", ["sdpa", "flash_attention_2"])
def test_vision_attention_dispatches_through_hf_interfaces(
    monkeypatch, attn_implementation
):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = make_model()
    pixels = torch.randn(12, 24)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    expected = model.get_image_features(pixels, grids)

    calls = []

    def flash_interface(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs)
        return _segment_attention(module, query, key, value, attention_mask, **kwargs)

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "flash_attention_2", flash_interface)
    model.config.vision_config._attn_implementation = attn_implementation
    actual = model.get_image_features(pixels, grids)

    torch.testing.assert_close(torch.cat(actual), torch.cat(expected), atol=2e-6, rtol=2e-5)
    if attn_implementation == "flash_attention_2":
        assert len(calls) == len(model.model.visual.blocks)
        assert calls[0]["cu_seq_lens_q"].tolist() == [0, 4, 12]
        assert calls[0]["cu_seq_lens_k"] is calls[0]["cu_seq_lens_q"]
        assert int(calls[0]["max_length_q"]) == 8
        assert calls[0]["is_causal"] is False
    else:
        assert calls == []


@pytest.mark.parametrize("attn_implementation", ["eager", "flash_attention_2"])
def test_on_parallelized_switches_text_but_leaves_vision_alone(
    monkeypatch, attn_implementation
):
    import torchturbo.models.bailing_moe_v3_vl.model as wrapper
    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        Qwen3VLMoeVisionAttention,
    )

    model = make_model()
    patched = []
    monkeypatch.setattr(wrapper, "patch_rms_norm", lambda *args: None)
    monkeypatch.setattr(
        wrapper, "patch_kda_forward", lambda *args: patched.append(args)
    )
    registered = []

    def register_sp_attention_interface(sp_info):
        config = model.get_sp_attention_config()
        registered.append((config, sp_info["sp_size"]))
        config._attn_implementation = "turbo_ulysses"

    monkeypatch.setattr(
        model, "register_sp_attention_interface", register_sp_attention_interface
    )
    model.config.vision_config._attn_implementation = attn_implementation
    model.on_parallelized({"sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}})

    assert patched == [(wrapper.BailingMoeV3KimiDeltaAttention, None, 1, 0)]
    # Only the text config switches to the registered SP interface; the
    # vision tower keeps resolving its own subconfig implementation.
    assert registered == [(model.config.text_config, 1)]
    assert model.config.text_config._attn_implementation == "turbo_ulysses"
    vision_attention = model.model.visual.blocks[0].attn
    assert vision_attention.config._attn_implementation == attn_implementation
    assert Qwen3VLMoeVisionAttention.__dict__.get("attention_interface") is None
