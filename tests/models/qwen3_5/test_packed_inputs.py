import importlib
from types import SimpleNamespace

import pytest
import torch

import torchturbo.ops as turbo_ops
from torchturbo.models.modules.attention.gdn import build_gdn_sp_forward
from torchturbo.models.modules.attention.ulysses import ulysses_pad_inputs
from torchturbo.models.qwen3_5 import modeling_qwen3_5 as dense
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.swift_template import _post_process_inputs
from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.ops.dispatcher import OpDispatcher
from torchturbo.runtime.cpu import CpuRuntime


@pytest.mark.parametrize(
    "model_cls",
    [TurboQwen3_5ForConditionalGeneration, TurboQwen3_5MoeForConditionalGeneration],
)
def test_sp_padding_keeps_swift_text_positions_aligned(model_cls):
    model = SimpleNamespace(model=SimpleNamespace(language_model=torch.nn.Module()))
    spec = model_cls.get_sequence_parallel_input_slicing_spec(model)
    batch = ulysses_pad_inputs(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
            "position_ids": torch.tensor([[[0, 1, 0, 1, 2]]] * 3),
            "text_position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
            "mm_token_type_ids": torch.zeros(1, 5, dtype=torch.long),
        },
        spec,
        sp_size=2,
    )

    # Swift's pre-forward hook merges both position layouts at model call time.
    position_ids = _post_process_inputs(batch)["position_ids"]

    assert position_ids.shape == (4, 1, 6)
    assert position_ids[0].tolist() == [[0, 1, 0, 1, 2, 0]]


@pytest.mark.parametrize("text_model", [dense.Qwen3_5TextModel, moe.Qwen3_5MoeTextModel])
def test_text_model_preserves_generation_padding_mask(text_model):
    captured = []

    class Decoder:
        layer_type = "full_attention"

        def __call__(self, hidden_states, **kwargs):
            captured.append(kwargs["attention_mask"])
            return hidden_states

    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=False, num_hidden_layers=1),
        gradient_checkpointing=False,
        training=False,
        layers=[Decoder()],
        rotary_emb=lambda *args: None,
        norm=lambda x: x,
    )
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    text_model.forward(model, inputs_embeds=torch.ones(2, 4, 4), attention_mask=mask)

    assert captured == [mask]


def _attention():
    return SimpleNamespace(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=2,
        head_v_dim=2,
        key_dim=4,
        value_dim=4,
        conv_kernel_size=2,
        conv1d=SimpleNamespace(weight=torch.ones(12, 1, 2), bias=None),
        activation="silu",
        in_proj_qkv=lambda x: x.repeat(1, 1, 3),
        in_proj_z=lambda x: x,
        in_proj_b=lambda x: x[..., :2],
        in_proj_a=lambda x: x[..., :2],
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        norm=lambda x, z: x,
        out_proj=lambda x: x,
    )


@pytest.mark.parametrize("sp_size", [1, 2])
def test_gated_delta_ops_cover_padding_and_share_metadata(monkeypatch, sp_size):
    captured = []

    def dispatch(op_name, *args, **kwargs):
        packed = kwargs["packed_params"]
        assert kwargs.get("cu_seqlens") is None
        assert packed.cu_seqlens_list[-1] == args[0].size(1)
        captured.append(packed)
        if op_name == "causal_conv1d":
            return args[0], None
        assert op_name == "gated_delta_rule"
        return args[2], None

    # Keep public op normalization real; replace only the unavailable kernels.
    dispatcher = OpDispatcher.__new__(OpDispatcher)
    monkeypatch.setattr(dispatcher, "_dispatch", dispatch)
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)
    attention = _attention()
    if sp_size > 1:
        ulysses = importlib.import_module("torchturbo.models.modules.attention.ulysses.ulysses")

        def gather_sequence(x, seq_dim, head_dim, **kwargs):
            return torch.cat([x, x], dim=seq_dim).narrow(head_dim, 0, x.size(head_dim) // 2)

        def gather_heads(x, seq_dim, head_dim, **kwargs):
            return torch.cat([x, x], dim=head_dim).narrow(seq_dim, 0, x.size(seq_dim) // 2)

        monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", gather_sequence)
        monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", gather_heads)
        gdn_forward = importlib.import_module("torchturbo.models.modules.attention.gdn.forward")
        monkeypatch.setattr(gdn_forward, "get_device_runtime", CpuRuntime)

    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    forward = build_gdn_sp_forward(None, sp_size, 0)
    output = forward(attention, torch.ones(1, 6 // sp_size, 4), cu_seq_lens=cu)

    assert output.shape == (1, 6 // sp_size, 4)
    assert captured[0] is captured[1]
    assert captured[0].cu_seqlens_list == [0, 2, 5, 6]
    assert cu.tolist() == [0, 2, 5]


def _rope_index_model(
    model_cls=TurboQwen3_5MoeForConditionalGeneration, spatial_merge_size=1
):
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
        image_token_id=99,
        video_token_id=98,
        vision_start_token_id=97,
    )
    model = SimpleNamespace(config=config)
    if model_cls is TurboQwen3_5ForConditionalGeneration:
        model.get_rope_index = lambda *args, **kwargs: dense.Qwen3_5Model.get_rope_index(model, *args, **kwargs)
        return model
    model.get_vision_position_ids = lambda *args, **kwargs: moe.Qwen3_5MoeModel.get_vision_position_ids(model, *args, **kwargs)
    model.get_rope_index = lambda *args, **kwargs: moe.Qwen3_5MoeModel.get_rope_index(model, *args, **kwargs)
    TurboQwen3_5MoeForConditionalGeneration._post_init_patches(SimpleNamespace(model=model, config=config))
    return model


def _input_hook_model(model_cls):
    # Exercise the actual input hook and RoPE code without model weights.
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.model = _rope_index_model(model_cls, spatial_merge_size=2)
    model.config = model.model.config
    return model


@pytest.mark.parametrize(
    "model_cls",
    [TurboQwen3_5ForConditionalGeneration, TurboQwen3_5MoeForConditionalGeneration],
)
@pytest.mark.parametrize("tail_padding", [0, 2])
@pytest.mark.parametrize("via_engine", [False, True])
def test_input_hook_matches_independent_multimodal_samples(
    model_cls, tail_padding, via_engine
):
    model = _input_hook_model(model_cls)
    # Text-only samples, multiple images, and interleaved image/video entries
    # make sample indices differ from both image and video grid indices.
    samples = [
        ([10, 11], None, None),
        ([10, 97, 99, 99, 96, 11, 97, 99, 99, 99, 99, 96, 12],
         [[1, 2, 4], [1, 4, 4]], None),
        ([12, 13, 14], None, None),
        ([10, 97, 98, 98, 96, 11, 97, 99, 99, 96, 12, 97, 98, 98, 98, 98, 96, 13],
         [[1, 4, 2]], [[1, 2, 4], [1, 4, 4]]),
        ([10, 97, 99, 99, 96, 11, 97, 98, 98, 96, 12],
         [[1, 2, 4]], [[1, 4, 2]]),
    ]
    _assert_prepared_positions_match_samples(model, samples, tail_padding, via_engine)


def test_input_hook_keeps_multiframe_video_grids_with_their_sample():
    model = _input_hook_model(TurboQwen3_5ForConditionalGeneration)
    samples = [
        ([10, 11], None, None),
        ([10, 97, 98, 98, 96, 11, 97, 98, 98, 96, 12],
         None, [[2, 2, 4]]),
        ([13, 97, 98, 98, 98, 98, 96, 14], None, [[1, 4, 4]]),
    ]
    _assert_prepared_positions_match_samples(model, samples, tail_padding=0)


def _assert_prepared_positions_match_samples(
    model, samples, tail_padding, via_engine=False
):
    input_ids, text_positions, expected_positions = [], [], []
    media_grids = {"image_grid_thw": [], "video_grid_thw": []}
    boundaries = [0]
    for ids, images, videos in samples:
        ids = torch.tensor([ids])
        sample_grids = {}
        for key, grids in zip(media_grids, (images, videos)):
            if grids is not None:
                sample_grids[key] = torch.tensor(grids)
                media_grids[key].append(sample_grids[key])
        positions, _ = model.model.get_rope_index(input_ids=ids, **sample_grids)
        input_ids.append(ids)
        text_positions.append(torch.arange(ids.size(-1)).unsqueeze(0))
        expected_positions.append(positions)
        boundaries.append(boundaries[-1] + ids.size(-1))

    text_positions = torch.nn.functional.pad(
        torch.cat(text_positions, dim=-1), (0, tail_padding)
    )
    cu = torch.tensor(boundaries, dtype=torch.int32)
    inputs = {
        "input_ids": torch.nn.functional.pad(
            torch.cat(input_ids, dim=-1), (0, tail_padding)
        ),
        "position_ids": text_positions,
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": max(len(ids) for ids, _, _ in samples),
        "max_length_k": max(len(ids) for ids, _, _ in samples),
        **{key: torch.cat(grids) for key, grids in media_grids.items() if grids},
    }
    original_grids = {key: inputs[key].clone() for key in media_grids if key in inputs}

    if via_engine:
        from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
        from torchturbo.engine.core import TorchTurboEngine

        engine = TorchTurboEngine.__new__(TorchTurboEngine)
        engine.model = model
        engine.backend = FSDP2Backend.__new__(FSDP2Backend)
        engine.backend.get_sp_info = lambda: {"sp_size": 1}
        engine.get_sp_info = lambda: {"sp_size": 1}
        prepared = engine.prepare_packed_micro_batch(
            {
                "input_ids": inputs["input_ids"].squeeze(0),
                "cu_seqlens": cu,
                **{key: inputs[key] for key in original_grids},
            },
            cu.diff().tolist(),
        )
    else:
        prepared = model.on_prepare_inputs(inputs)

    expected = torch.cat(
        (text_positions.unsqueeze(0), torch.nn.functional.pad(
            torch.cat(expected_positions, dim=-1), (0, tail_padding)
        )), dim=0,
    )
    torch.testing.assert_close(prepared["position_ids"], expected)
    for key, grids in original_grids.items():
        torch.testing.assert_close(prepared[key], grids)


@pytest.mark.parametrize(
    "model_cls",
    [TurboQwen3_5ForConditionalGeneration, TurboQwen3_5MoeForConditionalGeneration],
)
@pytest.mark.parametrize("position_rows", [1, 3, 4])
def test_input_hook_preserves_text_and_precomputed_positions(model_cls, position_rows):
    model = _input_hook_model(model_cls)
    positions = torch.tensor([[0, 1, 0, 1, 2]])
    if position_rows > 1:
        positions = positions.unsqueeze(0).expand(position_rows, -1, -1)
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "position_ids": positions,
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
    }

    assert model.on_prepare_inputs(inputs)["position_ids"] is positions


def test_moe_rope_index_restarts_positions_per_mask_block():
    position_ids, _ = _rope_index_model().get_rope_index(
        input_ids=torch.tensor([[10, 99, 99, 11, 12, 99, 99, 13, 0]]),
        image_grid_thw=torch.tensor([[1, 1, 2], [1, 1, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 0]]),
    )

    expected = torch.tensor([
        [[0, 1, 1, 3, 0, 1, 1, 3, 0]],
        [[0, 1, 1, 3, 0, 1, 1, 3, 0]],
        [[0, 1, 2, 3, 0, 1, 2, 3, 0]],
    ])
    torch.testing.assert_close(position_ids, expected)


def test_moe_rope_index_aligns_truncated_images_from_collator_masks():
    # The LF collator passes a binary float mask; truncation can keep only a
    # prefix of an image's placeholder tokens.
    with pytest.warns(UserWarning, match="placeholder_len=2 != grid_len=4"):
        position_ids, _ = _rope_index_model().get_rope_index(
            input_ids=torch.tensor([[10, 11, 99, 99]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            attention_mask=torch.ones(1, 4),
        )

    expected = torch.tensor([[[0, 1, 2, 2]], [[0, 1, 2, 2]], [[0, 1, 2, 3]]])
    torch.testing.assert_close(position_ids, expected)


def test_moe_mtp_attention_receives_prefix_boundaries():
    captured = {}

    def attention(hidden_states, **kwargs):
        captured.update(kwargs)
        return hidden_states, None

    layer = SimpleNamespace(
        input_layernorm=lambda x: x,
        self_attn=attention,
        post_attention_layernorm=lambda x: x,
        mlp=lambda x: x,
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    moe.Qwen3_5MoeMTPLayer.forward(
        layer,
        torch.ones(1, 6, 4),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=3,
        max_length_k=3,
    )

    assert captured["cu_seq_lens_q"] is cu
    assert captured["cu_seq_lens_k"] is cu
    assert captured["max_length_q"] == captured["max_length_k"] == 3
