from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch
import pytest

from torchturbo.adapters.llamafactory_v0.patches import (
    AcceleratorPatch,
    PackedInputCollatorPatch,
    RunExpPatch,
)
from torchturbo.models.qwen3_5_moe.lf_template import Qwen35DummyMediaPatch


def _tokenizer(pad_token_id: int = 99):
    return SimpleNamespace(
        pad_token_id=pad_token_id,
        padding_side="right",
        convert_tokens_to_ids=lambda _token: -1,
    )


@pytest.mark.parametrize(
    "stage,ignore_padding,options,error",
    [
        ("sft", False, {}, "padding"),
        ("sft", True, {}, None),
        ("pt", False, {}, None),
        ("dpo", False, {}, None),
        ("rm", False, {"do_eval": True}, "native"),
    ],
)
def test_run_exp_validates_padding_before_engine_setup(
    monkeypatch, stage, ignore_padding, options, error,
):
    from torchturbo.adapters.llamafactory_v0 import utils
    from torchturbo.engine import args as engine_args
    from torchturbo import logging as turbo_logging

    _install_fake_llamafactory(monkeypatch)
    training_args = SimpleNamespace(**options)
    data_args = SimpleNamespace(ignore_pad_token_for_loss=ignore_padding)
    finetuning_args = SimpleNamespace(stage=stage, finetuning_type="full")
    hparams = ModuleType("llamafactory.hparams")
    hparams.read_args = lambda args: args
    hparams.get_train_args = lambda args: (
        SimpleNamespace(), data_args, training_args, finetuning_args, SimpleNamespace(),
    )
    train = ModuleType("llamafactory.train")
    train.__path__ = []
    tuner = ModuleType("llamafactory.train.tuner")
    tuner.run_exp = lambda *args: "native"
    train.tuner = tuner
    launcher = ModuleType("llamafactory.launcher")
    launcher.run_exp = tuner.run_exp
    for name, module in (
        ("llamafactory.hparams", hparams),
        ("llamafactory.train", train),
        ("llamafactory.train.tuner", tuner),
        ("llamafactory.launcher", launcher),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        engine_args, "get_torchturbo_args", lambda: SimpleNamespace(log_level="INFO"),
    )
    monkeypatch.setattr(turbo_logging, "logging_configure", lambda level: None)
    accepted = []

    def stop_before_engine_setup(**kwargs):
        accepted.append(kwargs)
        raise RuntimeError("configuration accepted")

    monkeypatch.setattr(utils, "build_config_from_lf_args", stop_before_engine_setup)
    patch = RunExpPatch(SimpleNamespace())
    patch.patch()
    try:
        if error == "padding":
            with pytest.raises(ValueError, match="requires ignore_pad_token_for_loss=True"):
                tuner.run_exp({})
            assert not accepted
        elif error == "native":
            assert tuner.run_exp({}) == "native"
            assert not accepted
        else:
            with pytest.raises(RuntimeError, match="configuration accepted"):
                tuner.run_exp({})
            assert len(accepted) == 1
        assert data_args.ignore_pad_token_for_loss is ignore_padding
    finally:
        patch.revert()


def test_accelerator_patch_keeps_engine_optimizer_state_placement(monkeypatch):
    calls = []

    class Accelerator:
        def prepare(self, *args, device_placement=None):
            return args

        def prepare_model(self, model, device_placement=None, evaluation_mode=False):
            return model

        def prepare_optimizer(self, optimizer, device_placement=None):
            calls.append((optimizer, device_placement))
            return optimizer

        def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
            return None

        def backward(self, loss, **kwargs):
            return None

    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = Accelerator
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    raw_optimizer = object()
    engine = SimpleNamespace(
        optimizer=raw_optimizer,
        backend=SimpleNamespace(clip_grad_norm_=lambda *_args: None),
    )
    patch = AcceleratorPatch()
    patch.patch(engine)
    try:
        accelerator = Accelerator()
        assert accelerator.prepare_optimizer(raw_optimizer) is raw_optimizer
        assert calls == [(raw_optimizer, False)]
    finally:
        patch.revert()


def test_engine_patch_routes_lf095_packed_mask_through_transformers_varlen(
    monkeypatch,
):
    import transformers.integrations.flash_attention as flash_integration
    import transformers.modeling_flash_attention_utils as fa_utils

    from torchturbo.engine import TorchTurboEngine
    from torchturbo.utils.flash_attn_utils import get_unpad_data

    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])

    # Verify the engine installs packing support for sequence-ID masks.
    monkeypatch.setattr(
        flash_integration, "_flash_attention_forward", flash_integration._flash_attention_forward,
    )
    monkeypatch.setattr(fa_utils, "_get_unpad_data", lambda _mask: None)
    engine = object.__new__(TorchTurboEngine)
    engine._patch_transformers()
    assert fa_utils._get_unpad_data is get_unpad_data

    states = torch.arange(64, dtype=torch.float32).reshape(1, 8, 2, 4)

    def unexpected_unpad(*_args, **_kwargs):
        raise AssertionError("query/key lengths match; fallback unpad must not run")

    unpacked = fa_utils._upad_input(
        states,
        states,
        states,
        attention_mask,
        query_length=8,
        unpad_input_func=unexpected_unpad,
    )
    query, key, value, indices, (cu_q, cu_k), (max_q, max_k) = unpacked

    assert query.shape == key.shape == value.shape == (5, 2, 4)
    assert indices.tolist() == [0, 1, 2, 3, 4]
    assert cu_q.tolist() == cu_k.tolist() == [0, 2, 5]
    assert int(max_q) == int(max_k) == 3

def _install_fake_llamafactory(monkeypatch):
    collator_module = ModuleType("llamafactory.data.collator")

    class MultiModalDataCollatorForSeq2Seq:
        compute_dtype = torch.float32

        def __call__(self, features):
            self.rope_seen_by_lf = self.get_rope_func
            self.plugin_tokens_seen_by_lf = (
                self.template.mm_plugin.image_token,
                self.template.mm_plugin.audio_token,
            )
            for feature in features:
                feature.pop("packing_params", None)
            fake_input_ids = self.__dict__.get("fake_input_ids", [])
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] += fake_input_ids
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
            return {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self.parent_batch.items()
            }

    class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
        @staticmethod
        def _unpad_packed_features(result):
            mask = result["attention_mask"]
            valid = torch.nonzero(mask[0] != 0, as_tuple=False).flatten()
            seq_len = mask.size(-1)
            for key, value in list(result.items()):
                if not torch.is_tensor(value):
                    continue
                is_sequence_field = (
                    key == "position_ids" or value.dim() == 2
                ) and value.size(-1) == seq_len
                if is_sequence_field:
                    result[key] = value.index_select(-1, valid)

        def __call__(self, features):
            result = super().__call__(features)
            has_dummy_image = result.pop("has_dummy_image", False)
            if self.neat_packing and self.attn_implementation == "flash_attention_2":
                assert result["input_ids"].shape[0] == 1
                if not has_dummy_image:
                    self._unpad_packed_features(result)
                result["attention_mask"] = None
            return result

    collator_module.MultiModalDataCollatorForSeq2Seq = MultiModalDataCollatorForSeq2Seq
    collator_module.SFTDataCollatorWith4DAttentionMask = (
        SFTDataCollatorWith4DAttentionMask
    )
    collator_module.MROPE_MODELS = {"qwen3_5", "qwen3_5_moe"}

    lf_module = ModuleType("llamafactory")
    lf_module.__path__ = []
    data_module = ModuleType("llamafactory.data")
    data_module.__path__ = []
    data_module.collator = collator_module
    lf_module.data = data_module
    monkeypatch.setitem(sys.modules, "llamafactory", lf_module)
    monkeypatch.setitem(sys.modules, "llamafactory.data", data_module)
    monkeypatch.setitem(sys.modules, "llamafactory.data.collator", collator_module)
    return (
        collator_module,
        MultiModalDataCollatorForSeq2Seq,
        SFTDataCollatorWith4DAttentionMask,
    )


def test_lf095_path_preserves_native_mrope_and_compacts_packed_tokens(monkeypatch):
    module, parent_cls, sft_cls = _install_fake_llamafactory(monkeypatch)
    original_parent_call = parent_cls.__call__
    original_sft_call = sft_cls.__call__
    qwen_patch = Qwen35DummyMediaPatch()
    packed_patch = PackedInputCollatorPatch()
    qwen_patch.patch()
    packed_patch.patch()

    rope_func = object()
    plugin = SimpleNamespace(image_token="<image>", audio_token="<audio>")
    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(mm_plugin=plugin)
    collator.get_rope_func = rope_func
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    position_ids = torch.zeros(4, 1, 8, dtype=torch.long)
    position_ids[0, 0, :5] = torch.tensor([0, 1, 0, 1, 2])
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": position_ids,
        "token_type_ids": torch.ones(1, 8, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{
        "input_ids": list(range(8)), "images": None, "videos": None, "audios": None,
        "attention_mask": [1, 1, 2, 2, 2, 0, 0, 0],
        "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
    }])

    assert collator.rope_seen_by_lf is rope_func
    assert collator.plugin_tokens_seen_by_lf == (None, None)
    assert plugin.image_token == "<image>"
    assert plugin.audio_token == "<audio>"
    assert module.MROPE_MODELS == {"qwen3_5", "qwen3_5_moe"}
    assert result["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    torch.testing.assert_close(result["position_ids"], position_ids[:, :, :5])
    assert result["labels"].tolist() == [[-100, 1, -100, 3, 4]]
    assert "token_type_ids" not in result
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == result["cu_seq_lens_k"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == result["max_length_k"] == 3
    assert collator.neat_packing is True

    packed_patch.revert()
    qwen_patch.revert()
    assert parent_cls.__call__ is original_parent_call
    assert sft_cls.__call__ is original_sft_call


def test_qwen_dummy_patch_leaves_bailing_media_tokens_enabled(monkeypatch):
    _, parent_cls, _ = _install_fake_llamafactory(monkeypatch)
    original_call = parent_cls.__call__
    patch = Qwen35DummyMediaPatch()
    patch.patch()

    plugin = SimpleNamespace(image_token="<image>", audio_token=None)
    collator = parent_cls()
    collator.model = SimpleNamespace(
        config=SimpleNamespace(model_type="bailing_moe_v3_vl")
    )
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(mm_plugin=plugin)
    collator.get_rope_func = None
    collator.parent_batch = {"input_ids": torch.tensor([[1, 2, 3]])}

    try:
        collator(
            [
                {
                    "input_ids": [1, 2, 3],
                    "images": None,
                    "videos": None,
                    "audios": None,
                }
            ]
        )
        assert collator.plugin_tokens_seen_by_lf == ("<image>", None)
        assert plugin.image_token == "<image>"
    finally:
        patch.revert()

    assert parent_cls.__call__ is original_call


def test_lf095_kimi_packed_fa2_uses_cu_metadata(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedInputCollatorPatch()
    patch.patch()

    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="kimi_k25"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token=None, audio_token=None)
    )
    collator.get_rope_func = None
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{
        "input_ids": list(range(8)), "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
        "attention_mask": [1, 1, 2, 2, 2, 0, 0, 0],
    }])

    assert result["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert result["labels"].tolist() == [[-100, 1, -100, 3, 4]]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert collator.neat_packing is True
    patch.revert()


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "turbo_ulysses"])
def test_lf095_packed_fa_kernels_compact_other_models(monkeypatch, attn_implementation):
    """LF 0.9.5 reads the implementation name off model.config after load,
    so SP registration surfaces as "turbo_ulysses" here; the packed
    normalization must still run for it."""
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedInputCollatorPatch()
    patch.patch()

    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="llada2_moe"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token=None, audio_token=None)
    )
    collator.get_rope_func = None
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = attn_implementation
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{
        "input_ids": list(range(8)), "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
        "attention_mask": [1, 1, 2, 2, 2, 0, 0, 0],
    }])

    assert result["input_ids"].shape == (1, 5)
    assert result["position_ids"].shape == (1, 5)
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert collator.neat_packing is True
    patch.revert()


def _packed_collator(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedInputCollatorPatch()
    patch.patch()
    collator = sft_cls()
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token="<image>", audio_token="<audio>")
    )
    collator.get_rope_func = None
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    return collator, patch


def test_lf095_unpacked_train_batch_flattens_rows_with_cu(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.compute_dtype = torch.bfloat16
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 0, 0], [20, 21, 22, 0]]),
        "labels": torch.tensor([[10, 11, -100, -100], [20, 21, 22, -100]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
    }
    try:
        result = collator([
            {"input_ids": [10, 11], "labels": [10, 11]},
            {"input_ids": [20, 21, 22], "labels": [20, 21, 22]},
        ])
    finally:
        patch.revert()

    # Rows flatten in order; every batch row is one cu segment.
    assert result["input_ids"].tolist() == [[10, 11, 20, 21, 22]]
    assert result["labels"].tolist() == [[-100, 11, -100, 21, 22]]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["cu_seq_lens_k"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == result["max_length_k"] == 3
    assert "attention_mask" not in result


@pytest.mark.parametrize("layout", ["text", "mrope"])
@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "turbo_ulysses"])
def test_lf095_ordinary_packing_isolates_samples_at_position_resets(
    monkeypatch, attn_implementation, layout,
):
    from transformers.modeling_flash_attention_utils import (
        prepare_fa_kwargs_from_position_ids,
    )

    collator, patch = _packed_collator(monkeypatch)
    collator.neat_packing = False
    collator.attn_implementation = attn_implementation
    # LF pads ordinary packed rows with mask-one tokens at position zero.
    text_positions = torch.tensor([[0, 1, 0, 1, 0, 0], [0, 1, 2, 0, 1, 0]])
    position_ids = text_positions
    if layout == "mrope":
        # LF recomputes mRoPE across each packed row, overwriting text resets.
        position_ids = torch.arange(6).expand(4, 2, 6)
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 99, 99], [20, 21, 22, 23, 24, 99]]),
        "labels": torch.tensor([[-100, 11, -100, 13, -100, -100], [-100, 21, 22, -100, 24, -100]]),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "position_ids": position_ids,
    }
    features = [
        {
            **{key: value[row].tolist() for key, value in collator.parent_batch.items() if value.dim() == 2},
            "position_ids": text_positions[row].tolist(),
        }
        for row in range(2)
    ]
    try:
        result = collator(features)
    finally:
        patch.revert()

    # The binary mask carries no sample boundaries; restarted text positions
    # do, exactly as HF's position-based varlen detection reads them.
    (native_cu, _), (native_max, _) = prepare_fa_kwargs_from_position_ids(
        text_positions.reshape(1, -1)
    )
    assert result["cu_seq_lens_q"].tolist() == native_cu.tolist() == [0, 2, 4, 5, 6, 9, 11, 12]
    assert result["cu_seq_lens_k"].tolist() == native_cu.tolist()
    assert result["max_length_q"] == result["max_length_k"] == int(native_max) == 3
    for key in ("input_ids", "labels"):
        torch.testing.assert_close(result[key], collator.parent_batch[key].reshape(1, -1))
    expected_positions = position_ids.flatten(-2).unsqueeze(-2)
    torch.testing.assert_close(result["position_ids"], expected_positions)
    assert "attention_mask" not in result


@pytest.mark.parametrize("neat_packing", [False, True])
@pytest.mark.parametrize("num_predictions", [1, 3])
def test_packed_supervision_matches_trainer_count_and_backend_loss(
    monkeypatch, neat_packing, num_predictions,
):
    import importlib

    from transformers import Trainer

    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime

    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    collator, patch = _packed_collator(monkeypatch)
    collator.neat_packing = neat_packing
    input_ids = torch.arange(1, 9).unsqueeze(0)
    collator.parent_batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])
        if neat_packing else torch.ones_like(input_ids),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
    }
    try:
        batch = collator([{
            key: value[0].tolist() for key, value in collator.parent_batch.items()
        }])
    finally:
        patch.revert()

    trainer = object.__new__(Trainer)
    trainer.model_accepts_loss_kwargs = True
    trainer.compute_loss_func = None
    trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)
    trainer.accelerator = SimpleNamespace(parallelism_config=None)
    num_items = trainer._get_num_items_in_batch([batch], torch.device("cpu"))

    boundaries = [(0, 4), (4, 8)]
    main_targets = torch.cat([input_ids[0, start + 1:end] for start, end in boundaries])
    expected_num_items = main_targets.ne(-100).sum()
    torch.testing.assert_close(num_items, expected_num_items)

    generator = torch.Generator().manual_seed(1)
    logits = [
        torch.randn(1, 8, 16, generator=generator, requires_grad=True)
        for _ in range(num_predictions)
    ]
    reference_logits = [value.detach().clone().requires_grad_() for value in logits]
    reference_losses = []
    for shift, prediction in enumerate(reference_logits, 1):
        targets = torch.cat([input_ids[0, start + shift:end] for start, end in boundaries])
        valid_logits = torch.cat([prediction[0, start:end - shift] for start, end in boundaries])
        reference_losses.append(
            torch.nn.functional.cross_entropy(valid_logits, targets, reduction="sum")
            / expected_num_items
        )
    reference_total = reference_losses[0] + sum(reference_losses[1:]) * 0.2

    backend = object.__new__(FSDP2Backend)
    backend.config = SimpleNamespace(mtp=SimpleNamespace(enabled=num_predictions > 1))
    loss, components = backend._compute_lm_loss_with_sp(
        model=None,
        outputs=SimpleNamespace(logits=logits[0], mtp_logits=logits[1:]),
        labels=batch["labels"],
        chunk_hidden_states=None,
        sp_size=1,
        sp_group=None,
        sp_rank=0,
        num_items_in_batch=num_items,
        mtp_loss_scale=0.2,
        cu_seq_lens=batch["cu_seq_lens_q"],
    )
    torch.testing.assert_close(loss, reference_total)
    for actual, expected in zip(components, reference_losses):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        torch.autograd.grad(loss, logits),
        torch.autograd.grad(reference_total, reference_logits),
    ):
        torch.testing.assert_close(actual, expected)


def test_unpacked_dummy_tokens_still_match_qwen_image_features(monkeypatch):
    from torchturbo.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    collator, patch = _packed_collator(monkeypatch)
    collator.fake_input_ids = [70, 70]
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 70, 70]]),
        "labels": torch.tensor([[10, 11, -100, -100]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "pixel_values": torch.ones(2, 4),
    }
    try:
        result = collator([{"input_ids": [10, 11], "labels": [10, 11]}])
    finally:
        patch.revert()

    model = SimpleNamespace(config=SimpleNamespace(image_token_id=70, video_token_id=71))
    image_mask, _ = Qwen3_5Model.get_placeholder_mask(
        model,
        result["input_ids"],
        torch.ones(1, 4, 4),
        image_features=result["pixel_values"],
    )
    assert image_mask[..., 0].tolist() == [[False, False, True, True]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2]


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_multimodal_pretraining_retains_dummy_media(monkeypatch, padding_side):
    from torchturbo.adapters.llamafactory_v0.trainer import LlamaFactoryInputCollator

    _, parent_cls, _ = _install_fake_llamafactory(monkeypatch)
    collator = parent_cls()
    collator.tokenizer = _tokenizer()
    collator.tokenizer.padding_side = padding_side
    collator.get_rope_func = None
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token="<image>", audio_token=None)
    )
    collator.fake_input_ids = [70, 70]
    if padding_side == "right":
        input_ids = [[10, 11, 70, 70], [20, 21, 22, 99]]
        labels = [[10, 11, -100, -100], [20, 21, 22, -100]]
        mask = [[1, 1, 0, 0], [1, 1, 1, 0]]
    else:
        input_ids = [[70, 70, 10, 11], [99, 20, 21, 22]]
        labels = [[-100, -100, 10, 11], [-100, 20, 21, 22]]
        mask = [[0, 0, 1, 1], [0, 1, 1, 1]]
    collator.parent_batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
        "pixel_values": torch.ones(2, 4),
    }

    result = LlamaFactoryInputCollator(collator)([
        {"input_ids": [10, 11], "labels": [10, 11]},
        {"input_ids": [20, 21, 22], "labels": [20, 21, 22]},
    ])

    assert result["input_ids"].tolist() == [[10, 11, 20, 21, 22, 70, 70]]
    assert result["labels"].tolist() == [[-100, 11, -100, 21, 22, -100, -100]]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    torch.testing.assert_close(result["pixel_values"], collator.parent_batch["pixel_values"])


def test_lf095_packed_fa2_flattens_rows_without_cross_row_loss(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.compute_dtype = torch.bfloat16
    positions = torch.arange(40).reshape(4, 2, 5)
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 12, 99, 99], [20, 21, 22, 23, 99]]),
        "labels": torch.tensor([[10, 11, 12, -100, -100], [20, 21, 22, 23, -100]]),
        "attention_mask": torch.tensor([[1, 1, 2, 0, 0], [1, 2, 2, 2, 0]]),
        "position_ids": positions,
        "token_type_ids": torch.arange(10).reshape(2, 5),
        "mm_token_type_ids": torch.arange(10).reshape(2, 5),
        "cross_attention_mask": torch.arange(10).reshape(2, 5, 1, 1),
        "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
    }
    try:
        result = collator([
            {
                "input_ids": [10, 11, 12, 99, 99], "attention_mask": [1, 1, 2, 0, 0],
                "packing_params": {"sequence_boundaries": [0, 2, 3, 5]},
            },
            {
                "input_ids": [20, 21, 22, 23, 99], "attention_mask": [1, 2, 2, 2, 0],
                "packing_params": {"sequence_boundaries": [0, 1, 4, 5]},
            },
        ])
    finally:
        patch.revert()

    assert result["input_ids"].tolist() == [[10, 11, 12, 20, 21, 22, 23]]
    # Every attention segment starts without a supervised preceding token.
    assert result["labels"].tolist() == [[-100, 11, -100, -100, -100, 22, 23]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 3, 4, 7]
    assert result["max_length_q"] == result["max_length_k"] == 3
    expected_positions = torch.cat([positions[:, :1, :3], positions[:, 1:, :4]], dim=-1)
    torch.testing.assert_close(result["position_ids"], expected_positions)
    assert "token_type_ids" not in result
    assert result["mm_token_type_ids"].tolist() == [[0, 1, 2, 5, 6, 7, 8]]
    assert result["cross_attention_mask"].flatten().tolist() == [0, 1, 2, 5, 6, 7, 8]
    assert result["pixel_values"].dtype == torch.bfloat16
    assert result["pixel_values"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert result["image_grid_thw"].tolist() == [[1, 2, 3], [1, 4, 5]]
    assert "attention_mask" not in result

    logits = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(0))
    original_loss = torch.nn.functional.cross_entropy(
        torch.cat([logits[0, :1], logits[1, 1:3]]),
        torch.tensor([11, 22, 23]),
        reduction="sum",
    )
    packed_logits = logits.flatten(0, 1)[[0, 1, 2, 5, 6, 7, 8]]
    packed_loss = torch.nn.functional.cross_entropy(
        packed_logits[:-1], result["labels"][0, 1:], reduction="sum",
    )
    torch.testing.assert_close(packed_loss, original_loss)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
def test_lf095_dummy_media_stays_at_masked_tail(monkeypatch, mrope, packed):
    collator, patch = _packed_collator(monkeypatch)
    collator.fake_input_ids = [70, 71]
    attention_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
    positions = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 0]])
    if mrope:
        positions = torch.arange(64).reshape(4, 2, 8)
        # LF appends an extra zero mask tail in its dummy-image mRoPE branch.
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 2))
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 99, 99, 70, 71, 99, 99], [20, 21, 22, 99, 99, 99, 99, 99]]),
        "labels": torch.tensor([[10, 11, -100, -100, -100, -100, -100, -100], [20, 21, 22, -100, -100, -100, -100, -100]]),
        "attention_mask": attention_mask,
        "position_ids": positions,
        "has_dummy_image": mrope,
        "pixel_values": torch.tensor([[7.0, 8.0]]),
    }
    features = [
        {
            "input_ids": [10, 11, 99, 99], "labels": [10, 11, -100, -100],
            "attention_mask": [1, 1, 0, 0],
        },
        {
            "input_ids": [20, 21, 22, 99], "labels": [20, 21, 22, -100],
            "attention_mask": [1, 1, 1, 0],
        },
    ]
    if packed:
        features[0]["packing_params"] = {"sequence_boundaries": [0, 2, 4]}
        features[1]["packing_params"] = {"sequence_boundaries": [0, 3, 4]}
    try:
        result = collator(features)
    finally:
        patch.revert()

    assert result["input_ids"].tolist() == [[10, 11, 20, 21, 22, 70, 71]]
    assert result["labels"].tolist() == [[-100, 11, -100, 21, 22, -100, -100]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == 3
    if mrope:
        expected = torch.cat([positions[:, :1, :2], positions[:, 1:, :3], positions[:, :1, 4:6]], dim=-1)
        torch.testing.assert_close(result["position_ids"], expected)
    else:
        assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0]]
    assert result["pixel_values"].tolist() == [[7.0, 8.0]]
    assert "has_dummy_image" not in result
    assert "attention_mask" not in result


def test_lf095_mismatched_text_positions_rejected(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 12, 99]]),
        "labels": torch.tensor([[10, 11, 12, -100]]),
        "attention_mask": torch.tensor([[1, 1, 2, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 0]]),
    }
    try:
        with pytest.raises(ValueError, match="position resets"):
            collator([{
                "input_ids": [10, 11, 12, 99], "packing_params": {"sequence_boundaries": [0, 2, 3, 4]},
                "attention_mask": [1, 1, 2, 0],
            }])
    finally:
        patch.revert()


def test_adapter_discovers_and_registers_model_lf_patches(monkeypatch):
    from torchturbo.adapters.llamafactory_v0 import LlamaFactoryV0Adapter
    from torchturbo.models import registry
    from torchturbo.registry.adapter import BaseAdapter

    monkeypatch.setattr(registry, "get_all_models", lambda: {"qwen3_5_moe": object})

    adapter = object.__new__(LlamaFactoryV0Adapter)
    BaseAdapter.__init__(adapter)

    adapter._register_lf_templates()
    assert [p.name for p in adapter._patches] == ["qwen3_5_dummy_media"]
