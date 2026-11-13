"""Regression checks using an installed LlamaFactory and local image inputs."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("packing", ["unpacked", "ordinary", "ordinary_batched"])
def test_native_image_training_keeps_mrope(tmp_path, padding_side, packing):
    pytest.importorskip("llamafactory", minversion="0.9.5")

    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.mm_plugin import Qwen3VLPlugin
    from llamafactory.hparams import FinetuningArguments, TrainingArguments
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        GenerationConfig,
        PretrainedConfig,
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from torchturbo.adapters.llamafactory_v0.patches import PackedInputCollatorPatch
    from torchturbo.adapters.llamafactory_v0.trainer import make_torchturbo_trainer
    from torchturbo.models.base import TurboModel
    from torchturbo.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    class Model(TurboModel, torch.nn.Module):
        main_input_name = "input_ids"
        is_diffusion_model = False
        get_rope_index = Qwen3_5Model.get_rope_index

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = PretrainedConfig(
                image_token_id=70,
                video_token_id=71,
                vision_start_token_id=90,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )
            self.config.model_type = "qwen3_5"
            self.generation_config = GenerationConfig(pad_token_id=0, eos_token_id=2)
            self.forward_positions = None
            self.forward_cu = None

        def forward(self, input_ids, position_ids, labels=None, **kwargs):
            self.forward_positions = position_ids.detach().clone()
            self.forward_cu = kwargs["cu_seq_lens_q"]
            logits = self.weight * torch.zeros(*input_ids.shape, 128)
            return CausalLMOutputWithPast(loss=self.weight * 0, logits=logits)

    class Engine:
        def compute_loss(self, model, inputs, **kwargs):
            outputs = model(**inputs)
            return outputs.loss, outputs

        def get_sp_info(self):
            return {"sp_size": 1, "sp_group": None, "sp_rank": 0}

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        padding_side=padding_side,
    )
    image_processor = Qwen2VLImageProcessor(
        patch_size=14, temporal_patch_size=2, merge_size=2,
        min_pixels=56 * 56, max_pixels=56 * 56,
    )
    processor = SimpleNamespace(image_processor=image_processor)
    template = SimpleNamespace(
        mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token=None, audio_token=None)
    )
    image = Image.new("RGB", (56, 56), color=(30, 60, 90))
    input_ids = [10, 90, 70, 70, 70, 70, 91, 11]
    features = [
        {"input_ids": input_ids, "attention_mask": [1] * 8, "labels": input_ids, "images": [image]},
        {"input_ids": [20, 21, 22], "attention_mask": [1] * 3, "labels": [20, 21, 22]},
    ]
    if packing != "unpacked":
        packed_ids = input_ids + [20, 21, 22]
        features = [{
            "input_ids": packed_ids,
            "attention_mask": [1] * 11,
            "labels": packed_ids,
            "position_ids": list(range(8)) + list(range(3)),
            "images": [image],
        }]
        if packing == "ordinary_batched":
            features.append(deepcopy(features[0]))
    model = Model()
    collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        template=template,
        processor=processor,
        model=model,
        pad_to_multiple_of=8,
        attn_implementation="flash_attention_2",
    )
    native_batch = collator(deepcopy(features))
    valid_indices = torch.nonzero(native_batch["attention_mask"].flatten()).flatten()
    native_positions = (
        native_batch["position_ids"].flatten(1).index_select(1, valid_indices).unsqueeze(1)
    )
    if packing != "unpacked":
        # Native LF recomputes RoPE for the whole packed row, erasing the
        # second sample's position reset from its original text coordinates.
        assert native_positions[0, 0, 8].item() == 6
    trainer_cls = make_torchturbo_trainer(CustomSeq2SeqTrainer, engine=Engine(), stage="sft")
    patch = PackedInputCollatorPatch()
    patch.patch()
    try:
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            finetuning_args=FinetuningArguments(finetuning_type="full"),
            args=TrainingArguments(
                output_dir=str(tmp_path),
                do_train=True,
                use_cpu=True,
                report_to="none",
                remove_unused_columns=False,
                disable_tqdm=True,
            ),
            data_collator=collator,
            train_dataset=features,
        )
        training_collator = trainer.data_collator
        assert training_collator.model is model
        assert training_collator is collator
        assert collator.get_rope_func is not None

        training_batch = training_collator(deepcopy(features))
        num_packed_rows = 2 if packing == "ordinary_batched" else 1
        assert training_batch["image_grid_thw"].tolist() == [[1, 4, 4]] * num_packed_rows
        expected_positions = torch.tensor([
            [0, 1, 2, 2, 2, 2, 4, 5, 0, 1, 2],
            [0, 1, 2, 2, 3, 3, 4, 5, 0, 1, 2],
            [0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 2],
        ]).unsqueeze(1)
        if packing != "unpacked":
            expected_positions[..., 8:] += 6
        expected_positions = expected_positions.repeat(1, 1, num_packed_rows)
        torch.testing.assert_close(training_batch["position_ids"][1:], expected_positions)
        torch.testing.assert_close(training_batch["position_ids"], native_positions)
        expected_cu = [0, 8, 11, 19, 22] if packing == "ordinary_batched" else [0, 8, 11]
        assert training_batch["cu_seq_lens_q"].tolist() == expected_cu
        assert training_batch["labels"][0, expected_cu[:-1]].eq(-100).all()
        trainer.current_gradient_accumulation_steps = 1
        trainer.compute_loss(model, training_batch)
        torch.testing.assert_close(model.forward_positions[1:], expected_positions)
        assert model.forward_cu.tolist() == expected_cu
    finally:
        patch.revert()
