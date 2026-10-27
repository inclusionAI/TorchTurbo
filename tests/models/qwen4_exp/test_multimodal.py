"""Packed multimodal training through the original HF vision dispatch."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from torchturbo.models.qwen4_exp.model import TurboQwen4ExpForConditionalGeneration

from .test_model import _small_config


def _cpu_flash(query, key, value, attention_mask, query_length, is_causal, **kwargs):
    outputs = []
    cu_q, cu_k = kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"]
    for index in range(cu_q.numel() - 1):
        q = query[:, cu_q[index] : cu_q[index + 1]].transpose(1, 2)
        k = key[:, cu_k[index] : cu_k[index + 1]].transpose(1, 2)
        v = value[:, cu_k[index] : cu_k[index + 1]].transpose(1, 2)
        repeats = q.shape[1] // k.shape[1]
        outputs.append(
            F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeats, dim=1),
                v.repeat_interleave(repeats, dim=1),
                is_causal=is_causal,
                scale=kwargs.get("softmax_scale"),
            ).transpose(1, 2)
        )
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("modality,frames", [("image", 1), ("video", 2)])
def test_packed_media_preserves_vision_metadata_and_gradients(
    monkeypatch, modality, frames
):
    import transformers.modeling_utils as hf_modeling

    # Keep HF dispatch real; CPU stand-ins replace unavailable flash kernels.
    monkeypatch.setitem(
        hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
        "general_availability_check",
        lambda: True,
    )
    monkeypatch.setattr(
        hf_modeling.PreTrainedModel, "_flash_attn_import_error", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hf_modeling, "lazy_import_flash_attention", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "transformers.integrations.flash_attention._flash_attention_forward", _cpu_flash
    )
    monkeypatch.setattr("torchturbo.models.qwen4_exp.qsa.attention_forward", _cpu_flash)
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4
    config.image_token_id, config.video_token_id = 20, 22
    config.vision_start_token_id, config.vision_end_token_id = 19, 21
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(29)
    packed_model = TurboQwen4ExpForConditionalGeneration(config).double().train()
    reference = copy.deepcopy(packed_model)
    token = config.image_token_id if modality == "image" else config.video_token_id
    first = torch.tensor([[3, 19, *([token] * frames), 21, 5]])
    second = torch.tensor([[6, 19, *([token] * frames), 21, 8, 9]])
    grids = torch.tensor([[frames, 2, 2], [frames, 2, 2]])
    pixels = torch.randn(frames * 8, 1536, dtype=torch.double, requires_grad=True)
    ref_pixels = pixels.detach().clone().requires_grad_()
    indices, weights = get_vision_bilinear_indices_and_weights(grids, 8, 2)
    weights = weights.double().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    vision_positions = get_vision_position_ids(grids, 2)

    def run(model, ids, lengths, media, grid, corners, coefficients, positions):
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        text_positions = torch.cat([torch.arange(length) for length in lengths])
        media_kwargs = {
            "pixel_values" if modality == "image" else "pixel_values_videos": media,
            f"{modality}_grid_thw": grid,
            f"{modality}_bilinear_indices": corners,
            f"{modality}_bilinear_weights": coefficients,
            f"{modality}_position_ids": positions,
            f"{modality}_cu_seqlens": get_vision_cu_seqlens(grid),
            f"{modality}_max_seqlen": 4,
        }
        return model(
            input_ids=ids,
            position_ids=text_positions.view(1, 1, -1).expand(3, -1, -1),
            use_cache=False,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
            **media_kwargs,
        ).logits

    actual = run(
        packed_model,
        torch.cat((first, second), dim=1),
        [first.shape[1], second.shape[1]],
        pixels,
        grids,
        indices,
        weights,
        vision_positions,
    )
    independent = []
    for index, ids in enumerate((first, second)):
        window = slice(index * frames * 4, (index + 1) * frames * 4)
        independent.append(
            run(
                reference,
                ids,
                [ids.shape[1]],
                ref_pixels[window],
                grids[index : index + 1],
                indices[:, window],
                ref_weights[:, window],
                vision_positions[window],
            )
        )
    expected = torch.cat(independent, dim=1)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(pixels.grad, ref_pixels.grad)
    assert weights.grad is not None and torch.count_nonzero(weights.grad) > 0
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in packed_model.named_parameters():
        if name.startswith("model.visual."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)
