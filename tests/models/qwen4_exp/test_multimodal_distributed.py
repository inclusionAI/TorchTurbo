"""Full packed Qwen4 media training with SP and decoder checkpointing."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from torchturbo.models.modules.attention.ulysses import (
    patch_forward_for_sp_input_slicing,
    ulysses_pad_inputs,
)
from torchturbo.models.modules.cpu_resident import CPUResidentParameter
from torchturbo.models.qwen4_exp.model import TurboQwen4ExpForConditionalGeneration
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

from .test_model import _small_config
from .test_multimodal import _cpu_flash
from .test_ple_distributed import _init_process_group


def _check_full_model_sp(rank, init_file, modality, reentrant):
    import transformers.modeling_utils as hf_modeling

    get_or_create_ops(device_runtime=CpuRuntime(), default_policy=OpPolicy.NATIVE)
    _init_process_group(rank, 2, init_file)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            # Preserve HF vision dispatch and production QSA/PLE/SP paths.
            # Only unavailable FlashAttention kernels and checks use CPU substitutes.
            monkeypatch.setitem(
                hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
                "general_availability_check",
                lambda: True,
            )
            monkeypatch.setattr(
                hf_modeling.PreTrainedModel,
                "_flash_attn_import_error",
                lambda *a, **k: None,
            )
            monkeypatch.setattr(
                hf_modeling, "lazy_import_flash_attention", lambda *a, **k: None
            )
            monkeypatch.setattr(
                "transformers.integrations.flash_attention._flash_attention_forward",
                _cpu_flash,
            )
            monkeypatch.setattr(
                "torchturbo.models.qwen4_exp.qsa.attention_forward", _cpu_flash
            )
            _compare_full_model_sp(rank, modality, reentrant, monkeypatch)
    finally:
        dist.destroy_process_group()


def _compare_full_model_sp(rank, modality, reentrant, monkeypatch):
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4
    config.image_token_id, config.video_token_id = 20, 22
    config.vision_start_token_id, config.vision_end_token_id = 19, 21
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(53)
    model = TurboQwen4ExpForConditionalGeneration(config).double().train()
    with torch.no_grad():
        model.model.language_model.layers[1].ple.conv1d.weight.normal_(std=0.03)
    reference = (
        TurboQwen4ExpForConditionalGeneration(copy.deepcopy(config)).double().train()
    )
    reference.load_state_dict(model.state_dict())
    embedding = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    reference_embedding = reference.model.language_model.layers[
        1
    ].ple.ple_embedding.ngram_embedding
    table = embedding.weight
    reference_table = reference_embedding.weight
    original_rows = table.clone()
    assert table is not reference_table
    assert table.data_ptr() != reference_table.data_ptr()
    torch.testing.assert_close(table, reference_table)
    assert table.device.type == "cpu" and not table.requires_grad
    parameters = list(reference_embedding.parameters())
    assert len(parameters) == 1
    assert all(isinstance(parameter, CPUResidentParameter) for parameter in parameters)
    assert all(parameter.device.type == "cpu" for parameter in parameters)
    assert all(not parameter.requires_grad for parameter in parameters)

    group = dist.group.WORLD
    spec = model.get_sequence_parallel_input_slicing_spec()
    patch_forward_for_sp_input_slicing(spec, group, 2, rank)
    model.on_parallelized({"sp": {"sp_group": group, "sp_size": 2, "sp_rank": rank}})
    model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": reentrant}
    )
    assert all(
        layer.gradient_checkpointing for layer in model.model.language_model.layers
    )
    decoder_calls = [0] * 4
    handles = []
    for index, layer in enumerate(model.model.language_model.layers):

        def count_decoder(module, args, index=index):
            decoder_calls[index] += 1

        handles.append(layer.register_forward_pre_hook(count_decoder))
    lookups = []
    original_gather = embedding.gather

    def record_lookup(indices):
        lookups.append(indices.clone())
        return original_gather(indices)

    monkeypatch.setattr(embedding, "gather", record_lookup)
    reference_lookups = []
    reference_gather = reference_embedding.gather

    def record_reference_lookup(indices):
        reference_lookups.append(indices.clone())
        return reference_gather(indices)

    monkeypatch.setattr(reference_embedding, "gather", record_reference_lookup)
    frames = 1 if modality == "image" else 2
    token = config.image_token_id if modality == "image" else config.video_token_id
    first = [3, 19, *([token] * frames), 21, 5]
    second = [6, 7, 19, *([token] * (4 * frames)), 21, 8, 9, 10, 11]
    if modality == "video":
        second.append(12)
    ids = torch.tensor([[*first, *second]])
    lengths = [len(first), len(second)]
    cu = torch.tensor([0, len(first), sum(lengths)], dtype=torch.int32)
    positions = torch.cat([torch.arange(length) for length in lengths])
    grids = torch.tensor([[frames, 2, 2], [frames, 4, 4]])
    pixels = torch.randn(20 * frames, 1536, dtype=torch.double, requires_grad=True)
    ref_pixels = pixels.detach().clone().requires_grad_()
    indices, weights = get_vision_bilinear_indices_and_weights(grids, 8, 2)
    weights = weights.double().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    media_name = "pixel_values" if modality == "image" else "pixel_values_videos"
    batch = {
        "input_ids": ids,
        "mm_token_type_ids": torch.where(
            ids == token, 1 if modality == "image" else 2, 0
        ),
        "position_ids": positions.view(1, 1, -1).expand(3, -1, -1),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": max(lengths),
        "max_length_k": max(lengths),
        "use_cache": False,
        media_name: pixels,
        f"{modality}_grid_thw": grids,
        f"{modality}_bilinear_indices": indices,
        f"{modality}_bilinear_weights": weights,
        f"{modality}_position_ids": get_vision_position_ids(grids, 2),
        f"{modality}_cu_seqlens": get_vision_cu_seqlens(grids),
        f"{modality}_max_seqlen": 16,
    }
    padded = ulysses_pad_inputs(batch, spec, 2)
    assert padded["input_ids"].shape[1] == ids.shape[1] + 1
    assert padded["input_ids"][0, -1] == 0
    assert padded["mm_token_type_ids"][0, -1] == 0
    assert padded["cu_seq_lens_q"] is cu
    assert padded[media_name] is pixels
    local_length = padded["input_ids"].shape[1] // 2
    assert padded["input_ids"][0, local_length - 1] == token
    assert padded["input_ids"][0, local_length] == token
    local_slice = slice(rank * local_length, (rank + 1) * local_length)

    reference_batch = dict(padded)
    reference_batch[media_name] = ref_pixels
    reference_batch[f"{modality}_bilinear_weights"] = ref_weights
    expected = reference(**reference_batch).logits
    assert len(reference_lookups) == 1
    assert lookups == []
    probe = torch.randn_like(expected)
    probe[:, ids.shape[1] :] = 0
    expected.backward(probe)
    assert len(reference_lookups) == 1
    assert lookups == []
    actual = model(**padded).logits
    context = (
        f"rank={rank}, modality={modality}, checkpoint={reentrant}"
    )
    torch.testing.assert_close(
        actual, expected[:, local_slice], atol=2e-6, rtol=2e-5, msg=context
    )
    assert len(lookups) == 1
    assert decoder_calls == [1] * 4
    actual.backward(probe[:, local_slice])
    assert len(lookups) == 1
    assert all(calls >= 2 for calls in decoder_calls)
    for handle in handles:
        handle.remove()
    dist.all_reduce(pixels.grad, group=group)
    dist.all_reduce(weights.grad, group=group)
    torch.testing.assert_close(
        pixels.grad, ref_pixels.grad, atol=2e-6, rtol=2e-5, msg=context
    )
    assert torch.count_nonzero(weights.grad) > 0
    torch.testing.assert_close(
        weights.grad, ref_weights.grad, atol=2e-6, rtol=2e-5, msg=context
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected_gradient = reference_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None, (context, name)
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (
            context,
            name,
        )
        dist.all_reduce(parameter.grad, group=group)
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            atol=2e-6,
            rtol=2e-5,
            msg=f"{context}, {name}",
        )
    assert torch.count_nonzero(model.model.visual.patch_embed.proj.weight.grad) > 0
    assert (
        torch.count_nonzero(
            model.model.language_model.layers[1].ple.value_proj.weight.grad
        )
        > 0
    )
    assert (
        torch.count_nonzero(model.model.language_model.layers[1].ple.conv1d.weight.grad)
        > 0
    )
    assert (
        torch.count_nonzero(
            model.model.language_model.layers[1].self_attn.k_proj.weight.grad
        )
        > 0
    )
    assert table.grad is None and torch.equal(table, original_rows)


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_full_model_sp_media_fusion_with_decoder_checkpointing(
    tmp_path, modality, reentrant
):
    mp.spawn(
        _check_full_model_sp,
        args=(str(tmp_path / "full-model-init"), modality, reentrant),
        nprocs=2,
        join=True,
    )
