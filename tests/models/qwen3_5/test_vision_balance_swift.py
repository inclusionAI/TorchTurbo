"""Swift text/media training with real Qwen towers and CPU/Gloo balancing.

SP sizes here exercise media ownership, not text sequence sharding or FSDP.
"""

import copy
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.modules.vision import register_vision_balance
from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.swift_template import _make_kwarg_preserving_template_cls
from torchturbo.models.qwen3_5.vision_balance import Qwen3_5VisionBalanceIO
from torchturbo.parallel import VisionBalanceMode, VisionBalanceSpec


def _model():
    config = Qwen3_5Config(
        image_token_id=30,
        video_token_id=31,
        vision_start_token_id=29,
        text_config={
            "vocab_size": 32,
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "layer_types": ["full_attention"],
            "mtp_num_hidden_layers": 0,
            "use_cache": False,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 0],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 12,
            "num_position_embeddings": 16,
        },
    )
    config._attn_implementation = "eager"
    return TurboQwen3_5ForConditionalGeneration(config).train()


def _template():
    from transformers import Qwen2VLImageProcessor

    # Supply only the processor needed by the real training hook; no hub access.
    template = object.__new__(_make_kwarg_preserving_template_cls())
    template.mode = "train"
    template.padding_free = False
    template.processor = SimpleNamespace(
        image_processor=Qwen2VLImageProcessor(
            min_pixels=16, max_pixels=64, patch_size=2,
            temporal_patch_size=2, merge_size=2,
        )
    )
    return template


def _batch(rank, sp_size, case):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    batch = {"input_ids": ids, "position_ids": torch.arange(7).unsqueeze(0)}
    if case.startswith("mixed") and (rank // sp_size) % 2:
        ids[0, 1:3] = 30
        ids[0, 4:6] = 31
        batch.update(
            pixel_values=torch.arange(8 * 24).float().reshape(8, 24) / 500,
            image_grid_thw=torch.tensor(
                [[1, 2, 2], [1, 2, 2]] if case == "mixed_dense" else [[1, 2, 4]]
            ),
            pixel_values_videos=torch.arange(8 * 24).float().reshape(8, 24) / 300,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
        )
    batch["labels"] = ids.clone()
    return batch


def _distributed_swift_training(rank, init_file, world_size, sp_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(init_file).as_uri(), rank=rank,
        world_size=world_size, timeout=timedelta(seconds=40),
    )
    try:
        template = _template()
        for mode in VisionBalanceMode:
            for frozen in (False, True):
                torch.manual_seed(42)
                model = _model()
                model.visual.requires_grad_(not frozen)
                reference = copy.deepcopy(model)
                observed = []
                model.visual.patch_embed.register_forward_pre_hook(
                    lambda module, args, observed=observed: observed.append(args[0].size(0))
                )
                register_vision_balance(
                    model.visual,
                    spec=VisionBalanceSpec(mode=mode, io=Qwen3_5VisionBalanceIO()),
                    sp_size=sp_size, sp_rank=rank % sp_size,
                )
                # Reuse hooks across steps to catch stale plans or dummy state.
                for case in ("text", "mixed", "mixed_dense", "text"):
                    model.zero_grad(set_to_none=True)
                    reference.zero_grad(set_to_none=True)
                    observed.clear()
                    batch = _batch(rank, sp_size, case)
                    args, prepared = template.pre_forward_hook(model, (), dict(batch))
                    ref_args, ref_prepared = template.pre_forward_hook(reference, (), dict(batch))
                    assert len(observed) == 1
                    assert prepared["input_ids"] is batch["input_ids"]
                    torch.testing.assert_close(prepared["inputs_embeds"], ref_prepared["inputs_embeds"])
                    if case == "text":
                        torch.testing.assert_close(
                            prepared["inputs_embeds"], model.get_input_embeddings()(batch["input_ids"])
                        )
                    actual = model(*args, **prepared)
                    expected = reference(*ref_args, **ref_prepared)
                    assert len(observed) == 1
                    assert torch.isfinite(actual.loss)
                    torch.testing.assert_close(actual.logits, expected.logits)
                    torch.testing.assert_close(actual.loss, expected.loss)
                    actual.loss.backward()
                    expected.loss.backward()
                    for (name, param), (_, ref) in zip(
                        model.named_parameters(), reference.named_parameters(), strict=True
                    ):
                        if not param.requires_grad:
                            assert param.grad is None and ref.grad is None
                            continue
                        assert param.grad is not None, name
                        assert ref.grad is not None, name
                        # Compare the parameter reduction performed by FSDP.
                        dist.all_reduce(param.grad)
                        dist.all_reduce(ref.grad)
                        torch.testing.assert_close(
                            param.grad, ref.grad, rtol=2e-4, atol=2e-6, msg=name
                        )
                        if case == "text" and name.startswith("model.visual."):
                            assert torch.count_nonzero(param.grad) == 0, name
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size, sp_size", [(2, 1), (4, 2)])
def test_swift_text_and_mixed_media_training(tmp_path: Path, world_size: int, sp_size: int) -> None:
    pytest.importorskip("swift")
    context = mp.spawn(
        _distributed_swift_training,
        args=(str(tmp_path / "gloo_init"), world_size, sp_size),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Qwen3.5 Swift vision balance worker timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
