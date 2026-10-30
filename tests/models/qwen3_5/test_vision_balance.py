"""Qwen3.5 vision balancing with real towers and distributed gradients."""

import copy
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.modeling_outputs import BaseModelOutputWithPooling

from torchturbo.engine.args import parse_torchturbo_args
from torchturbo.engine.config import build_from_args
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.base import TurboModel
from torchturbo.models.modules.vision import VisionBatch, register_vision_balance
from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from torchturbo.models.qwen3_5.vision_balance import Qwen3_5VisionBalanceIO
from torchturbo.parallel import VisionBalanceMode, VisionBalanceSpec


def _tower():
    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=12,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    return Qwen3_5VisionModel(config).double().train()


class _DensePlanModel(TurboModel):
    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        return TurboQwen3_5ForConditionalGeneration.get_default_parallel_plan()


@pytest.mark.parametrize("mode", [None, "patch", "patch_sq"])
def test_yaml_merges_dense_vision_balance_without_losing_sharding(tmp_path, mode):
    path = tmp_path / "vision.yaml"
    spec = f"        vision_balance:\n          mode: {mode}\n" if mode else ""
    path.write_text(
        "engine_config:\n  parallel_plan:\n    sp_size: 8\n    ep_size: 8\n"
        "    plan:\n      model.visual:\n        standalone_world_shard: true\n"
        + spec
    )
    args, _ = parse_torchturbo_args(["--tt_config", str(path)])
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.config = build_from_args(args)
    plan = engine._resolve_parallel_plan(_DensePlanModel())
    visual = plan.plan["model.visual"]
    assert plan.sp_size == plan.ep_size == 8
    assert visual.standalone_world_shard is True
    assert plan.plan["model.visual.blocks.*"].activation_checkpoint is True
    assert isinstance(visual.vision_balance.io, Qwen3_5VisionBalanceIO)
    assert visual.vision_balance.mode == (VisionBalanceMode(mode) if mode else None)
    default = _DensePlanModel.get_default_parallel_plan()
    assert default.plan["model.visual"].vision_balance.mode is None
    assert default.ep_size == 1


@pytest.mark.parametrize("style", ["positional", "mixed", "keyword"])
def test_io_preserves_call_and_both_output_streams(style):
    tower = _tower()
    io = Qwen3_5VisionBalanceIO()
    pixels = torch.randn(20, 24, dtype=torch.float64)
    grid = torch.tensor([[1, 2, 2], [2, 2, 4]])
    kwargs = {"return_dict": True}
    if style == "positional":
        args = (pixels, grid)
    elif style == "mixed":
        args = (pixels,)
        kwargs["grid_thw"] = grid
    else:
        args = ()
        kwargs.update(hidden_states=pixels, grid_thw=grid)
    batch = io.read_inputs(tower, args, kwargs)
    assert batch.output_lengths == {
        "last_hidden_state": [4, 16], "pooler_output": [1, 4]
    }
    reordered = VisionBatch(
        torch.cat((pixels[4:], pixels[:4])), grid.flip(0),
        {"last_hidden_state": [16, 4], "pooler_output": [4, 1]},
    )
    new_args, new_kwargs = io.write_inputs(args, kwargs, reordered)
    assert len(new_args) == len(args)
    assert new_kwargs["return_dict"] is True
    assert io.read_inputs(tower, args, kwargs).pixels is pixels
    actual = tower(*new_args, **new_kwargs)
    assert isinstance(actual, BaseModelOutputWithPooling)
    expected = tower(pixels, grid_thw=grid)
    raw = torch.cat((actual.last_hidden_state[16:], actual.last_hidden_state[:16]))
    pooled = torch.cat((actual.pooler_output[4:], actual.pooler_output[:4]))
    restored = io.write_outputs(
        actual, {"last_hidden_state": raw, "pooler_output": pooled}, batch
    )
    torch.testing.assert_close(restored.last_hidden_state, expected.last_hidden_state)
    torch.testing.assert_close(restored.pooler_output, expected.pooler_output)
    assert restored[0] is restored.last_hidden_state
    assert restored["pooler_output"] is restored.pooler_output


def test_dummy_runs_both_streams_and_connects_gradients():
    tower = _tower()
    io = Qwen3_5VisionBalanceIO()
    dummy = io.make_dummy(tower)
    assert dummy.pixels.shape == (4, 24)
    assert dummy.pixels.dtype == torch.float64
    assert dummy.grid_thw.tolist() == [[1, 2, 2]]
    assert dummy.output_lengths == {"last_hidden_state": [4], "pooler_output": [1]}
    output = tower(dummy.pixels, grid_thw=dummy.grid_thw)
    assert output.last_hidden_state.shape == (4, 16)
    assert output.pooler_output.shape == (1, 12)
    output.pooler_output.sum().backward()
    assert all(p.grad is not None for p in tower.parameters())


def _rank_inputs(rank, sp_size, case):
    group = rank // sp_size
    if case == "scarce":
        rows = [[1, 2, 2]]
    elif group % 2:
        rows = [[3, 2, 2], [1, 4, 4]]
    else:
        rows = [[1, 2, 2], [2, 2, 4], [1, 4, 2]]
    grid = torch.tensor(rows)
    count = int(grid.prod(-1).sum())
    pixels = torch.arange(count * 24, dtype=torch.float64).reshape(count, 24) / 500
    return pixels + group, grid


def _loss(output, rank, loss_mode):
    if loss_mode == "pooled":
        return output.pooler_output.square().sum()
    if loss_mode == "rank_specific":
        features = output.last_hidden_state if rank % 2 else output.pooler_output
        return features.square().sum()
    return (rank + 1) * (
        output.last_hidden_state.square().sum() + output.pooler_output.square().sum()
    )


def _distributed_round_trip(rank, init_file, world_size, sp_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(init_file).as_uri(), rank=rank,
        world_size=world_size, timeout=timedelta(seconds=40),
    )
    try:
        for mode in VisionBalanceMode:
            for case, loss_mode, checkpoint, frozen in [
                ("uneven", "all", False, False),
                ("uneven", "pooled", False, False),
                ("uneven", "rank_specific", False, False),
                ("scarce", "pooled", True, False),
                ("scarce", "pooled", False, True),
            ]:
                torch.manual_seed(42)
                tower = _tower().requires_grad_(not frozen)
                reference = copy.deepcopy(tower)
                if checkpoint:
                    tower.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                observed = []
                tower.patch_embed.register_forward_pre_hook(
                    lambda module, args: observed.append(args[0].size(0))
                )
                register_vision_balance(
                    tower,
                    spec=VisionBalanceSpec(mode=mode, io=Qwen3_5VisionBalanceIO()),
                    sp_size=sp_size, sp_rank=rank % sp_size,
                )
                pixels, grid = _rank_inputs(rank, sp_size, case)
                actual = tower(pixels, grid_thw=grid)
                expected = reference(pixels, grid_thw=grid)
                if case == "scarce":
                    assert observed == [4]
                assert tower.state_dict().keys() == reference.state_dict().keys()
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output)
                if not frozen:
                    _loss(actual, rank, loss_mode).backward()
                    _loss(expected, rank, loss_mode).backward()
                    for (name, p), (_, ref) in zip(
                        tower.named_parameters(), reference.named_parameters(), strict=True
                    ):
                        # Compare the reduction FSDP performs: a rank can own
                        # different media, or leave a stream unused locally.
                        grad = p.grad if p.grad is not None else torch.zeros_like(p)
                        ref_grad = ref.grad if ref.grad is not None else torch.zeros_like(ref)
                        dist.all_reduce(grad)
                        dist.all_reduce(ref_grad)
                        torch.testing.assert_close(grad, ref_grad, msg=name)
                with torch.no_grad():
                    local = tower(pixels, grid_thw=grid)
                    assert observed[-1] == pixels.size(0)
                    torch.testing.assert_close(local.pooler_output, expected.pooler_output)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size, sp_size", [(2, 1), (2, 2), (4, 2), (8, 8)])
def test_real_tower_distributed_outputs_and_gradients(tmp_path, world_size, sp_size):
    context = mp.spawn(
        _distributed_round_trip,
        args=(str(tmp_path / "gloo_init"), world_size, sp_size),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Qwen3.5 vision balance worker timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
