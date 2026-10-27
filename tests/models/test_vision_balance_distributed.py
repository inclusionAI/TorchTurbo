"""Exercise model-owned vision I/O through real CPU collectives."""

import copy
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from torchturbo.models.modules.vision import (
    BaseVisionBalanceIO,
    VisionBatch,
    register_vision_balance,
)
from torchturbo.parallel import VisionBalanceMode, VisionBalanceSpec


@dataclass
class _VisionOutput:
    raw: torch.Tensor
    pooled: torch.Tensor
    marker: str


class _TemporalPoolTower(nn.Module):
    """Return patch and temporally pooled streams with three-dimensional rows."""

    def __init__(self, detach_raw: bool = False) -> None:
        super().__init__()
        self.detach_raw = detach_raw
        self.weight = nn.Parameter(
            torch.tensor([[0.2, 0.3, -0.4], [0.5, -0.6, 0.7]], dtype=torch.float64)
        )
        self.observed_rows = -1

    def forward(
        self,
        patches: torch.Tensor,
        *,
        layout: torch.Tensor,
        scale: float,
        marker: str,
    ) -> _VisionOutput:
        self.observed_rows = patches.size(0)
        raw = (patches @ self.weight) * scale
        pooled = []
        start = 0
        for t, h, w in layout.tolist():
            end = start + t * h * w
            pooled.append(raw[start:end].reshape(t, h * w, 2, 3).mean(0))
            start = end
        merged = torch.cat(pooled) if pooled else raw[:0]
        return _VisionOutput(raw.detach() if self.detach_raw else raw, merged, marker)


class _TemporalPoolIO(BaseVisionBalanceIO):
    def read_inputs(
        self, tower: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> VisionBatch:
        pixels = args[0] if args else kwargs["patches"]
        grid = kwargs["layout"]
        return VisionBatch(
            pixels,
            grid,
            {
                "raw": grid.prod(-1).tolist(),
                "pooled": grid[:, 1:].prod(-1).tolist(),
            },
        )

    def write_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        batch: VisionBatch,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        updated = dict(kwargs, layout=batch.grid_thw)
        if args:
            return (batch.pixels, *args[1:]), updated
        updated["patches"] = batch.pixels
        return args, updated

    def read_outputs(self, output: _VisionOutput) -> dict[str, torch.Tensor]:
        return {"raw": output.raw, "pooled": output.pooled}

    def write_outputs(
        self,
        output: _VisionOutput,
        features: dict[str, torch.Tensor],
        original_batch: VisionBatch,
    ) -> _VisionOutput:
        return _VisionOutput(features["raw"], features["pooled"], output.marker)

    def make_dummy(self, tower: nn.Module) -> VisionBatch:
        parameter = next(tower.parameters())
        return VisionBatch(
            parameter.new_zeros(2, 2, 2),
            torch.tensor([[2, 1, 1]], device=parameter.device),
            {"raw": [2], "pooled": [1]},
        )


def _rank_inputs(rank: int, sp_size: int, case: str) -> tuple[torch.Tensor, torch.Tensor]:
    if case == "empty":
        rows = []
    elif case == "one":
        rows = [[2, 1, 2]] if sp_size == 2 or rank == 1 else []
    elif sp_size == 2 or rank == 0:
        rows = [[1, 1, 2], [2, 1, 4], [1, 1, 4]]
    else:
        rows = [[3, 1, 6], [1, 1, 2]]
    grid = torch.tensor(rows, dtype=torch.long).reshape(-1, 3)
    num_patches = int(grid.prod(-1).sum())
    pixels = torch.arange(num_patches * 4, dtype=torch.float64).reshape(-1, 2, 2) / 37
    return pixels + rank // sp_size, grid


def _loss(output: _VisionOutput, rank: int, loss_mode: str) -> torch.Tensor:
    # Even empty outputs retain their exchange in the backward graph.
    if loss_mode == "pooled":
        return output.pooled.square().sum()
    if loss_mode == "rank_specific":
        selected = output.raw if rank == 0 else output.pooled
        return selected.square().sum()
    return (rank + 1) * (output.raw.square().sum() + 0.7 * output.pooled.square().sum())


def _distributed_round_trip(rank: int, init_file: str, sp_size: int) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        for mode in VisionBalanceMode:
            for case, loss_mode in product(
                ("uneven", "one", "empty"),
                ("all", "pooled", "rank_specific", "detached_raw"),
            ):
                tower = _TemporalPoolTower(detach_raw=loss_mode == "detached_raw")
                reference = copy.deepcopy(tower)
                register_vision_balance(
                    tower,
                    spec=VisionBalanceSpec(mode=mode, io=_TemporalPoolIO()),
                    sp_size=sp_size,
                    sp_rank=rank % sp_size,
                )
                assert tower.state_dict().keys() == reference.state_dict().keys()
                pixels, grid = _rank_inputs(rank, sp_size, case)
                kwargs = {"layout": grid, "scale": 1.25, "marker": f"rank-{rank}"}
                expected = reference(pixels, **kwargs)
                if case == "one":
                    actual = tower(patches=pixels, **kwargs)
                    assert tower.observed_rows == (4 if rank == 0 else 2)
                else:
                    actual = tower(pixels, **kwargs)
                    if case == "empty":
                        assert tower.observed_rows == 2
                assert isinstance(actual, _VisionOutput)
                assert actual.marker == expected.marker
                assert actual.raw.requires_grad == expected.raw.requires_grad
                assert actual.pooled.requires_grad == expected.pooled.requires_grad
                torch.testing.assert_close(actual.raw, expected.raw)
                torch.testing.assert_close(actual.pooled, expected.pooled)

                _loss(actual, rank, loss_mode).backward()
                _loss(expected, rank, loss_mode).backward()
                # FSDP would reduce these parameter gradients after the hook's
                # differentiable exchanges combine SP consumers on each owner.
                dist.all_reduce(tower.weight.grad)
                dist.all_reduce(reference.weight.grad)
                torch.testing.assert_close(tower.weight.grad, reference.weight.grad)

                with torch.no_grad():
                    local = tower(pixels, **kwargs)
                    assert tower.observed_rows == pixels.size(0)
                    torch.testing.assert_close(local.raw, expected.raw)
                    torch.testing.assert_close(local.pooled, expected.pooled)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sp_size", [1, 2])
def test_model_io_restores_streams_and_gradients_across_ranks(tmp_path, sp_size):
    context = mp.spawn(
        _distributed_round_trip,
        args=(str(tmp_path / "gloo_init"), sp_size),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("vision balance worker timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
