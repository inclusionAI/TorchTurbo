"""Empty-rank vision exchanges, including their inverse gradient exchanges."""

import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.modules.comm.comm import all_to_all


def _counts(world: int, case: str) -> list[list[int]]:
    if case == "empty":
        return [[0] * world for _ in range(world)]
    if case == "self_only":
        return [
            [src + 1 if src == dst else 0 for dst in range(world)]
            for src in range(world)
        ]
    if case == "one_sender":
        return [
            [2 if src == 0 and dst == 1 else 0 for dst in range(world)]
            for src in range(world)
        ]
    if case == "sp2":
        return [
            [3 if src % 2 == 0 and dst == src // 2 else 0 for dst in range(world)]
            for src in range(world)
        ]
    return [
        [1 + src if dst == (src + 1) % world else 0 for dst in range(world)]
        for src in range(world)
    ]


def _rows(src: int, dst: int, count: int) -> torch.Tensor:
    # Non-contiguous three-dimensional rows exercise more than flat features.
    return (
        torch.arange(count * 6, dtype=torch.float32)
        .reshape(count, 3, 2)
        .transpose(1, 2)
        / 32
        + src
        + dst / 8
    )


def _distributed_exchange(rank: int, init_file: str, world: int, backend: str) -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)
    dist.init_process_group(
        backend,
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    original = dist.all_to_all_single
    calls = []

    def checked(output, input, **kwargs):
        assert input.size(0) == sum(kwargs["input_split_sizes"])
        assert output.size(0) == sum(kwargs["output_split_sizes"])
        calls.append((kwargs["input_split_sizes"], kwargs["output_split_sizes"]))
        return original(output, input, **kwargs)

    try:
        for case in ("empty", "self_only", "one_sender", "sp2", "nonempty", "empty"):
            counts = _counts(world, case)
            send = counts[rank]
            recv = [counts[src][rank] for src in range(world)]
            send_before, recv_before = send.copy(), recv.copy()
            values = torch.cat(
                [_rows(rank, dst, count) for dst, count in enumerate(send)]
            )
            # Preserve a non-contiguous input to the exchange.
            values = values.transpose(1, 2).contiguous().transpose(1, 2)
            expected = torch.cat(
                [_rows(src, rank, count) for src, count in enumerate(recv)]
            )
            for dtype in (torch.float32, torch.bfloat16):
                for trainable in (False, True):
                    rows = (
                        values.to(device=device, dtype=dtype)
                        .detach()
                        .requires_grad_(trainable)
                    )
                    calls.clear()
                    with patch.object(dist, "all_to_all_single", side_effect=checked):
                        actual = all_to_all(dist.group.WORLD, rows, recv, send)
                        torch.testing.assert_close(
                            actual.cpu(), expected.to(dtype=dtype)
                        )
                        assert actual.requires_grad == trainable
                        if trainable:
                            (actual.square().sum() * (rank + 1)).backward()
                            expected_grad = torch.cat(
                                [
                                    2
                                    * _rows(rank, dst, count).to(dtype=dtype)
                                    * (dst + 1)
                                    for dst, count in enumerate(send)
                                ]
                            )
                            torch.testing.assert_close(rows.grad.cpu(), expected_grad)
                    # Backward transposes the original counts, including zeros.
                    assert calls == (
                        [(send, recv), (recv, send)]
                        if trainable
                        else [(send, recv)]
                    )
                    assert send == send_before and recv == recv_before
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend",
    [
        "gloo",
        pytest.param(
            "hccl",
            marks=pytest.mark.skipif(
                os.environ.get("TORCH_TURBO_TEST_VISION_HCCL") != "1",
                reason="opt-in four-device HCCL regression",
            ),
        ),
    ],
)
def test_empty_buffers_forward_and_backward(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "hccl":
        # Work around the observed CANN 9.0/A3 AlltoAllV cache metadata failure.
        # An explicit environment override can reproduce that backend failure.
        monkeypatch.setenv(
            "HCCL_OP_EXPANSION_MODE",
            os.environ.get("HCCL_OP_EXPANSION_MODE", "AICPU_CacheDisable"),
        )
    context = mp.spawn(
        _distributed_exchange,
        args=(str(tmp_path / "init"), 4, backend),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("vision communication worker timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_single_rank_empty_exchange_preserves_gradient(tmp_path: Path) -> None:
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "init").as_uri(), rank=0, world_size=1
    )
    try:
        rows = torch.empty(0, 3, requires_grad=True)
        result = all_to_all(dist.group.WORLD, rows, [0], [0])
        assert result.shape == rows.shape
        result.sum().backward()
        assert rows.grad is not None and rows.grad.shape == rows.shape
    finally:
        dist.destroy_process_group()
