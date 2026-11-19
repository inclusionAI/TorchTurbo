"""Fresh-process statistics probe with each input allocated first."""

from __future__ import annotations

import argparse
import os

import torch

from torchturbo.runtime.npu import NpuRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", choices=("counts", "prefixes", "boundaries", "copies"), required=True)
    args = parser.parse_args()
    runtime = NpuRuntime()
    runtime.initialize()
    runtime.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from torchturbo.ops.providers.npu._kernels.virtual_expert_planner import _statistics_kernel

    def allocate(name: str) -> torch.Tensor:
        if name == "copies":
            return torch.full((2, 2), -1, device="npu", dtype=torch.int64)
        shape = (4, 2) if name == "boundaries" else (2, 4)
        return torch.zeros(shape, device="npu", dtype=torch.int64)

    buffers = {args.first: allocate(args.first)}
    assert buffers[args.first].data_ptr() % (2 * 1024 * 1024) == 0, (
        "The first input must begin at a 2 MiB expandable allocation boundary"
    )
    for name in ("counts", "prefixes", "boundaries", "copies"):
        if name != args.first:
            buffers[name] = allocate(name)
    statistics = torch.empty(10, device="npu", dtype=torch.int64)

    def check(expected: list[list[int]]) -> None:
        for rank in (0, 1):
            for block in (4, 8):
                _statistics_kernel[(2,)](
                    buffers["counts"], buffers["prefixes"], buffers["boundaries"],
                    buffers["copies"], statistics, rank, 2, 4, 2, block, block,
                )
                runtime.synchronize()
                torch.testing.assert_close(
                    statistics.cpu(), torch.tensor(expected[rank], device="cpu", dtype=torch.int64),
                    rtol=0, atol=0,
                )

    check([[0] * 10, [0] * 10])
    # Native traffic plus inactive replicas verifies that mapping masked
    # addresses to expert zero does not introduce expert-zero contributions.
    buffers["counts"].copy_(torch.tensor([[3, 0, 1, 0], [2, 0, 4, 0]], device="npu"))
    buffers["prefixes"].copy_(torch.tensor([[0, 0, 0, 0], [3, 0, 1, 0]], device="npu"))
    buffers["boundaries"].copy_(torch.tensor([[5, 5], [0, 0], [0, 5], [0, 0]], device="npu"))
    check([[3, 1, 3, 0, 0, 0, 2, 0, 0, 0], [2, 4, 1, 0, 0, 0, 4, 0, 0, 0]])
    print(f"STATISTICS_EXPANDABLE_BOUNDARY_PASS first={args.first}", flush=True)


if __name__ == "__main__":
    main()
