"""Run an external callable through the public dispatcher and backward."""

from pathlib import Path

import torch

from torchturbo.engine.args import parse_torchturbo_args
from torchturbo.engine.config import build_from_args
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


def main() -> None:
    args, _ = parse_torchturbo_args([
        "--tt_config", str(Path(__file__).with_name("config.yaml")),
    ])
    config = build_from_args(args)
    ops = get_or_create_ops(
        device_runtime=CpuRuntime(), plugins=config.ops.plugins,
    )
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    output = ops.scaled_square(x, scale=2.0)
    output.sum().backward()
    torch.testing.assert_close(output, 2 * x.square())
    torch.testing.assert_close(x.grad, 4 * x)
    ops.log_ops_selection()
    print(f"output={output.detach().tolist()}, gradient={x.grad.tolist()}")


if __name__ == "__main__":
    main()
