import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.engine.backends.fsdp2.checkpoint_input import (
    checkpoint_with_cpu_input,
    checkpoint_with_large_input_offload,
)
from torchturbo.engine.config import ActivationCheckpointingConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_input_offload(dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("An NPU is required.")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    x = torch.randn(2, 16, 4, 32, device="npu", dtype=dtype, requires_grad=True)
    w = torch.randn(32, 32, device="npu", dtype=dtype, requires_grad=True)

    def compute(value):
        return (value @ w).sin()

    expected = checkpoint(compute, x, use_reentrant=False)
    expected.float().square().sum().backward()
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    actual = checkpoint_with_cpu_input(compute, x, use_reentrant=False)
    actual.float().square().sum().backward()
    for a, b in [(actual, expected), (x.grad, dx), (w.grad, dw)]:
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_reentrant_offload_rejected():
    with pytest.raises(ValueError, match="non-reentrant"):
        ActivationCheckpointingConfig(use_reentrant=True, offload_input_to_cpu=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reentrant_large_input_offload(dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("An NPU is required.")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    x = torch.randn(2, 16, 4, 32, device="npu", dtype=dtype, requires_grad=True)
    w = torch.randn(32, 32, device="npu", dtype=dtype, requires_grad=True)

    def compute(value):
        return (value @ w).sin()

    expected = checkpoint(compute, x, use_reentrant=True)
    expected.float().square().sum().backward()
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    actual = checkpoint_with_large_input_offload(
        compute, x, use_reentrant=True, min_bytes=1024
    )
    actual.float().square().sum().backward()
    # The offload only changes where the saved input lives; the unpacked
    # values are bitwise equal, so gradients must match exactly.
    for a, b in [(actual, expected), (x.grad, dx), (w.grad, dw)]:
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_large_input_offload_requires_reentrant():
    with pytest.raises(ValueError, match="use_reentrant=True"):
        checkpoint_with_large_input_offload(
            lambda value: value, torch.zeros(1), use_reentrant=False
        )


def test_large_input_offload_config_requires_reentrant():
    ActivationCheckpointingConfig(
        use_reentrant=True, offload_large_inputs_to_cpu=True
    )
    with pytest.raises(ValueError, match="reentrant"):
        ActivationCheckpointingConfig(
            use_reentrant=False, offload_large_inputs_to_cpu=True
        )
