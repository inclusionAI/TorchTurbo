from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from torchturbo.ops.providers.npu.grouped_matmul import TorchNpuGroupedMatmul
from torchturbo.ops.providers.npu.matmul_add import TorchNpuMatmulAdd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_wgrad_preserves_fp32_accumulation(dtype):
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(128, 32, generator=generator).to(dtype)
    dy = torch.randn(128, 48, generator=generator).to(dtype)
    accumulator = torch.zeros(32, 48, dtype=torch.float32, device="npu")
    TorchNpuMatmulAdd()(accumulator, x.to("npu"), dy.to("npu"), transpose_lhs=True)
    expected = (x.double().T @ dy.double()).float()
    torch.testing.assert_close(accumulator.cpu(), expected, atol=2e-4, rtol=1e-5)
    assert (accumulator.cpu() - expected.to(dtype).float()).abs().max() > 1e-3


def _reference_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: list[int],
    trans_b: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_index, count in enumerate(counts):
        end = start + count
        expert_weight = weight[expert_index]
        if trans_b:
            expert_weight = expert_weight.transpose(0, 1)
        outputs.append(x[start:end] @ expert_weight)
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return x.new_empty((0, weight.shape[-1]))


def _reference_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: list[int],
    trans_b: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_size = weight.shape[1] if trans_b else weight.shape[2]
    grad_output = torch.ones((sum(counts), output_size), dtype=x.dtype)
    grad_x = torch.zeros_like(x)
    grad_weight = torch.zeros_like(weight)
    start = 0
    for expert_index, count in enumerate(counts):
        end = start + count
        expert_weight = weight[expert_index]
        if trans_b:
            grad_x[start:end] = grad_output[start:end] @ weight[expert_index]
            grad_weight[expert_index] = (
                x[start:end].T @ grad_output[start:end]
            ).T
        else:
            grad_x[start:end] = grad_output[start:end] @ expert_weight.T
            grad_weight[expert_index] = x[start:end].T @ grad_output[start:end]
        start = end
    return grad_x, grad_weight


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("counts_device", ["cpu", "npu"])
def test_grouped_matmul_forward_and_backward(
    trans_b: bool, counts_device: str
) -> None:
    counts = [0, 5, 0, 11]
    input_size = 8
    output_size = 6
    x_values = torch.randn(sum(counts), input_size, dtype=torch.float32)
    weight_shape = (
        (len(counts), output_size, input_size)
        if trans_b
        else (len(counts), input_size, output_size)
    )
    weight_values = torch.randn(*weight_shape, dtype=torch.float32)

    x = x_values.to(device="npu").requires_grad_(True)
    weight = weight_values.to(device="npu").requires_grad_(True)
    group_list = torch.tensor(counts, dtype=torch.int64, device=counts_device)

    actual = TorchNpuGroupedMatmul()(x, weight, group_list, trans_b=trans_b)
    expected = _reference_forward(x_values, weight_values, counts, trans_b)
    assert torch.allclose(actual.float().cpu(), expected, atol=1e-4, rtol=1e-4)

    actual.backward(torch.ones_like(actual))
    expected_grad_x, expected_grad_weight = _reference_backward(
        x_values, weight_values, counts, trans_b
    )
    assert x.grad is not None
    assert weight.grad is not None
    assert torch.allclose(x.grad.float().cpu(), expected_grad_x, atol=1e-4, rtol=1e-4)
    assert torch.allclose(
        weight.grad.float().cpu(), expected_grad_weight, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [False, True])
def test_grouped_matmul_autocast_matches_matmul_outputs_and_gradients(dtype, trans_b):
    counts = [0, 3, 5]
    generator = torch.Generator().manual_seed(127)
    inputs = torch.randn(8, 16, generator=generator).to("npu").requires_grad_()
    weight_shape = (3, 12, 16) if trans_b else (3, 16, 12)
    weights = torch.randn(weight_shape, generator=generator).to("npu").requires_grad_()
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("npu", dtype=dtype):
        output = TorchNpuGroupedMatmul()(inputs, weights, torch.tensor(counts), trans_b=trans_b)
        expected = _reference_forward(ref_inputs, ref_weights, counts, trans_b)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(weights.grad, ref_weights.grad, atol=2e-2, rtol=1e-2)
