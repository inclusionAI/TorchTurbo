from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from torchturbo.ops.providers.npu.grouped_matmul import TorchNpuGroupedMatmul


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
