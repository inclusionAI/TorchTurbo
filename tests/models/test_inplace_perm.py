import numpy as np
import pytest
import torch

from torchturbo.models.modules.moe.inplace_perm import (
    InplaceRowPermuteFn,
    inverse_perm,
)


@pytest.mark.parametrize("num_rows", [17, 128, 4097])
def test_inplace_permutation_matches_index_select_forward_and_backward(num_rows):
    generator = torch.Generator().manual_seed(num_rows)
    row_perm = torch.randperm(num_rows, generator=generator).numpy()
    inverse = inverse_perm(row_perm)
    actual_source = torch.randn(num_rows, 4, generator=generator, requires_grad=True)
    expected_base = actual_source.detach().clone().requires_grad_()
    weights = torch.arange(1, num_rows + 1).unsqueeze(1)
    actual = InplaceRowPermuteFn.apply(actual_source * 1, row_perm, inverse)
    expected = expected_base[row_perm]

    torch.testing.assert_close(actual, expected)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual_source.grad, expected_base.grad)


def test_inplace_permutation_backward_restores_gradient_order():
    row_perm = np.array([2, 0, 1])
    inverse = inverse_perm(row_perm)
    base = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
    output = InplaceRowPermuteFn.apply(base * 1, row_perm, inverse)
    output.backward(torch.tensor([[3.0], [5.0], [7.0]]))

    torch.testing.assert_close(base.grad, torch.tensor([[5.0], [7.0], [3.0]]))
