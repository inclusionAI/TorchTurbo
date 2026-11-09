"""Private Ascend Triton probability packing with inverse-only autograd state."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

import triton
import triton.language as tl


@triton.jit
def _scatter_probabilities(
    probabilities, inverse, output, num_elements,
    num_columns: tl.constexpr, row_stride, column_stride, inverse_stride,
    block_size: tl.constexpr,
):
    assignment = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    mask = assignment < num_elements
    token = assignment // num_columns
    column = assignment % num_columns
    value = tl.load(
        probabilities + token * row_stride + column * column_stride, mask, other=0,
    )
    packed_row = tl.load(inverse + assignment * inverse_stride, mask, other=0).to(tl.int64)
    tl.store(output + packed_row, value, mask)


@triton.jit
def _gather_probability_gradients(
    gradient, inverse, output, num_elements, gradient_stride, inverse_stride,
    block_size: tl.constexpr,
):
    assignment = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    mask = assignment < num_elements
    packed_row = tl.load(inverse + assignment * inverse_stride, mask, other=0).to(tl.int64)
    value = tl.load(gradient + packed_row * gradient_stride, mask, other=0)
    tl.store(output + assignment, value, mask)


class _PackProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs, inverse, block_size):
        ctx.save_for_backward(inverse)
        ctx.input_shape = probs.shape
        ctx.block_size = block_size
        num_elements = probs.numel()
        output = probs.new_empty(num_elements)
        if num_elements:
            if probs.ndim == 1:
                num_columns = 1
                row_stride, column_stride = probs.stride(0), 0
            else:
                num_columns = probs.shape[1]
                row_stride, column_stride = probs.stride()
            _scatter_probabilities[((num_elements + block_size - 1) // block_size,)](
                probs, inverse, output, num_elements, num_columns,
                row_stride, column_stride, inverse.stride(0), block_size,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (inverse,) = ctx.saved_tensors
        num_elements = inverse.numel()
        grad_input = grad_output.new_empty(num_elements)
        if num_elements:
            block_size = ctx.block_size
            _gather_probability_gradients[((num_elements + block_size - 1) // block_size,)](
                grad_output, inverse, grad_input, num_elements,
                grad_output.stride(0), inverse.stride(0), block_size,
            )
        return grad_input.view(ctx.input_shape), None, None


def pack_probabilities(
    probs: torch.Tensor, inverse: torch.Tensor,
) -> torch.Tensor:
    """Pack [N] or [tokens, topk] probabilities using a flat int32 inverse.

    Input and inverse may be strided; output is contiguous and keeps the input
    probability dtype. Each inverse entry must be a unique packed row in
    [0, probs.numel()), so there is no atomic update or dropping of zero weights.
    First-order autograd saves only the inverse and gathers packed gradients.
    Requires matching NPU tensors and an int32 inverse permutation.
    Higher-order derivatives are unsupported.
    """
    if probs.device.type != "npu" or inverse.device != probs.device:
        raise ValueError("Probability packing requires matching NPU tensors.")
    if probs.ndim not in (1, 2) or probs.dtype not in (
        torch.float16, torch.bfloat16, torch.float32,
    ):
        raise ValueError("Probability input must be a one- or two-dimensional floating tensor.")
    if inverse.ndim != 1 or inverse.dtype != torch.int32 or inverse.numel() != probs.numel():
        raise ValueError("The inverse must be a flat int32 permutation with one entry per probability.")
    if probs.numel() >= 2**31:
        raise ValueError("Int32 probability packing requires fewer than 2**31 assignments.")
    return _PackProbabilities.apply(probs, inverse, 512)
