import torch

from torchturbo.models.modules.attention.gdn.forward import (
    _interleave_qkv_conv_parameter,
)


def test_interleave_qkv_conv_parameter_matches_per_head_channel_order():
    num_key_heads = 2
    q = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    k = torch.arange(12, 24, dtype=torch.float64).reshape(4, 3).requires_grad_()
    v = torch.arange(24, 60, dtype=torch.float64).reshape(12, 3).requires_grad_()

    expected = torch.cat(
        (
            q.reshape(num_key_heads, 2, 3),
            k.reshape(num_key_heads, 2, 3),
            v.reshape(num_key_heads, 6, 3),
        ),
        dim=1,
    ).reshape(20, 3)
    actual = _interleave_qkv_conv_parameter(q, k, v, num_key_heads)

    torch.testing.assert_close(actual, expected)
    multiplier = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * multiplier).sum(), (q, k, v))
    expected_grads = torch.autograd.grad((expected * multiplier).sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
