from __future__ import annotations

import os
import sys
from collections.abc import Callable
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from torchturbo.ops.providers.npu.cross_entropy_loss import NpuCrossEntropyLoss


def _install_fake_torch_npu(
    monkeypatch, cross_entropy_loss: Callable[..., torch.Tensor]
) -> None:
    torch_npu = ModuleType("torch_npu")
    torch_npu.npu_cross_entropy_loss = cross_entropy_loss
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)


@pytest.mark.parametrize("shape", [(1,), (2, 1)])
def test_npu_cross_entropy_preserves_unreduced_shape(monkeypatch, shape):
    expected = torch.ones(shape)
    _install_fake_torch_npu(monkeypatch, lambda *args, **kwargs: expected)

    actual = NpuCrossEntropyLoss()(
        torch.randn(2, 3),
        torch.tensor([0, 1]),
        reduction="none",
    )

    assert actual.shape == expected.shape


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_npu_cross_entropy_scalarizes_reduced_loss(monkeypatch, reduction):
    _install_fake_torch_npu(monkeypatch, lambda *args, **kwargs: torch.ones(1))

    actual = NpuCrossEntropyLoss()(
        torch.randn(2, 3),
        torch.tensor([0, 1]),
        reduction=reduction,
    )

    assert actual.shape == torch.Size([])


def _precision_inputs(dtype):
    generator = torch.Generator().manual_seed(73)
    logits = torch.randn(256, 1024, generator=generator).to(dtype)
    labels = torch.randint(1024, (256,), generator=generator)
    labels[[124, 255]] = -100
    return logits, labels


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_npu_cross_entropy_matches_float32_loss_and_gradients(monkeypatch, dtype):
    # CPU CE preserves its input dtype outside autocast, exposing premature
    # low-precision reduction just as a dtype-preserving fused kernel would.
    _install_fake_torch_npu(monkeypatch, F.cross_entropy)
    logits, labels = _precision_inputs(dtype)
    num_valid_tokens = int((labels != -100).sum())
    logits.requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()

    actual = NpuCrossEntropyLoss()(logits, labels, reduction="sum")
    expected = F.cross_entropy(reference_logits.float(), labels, reduction="sum")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    (actual_grad,) = torch.autograd.grad(actual / num_valid_tokens, logits)
    (expected_grad,) = torch.autograd.grad(
        expected / num_valid_tokens, reference_logits
    )
    assert actual_grad.dtype == dtype
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert torch.count_nonzero(actual_grad[labels == -100]) == 0


@pytest.mark.skipif(
    os.environ.get("TORCH_TURBO_TEST_NPU_CROSS_ENTROPY") != "1",
    reason="opt-in native NPU cross-entropy forward/backward",
)
@pytest.mark.parametrize("num_partitions", [1, 2, 8])
def test_native_npu_cross_entropy_preserves_partitioned_loss_and_gradients(
    num_partitions,
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    host_logits, host_labels = _precision_inputs(torch.bfloat16)
    num_valid_tokens = int((host_labels != -100).sum())
    labels = host_labels.to("npu")
    reference_logits = host_logits.to("npu").requires_grad_()
    with torch.autocast("npu", dtype=torch.bfloat16):
        expected = F.cross_entropy(
            reference_logits.float(), labels, reduction="sum"
        ) / num_valid_tokens
    (expected_grad,) = torch.autograd.grad(expected, reference_logits)

    logits = host_logits.to("npu").requires_grad_()
    # Autocast must not cast the FP32 logits back to BF16 before the fused op.
    with torch.autocast("npu", dtype=torch.bfloat16):
        losses = [
            NpuCrossEntropyLoss()(part, targets, reduction="sum")
            for part, targets in zip(
                logits.chunk(num_partitions),
                labels.chunk(num_partitions),
                strict=True,
            )
        ]
        assert all(loss.dtype == torch.float32 for loss in losses)
        actual = torch.stack(losses).sum() / num_valid_tokens
    (actual_grad,) = torch.autograd.grad(actual, logits)
    assert actual_grad.dtype == torch.bfloat16
    # FP32 reduction order may differ across partitions and CE kernels.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    # Compare after the cast's backward rounds gradients to BF16. Permit
    # one BF16 relative rounding unit, with a floor for small probabilities.
    torch.testing.assert_close(
        actual_grad.float(), expected_grad.float(), rtol=0.008, atol=1e-9
    )
    assert torch.count_nonzero(actual_grad[labels == -100]) == 0
