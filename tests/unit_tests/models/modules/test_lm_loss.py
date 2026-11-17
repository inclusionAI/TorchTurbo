"""Tests for regular and chunked language-model loss computation."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchturbo.models.modules.loss import (
    chunk_loss,
    logits_loss,
)
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

IGNORE = -100


@pytest.fixture(autouse=True)
def _process_ops(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())
    yield


def _make_inputs(b=2, s=6, h=4, v=7, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(b, s, h, dtype=dtype, requires_grad=True)
    weight = (torch.randn(v, h, dtype=dtype) / 2).requires_grad_(True)
    labels = torch.randint(0, v, (b, s))
    return hidden, weight, labels


def _reference(hidden, weight, labels, normalizer, ignore_index=IGNORE):
    hidden = hidden.detach().clone().requires_grad_(hidden.requires_grad)
    weight = weight.detach().clone().requires_grad_(weight.requires_grad)
    logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight).float()
    loss = F.cross_entropy(
        logits,
        labels.reshape(-1),
        reduction="sum",
        ignore_index=ignore_index,
    ) / normalizer
    return loss, hidden, weight


def _single_loss(hidden, lm_head, labels, *, normalizer, chunk_size):
    loss, _ = chunk_loss(
        [hidden],
        [labels],
        lm_head=lm_head,
        normalizers=[normalizer],
        loss_scales=[1.0],
        chunk_size=chunk_size,
    )
    return loss


@pytest.mark.parametrize("chunk_size", [12, 5, 3, 1, 99])
def test_matches_reference_loss_and_grads(chunk_size):
    hidden, weight, labels = _make_inputs()

    loss = _single_loss(
        hidden, weight, labels, normalizer=3.0, chunk_size=chunk_size
    )
    loss.backward()

    ref_loss, ref_hidden, ref_weight = _reference(hidden, weight, labels, 3.0)
    ref_loss.backward()

    assert torch.allclose(loss, ref_loss, atol=1e-5, rtol=1e-5)
    assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-5, rtol=1e-5)


def test_chunks_flattened_tokens_across_batch(monkeypatch):
    hidden, weight, labels = _make_inputs(b=2, s=3)
    chunk_lengths = []
    original_linear = F.linear

    def record_linear(input, linear_weight, bias=None):
        chunk_lengths.append(input.shape[0])
        return original_linear(input, linear_weight, bias)

    monkeypatch.setattr(F, "linear", record_linear)
    _single_loss(
        hidden, weight, labels, normalizer=1.0, chunk_size=4
    )

    assert chunk_lengths == [4, 2]


def test_autograd_stops_at_linear_output(monkeypatch):
    hidden, weight, labels = _make_inputs(dtype=torch.bfloat16)
    grad_targets = []
    original_grad = torch.autograd.grad

    def record_grad(outputs, inputs, *args, **kwargs):
        inputs = (inputs,) if torch.is_tensor(inputs) else tuple(inputs)
        grad_targets.extend(inputs)
        return original_grad(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    _single_loss(
        hidden, weight, labels, normalizer=1.0, chunk_size=5
    )

    assert [target.shape for target in grad_targets] == [
        torch.Size([5, weight.shape[0]]),
        torch.Size([5, weight.shape[0]]),
        torch.Size([2, weight.shape[0]]),
    ]
    assert all(target.dtype == torch.bfloat16 for target in grad_targets)
    assert all(target is not weight for target in grad_targets)


def test_default_chunk_size_is_1024(monkeypatch):
    hidden, weight, labels = _make_inputs(b=1, s=1025, h=4, v=16)
    chunk_lengths = []
    original_linear = F.linear

    def record_linear(input, linear_weight, bias=None):
        chunk_lengths.append(input.shape[0])
        return original_linear(input, linear_weight, bias)

    monkeypatch.setattr(F, "linear", record_linear)
    chunk_loss(
        [hidden],
        [labels],
        lm_head=weight,
        normalizers=[1.0],
        loss_scales=[1.0],
    )

    assert chunk_lengths == [1024, 1]


def test_multiple_inputs_match_one_combined_reference_gradient():
    hidden_a, weight, labels_a = _make_inputs(seed=1)
    hidden_b, _, labels_b = _make_inputs(s=4, seed=2)
    weight_ref = weight.detach().clone().requires_grad_(True)
    hidden_a_ref = hidden_a.detach().clone().requires_grad_(True)
    hidden_b_ref = hidden_b.detach().clone().requires_grad_(True)
    scales = [1.0, 0.3]
    normalizers = [5.0, 3.0]

    total, components = chunk_loss(
        [hidden_a, hidden_b],
        [labels_a, labels_b],
        lm_head=weight,
        normalizers=normalizers,
        loss_scales=scales,
        chunk_size=3,
    )
    total.backward()

    ref_a = F.cross_entropy(
        F.linear(hidden_a_ref, weight_ref).float().flatten(0, 1),
        labels_a.flatten(),
        reduction="sum",
    ) / normalizers[0]
    ref_b = F.cross_entropy(
        F.linear(hidden_b_ref, weight_ref).float().flatten(0, 1),
        labels_b.flatten(),
        reduction="sum",
    ) / normalizers[1]
    ref_total = ref_a * scales[0] + ref_b * scales[1]
    ref_total.backward()

    assert torch.allclose(total, ref_total, atol=1e-5, rtol=1e-5)
    assert torch.allclose(components, torch.stack([ref_a, ref_b]))
    assert components.requires_grad is False
    assert torch.allclose(hidden_a.grad, hidden_a_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(hidden_b.grad, hidden_b_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=1e-5, rtol=1e-5)


def test_zero_scaled_chunk_stream_is_excluded_from_backward(monkeypatch):
    hidden_main, weight, labels_main = _make_inputs(b=1, s=5, seed=3)
    hidden_mtp, _, labels_mtp = _make_inputs(b=1, s=4, seed=4)
    mtp_backward = []
    grad_calls = 0
    original_grad = torch.autograd.grad

    hidden_mtp.register_hook(mtp_backward.append)

    def record_grad(*args, **kwargs):
        nonlocal grad_calls
        grad_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    total, components = chunk_loss(
        [hidden_main, hidden_mtp],
        [labels_main, labels_mtp],
        lm_head=weight,
        normalizers=[5.0, 4.0],
        loss_scales=[1.0, 0.0],
        chunk_size=3,
    )
    total.backward()

    main_ref, hidden_main_ref, weight_ref = _reference(
        hidden_main, weight, labels_main, 5.0
    )
    mtp_ref, _, _ = _reference(hidden_mtp, weight, labels_mtp, 4.0)
    main_ref.backward()

    assert torch.allclose(total, main_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        components, torch.stack([main_ref, mtp_ref]), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        hidden_main.grad, hidden_main_ref.grad, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(weight.grad, weight_ref.grad, atol=1e-5, rtol=1e-5)
    assert hidden_mtp.grad is None
    assert mtp_backward == []
    assert grad_calls == 2


def test_multiple_logits_inputs_match_combined_reference_gradient():
    torch.manual_seed(4)
    logits_a = torch.randn(2, 5, 7, requires_grad=True)
    logits_b = torch.randn(2, 3, 7, requires_grad=True)
    labels_a = torch.randint(0, 7, (2, 5))
    labels_b = torch.randint(0, 7, (2, 3))
    labels_a[0, 2] = IGNORE
    scales = [1.0, 0.3]
    normalizers = [4.0, 2.0]

    total, components = logits_loss(
        [logits_a, logits_b],
        [labels_a, labels_b],
        normalizers=normalizers,
        loss_scales=scales,
    )
    total.backward()

    logits_a_ref = logits_a.detach().clone().requires_grad_(True)
    logits_b_ref = logits_b.detach().clone().requires_grad_(True)
    ref_a = F.cross_entropy(
        logits_a_ref.flatten(0, 1),
        labels_a.flatten(),
        reduction="sum",
        ignore_index=IGNORE,
    ) / normalizers[0]
    ref_b = F.cross_entropy(
        logits_b_ref.flatten(0, 1),
        labels_b.flatten(),
        reduction="sum",
        ignore_index=IGNORE,
    ) / normalizers[1]
    ref_total = ref_a * scales[0] + ref_b * scales[1]
    ref_total.backward()

    assert torch.allclose(total, ref_total)
    assert torch.allclose(components, torch.stack([ref_a, ref_b]))
    assert components.requires_grad is False
    assert torch.allclose(logits_a.grad, logits_a_ref.grad)
    assert torch.allclose(logits_b.grad, logits_b_ref.grad)


def test_zero_scaled_logits_stream_is_excluded_from_backward():
    torch.manual_seed(5)
    logits_main = torch.randn(1, 5, 7, requires_grad=True)
    logits_mtp = torch.randn(1, 4, 7, requires_grad=True)
    labels_main = torch.randint(0, 7, (1, 5))
    labels_mtp = torch.randint(0, 7, (1, 4))
    mtp_backward = []
    logits_mtp.register_hook(mtp_backward.append)

    total, components = logits_loss(
        [logits_main, logits_mtp],
        [labels_main, labels_mtp],
        normalizers=[5.0, 4.0],
        loss_scales=[1.0, 0.0],
    )
    total.backward()

    logits_main_ref = logits_main.detach().clone().requires_grad_(True)
    main_ref = F.cross_entropy(
        logits_main_ref.flatten(0, 1), labels_main.flatten(), reduction="sum"
    ) / 5.0
    mtp_ref = F.cross_entropy(
        logits_mtp.detach().flatten(0, 1),
        labels_mtp.flatten(),
        reduction="sum",
    ) / 4.0
    main_ref.backward()

    assert torch.allclose(total, main_ref)
    assert torch.allclose(components, torch.stack([main_ref, mtp_ref]))
    assert torch.allclose(logits_main.grad, logits_main_ref.grad)
    assert logits_mtp.grad is None
    assert mtp_backward == []


def test_tensor_normalizer_and_upstream_scale():
    hidden, weight, labels = _make_inputs()
    loss = _single_loss(
        hidden,
        weight,
        labels,
        normalizer=torch.tensor(5.0),
        chunk_size=2,
    )
    (loss * 3.0).backward()

    ref_loss, ref_hidden, ref_weight = _reference(hidden, weight, labels, 5.0)
    (ref_loss * 3.0).backward()

    assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-5, rtol=1e-5)


def test_fully_masked_chunk_contributes_zero_gradient():
    hidden, weight, labels = _make_inputs(b=1, s=6)
    labels[:, 2:4] = IGNORE

    loss = _single_loss(
        hidden, weight, labels, normalizer=1.0, chunk_size=2
    )
    loss.backward()

    ref_loss, ref_hidden, ref_weight = _reference(hidden, weight, labels, 1.0)
    ref_loss.backward()
    assert torch.count_nonzero(hidden.grad[:, 2:4]) == 0
    assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-5, rtol=1e-5)


def test_all_labels_masked_gives_zero_loss_and_grads():
    hidden, weight, labels = _make_inputs(b=1, s=4)
    labels[:] = IGNORE

    loss = _single_loss(
        hidden, weight, labels, normalizer=1.0, chunk_size=2
    )
    loss.backward()

    assert loss.detach().item() == 0.0
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


def test_frozen_inputs_only_compute_requested_grads():
    hidden, weight, labels = _make_inputs()
    frozen_weight = weight.detach()
    loss = _single_loss(
        hidden, frozen_weight, labels, normalizer=1.0, chunk_size=2
    )
    loss.backward()
    assert frozen_weight.grad is None
    assert hidden.grad is not None

    hidden, weight, labels = _make_inputs()
    frozen_hidden = hidden.detach()
    loss = _single_loss(
        frozen_hidden, weight, labels, normalizer=1.0, chunk_size=2
    )
    loss.backward()
    assert frozen_hidden.grad is None
    assert weight.grad is not None


def test_bfloat16_grad_dtypes_match_inputs():
    hidden, weight, labels = _make_inputs(dtype=torch.bfloat16)

    loss = _single_loss(
        hidden, weight, labels, normalizer=1.0, chunk_size=2
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert hidden.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16


def test_biasless_linear_module_is_supported():
    hidden, _, labels = _make_inputs()
    lm_head = nn.Linear(hidden.shape[-1], 7, bias=False)

    _single_loss(
        hidden, lm_head, labels, normalizer=1.0, chunk_size=2
    ).backward()

    assert lm_head.weight.grad is not None


def test_rejects_unsupported_inputs():
    hidden, weight, labels = _make_inputs()
    with pytest.raises(NotImplementedError, match="bias"):
        _single_loss(
            hidden,
            nn.Linear(hidden.shape[-1], 7, bias=True),
            labels,
            normalizer=1.0,
            chunk_size=2,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        _single_loss(
            hidden,
            torch.randn(7, 2),
            labels,
            normalizer=1.0,
            chunk_size=2,
        )

    with pytest.raises(ValueError, match="labels"):
        _single_loss(
            hidden,
            weight,
            labels[:, :-1],
            normalizer=1.0,
            chunk_size=2,
        )
    with pytest.raises(ValueError, match="normalizer"):
        _single_loss(
            hidden, weight, labels, normalizer=0.0, chunk_size=2
        )
    with pytest.raises(ValueError, match="chunk_size"):
        _single_loss(
            hidden, weight, labels, normalizer=1.0, chunk_size=0
        )
    with pytest.raises(ValueError, match=r"\[B, S, H\]"):
        _single_loss(
            hidden.flatten(0, 1),
            weight,
            labels,
            normalizer=1.0,
            chunk_size=2,
        )


def test_rejects_noncontiguous_hidden_states():
    hidden, weight, labels = _make_inputs()

    with pytest.raises(ValueError, match="must be contiguous"):
        _single_loss(
            hidden.transpose(0, 1),
            weight,
            labels.transpose(0, 1),
            normalizer=1.0,
            chunk_size=2,
        )


def test_requires_matching_nonempty_inputs():
    hidden, weight, labels = _make_inputs()
    with pytest.raises(ValueError, match="at least one"):
        chunk_loss(
            [], [], lm_head=weight, normalizers=[], loss_scales=[], chunk_size=2
        )
    with pytest.raises(ValueError, match="same length"):
        chunk_loss(
            [hidden],
            [labels],
            lm_head=weight,
            normalizers=[1.0],
            loss_scales=[],
            chunk_size=2,
        )
