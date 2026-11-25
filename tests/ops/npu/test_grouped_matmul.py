from __future__ import annotations

import importlib
from unittest.mock import Mock, patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from torchturbo.models.modules.moe.grouped_linear import grouped_linear
from torchturbo.ops import get_or_create_ops
from torchturbo.ops.providers.npu.grouped_matmul import TorchNpuGroupedMatmul
from torchturbo.ops.providers.npu.matmul_add import TorchNpuMatmulAdd
from torchturbo.runtime.npu import NpuRuntime


@pytest.fixture
def npu_dispatch(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    return get_or_create_ops(device_runtime=NpuRuntime())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("impl", [None, "torch_npu_grouped_addmm"], ids=["auto", "native-addmm"])
def test_grouped_wgrad_adds_to_existing_accumulator(npu_dispatch, dtype, impl):
    counts = torch.tensor([2, 0, 0, 5, 0], device="npu")
    x = torch.ones((7, 64), dtype=dtype, device="npu")
    dy = torch.full((7, 48), 2, dtype=dtype, device="npu")
    accumulator = torch.full((5, 64, 48), 0.25, dtype=torch.float32, device="npu")
    expected = accumulator.cpu() + 2 * counts.cpu().view(-1, 1, 1)
    npu_dispatch.grouped_matmul_add(accumulator, x, dy, counts, impl=impl)
    torch.testing.assert_close(accumulator.cpu(), expected, atol=0, rtol=0)


def _active_weights(native, replica, counts, active):
    if active == "native":
        return native, counts[:len(native)]
    if active == "replica":
        return replica, counts[len(native):]
    return (native, replica), counts


@pytest.mark.parametrize(
    "counts,active",
    [([0, 9, 0, 0, 13, 0], "both"), ([0, 0, 0, 0, 13, 0], "replica"), ([0] * 6, "native")],
    ids=["mixed", "empty-owner", "empty"],
)
def test_grouped_linear_stores_full_owner_and_replica_wgrad(npu_dispatch, counts, active):
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(sum(counts), 64, generator=generator).bfloat16()
    dy = torch.randn(sum(counts), 48, generator=generator).bfloat16()
    native = torch.ones(3, 64, 48, device="npu", dtype=torch.bfloat16, requires_grad=True)
    replica = torch.ones_like(native, requires_grad=True)
    store = Mock()
    full_counts = torch.tensor(counts)
    weight, forward_counts = _active_weights(native, replica, full_counts, active)
    output = grouped_linear(
        x.npu(), forward_counts, weight=weight, wgrad_store=store, wgrad_counts=full_counts,
    )
    with patch.object(
        torch_npu, "npu_grouped_matmul_add_", wraps=torch_npu.npu_grouped_matmul_add_,
    ) as add:
        output.backward(dy.npu())

    assert add.call_count == int(bool(sum(counts)))
    store.assert_called_once()
    (wgrad,) = store.call_args.args
    assert wgrad.is_contiguous()
    actual = wgrad.cpu()
    expected = torch.zeros(6, 64, 48)
    offset = 0
    for expert, count in enumerate(counts):
        end = offset + count
        expected[expert] = x[offset:end].float().T @ dy[offset:end].float()
        offset = end
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-5)
    if sum(counts):
        assert (actual - expected.bfloat16().float()).abs().max() > 1e-2


@pytest.mark.parametrize(
    "counts,active",
    [([3, 0, 5, 0, 7], "both"), ([0, 0, 0, 6, 0], "replica"), ([0] * 5, "native")],
)
def test_grouped_linear_reads_active_weight_banks_in_one_gemm(npu_dispatch, counts, active):
    generator = torch.Generator().manual_seed(271)
    x = torch.randn(sum(counts), 64, generator=generator).bfloat16()
    dy = torch.randn(sum(counts), 48, generator=generator).bfloat16()
    weights = torch.randn(5, 64, 48, generator=generator).bfloat16()
    native = weights[:3].npu().requires_grad_()
    # A slice of a second allocation models an FC section inside the SHMEM arena.
    replica = torch.cat((weights[:1], weights[3:])).npu()[1:]
    device_x = x.npu().requires_grad_()
    store = Mock()
    full_counts = torch.tensor(counts)
    weight, forward_counts = _active_weights(native, replica, full_counts, active)
    with patch.object(torch_npu, "npu_grouped_matmul", wraps=torch_npu.npu_grouped_matmul) as gmm:
        y = grouped_linear(
            device_x, forward_counts, weight=weight, wgrad_store=store, wgrad_counts=full_counts,
        )
        y.backward(dy.npu())

    assert gmm.call_count == (2 if sum(counts) else 0)
    if active == "both":
        pointers = [w.data_ptr() for bank in (native, replica) for w in bank.unbind()]
        for call in gmm.call_args_list:
            assert [w.data_ptr() for w in call.args[1]] == pointers
    elif active == "replica":
        for call in gmm.call_args_list:
            assert [w.data_ptr() for w in call.args[1]] == [replica.data_ptr()]
    expected_y = torch.cat([xx.float() @ w.float() for xx, w in zip(x.split(counts), weights)])
    expected_dx = torch.cat([dd.float() @ w.float().T for dd, w in zip(dy.split(counts), weights)])
    expected_dw = torch.stack([
        xx.float().T @ dd.float() for xx, dd in zip(x.split(counts), dy.split(counts))
    ])
    torch.testing.assert_close(y.cpu(), expected_y.bfloat16())
    torch.testing.assert_close(device_x.grad.cpu(), expected_dx.bfloat16())
    store.assert_called_once()
    (wgrad,) = store.call_args.args
    torch.testing.assert_close(wgrad.cpu(), expected_dw)


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


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("counts", [[0, 3, 0, 6], [0, 0, 0, 0]])
def test_grouped_matmul_with_separate_weight_banks(npu_dispatch, trans_b, counts):
    generator = torch.Generator().manual_seed(51)
    x = torch.randn(sum(counts), 32 if trans_b else 64, generator=generator).bfloat16()
    weights = torch.randn(4, 64, 32, generator=generator).bfloat16()
    device_x = x.npu().requires_grad_()
    banks = (weights[:2].npu().requires_grad_(), weights[2:].npu().requires_grad_())
    output = npu_dispatch.grouped_matmul(device_x, banks, torch.tensor(counts), trans_b=trans_b)
    expected = _reference_forward(x.float(), weights.float(), counts, trans_b)
    torch.testing.assert_close(output.cpu(), expected.bfloat16())
    output.backward(torch.ones_like(output))
    expected_dx, expected_dw = _reference_backward(x.float(), weights.float(), counts, trans_b)
    torch.testing.assert_close(device_x.grad.cpu(), expected_dx.bfloat16())
    torch.testing.assert_close(torch.cat([bank.grad for bank in banks]).cpu(), expected_dw.bfloat16())


def test_grouped_matmul_over_npu_weight_list_limit(npu_dispatch):
    x = torch.ones(131, 32, dtype=torch.bfloat16, device="npu")
    eye = torch.eye(32).bfloat16().npu()
    banks = (eye.repeat(128, 1, 1), (2 * eye).repeat(3, 1, 1))
    actual = npu_dispatch.grouped_matmul(x, banks, torch.ones(131, dtype=torch.int64))
    torch.testing.assert_close(actual[:128], x[:128])
    torch.testing.assert_close(actual[128:], 2 * x[128:])


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
