"""Compare SparseFlashMla precision for padded and ragged TND inputs."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU is required for SparseFlashMla precision tests", allow_module_level=True)


@pytest.fixture(scope="module")
def npu_ops():
    from torchturbo.ops import OpDispatcher
    from torchturbo.ops.registry import registry
    from torchturbo.runtime.npu import NpuRuntime

    runtime = NpuRuntime()
    runtime.initialize()
    return OpDispatcher(device_runtime=runtime, registry=registry)


HEAD_DIM = 512
NUM_Q_HEADS = 1
RATIO = 128
DTYPE = torch.float16


def _cu_seqlens(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device="npu")


def _run_sfa(ops, q, ori_kv, cmp_kv, q_lens, ori_lens, cmp_lens, residuals, sinks):
    return ops.sparse_flash_mla(
        query=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=None,
        cmp_residual_kv=torch.tensor(residuals, dtype=torch.int32, device="npu"),
        sinks=sinks,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        cmp_ratio=RATIO,
        ori_win_left=RATIO - 1,
        ori_win_right=0,
        cu_seq_lens_q=_cu_seqlens(q_lens),
        cu_seq_lens_ori_kv=_cu_seqlens(ori_lens),
        cu_seq_lens_cmp_kv=_cu_seqlens(cmp_lens),
    )


def _make_cases():
    generator = torch.Generator().manual_seed(20260922)

    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=DTYPE).to("npu")

    # Two logical requests: one ends at 127 tokens, the other at 129 tokens.
    q_a, q_b = rand((127, NUM_Q_HEADS, HEAD_DIM)), rand((129, NUM_Q_HEADS, HEAD_DIM))
    ori_a, ori_b = rand((127, 1, HEAD_DIM)), rand((129, 1, HEAD_DIM))
    cmp_a, cmp_b, cmp_b_tail = (
        rand((1, 1, HEAD_DIM)),
        rand((1, 1, HEAD_DIM)),
        rand((1, 1, HEAD_DIM)),
    )
    zero_q_a = torch.zeros((1, NUM_Q_HEADS, HEAD_DIM), dtype=DTYPE, device="npu")
    zero_q_b = torch.zeros((127, NUM_Q_HEADS, HEAD_DIM), dtype=DTYPE, device="npu")
    zero_ori_a = torch.zeros((1, 1, HEAD_DIM), dtype=DTYPE, device="npu")
    zero_ori_b = torch.zeros((127, 1, HEAD_DIM), dtype=DTYPE, device="npu")

    ragged = {
        "q": torch.cat((q_a, q_b), dim=0).detach().requires_grad_(),
        "ori_kv": torch.cat((ori_a, ori_b), dim=0).detach().requires_grad_(),
        "cmp_kv": cmp_b.detach().requires_grad_(),
        "q_lens": [127, 129],
        "ori_lens": [127, 129],
        "cmp_lens": [0, 1],
        "residuals": [127, 1],
    }
    padded = {
        "q": torch.cat((q_a, zero_q_a, q_b, zero_q_b), dim=0).detach().requires_grad_(),
        "ori_kv": torch.cat((ori_a, zero_ori_a, ori_b, zero_ori_b), dim=0)
        .detach()
        .requires_grad_(),
        "cmp_kv": torch.cat((cmp_a, cmp_b, cmp_b_tail), dim=0).detach().requires_grad_(),
        "q_lens": [128, 256],
        "ori_lens": [128, 256],
        "cmp_lens": [1, 2],
        "residuals": [0, 0],
    }
    return ragged, padded, rand((127 + 129, NUM_Q_HEADS, HEAD_DIM))


def test_sfa_hca_padded_and_ragged_precision(npu_ops):
    ragged, padded, grad = _make_cases()
    ragged_sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device="npu", requires_grad=True)
    padded_sinks = ragged_sinks.detach().clone().requires_grad_()

    ragged_out = _run_sfa(npu_ops, **ragged, sinks=ragged_sinks)
    padded_out = _run_sfa(npu_ops, **padded, sinks=padded_sinks)
    torch.npu.synchronize()

    padded_actual = torch.cat((padded_out[:127], padded_out[128 : 128 + 129]), dim=0)
    torch.testing.assert_close(ragged_out, padded_actual, atol=5e-2, rtol=5e-2)

    padded_grad = torch.cat(
        (
            grad[:127],
            torch.zeros((1, NUM_Q_HEADS, HEAD_DIM), dtype=DTYPE, device="npu"),
            grad[127:],
            torch.zeros((127, NUM_Q_HEADS, HEAD_DIM), dtype=DTYPE, device="npu"),
        ),
        dim=0,
    )
    (ragged_out * grad).sum().backward()
    (padded_out * padded_grad).sum().backward()
    torch.npu.synchronize()

    padded_q_grad = torch.cat((padded["q"].grad[:127], padded["q"].grad[128 : 128 + 129]), dim=0)
    padded_ori_grad = torch.cat(
        (padded["ori_kv"].grad[:127], padded["ori_kv"].grad[128 : 128 + 129]), dim=0
    )
    torch.testing.assert_close(ragged["q"].grad, padded_q_grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(ragged["ori_kv"].grad, padded_ori_grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(ragged["cmp_kv"].grad[0], padded["cmp_kv"].grad[1], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(ragged_sinks.grad, padded_sinks.grad, atol=5e-2, rtol=5e-2)
