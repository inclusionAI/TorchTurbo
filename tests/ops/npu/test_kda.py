"""Numerical alignment of the ``ascendc_kda`` provider against the fla
triton baseline in bailing_hybrid's production config (dense and packed varlen,
``chunk_size=64``, in-kernel l2norm/gate/sigmoid), on NPU hosts with
``fla`` and ``fla_npu`` installed.

Bound to fla 0.6.0 (main @ 954438d1) and fla_npu v26.10.0-feature.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fla_npu")

from torchturbo.ops import OpPolicy, get_or_create_ops  # noqa: E402
from torchturbo.runtime import TurboDevice  # noqa: E402
from torchturbo.runtime.npu import NpuRuntime  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_ops_singleton():
    # get_or_create_ops is a process-wide singleton; drop whatever the CPU
    # dispatch tests installed so this module can build the NPU dispatcher.
    # ``import torchturbo.ops.ops as x`` would bind the module-level ``ops``
    # proxy instead (it shadows the submodule attribute), so go through
    # sys.modules for the real module.
    import sys

    _ops_module = sys.modules["torchturbo.ops.ops"]
    _ops_module._OPS = None
    try:
        yield
    finally:
        _ops_module._OPS = None


def _make_inputs(device, dtype, num_sequences, batch=1, heads=2, head_dim=128):
    torch.manual_seed(0)
    seq_len = 128 if num_sequences is None else 64 * num_sequences
    # Flat [B, T, H*D] Q/K/V; the provider splits heads from g's shape.
    q = torch.randn(batch, seq_len, heads * head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads * head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads * head_dim, device=device, dtype=dtype)
    g = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads * head_dim, device=device, dtype=torch.float32)
    cu_seqlens = None
    if num_sequences is not None:
        cu_seqlens = torch.arange(0, seq_len + 1, 64, dtype=torch.int32, device=device)
    return q, k, v, g, beta, A_log, dt_bias, cu_seqlens


def _run_and_backward(impl, device, dtype, num_sequences):
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = _make_inputs(device, dtype, num_sequences)
    for t in (q, k, v, g, beta, A_log, dt_bias):
        t.requires_grad_(True)

    ops = get_or_create_ops(
        device_runtime=NpuRuntime(),
        default_policy=OpPolicy.AUTO,
    )

    o, _ = ops.kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        impl=impl,
    )
    o.float().sum().backward()

    return (
        o.detach(),
        q.grad.detach(), k.grad.detach(), v.grad.detach(),
        g.grad.detach(), beta.grad.detach(),
        A_log.grad.detach(), dt_bias.grad.detach(),
    )


# The AscendC kernels are bf16-only (fp32 inputs fail inside the operator
# with a CANN error); production also runs bailing in bf16, so the numeric
# alignment is a bf16 comparison.
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_sequences", [None, 2, 1025], ids=["dense", "packed", "packed_over_1024"])
def test_ascendc_chunk_kda_matches_fl_triton(dtype, num_sequences):
    device = "npu"
    if not torch.npu.is_available():  # pragma: no cover - guard for hosts
        pytest.skip("NPU device not available")

    baseline = _run_and_backward("fla_kda", device, dtype, num_sequences)
    ascendc = _run_and_backward("ascendc_kda", device, dtype, num_sequences)

    tolerance = 1e-3 if dtype == torch.float32 else 2e-2
    names = ("o", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "dt_bias")
    for name, expected, actual in zip(names, baseline, ascendc):
        torch.testing.assert_close(
            actual.float(), expected.float(),
            rtol=tolerance, atol=tolerance,
            msg=lambda m: f"{name}: {m}",
        )
