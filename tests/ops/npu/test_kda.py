"""AscendC kda numerical alignment against the fla triton baseline.

Runs only on an NPU host with ``fla_npu`` and ``fla`` installed.  The
shapes mirror the production configuration exercised by bailing_hybrid:
packed varlen (``cu_seqlens``), ``chunk_size=64``, in-kernel l2norm and
gate, ``disable_recompute=True``, head-major Q/K/V.
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


def _make_inputs(device, dtype, batch=1, heads=2, head_dim=128, seq_len=128):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    # g/beta are sequence-major per the operator contract.
    g = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads * head_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
    return q, k, v, g, beta, A_log, dt_bias, cu_seqlens


def _run_and_backward(impl, device, dtype):
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = _make_inputs(device, dtype)
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
        disable_recompute=True, head_major_qkv=True,
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
def test_ascendc_chunk_kda_matches_fl_triton(dtype):
    device = "npu"
    if not torch.npu.is_available():  # pragma: no cover - guard for hosts
        pytest.skip("NPU device not available")

    baseline = _run_and_backward("fla_kda", device, dtype)
    ascendc = _run_and_backward("ascendc_kda", device, dtype)

    tolerance = 1e-3 if dtype == torch.float32 else 2e-2
    names = ("o", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "dt_bias")
    for name, expected, actual in zip(names, baseline, ascendc):
        torch.testing.assert_close(
            actual.float(), expected.float(),
            rtol=tolerance, atol=tolerance,
            msg=lambda m: f"{name}: {m}",
        )
