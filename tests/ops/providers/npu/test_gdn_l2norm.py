"""Native L2Norm values, gradients, tails and packed-length cache reuse."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

from fla_npu.ops.triton import l2norm_bwd as upstream_bwd
from fla_npu.ops.triton import l2norm_fwd as upstream_fwd
from triton.runtime.jit import JITFunction

from torchturbo.ops.providers.npu._fla_gdn import l2norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,dim", [(640, 128), (17, 128), (639, 96)])
def test_values_and_gradients(dtype, rows, dim):
    x = torch.randn(rows, dim, device="npu", dtype=dtype)
    dy = torch.randn_like(x)
    y, rstd = l2norm.l2norm_fwd(x)
    dx = l2norm.l2norm_bwd(y, rstd, dy)
    expected_rstd = torch.rsqrt(x.float().square().sum(-1) + 1e-6)
    expected_y = (x.float() * expected_rstd[:, None]).to(dtype)
    expected_dx = (
        dy.float() * rstd[:, None]
        - (dy.float() * y.float()).sum(-1)[:, None] * y.float() * rstd[:, None]
    ).to(dtype)
    tolerance = 0.004 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(y, expected_y, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(dx, expected_dx, rtol=tolerance, atol=tolerance)
    if rows % 16 == 0:
        old_y, old_rstd = upstream_fwd(x)
        old_dx = upstream_bwd(old_y, old_rstd, dy)
        torch.testing.assert_close(y, old_y, rtol=0, atol=0)
        torch.testing.assert_close(rstd, old_rstd, rtol=0, atol=0)
        torch.testing.assert_close(dx, old_dx, rtol=0, atol=0)


def test_packed_lengths_reuse_compiled_kernels(monkeypatch):
    def run(rows):
        x = torch.randn(rows, 128, device="npu", dtype=torch.bfloat16)
        y, rstd = l2norm.l2norm_fwd(x)
        l2norm.l2norm_bwd(y, rstd, x)
        torch.npu.synchronize()

    run(32)
    misses = []

    def cache_miss(**kwargs):
        if kwargs["fn"].module == l2norm.__name__:
            misses.append(kwargs["fn"].name)
        return False

    monkeypatch.setattr(JITFunction, "cache_hook", cache_miss)
    for rows in (17, 640, 786000, 786384):
        run(rows)
    assert misses == []
