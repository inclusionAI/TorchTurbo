"""GDN L2 normalization with a runtime packed sequence length."""

import torch
import triton
import triton.language as tl
from fla_npu.ops.triton import l2norm_bwd as _upstream_bwd
from fla_npu.ops.triton import l2norm_fwd as _upstream_fwd


@triton.jit(do_not_specialize=["T"])
def _l2norm_fwd_kernel(
    x, y, rstd, eps, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, bt_size
):
    block = tl.program_id(0)
    for offset in range(bt_size):
        row = (block * bt_size + offset) * BT
        if row < T:
            p_x = tl.make_block_ptr(x, (T, D), (D, 1), (row, 0), (BT, BD), (1, 0))
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (row, 0), (BT, BD), (1, 0))
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (row,), (BT,), (0,))
            b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
            b_y = b_x * b_rstd[:, None]
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_rstd, b_rstd, boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def _l2norm_bwd_kernel(
    y, rstd, dy, dx, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, bt_size
):
    block = tl.program_id(0)
    num_tiles = tl.cdiv(T, BT)
    base = num_tiles // bt_size
    remainder = num_tiles % bt_size
    num_tasks = base + (block < remainder)
    start = block * base + tl.minimum(block, remainder)
    for task in range(num_tasks):
        row = (start + task) * BT
        p_y = tl.make_block_ptr(y, (T, D), (D, 1), (row, 0), (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (row, 0), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (row, 0), (BT, BD), (1, 0))
        p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (row,), (BT,), (0,))
        b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        b_rstd = tl.load(p_rstd, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )
        b_dx = (
            b_dy * b_rstd[:, None]
            - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
        )
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows without compiling a new kernel for each packed length."""
    if x.shape[-1] > 512:
        return _upstream_fwd(x, eps, output_dtype)
    shape = x.shape
    x = x.view(-1, shape[-1])
    rows, dim = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    _l2norm_fwd_kernel[(triton.cdiv(rows, 16 * 32),)](
        x,
        y,
        rstd,
        eps,
        rows,
        dim,
        triton.next_power_of_2(dim),
        16,
        32,
        num_warps=4,
    )
    return y.view(shape), rstd.view(shape[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiate normalized rows, including the final partial tile."""
    if y.shape[-1] > 512:
        return _upstream_bwd(y, rstd, dy, eps)
    shape = y.shape
    y = y.view(-1, shape[-1])
    dy = dy.view_as(y)
    rows, dim = y.shape
    dx = torch.empty_like(y)
    _l2norm_bwd_kernel[(40,)](
        y,
        rstd,
        dy,
        dx,
        rows,
        dim,
        triton.next_power_of_2(dim),
        16,
        40,
        num_warps=4,
    )
    return dx.view(shape)
