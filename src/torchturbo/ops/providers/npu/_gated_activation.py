"""Fused pointwise kernels; imported only when the NPU provider is used."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
from torch.autograd.function import once_differentiable


@lru_cache
def _vector_cores(device_index: int) -> int:
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(device_index)["num_vectorcore"]


@triton.jit
def _upper_limit(x, LIMIT: tl.constexpr):
    # Match tl.minimum's default NaN behavior explicitly. An ordered
    # comparison alone is insufficient on this compiler.
    return tl.where(libdevice.isnan(x), LIMIT, tl.where(x > LIMIT, LIMIT, x))


@triton.jit
def _clamp_up(x, LIMIT: tl.constexpr):
    clipped = tl.where(x > LIMIT, LIMIT, tl.where(x < -LIMIT, -LIMIT, x))
    # tl.maximum(x, -LIMIT) selects -LIMIT when x is NaN.
    return tl.where(libdevice.isnan(x), -LIMIT, clipped)


@triton.jit
def _activation(x, GELU: tl.constexpr):
    if GELU:
        # torch_npu F.gelu uses the tanh approximation, including backward.
        u = 0.7978845608028654 * x * (1.0 + 0.044715 * x * x)
        tanh_u = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
        return 0.5 * x * (1.0 + tanh_u)
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _activation_grad(x, GELU: tl.constexpr):
    if GELU:
        u = 0.7978845608028654 * x * (1.0 + 0.044715 * x * x)
        tanh_u = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
        du = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
        return 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u * tanh_u) * du
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


@triton.jit
def _forward(X, Y, N, D: tl.constexpr, GELU: tl.constexpr,
             LIMIT: tl.constexpr, AFTER: tl.constexpr, BLOCK: tl.constexpr,
             ROWS: tl.constexpr = 4, TILES_PER_CORE=1):
    num_col_tiles: tl.constexpr = tl.cdiv(D, BLOCK)
    num_tiles = tl.cdiv(N // D, ROWS) * num_col_tiles
    start = tl.program_id(0) * TILES_PER_CORE
    end = tl.minimum(start + TILES_PER_CORE, num_tiles)
    for tile in range(start, end):
        row = (tile // num_col_tiles) * ROWS + tl.arange(0, ROWS)
        col = (tile - (tile // num_col_tiles) * num_col_tiles) * BLOCK + tl.arange(0, BLOCK)
        i = row[:, None] * D + col[None, :]
        offset = row[:, None] * (2 * D) + col[None, :]
        # Retain integer masks: floating index masks miscompile narrow rows.
        mask = (row[:, None] < N // D) & (col[None, :] < D)
        gate = tl.load(X + offset, mask, 0).to(tl.float32)
        up = tl.load(X + offset + D, mask, 0).to(tl.float32)
        if LIMIT > 0:
            up = _clamp_up(up, LIMIT)
            if AFTER:
                activated = _upper_limit(_activation(gate, GELU), LIMIT)
            else:
                activated = _activation(_upper_limit(gate, LIMIT), GELU)
        else:
            # Without limits the native activation rounds before multiplication.
            activated = _activation(gate, GELU).to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + i, activated * up, mask)


@triton.jit
def _backward(X, DY, DX, N, D: tl.constexpr, GELU: tl.constexpr,
              LIMIT: tl.constexpr, AFTER: tl.constexpr, BLOCK: tl.constexpr,
              ROWS: tl.constexpr = 4, TILES_PER_CORE=1):
    num_col_tiles: tl.constexpr = tl.cdiv(D, BLOCK)
    num_tiles = tl.cdiv(N // D, ROWS) * num_col_tiles
    start = tl.program_id(0) * TILES_PER_CORE
    end = tl.minimum(start + TILES_PER_CORE, num_tiles)
    for tile in range(start, end):
        row = (tile // num_col_tiles) * ROWS + tl.arange(0, ROWS)
        col = (tile - (tile // num_col_tiles) * num_col_tiles) * BLOCK + tl.arange(0, BLOCK)
        i = row[:, None] * D + col[None, :]
        offset = row[:, None] * (2 * D) + col[None, :]
        # Retain integer masks: floating index masks miscompile narrow rows.
        mask = (row[:, None] < N // D) & (col[None, :] < D)
        gate = tl.load(X + offset, mask, 0).to(tl.float32)
        up = tl.load(X + offset + D, mask, 0).to(tl.float32)
        dy = tl.load(DY + i, mask, 0).to(tl.float32)
        if LIMIT > 0:
            clipped_up = _clamp_up(up, LIMIT)
            if AFTER:
                raw_activation = _activation(gate, GELU)
                activated = _upper_limit(raw_activation, LIMIT)
                derivative = tl.where(raw_activation <= LIMIT, _activation_grad(gate, GELU), 0.0)
            else:
                clipped_gate = _upper_limit(gate, LIMIT)
                activated = _activation(clipped_gate, GELU)
                derivative = tl.where(gate <= LIMIT, _activation_grad(clipped_gate, GELU), 0.0)
            dg = dy * clipped_up * derivative
            du = tl.where((up >= -LIMIT) & (up <= LIMIT), dy * activated, 0.0)
        else:
            activated = _activation(gate, GELU).to(X.dtype.element_ty).to(tl.float32)
            dactivated = (dy * up).to(X.dtype.element_ty).to(tl.float32)
            dg = dactivated * _activation_grad(gate, GELU)
            du = dy * activated
        tl.store(DX + offset, dg, mask)
        tl.store(DX + offset + D, du, mask)


class FusedGatedActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gelu: bool, limit: float,
                after: bool) -> torch.Tensor:
        d = x.shape[-1] // 2
        y = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        n = y.numel()
        block = min(triton.next_power_of_2(d), 512)
        # SiLU's smaller live buffer set permits a larger tile in Ascend UB.
        rows = 8 if not gelu and d >= 128 else 4
        num_tiles = triton.cdiv(n // d, rows) * triton.cdiv(d, block)
        # Keep core boundaries aligned even for narrow, unaligned rows.
        num_cores = _vector_cores(x.device.index)
        tiles_per_core = triton.cdiv(triton.cdiv(num_tiles, num_cores), 4) * 4
        programs = triton.cdiv(num_tiles, tiles_per_core)
        _forward[(programs,)](
            x, y, n, d, gelu, limit, after, block, rows, tiles_per_core,
            enable_auto_bind_sub_block=False,
        )
        ctx.save_for_backward(x)
        ctx.gelu, ctx.limit, ctx.after = gelu, limit, after
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        n = grad_output.numel()
        d = x.shape[-1] // 2
        block = min(triton.next_power_of_2(d), 512)
        rows = 8 if not ctx.gelu and d >= 128 else 4
        num_tiles = triton.cdiv(n // d, rows) * triton.cdiv(d, block)
        # Keep core boundaries aligned even for narrow, unaligned rows.
        num_cores = _vector_cores(x.device.index)
        tiles_per_core = triton.cdiv(triton.cdiv(num_tiles, num_cores), 4) * 4
        programs = triton.cdiv(num_tiles, tiles_per_core)
        _backward[(programs,)](
            x, grad_output.contiguous(), dx, n, x.shape[-1] // 2,
            ctx.gelu, ctx.limit, ctx.after, block, rows, tiles_per_core,
            enable_auto_bind_sub_block=False,
        )
        return dx, None, None, None
