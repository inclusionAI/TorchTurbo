#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
CSA2 Compressor Backward (DeepSeek V4.1-Flash) — source aligned.

对应前向: CSA2_compressor_fwd_impl.py; 对应 golden: tests/ops/deepseek_v4_1/test_CSA2_compressor_bwd.py

分支 (与前向一一对应; kv_state/score_state 为不可微 cache, 只回传当前调用产生的梯度):
    ratio == 1:  out = norm(wkv(x))                → (dx, dwkv, None, dweight)
    ratio == 2, start_pos == 0 (prefill):
        完整组:  dlatent = rms_norm_backward(grad_out, bf16(latent), weight)
                 d_kv = dlatent * alpha ; d_score = softmax_backward(dlatent * kv, alpha)
        余数 token: 无输出 → 梯度 0
    ratio == 2, start_pos != 0 (decode):
        latent 由 state 池化, 仅当前 token 写入的槽位可微
    公共: dx = d_kv @ wkv + d_score @ wgate ; dwkv = Σ d_kv^T @ x ; dwgate = Σ d_score^T @ x
"""
import collections
import pypto
import torch
from torch._dynamo import allow_in_graph

RATIO = 2

# 静态分块参数 (kernel 内 ceil 分块 + 尾块 valid_shape; wrapper 直传精确形状)
BWD_G = 128                  # ratio>1 组块 (每块 G 个 group)
BWD_RB = BWD_G * RATIO       # ratio>1 组块行数 (256)
BWD_MB = 512                 # ratio>1 dx 行块
BWD_KB = 2048                # ratio>1 dwkv/dwgate K 块

R1_MB = 256                  # ratio1 行块
R1_KB = 2048                 # ratio1 dwkv K 块

CSA2CompressorBwdArgs = collections.namedtuple('CSA2CompressorBwdArgs', [
    'grad_out', 'x', 'wkv', 'wgate', 'weight', 'kv_state', 'score_state', 'start_pos',
    'cu_seqlens', 'ratio', 'eps',
], defaults=(None, 2, 1e-20))   # cu_seqlens (BSH 可省略) 位于标量参数 ratio/eps 之前


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "csa2_compressor_bwd(Tensor grad_out, Tensor x, Tensor wkv, Tensor? wgate, Tensor weight, \
    Tensor? kv_state, Tensor? score_state, Tensor start_pos, Tensor? cu_seqlens, \
    int ratio, float eps) -> (Tensor, Tensor, Tensor?, Tensor)"
)


@torch.library.impl(pyptolib, "csa2_compressor_bwd", "Meta")
def csa2_compressor_bwd(grad_out, x, wkv, wgate, weight, kv_state, score_state, start_pos,
                        cu_seqlens, ratio, eps):
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    dwkv = torch.empty(wkv.shape, dtype=wkv.dtype, device=wkv.device)
    dweight = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
    if ratio == 1:
        return dx, dwkv, None, dweight
    dwgate = torch.empty(wgate.shape, dtype=wgate.dtype, device=wgate.device)
    return dx, dwkv, dwgate, dweight


try:
    @torch.library.impl(pyptolib, "csa2_compressor_bwd", "NPU")
    def csa2_compressor_bwd(grad_out, x, wkv, wgate, weight, kv_state, score_state, start_pos,
                            cu_seqlens, ratio, eps):
        args = CSA2CompressorBwdArgs(
            grad_out=grad_out, x=x, wkv=wkv, wgate=wgate, weight=weight,
            kv_state=kv_state, score_state=score_state, start_pos=start_pos, ratio=ratio,
            eps=eps, cu_seqlens=cu_seqlens,
        )
        return npu_csa2_compressor_bwd(args)
except Exception as e:
    if "could not parse dispatch key: NPU" in str(e):
        print(f"Skip: torchair not installed, skip NPU registration for operator 'csa2_compressor_bwd'")
    else:
        print(f"Skip: Unexpected error : {e}")


def csa2_compressor_bwd_pypto(args: CSA2CompressorBwdArgs):
    return torch.ops.pypto.csa2_compressor_bwd(
        args.grad_out,
        args.x,
        args.wkv,
        args.wgate,
        args.weight,
        args.kv_state,
        args.score_state,
        args.start_pos,
        args.cu_seqlens,
        args.ratio,
        args.eps,
    )


def th_out_base(cu_seqlens, ratio):
    """TH 模式 per-batch 输出行前缀 (exclusive): out_base[b] = Σ_{i<b} L_i // ratio.

    与正向 impl 中同名 helper 一致 (两 impl 保持独立, 不跨文件引用)。
    性能路径: B=1 单 mul 归零短路; B>1 走 fp32 AiCore cumsum (int32 cumsum 落
    AICPU ~39us; cu/lens/ng 均 < 2^24 整数 fp32 精确, 先 floor 再前缀和)。
    """
    if cu_seqlens.shape[0] == 2:
        return cu_seqlens[:1] * 0
    cuf = cu_seqlens.to(torch.float32)
    lens = cuf[1:] - cuf[:-1]
    ng = (lens * (1.0 / ratio)).floor()
    cs = torch.cumsum(ng, 0)
    return (cs - ng).to(torch.int32)


def _csa2_compressor_bwd_bsh(args: CSA2CompressorBwdArgs):
    """BSH (x rank 3, (bsz, seq, h)) 反向分支; 返回 (dx, dwkv, dwgate, dweight)."""
    grad_out, x = args.grad_out, args.x
    wkv, wgate, weight = args.wkv, args.wgate, args.weight
    kv_state, score_state = args.kv_state, args.score_state
    start_pos, ratio, eps = args.start_pos, args.ratio, args.eps
    bsz, s1, h = x.shape
    d = weight.shape[0]
    # wkv/weight 接受 bf16/fp32: 内部归一到 kernel 注解 dtype (匹配时 .to() 零开销),
    # 输出按输入 dtype 返回 (对齐 golden 的 dwkv.to(wkv.dtype) 等契约)
    weight_f = weight.float()
    dweight = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)

    if ratio == 1:
        rows = bsz * s1
        x_in = x.reshape(1, rows, h)
        g_in = grad_out.reshape(1, rows, d)
        dx_out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
        dkv_all = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
        wkv_b = wkv.to(torch.bfloat16)
        dwkv = torch.empty(wkv.shape, dtype=torch.bfloat16, device=wkv.device)
        csa2_compressor_bwd_ratio1_kernel(
            g_in, x_in, wkv_b, weight_f, dx_out, dwkv, dweight, dkv_all, ratio, eps)
        return dx_out.view(bsz, s1, h), dwkv.to(wkv.dtype), None, dweight.to(weight.dtype)

    dwkv = torch.empty(wkv.shape, dtype=torch.float32, device=wkv.device)
    dwgate = torch.empty(wgate.shape, dtype=torch.float32, device=wgate.device)
    ng = s1 // ratio
    grad_in = grad_out.contiguous()
    rows = bsz * s1
    dx_out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    # d_kv/d_sc scratch: gb 循环覆盖完整组行, 余数行 (含 s1 < ratio 时全部行) 零填
    # (余数 token 无输出 → 梯度 0, dx/dw 段全行消费); 双 tensor 合并分配, 余数零填
    # 合并为单次 op。dtype 取 fp32 (bf16 GM scratch 裸参与 assemble/concat 触发标量回退)
    scratch = torch.empty((2, rows, d), dtype=torch.float32, device=x.device)
    d_kv_all, d_sc_all = scratch[0], scratch[1]
    if s1 % ratio:
        scratch.view(2, bsz, s1, d)[:, :, ng * ratio:, :] = 0
    # 权值单次 bf16 量化 (与 golden 同口径, 逐位一致)
    wkv_b = wkv.to(torch.bfloat16)
    wgate_b = wgate.to(torch.bfloat16)
    csa2_compressor_bwd_kernel(
        grad_in, x, wkv_b, wgate_b, weight_f,
        kv_state, score_state, start_pos,
        d_kv_all, d_sc_all, dx_out, dwkv, dwgate, dweight, ratio, eps)
    return (dx_out.view(bsz, s1, h), dwkv.to(wkv.dtype),
            dwgate.to(wgate.dtype), dweight.to(weight.dtype))


def _csa2_compressor_bwd_th(args: CSA2CompressorBwdArgs):
    """TH (BS 合轴, x rank 2, (t, h) + cu_seqlens) 反向分支; 返回 (dx, dwkv, dwgate, dweight)."""
    grad_out, x = args.grad_out, args.x
    wkv, wgate, weight = args.wkv, args.wgate, args.weight
    kv_state, score_state = args.kv_state, args.score_state
    start_pos, ratio, eps = args.start_pos, args.ratio, args.eps
    cu_seqlens = args.cu_seqlens
    t, h = x.shape
    d = weight.shape[0]
    weight_f = weight.float()
    dweight = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)

    if ratio == 1:
        rows = t
        x_in = x.reshape(1, rows, h)
        g_in = grad_out.reshape(1, rows, d)
        dx_out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
        dkv_all = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
        wkv_b = wkv.to(torch.bfloat16)
        dwkv = torch.empty(wkv.shape, dtype=torch.bfloat16, device=wkv.device)
        csa2_compressor_bwd_ratio1_kernel(
            g_in, x_in, wkv_b, weight_f, dx_out, dwkv, dweight, dkv_all, ratio, eps)
        return dx_out.view(t, h), dwkv.to(wkv.dtype), None, dweight.to(weight.dtype)

    dwkv = torch.empty(wkv.shape, dtype=torch.float32, device=wkv.device)
    dwgate = torch.empty(wgate.shape, dtype=torch.float32, device=wgate.device)
    rows = t
    grad_in = grad_out.contiguous()
    dx_out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    # d_kv/d_sc scratch: TH 余数行 (每段 L % ratio 行) 由 kernel 内按段零填
    scratch = torch.empty((2, rows, d), dtype=torch.float32, device=x.device)
    d_kv_all, d_sc_all = scratch[0], scratch[1]
    out_base = th_out_base(cu_seqlens, ratio)
    # 权值单次 bf16 量化 (与 golden 同口径, 逐位一致)
    wkv_b = wkv.to(torch.bfloat16)
    wgate_b = wgate.to(torch.bfloat16)
    csa2_compressor_bwd_th_kernel(
        grad_in, x, wkv_b, wgate_b, weight_f,
        cu_seqlens, out_base,
        kv_state, score_state, start_pos,
        d_kv_all, d_sc_all, dx_out, dwkv, dwgate, dweight, ratio, eps)
    return (dx_out, dwkv.to(wkv.dtype),
            dwgate.to(wgate.dtype), dweight.to(weight.dtype))


@allow_in_graph
def npu_csa2_compressor_bwd(args: CSA2CompressorBwdArgs):
    # state 允许 dim0 > bsz 预分配 (max_batch_size), 此处精确切片 (bwd 只读, 无写回语义);
    # batch 按模式判定: TH (cu_seqlens 给定) = 段数 (x dim0 为 token 总数 T, 不能当 batch 用),
    # BSH (cu_seqlens 为 None) = x dim0
    batch = args.cu_seqlens.shape[0] - 1 if args.cu_seqlens is not None else args.x.shape[0]
    if args.kv_state is not None and args.kv_state.shape[0] != batch:
        assert args.kv_state.shape[0] > batch and args.score_state.shape[0] > batch, (
            f"state dim0 is ({args.kv_state.shape[0]}, {args.score_state.shape[0]}), "
            f"expected == batch {batch} or > it (max_batch_size 预分配)")
        args = args._replace(kv_state=args.kv_state[:batch],
                             score_state=args.score_state[:batch])
    check_args(args)
    if args.cu_seqlens is not None:
        # TH (BS 合轴)
        return _csa2_compressor_bwd_th(args)
    # BSH
    return _csa2_compressor_bwd_bsh(args)


def _check_dtype(name, tensor, expect):
    assert tensor.dtype == expect, f"{name} dtype is {tensor.dtype}, expected {expect}"


def check_args(args: CSA2CompressorBwdArgs):
    """wrapper 入口完整校验: 输入 shape / dtype / device 契约."""
    x, grad_out = args.x, args.grad_out
    wkv, wgate, weight = args.wkv, args.wgate, args.weight
    kv_state, score_state, start_pos = args.kv_state, args.score_state, args.start_pos

    assert args.ratio in (1, RATIO), f"ratio is {args.ratio}, expected 1 or {RATIO}"
    assert isinstance(args.eps, (int, float)) and args.eps >= 0, (
        f"eps is {args.eps!r}, expected non-negative number")

    assert start_pos.dim() == 1 and start_pos.numel() == 1, (
        f"start_pos shape is {tuple(start_pos.shape)}, expected (1,)")
    assert start_pos.dtype == torch.int32, (
        f"start_pos dtype is {start_pos.dtype}, expected torch.int32")

    assert x.dim() in (2, 3), f"x dim is {x.dim()}, expected 2 (TH) or 3 (BSH)"
    assert x.dtype == torch.bfloat16, f"x dtype is {x.dtype}, expected torch.bfloat16"
    cu_seqlens = args.cu_seqlens
    if x.dim() == 3:
        assert cu_seqlens is None, (
            f"BSH (x rank 3) requires cu_seqlens to be None, "
            f"got shape {tuple(cu_seqlens.shape)}")
        bsz, s1, h = x.shape
        t = bsz * s1
    else:
        assert cu_seqlens is not None, "TH (x rank 2) requires cu_seqlens (bsz+1,)"
        assert cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 1, (
            f"cu_seqlens shape is {tuple(cu_seqlens.shape)}, expected (bsz+1,)")
        assert cu_seqlens.dtype == torch.int32, (
            f"cu_seqlens dtype is {cu_seqlens.dtype}, expected torch.int32")
        s1 = 0
        t, h = x.shape
        bsz = cu_seqlens.shape[0] - 1

    assert wkv.dim() == 2, f"wkv dim is {wkv.dim()}, expected 2 (head_dim, h)"
    d, h_kv = wkv.shape
    assert h_kv == h, f"wkv shape is {tuple(wkv.shape)}, h mismatch with x ({h})"
    assert wkv.dtype in (torch.bfloat16, torch.float32), (
        f"wkv dtype is {wkv.dtype}, expected bfloat16 or float32")

    assert weight.dim() == 1 and weight.shape[0] == d, (
        f"weight shape is {tuple(weight.shape)}, expected ({d},)")
    assert weight.dtype in (torch.bfloat16, torch.float32), (
        f"weight dtype is {weight.dtype}, expected bfloat16 or float32")

    if x.dim() == 3:
        n_out = s1 if args.ratio == 1 else max(s1 // args.ratio, 1)
        assert grad_out.dim() == 3, f"grad_out dim is {grad_out.dim()}, expected 3"
        assert tuple(grad_out.shape) == (bsz, n_out, d), (
            f"grad_out shape is {tuple(grad_out.shape)}, expected {(bsz, n_out, d)}")
    else:
        # TH: 输出 compact 压缩行 + pad, 总行数 = min(T, T//ratio + bsz)
        sr = t if args.ratio == 1 else min(t, t // args.ratio + bsz)
        assert grad_out.dim() == 2, f"grad_out dim is {grad_out.dim()}, expected 2 (TH)"
        assert tuple(grad_out.shape) == (sr, d), (
            f"grad_out shape is {tuple(grad_out.shape)}, expected {(sr, d)}")
    assert grad_out.dtype == x.dtype, (
        f"grad_out dtype is {grad_out.dtype}, expected {x.dtype}")

    if args.ratio == 1:
        assert wgate is None and kv_state is None and score_state is None, (
            "ratio == 1 requires wgate/kv_state/score_state to be None")
    else:
        assert wgate is not None and kv_state is not None and score_state is not None, (
            f"ratio == {RATIO} requires wgate/kv_state/score_state")
        assert tuple(wgate.shape) == (d, h), (
            f"wgate shape is {tuple(wgate.shape)}, expected {(d, h)}")
        assert wgate.dtype in (torch.bfloat16, torch.float32), (
            f"wgate dtype is {wgate.dtype}, expected bfloat16 or float32")
        assert tuple(kv_state.shape) == (bsz, RATIO, d), (
            f"kv_state shape is {tuple(kv_state.shape)}, expected {(bsz, RATIO, d)}")
        assert tuple(score_state.shape) == (bsz, RATIO, d), (
            f"score_state shape is {tuple(score_state.shape)}, expected {(bsz, RATIO, d)}")
        assert kv_state.dtype == torch.float32 and score_state.dtype == torch.float32, (
            f"state dtype is ({kv_state.dtype}, {score_state.dtype}), expected torch.float32")

    for name, t in (("grad_out", grad_out), ("wkv", wkv), ("weight", weight),
                    ("start_pos", start_pos), ("wgate", wgate),
                    ("kv_state", kv_state), ("score_state", score_state)):
        if t is not None:
            assert t.device == x.device, f"{name} device is {t.device}, expected {x.device}"


def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub, precision_type=pypto.PrecisionType.HIGH_PRECISION)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum, precision_type=pypto.PrecisionType.HIGH_PRECISION)
    return xdiv


def softmax_backward(dy: pypto.Tensor, y: pypto.Tensor, dim) -> pypto.Tensor:
    d = pypto.mul(dy, y)
    sum_d = pypto.sum(d, dim, keepdim=True)
    dx = pypto.mul(y, pypto.sub(dy, sum_d))
    return dx


def rms_norm_backward_2d(dy: pypto.Tensor, x: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-5,
                         rows_tile=16, gamma_2d=None):
    """RMSNorm backward, 2D [R, d] fp32; 返回 (dx [R, d], dgamma [1, d])."""
    d = gamma.shape[0]
    pypto.set_vec_tile_shapes(rows_tile, d)
    if gamma_2d is None:
        gamma_2d = pypto.reshape(gamma, [1, d])
    d_y = pypto.mul(dy, gamma_2d)
    rms = pypto.sum(pypto.mul(x, x), -1, keepdim=True)
    rms = pypto.mul(rms, 1.0 / d)
    rms = pypto.add(rms, epsilon)
    rms = pypto.rsqrt(rms, precision_type=pypto.PrecisionType.HIGH_PRECISION)
    term1 = pypto.mul(d_y, rms)
    sum_term = pypto.sum(pypto.mul(d_y, x), -1, keepdim=True)
    rms3 = pypto.mul(pypto.mul(rms, rms), rms)
    s = pypto.mul(pypto.mul(sum_term, rms3), 1.0 / d)
    term2 = pypto.mul(x, s)
    dx = pypto.sub(term1, term2)
    x_hat = pypto.mul(x, rms)
    dgamma = pypto.mul(dy, x_hat)
    dgamma = pypto.sum(dgamma, 0, keepdim=True)          # [1, d]
    return dx, dgamma


@pypto.frontend.jit(
    pass_options={
        # cube 合图 (靶向): func7_1 = C 段 (dx matmul/assemble); 键为编译期 hashOrder,
        # kernel 结构改动后需重采泳道确认键位
        "cube_nbuffer_setting": {"DEFAULT": 1, "func7_1": 4},
    },
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 64,
        "launch_sched_aicpu_num": 3,
    },
)
def csa2_compressor_bwd_ratio1_kernel(
    grad_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    dx: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    dwkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    dweight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    dkv_all: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    ratio,
    eps,
):
    """ratio == 1 反向: out = norm(wkv(x))。

    x/grad_out 精确行数传入 (bsz=1, s1=rows); ceil 分块, 尾块 valid_shape。
    结构: 主循环 (kv matmul → rms bwd → dx matmul → dkv 落 bf16 scratch)
          + D 段 (dwkv = Σ dkv^T @ x, K 分块大 matmul) + EPI
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    assert ratio == 1, f"ratio is {ratio}, expected 1"
    bsz, s1, h = x.shape
    d = weight.shape[0]
    rows = bsz * s1

    x2 = pypto.reshape(x, [rows, h], inplace=True)
    g2 = pypto.reshape(grad_out, [rows, d], inplace=True)

    pypto.set_vec_tile_shapes(64, 512)
    dwkv_acc = pypto.full([d, h], 0.0, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(1, d)
    dweight_acc = pypto.full([1, d], 0.0, pypto.DT_FP32)
    gamma_2d = pypto.reshape(weight, [1, d], inplace=True)

    for mb in pypto.loop((rows + R1_MB - 1) // R1_MB, name="LOOP_CSA2_BWD_R1", idx_name="mb"):
        base = mb * R1_MB
        valid = (rows - base).min(R1_MB)
        x_blk = pypto.view(x2, [R1_MB, h], [base, 0], valid_shape=[valid, h])
        g_blk = pypto.view(g2, [R1_MB, d], [base, 0], valid_shape=[valid, d])
        pypto.set_cube_tile_shapes([64, 128], [128, 256], [64, 64])
        kv = pypto.matmul(x_blk, wkv, pypto.DT_FP32, b_trans=True)      # [R1_MB, d]
        pypto.set_vec_tile_shapes(16, d)
        kv_q = pypto.cast(pypto.cast(kv, pypto.DT_BF16), pypto.DT_FP32)
        g_f = pypto.cast(g_blk, pypto.DT_FP32)
        dkv, dw = rms_norm_backward_2d(g_f, kv_q, weight, eps, rows_tile=8, gamma_2d=gamma_2d)
        pypto.set_vec_tile_shapes(1, d)
        dweight_acc[:] = dweight_acc + dw
        pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 256])
        dkv_b = pypto.cast(dkv, pypto.DT_BF16)
        dx_f = pypto.matmul(dkv_b, wkv, pypto.DT_FP32)                  # [R1_MB, h]
        pypto.set_vec_tile_shapes(16, 512)
        pypto.assemble(pypto.cast(dx_f, pypto.DT_BF16), [base, 0], dx)
        # assemble 源须为专用 cast (与上方 matmul 共享会触发循环偏移不重绑的前端 bug)
        pypto.set_vec_tile_shapes(64, 512)
        pypto.assemble(pypto.cast(dkv, pypto.DT_BF16), [base, 0], dkv_all)

    pypto.set_cube_tile_shapes([128, 512], [128, 128], [128, 256])
    for kb in pypto.loop((rows + R1_KB - 1) // R1_KB, name="LOOP_CSA2_BWD_R1_DW", idx_name="kb"):
        k0 = kb * R1_KB
        kvalid = (rows - k0).min(R1_KB)
        dkv_k = pypto.view(dkv_all, [R1_KB, d], [k0, 0], valid_shape=[kvalid, d])
        x_k = pypto.view(x2, [R1_KB, h], [k0, 0], valid_shape=[kvalid, h])
        pypto.set_vec_tile_shapes(16, 512)
        dwkv_acc[:] = dwkv_acc + pypto.matmul(dkv_k, x_k, pypto.DT_FP32, a_trans=True)

    pypto.set_vec_tile_shapes(32, 512)
    dwkv[:] = pypto.cast(dwkv_acc, pypto.DT_BF16)
    pypto.set_vec_tile_shapes(1, 512)
    pypto.assemble(pypto.reshape(dweight_acc, [d]), [0], dweight)
    pypto.set_vec_tile_shapes(32, 512)


@pypto.frontend.jit(
    pass_options={},
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 64,
        "launch_sched_aicpu_num": 3,
    }
)
def csa2_compressor_bwd_kernel(
    grad_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    wgate_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    kv_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    start_pos_dy: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    d_kv_all: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    d_sc_all: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    dx: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    dwkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dwgate: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dweight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    ratio,
    eps,
):
    """ratio == 2 反向 (prefill + decode)。

    x/grad_out 精确形状直传; 余数 token 行的 d_kv/d_sc 由 wrapper 零填, decode / s1 <
    ratio 全量零填。各段 ceil 分块, 尾块 valid_shape。wkv_b/wgate_b 为单次 bf16 量化
    权值 (与 golden 逐位一致); d_kv_all/d_sc_all (batch*s1, head_dim) fp32 scratch。

    结构: 组块段 (matmul + softmax/pool/rms bwd) → dx 段 (行分块 matmul)
          → dwkv/dwgate 段 (K 分块 matmul, [2d, h] 合并累加器)
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    assert ratio == RATIO, f"ratio is {ratio}, expected {RATIO}"
    bsz, s1, h = x.shape
    d = weight.shape[0]
    ng = s1 // ratio
    g_cols = grad_out.shape[1]
    rows = bsz * s1

    x2 = pypto.reshape(x, [rows, h], inplace=True)
    g2 = pypto.reshape(grad_out, [bsz * g_cols, d], inplace=True)

    pypto.set_vec_tile_shapes(1, d)
    dweight_acc = pypto.full([1, d], 0.0, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(16, 512)
    dwks_acc = pypto.full([2 * d, h], 0.0, pypto.DT_FP32)

    G = BWD_G
    RB = BWD_RB
    g_loop = (ng + G - 1) // G
    dx_loop = (rows + BWD_MB - 1) // BWD_MB
    kw_loop = (rows + BWD_KB - 1) // BWD_KB
    start = start_pos_dy[0]

    for b in pypto.loop(bsz, name="LOOP_CSA2_BWD_B", idx_name="b"):
        if start < 1:
            # 尾组块以 reshape(valid_shape=) 在 3D<->2D 形变处重标注 (纯元数据, 零搬运)。
            # ⚠️ 不可用 view 重标注: view 是 slice 抽取, 计算张量上整块抽取物化会超 UB
            for gb in pypto.loop(g_loop, name="LOOP_CSA2_BWD_G", idx_name="gb"):
                base = b * s1 + gb * RB
                valid_g = (ng - gb * G).min(G)
                valid_rb = valid_g * ratio
                x_blk = pypto.view(x2, [RB, h], [base, 0], valid_shape=[valid_rb, h])
                pypto.set_cube_tile_shapes([256, 256], [128, 256], [128, 128])
                kv2 = pypto.matmul(x_blk, wkv_b, pypto.DT_FP32, b_trans=True)       # [RB, d]
                sc2 = pypto.matmul(x_blk, wgate_b, pypto.DT_FP32, b_trans=True)     # [RB, d]
                pypto.set_vec_tile_shapes(8, ratio, d)
                kv3 = pypto.reshape(kv2, [G, ratio, d], valid_shape=[valid_g, ratio, d])
                sc3 = pypto.reshape(sc2, [G, ratio, d], valid_shape=[valid_g, ratio, d])
                alpha = softmax(sc3, 1)
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                        # [G, d]
                pypto.set_vec_tile_shapes(16, d)
                latent = pypto.cast(pypto.cast(latent, pypto.DT_BF16), pypto.DT_FP32)
                gr = pypto.view(g2, [G, d], [b * g_cols + gb * G, 0], valid_shape=[valid_g, d])
                gr = pypto.cast(gr, pypto.DT_FP32)
                dlatent, dw = rms_norm_backward_2d(gr, latent, weight, eps)
                pypto.set_vec_tile_shapes(1, d)
                dweight_acc[:] = dweight_acc + dw
                pypto.set_vec_tile_shapes(8, ratio, d)
                dl3 = pypto.reshape(dlatent, [G, 1, d], valid_shape=[valid_g, 1, d])
                d_alpha = pypto.mul(dl3, kv3)
                d_kv = pypto.mul(dl3, alpha)
                d_sc = softmax_backward(d_alpha, alpha, 1)
                # reshape 携带 valid, assemble 只写有效行 (防越界覆盖他 batch)
                pypto.assemble(pypto.reshape(d_kv, [RB, d], valid_shape=[valid_rb, d]),
                               [base, 0], d_kv_all, parallel=True)
                pypto.assemble(pypto.reshape(d_sc, [RB, d], valid_shape=[valid_rb, d]),
                               [base, 0], d_sc_all, parallel=True)
        else:
            # decode: latent 来自 state, 仅当前槽位可微
            slot = start % ratio
            st_kv = pypto.view(kv_state_total, [1, ratio, d], [b, 0, 0])
            st_sc = pypto.view(score_state_total, [1, ratio, d], [b, 0, 0])
            pypto.set_vec_tile_shapes(1, ratio, d)
            alpha = softmax(st_sc, 1)
            latent = pypto.sum(pypto.mul(alpha, st_kv), 1)                          # [1, d]
            pypto.set_vec_tile_shapes(1, d)
            latent = pypto.cast(pypto.cast(latent, pypto.DT_BF16), pypto.DT_FP32)
            gr = pypto.view(g2, [1, d], [b * g_cols, 0])
            gr = pypto.cast(gr, pypto.DT_FP32)
            dlatent, dw = rms_norm_backward_2d(gr, latent, weight, eps)
            dweight_acc[:] = dweight_acc + dw
            pypto.set_vec_tile_shapes(1, ratio, d)
            dl3 = pypto.reshape(dlatent, [1, 1, d])
            d_alpha = pypto.mul(dl3, st_kv)
            d_st_kv = pypto.mul(dl3, alpha)
            d_st_sc = softmax_backward(d_alpha, alpha, 1)
            d_st_kv2 = pypto.reshape(d_st_kv, [ratio, d])
            d_st_sc2 = pypto.reshape(d_st_sc, [ratio, d])
            # 取当前槽位: 2D 首维 offset (view 中维 offset 不可靠)
            d_kv2 = pypto.view(d_st_kv2, [1, d], [slot, 0])
            d_sc2 = pypto.view(d_st_sc2, [1, d], [slot, 0])
            pypto.assemble(d_kv2, [b * s1, 0], d_kv_all)
            pypto.assemble(d_sc2, [b * s1, 0], d_sc_all)

    # dx 段: 行分块大 matmul + 写回
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    for mb in pypto.loop(dx_loop, name="LOOP_CSA2_BWD_DX", idx_name="mb", submit_before_loop=True):
        m0 = mb * BWD_MB
        valid_m = (rows - m0).min(BWD_MB)
        kv_b = pypto.view(d_kv_all, [BWD_MB, d], [m0, 0], valid_shape=[valid_m, d])
        sc_b = pypto.view(d_sc_all, [BWD_MB, d], [m0, 0], valid_shape=[valid_m, d])
        kv_bf = pypto.cast(kv_b, pypto.DT_BF16)
        sc_bf = pypto.cast(sc_b, pypto.DT_BF16)
        dx_f = pypto.matmul(kv_bf, wkv_b, pypto.DT_FP32)                          # [BWD_MB, h]
        dx_f = pypto.add(dx_f, pypto.matmul(sc_bf, wgate_b, pypto.DT_FP32))
        pypto.set_vec_tile_shapes(16, 512)
        pypto.assemble(pypto.cast(dx_f, pypto.DT_BF16), [m0, 0], dx)

    # dwkv/dwgate 段: K 分块大 matmul, 双子块 concat 合并 ([2d, h] 累加器)
    pypto.set_cube_tile_shapes([256, 256], [128, 256], [128, 128])
    for kb in pypto.loop(kw_loop, name="LOOP_CSA2_BWD_DW", idx_name="kb", submit_before_loop=True):
        k0 = kb * BWD_KB
        valid_k = (rows - k0).min(BWD_KB)
        kv_b = pypto.view(d_kv_all, [BWD_KB, d], [k0, 0], valid_shape=[valid_k, d])
        sc_b = pypto.view(d_sc_all, [BWD_KB, d], [k0, 0], valid_shape=[valid_k, d])
        x_b = pypto.view(x2, [BWD_KB, h], [k0, 0], valid_shape=[valid_k, h])
        pypto.set_vec_tile_shapes(16, 512)
        kv_bf = pypto.cast(kv_b, pypto.DT_BF16)
        sc_bf = pypto.cast(sc_b, pypto.DT_BF16)
        ks_b = pypto.concat([kv_bf, sc_bf], 1)                                 # [BWD_KB, 2d] bf16
        dwks_acc[:] = dwks_acc + pypto.matmul(ks_b, x_b, pypto.DT_FP32, a_trans=True)   # [2d, h]

    pypto.set_vec_tile_shapes(16, 512)
    pypto.assemble(pypto.view(dwks_acc, [d, h], [0, 0]), [0, 0], dwkv)
    pypto.assemble(pypto.view(dwks_acc, [d, h], [d, 0]), [0, 0], dwgate)
    pypto.set_vec_tile_shapes(1, 512)
    pypto.assemble(pypto.reshape(dweight_acc, [d]), [0], dweight)


@pypto.frontend.jit(
    pass_options={},
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 64,
        "launch_sched_aicpu_num": 3,
    }
)
def csa2_compressor_bwd_th_kernel(
    grad_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    wgate_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    out_base: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    kv_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    start_pos_dy: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    d_kv_all: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    d_sc_all: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    dx: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    dwkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dwgate: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dweight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    ratio,
    eps,
):
    """ratio == 2 TH (BS 合轴) 反向: x/grad_out 2D、cu_seqlens 前缀和分段.

    prefill (start_pos < 1): 每 batch 段独立组块反向, 组块行写 token 布局 scratch
        [cu[b] + ...]; 段尾余数行按段零填 (无输出 → 梯度 0); grad_out 行取
        out_base[b] 起的 compact 前缀。
    decode  (start_pos >= 1): 每 batch 1 token (行序 = batch 序), latent 由 state 池化,
        仅当前槽位可微; grad 行 = batch 序。
    公共尾部 (dx / dwkv / dwgate 段) 按 token 行遍历, 与 BSH 版结构一致。
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    assert ratio == RATIO, f"ratio is {ratio}, expected {RATIO}"
    t, h = x.shape
    d = weight.shape[0]
    nb = cu_seqlens.shape[0] - 1
    rows = t

    pypto.set_vec_tile_shapes(1, d)
    dweight_acc = pypto.full([1, d], 0.0, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(16, 512)
    dwks_acc = pypto.full([2 * d, h], 0.0, pypto.DT_FP32)
    z_row = pypto.full([1, d], 0.0, pypto.DT_FP32)

    G = BWD_G
    RB = BWD_RB
    start = start_pos_dy[0]

    if start < 1:
        for b_idx in pypto.loop(nb, name="LOOP_CSA2_BWD_TH_B", idx_name="b_idx"):
            tok0 = cu_seqlens[b_idx]
            cur_len = cu_seqlens[b_idx + 1] - tok0
            ng_b = cur_len // ratio
            ob = out_base[b_idx]
            g_loop_b = (ng_b + G - 1) // G
            for gb in pypto.loop(g_loop_b, name="LOOP_CSA2_BWD_TH_G", idx_name="gb"):
                base = tok0 + gb * RB
                valid_g = (ng_b - gb * G).min(G)
                valid_rb = valid_g * ratio
                x_blk = pypto.view(x, [RB, h], [base, 0], valid_shape=[valid_rb, h])
                pypto.set_cube_tile_shapes([256, 256], [128, 256], [128, 128])
                kv2 = pypto.matmul(x_blk, wkv_b, pypto.DT_FP32, b_trans=True)       # [RB, d]
                sc2 = pypto.matmul(x_blk, wgate_b, pypto.DT_FP32, b_trans=True)     # [RB, d]
                pypto.set_vec_tile_shapes(8, ratio, d)
                kv3 = pypto.reshape(kv2, [G, ratio, d], valid_shape=[valid_g, ratio, d])
                sc3 = pypto.reshape(sc2, [G, ratio, d], valid_shape=[valid_g, ratio, d])
                alpha = softmax(sc3, 1)
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                        # [G, d]
                pypto.set_vec_tile_shapes(16, d)
                latent = pypto.cast(pypto.cast(latent, pypto.DT_BF16), pypto.DT_FP32)
                gr = pypto.view(grad_out, [G, d], [ob + gb * G, 0], valid_shape=[valid_g, d])
                gr = pypto.cast(gr, pypto.DT_FP32)
                dlatent, dw = rms_norm_backward_2d(gr, latent, weight, eps)
                pypto.set_vec_tile_shapes(1, d)
                dweight_acc[:] = dweight_acc + dw
                pypto.set_vec_tile_shapes(8, ratio, d)
                dl3 = pypto.reshape(dlatent, [G, 1, d], valid_shape=[valid_g, 1, d])
                d_alpha = pypto.mul(dl3, kv3)
                d_kv = pypto.mul(dl3, alpha)
                d_sc = softmax_backward(d_alpha, alpha, 1)
                pypto.assemble(pypto.reshape(d_kv, [RB, d], valid_shape=[valid_rb, d]),
                               [base, 0], d_kv_all, parallel=True)
                pypto.assemble(pypto.reshape(d_sc, [RB, d], valid_shape=[valid_rb, d]),
                               [base, 0], d_sc_all, parallel=True)
            if cur_len % ratio > 0:
                # 段尾余数行无输出 → 梯度 0 (kernel 内按段零填; ratio=2 时至多 1 行)
                pypto.assemble(z_row, [tok0 + ng_b * ratio, 0], d_kv_all)
                pypto.assemble(z_row, [tok0 + ng_b * ratio, 0], d_sc_all)
    else:
        # TH decode: 每 batch 1 token (行序 = batch 序); grad 行 = batch 序
        for b_idx in pypto.loop(nb, name="LOOP_CSA2_BWD_TH_D", idx_name="b_idx"):
            slot = start % ratio
            st_kv = pypto.view(kv_state_total, [1, ratio, d], [b_idx, 0, 0])
            st_sc = pypto.view(score_state_total, [1, ratio, d], [b_idx, 0, 0])
            pypto.set_vec_tile_shapes(1, ratio, d)
            alpha = softmax(st_sc, 1)
            latent = pypto.sum(pypto.mul(alpha, st_kv), 1)                          # [1, d]
            pypto.set_vec_tile_shapes(1, d)
            latent = pypto.cast(pypto.cast(latent, pypto.DT_BF16), pypto.DT_FP32)
            gr = pypto.view(grad_out, [1, d], [b_idx, 0])
            gr = pypto.cast(gr, pypto.DT_FP32)
            dlatent, dw = rms_norm_backward_2d(gr, latent, weight, eps)
            dweight_acc[:] = dweight_acc + dw
            pypto.set_vec_tile_shapes(1, ratio, d)
            dl3 = pypto.reshape(dlatent, [1, 1, d])
            d_alpha = pypto.mul(dl3, st_kv)
            d_st_kv = pypto.mul(dl3, alpha)
            d_st_sc = softmax_backward(d_alpha, alpha, 1)
            d_st_kv2 = pypto.reshape(d_st_kv, [ratio, d])
            d_st_sc2 = pypto.reshape(d_st_sc, [ratio, d])
            # 取当前槽位: 2D 首维 offset (view 中维 offset 不可靠)
            d_kv2 = pypto.view(d_st_kv2, [1, d], [slot, 0])
            d_sc2 = pypto.view(d_st_sc2, [1, d], [slot, 0])
            pypto.assemble(d_kv2, [b_idx, 0], d_kv_all)
            pypto.assemble(d_sc2, [b_idx, 0], d_sc_all)

    # dx 段: 行分块大 matmul + 写回 (token 行遍历, 与 BSH 版一致)
    dx_loop = (rows + BWD_MB - 1) // BWD_MB
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    for mb in pypto.loop(dx_loop, name="LOOP_CSA2_BWD_TH_DX", idx_name="mb", submit_before_loop=True):
        m0 = mb * BWD_MB
        valid_m = (rows - m0).min(BWD_MB)
        kv_b = pypto.view(d_kv_all, [BWD_MB, d], [m0, 0], valid_shape=[valid_m, d])
        sc_b = pypto.view(d_sc_all, [BWD_MB, d], [m0, 0], valid_shape=[valid_m, d])
        kv_bf = pypto.cast(kv_b, pypto.DT_BF16)
        sc_bf = pypto.cast(sc_b, pypto.DT_BF16)
        dx_f = pypto.matmul(kv_bf, wkv_b, pypto.DT_FP32)                          # [BWD_MB, h]
        dx_f = pypto.add(dx_f, pypto.matmul(sc_bf, wgate_b, pypto.DT_FP32))
        pypto.set_vec_tile_shapes(16, 512)
        pypto.assemble(pypto.cast(dx_f, pypto.DT_BF16), [m0, 0], dx)

    # dwkv/dwgate 段: K 分块大 matmul, 双子块 concat 合并 ([2d, h] 累加器)
    kw_loop = (rows + BWD_KB - 1) // BWD_KB
    pypto.set_cube_tile_shapes([256, 256], [128, 256], [128, 128])
    for kb in pypto.loop(kw_loop, name="LOOP_CSA2_BWD_TH_DW", idx_name="kb", submit_before_loop=True):
        k0 = kb * BWD_KB
        valid_k = (rows - k0).min(BWD_KB)
        kv_b = pypto.view(d_kv_all, [BWD_KB, d], [k0, 0], valid_shape=[valid_k, d])
        sc_b = pypto.view(d_sc_all, [BWD_KB, d], [k0, 0], valid_shape=[valid_k, d])
        x_b = pypto.view(x, [BWD_KB, h], [k0, 0], valid_shape=[valid_k, h])
        pypto.set_vec_tile_shapes(16, 512)
        kv_bf = pypto.cast(kv_b, pypto.DT_BF16)
        sc_bf = pypto.cast(sc_b, pypto.DT_BF16)
        ks_b = pypto.concat([kv_bf, sc_bf], 1)                                 # [BWD_KB, 2d] bf16
        dwks_acc[:] = dwks_acc + pypto.matmul(ks_b, x_b, pypto.DT_FP32, a_trans=True)   # [2d, h]

    pypto.set_vec_tile_shapes(16, 512)
    pypto.assemble(pypto.view(dwks_acc, [d, h], [0, 0]), [0, 0], dwkv)
    pypto.assemble(pypto.view(dwks_acc, [d, h], [d, 0]), [0, 0], dwgate)
    pypto.set_vec_tile_shapes(1, 512)
    pypto.assemble(pypto.reshape(dweight_acc, [d]), [0], dweight)
