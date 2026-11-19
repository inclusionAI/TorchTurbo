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
CSA2 Compressor (DeepSeek V4.1-Flash) — source aligned (ratio == 1 / prefill / decode).

源码: tests/ops/deepseek_v4_1/deepseekv41compressor.py (Compressor.forward)
Golden: tests/ops/deepseek_v4_1/test_CSA2_compressor_fwd.py (golden_compress_csa2)

分支 (与源码一一对应):
    ratio == 1:   out = norm(wkv(x))                      # bf16 投影, 无 gate/state/池化
                                                          # 输出 (bsz, seqlen, d), 永不 None
    ratio > 1:
        start_pos == 0 (prefill):
            完整组 [k*ratio, (k+1)*ratio) 用当前 token 做 softmax(gate) 加权池化;
            尾部余数 token 存入 kv_state/score_state[0:remainder];
            should_compress = seqlen >= ratio, 否则 out 不写 (源码返回 None)
        start_pos != 0 (decode, 每步 1 token):
            写 kv_state/score_state[start_pos % ratio];
            (start_pos+1) % ratio == 0 时对 state 的 ratio 个槽位做 softmax 池化
        输出 (bsz, max(seqlen // ratio, 1), head_dim), 均为 pre-RoPE latent
        out = norm(pooled.to(hidden_dtype))               # 池化结果先落回 bf16 再 RMSNorm

ratio 只支持 1 与 2: ratio == 1 走独立的纯投影 kernel, ratio == 2 走压缩 kernel。
"""
import collections
import pypto
import torch
from torch._dynamo import allow_in_graph

RATIO = 2

CSA2CompressorArgs = collections.namedtuple('CSA2CompressorArgs', [
    'x', 'wkv', 'wgate', 'weight', 'kv_state', 'score_state', 'start_pos', 'cu_seqlens',
    'ratio', 'eps',
], defaults=(None, 2, 1e-20))   # cu_seqlens (BSH 可省略) 位于标量参数 ratio/eps 之前


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "csa2_compressor(Tensor x, Tensor wkv, Tensor? wgate, Tensor weight, \
    Tensor? kv_state, Tensor? score_state, Tensor start_pos, Tensor? cu_seqlens, \
    int ratio, float eps) -> Tensor"
)


@torch.library.impl(pyptolib, "csa2_compressor", "Meta")
def csa2_compressor(x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens,
                    ratio, eps):
    d = weight.shape[0]
    if cu_seqlens is not None:
        # TH (BS 合轴): x (T, H); 输出 ratio2 为 compact 压缩行 + pad
        t = x.shape[0]
        if ratio == 1:
            return torch.empty((t, d), dtype=x.dtype, device=x.device)
        bsz = cu_seqlens.shape[0] - 1
        sr = min(t, t // ratio + bsz)
        return torch.empty((sr, d), dtype=x.dtype, device=x.device)
    if ratio == 1:
        out = torch.empty(
            (x.shape[0], x.shape[1], d), dtype=x.dtype, device=x.device)
    else:
        n_groups = max(x.shape[1] // ratio, 1)
        out = torch.empty(
            (x.shape[0], n_groups, d), dtype=x.dtype, device=x.device)
    return out


try:
    @torch.library.impl(pyptolib, "csa2_compressor", "NPU")
    def csa2_compressor(x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens,
                        ratio, eps):
        args = CSA2CompressorArgs(
            x=x, wkv=wkv, wgate=wgate, weight=weight,
            kv_state=kv_state, score_state=score_state, start_pos=start_pos, ratio=ratio,
            eps=eps, cu_seqlens=cu_seqlens,
        )
        return npu_csa2_compressor(args)
except Exception as e:
    if "could not parse dispatch key: NPU" in str(e):
        print(f"Skip: torchair not installed, skip NPU registration for operator 'csa2_compressor'")
    else:
        print(f"Skip: Unexpected error : {e}")


def csa2_compressor_pypto(args: CSA2CompressorArgs):
    return torch.ops.pypto.csa2_compressor(
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

    输入 cu_seqlens (bsz+1,) int32 设备前缀和; 返回 (bsz,) int32。设备侧小算子,
    无 host 同步, 供 kernel 写 compact 压缩行/反向读 grad_out 前缀使用。

    性能路径 (原 5 算子链的 int32 cumsum 落 AICPU, 单次 ~39us):
    - B=1: exclusive 前缀首项恒 0, 单个 mul 归零即得 (1 个 AiCore 小算子);
    - B>1: 走 fp32 cumsum (AiCore 实现): cu / lens / ng 均为 < 2^24 的整数, fp32
      表示精确, ratio=2 时乘子 0.5 精确; 先 floor 再前缀和 (floor 对求和不可分配)。
    """
    if cu_seqlens.shape[0] == 2:
        return cu_seqlens[:1] * 0
    cuf = cu_seqlens.to(torch.float32)
    lens = cuf[1:] - cuf[:-1]
    ng = (lens * (1.0 / ratio)).floor()
    cs = torch.cumsum(ng, 0)
    return (cs - ng).to(torch.int32)


def _csa2_compressor_bsh(args: CSA2CompressorArgs):
    """BSH (x rank 3, (bsz, seq, h)) 分支.

    ratio1 -> (bsz, seq, d); ratio2 -> (bsz, max(seq//2, 1), d)。
    """
    x, wkv, wgate, weight = args.x, args.wkv, args.wgate, args.weight
    kv_state, score_state = args.kv_state, args.score_state
    start_pos, ratio, eps = args.start_pos, args.ratio, args.eps

    if ratio == 1:
        # 纯投影分支: 每 token 一个 latent. kernel 以 (bsz*seq, d) 的展平行处理,
        # 输出按行序写回, 对任意 bsz/seq 成立.
        out2d = torch.empty(
            (x.shape[0] * x.shape[1], weight.shape[0]), dtype=x.dtype, device=x.device)
        csa2_compressor_ratio1_kernel(x, wkv, weight, out2d, ratio, eps)
        return out2d.reshape(x.shape[0], x.shape[1], weight.shape[0])

    n_groups = max(x.shape[1] // ratio, 1)
    out = torch.empty(
        (x.shape[0], n_groups, weight.shape[0]), dtype=x.dtype, device=x.device)
    # kv_state/score_state 既作输入(读旧状态)又作输出(写新 token), 依赖同一存储别名
    csa2_compressor_kernel(
        x, wkv, wgate, weight, kv_state, score_state, start_pos,
        kv_state, score_state, out, ratio, eps)
    return out


def _csa2_compressor_th(args: CSA2CompressorArgs):
    """TH (BS 合轴, x rank 2, (t, h) + cu_seqlens) 分支.

    ratio1 -> (t, d); ratio2 -> (min(t, t//ratio + bsz), d) (compact 压缩行 + pad)。
    """
    x, wkv, wgate, weight = args.x, args.wkv, args.wgate, args.weight
    kv_state, score_state = args.kv_state, args.score_state
    start_pos, ratio, eps = args.start_pos, args.ratio, args.eps
    t, h = x.shape
    d = weight.shape[0]

    if ratio == 1:
        # 纯投影对分段不敏感: 展平 (T,1,h) 复用 ratio1 kernel, 输出 (T, d)
        x3 = x.reshape(t, 1, h)
        out2d = torch.empty((t, d), dtype=x.dtype, device=x.device)
        csa2_compressor_ratio1_kernel(x3, wkv, weight, out2d, ratio, eps)
        return out2d

    bsz = args.cu_seqlens.shape[0] - 1
    sr = min(t, t // ratio + bsz)
    out = torch.empty((sr, d), dtype=x.dtype, device=x.device)
    out_base = th_out_base(args.cu_seqlens, ratio)
    # kv_state/score_state 既作输入(读旧状态)又作输出(写新 token), 依赖同一存储别名
    csa2_compressor_th_kernel(
        x, wkv, wgate, weight, args.cu_seqlens, out_base,
        kv_state, score_state, start_pos,
        kv_state, score_state, out, ratio, eps)
    return out


@allow_in_graph
def npu_csa2_compressor(args: CSA2CompressorArgs):
    check_args(args)
    if args.cu_seqlens is not None:
        # TH (BS 合轴): cu_seqlens 给定 (x (T, H) 2D)
        return _csa2_compressor_th(args)
    # BSH: cu_seqlens 为 None (x (bsz, seq, h) 3D)
    return _csa2_compressor_bsh(args)


def _check_dtype(name, tensor, expect):
    assert tensor.dtype == expect, f"{name} dtype is {tensor.dtype}, expected {expect}"


def check_args(args: CSA2CompressorArgs):
    """校验 wrapper 接口参数的 shape / dtype (JIT 注解之外在 host 侧显式校验).

    契约 (x 的 h 为静态维, 由注解/JIT 特化保证; 此处做 host 侧显式校验):
        BSH: x         (bsz, seq, h)               bf16, cu_seqlens 必须为 None
        TH : x         (t, h)                      bf16, cu_seqlens (bsz+1,) int32 前缀和
        wkv               (head_dim, h)               ratio==1: bf16 / ratio==2: fp32
        wgate             (head_dim, h)               fp32   (仅 ratio==2)
        weight            (head_dim,)                 fp32
        kv_state          (bsz, ratio, head_dim)      fp32   (仅 ratio==2)
        score_state       (bsz, ratio, head_dim)      fp32   (仅 ratio==2)
        start_pos         (1,)                        int32
    """
    x, wkv, wgate = args.x, args.wkv, args.wgate
    weight, kv_state, score_state = args.weight, args.kv_state, args.score_state
    start_pos = args.start_pos
    cu_seqlens = args.cu_seqlens

    assert args.ratio in (1, RATIO), f"ratio is {args.ratio}, expected 1 or {RATIO}"

    # ---- x: BSH (bsz, seq, h) / TH (t, h) bf16 ----
    _check_dtype("x", x, torch.bfloat16)
    if x.dim() == 3:
        assert cu_seqlens is None, (
            f"BSH (x rank 3) requires cu_seqlens to be None, "
            f"got shape {tuple(cu_seqlens.shape)}")
        bsz, seq, h = x.shape
    else:
        assert x.dim() == 2, f"x rank is {x.dim()}, expected 2 (TH) or 3 (BSH)"
        assert cu_seqlens is not None, "TH (x rank 2) requires cu_seqlens (bsz+1,)"
        assert cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 1, (
            f"cu_seqlens shape is {tuple(cu_seqlens.shape)}, expected (bsz+1,)")
        _check_dtype("cu_seqlens", cu_seqlens, torch.int32)
        t, h = x.shape
        bsz = cu_seqlens.shape[0] - 1

    # ---- wkv: (head_dim, h) ----
    assert wkv.dim() == 2, f"wkv rank is {wkv.dim()}, expected 2"
    head_dim, h_wkv = wkv.shape
    assert h_wkv == h, f"wkv.shape[1] is {h_wkv}, expected x.shape[2] ({h})"

    # ---- weight: (head_dim,) fp32 ----
    assert weight.dim() == 1, f"weight rank is {weight.dim()}, expected 1"
    assert weight.shape[0] == head_dim, (
        f"weight numel is {weight.shape[0]}, expected head_dim ({head_dim})"
    )
    _check_dtype("weight", weight, torch.float32)

    # ---- start_pos: (1,) int32 ----
    assert tuple(start_pos.shape) == (1,), (
        f"start_pos shape is {tuple(start_pos.shape)}, expected (1,)"
    )
    _check_dtype("start_pos", start_pos, torch.int32)

    if args.ratio == 1:
        # 源码 ratio==1: wkv 保持 bf16, 无 wgate/kv_state/score_state
        _check_dtype("wkv", wkv, torch.bfloat16)
        assert wgate is None and kv_state is None and score_state is None, (
            "ratio == 1 requires wgate/kv_state/score_state to be None"
        )
        return

    # ---- ratio == 2 ----
    assert wgate is not None and kv_state is not None and score_state is not None, (
        f"ratio == {RATIO} requires wgate/kv_state/score_state"
    )
    # 源码 ratio>1: wkv/wgate 为 fp32
    _check_dtype("wkv", wkv, torch.float32)

    assert tuple(wgate.shape) == (head_dim, h), (
        f"wgate shape is {tuple(wgate.shape)}, expected {(head_dim, h)}"
    )
    _check_dtype("wgate", wgate, torch.float32)

    expected_state = (bsz, RATIO, head_dim)
    assert tuple(kv_state.shape) == expected_state, (
        f"kv_state shape is {tuple(kv_state.shape)}, expected {expected_state}"
    )
    assert tuple(score_state.shape) == expected_state, (
        f"score_state shape is {tuple(score_state.shape)}, expected {expected_state}"
    )
    _check_dtype("kv_state", kv_state, torch.float32)
    _check_dtype("score_state", score_state, torch.float32)


def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub, precision_type=pypto.PrecisionType.HIGH_PRECISION)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum, precision_type=pypto.PrecisionType.HIGH_PRECISION)
    return xdiv


def rms_norm(
    input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-5
) -> pypto.Tensor:
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.rsqrt(y, precision_type=pypto.PrecisionType.HIGH_PRECISION)   # 对齐 golden 的 torch.rsqrt
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y


def scatter_update_3d(input_tensor, index, src):
    """把 src 的 1 行写进 input 的 index 指定行 (dim=-2)"""
    input_shape = input_tensor.shape
    d = src.shape[2]
    pypto.set_vec_tile_shapes(1, 24, d)
    src = pypto.reshape(src, [src.shape[0] * src.shape[1], src.shape[2]])
    input_tensor = pypto.reshape(input_tensor, [input_tensor.shape[0] * input_tensor.shape[1], input_tensor.shape[2]])
    pypto.set_vec_tile_shapes(24, d)
    output = pypto.scatter_update(input_tensor, -2, index, src)
    return pypto.reshape(output, input_shape)


@pypto.frontend.jit(
    pass_options={},
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def csa2_compressor_ratio1_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    ratio,
    eps,
):
    """ratio == 1 纯投影分支 (源码: return self.norm(self.wkv(x))).

    x     (batch, seq, 5120) bf16, seq 为动态轴
    wkv   (head_dim, 5120)   bf16      # 源码 ratio==1 时 wkv 保持 bf16
    weight(head_dim,)        fp32      # kv_norm.weight
    out   (batch*seq, head_dim) bf16   # 展平行; 每 token 一个 pre-RoPE latent
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    bsz, s1, h = x.shape
    d = weight.shape[0]
    # 把 (batch, seq) 展平成行: x -> [bsz*s1, h]. 每块 256 个连续行送一次 matmul,
    # 这样 matmul 的 A 操作数是一段连续 2D view (无需物化 reshape 拷贝), 与
    # (batch, t_tile) 分块无关; 输出按同一行序写回 out. 对任意 bsz/s1 成立.
    rows = 512
    n_rows = bsz * s1
    r_loop = (n_rows + rows - 1) // rows
    x2d = pypto.reshape(x, [n_rows, h], inplace=True)
    for r_idx in pypto.loop(r_loop, submit_before_loop=True, name="LOOP_CSA2_R1", idx_name="r_idx"):
        r_off = r_idx * rows
        r_valid = (n_rows - r_off).min(rows)
        x2 = pypto.view(x2d, [rows, h], [r_off, 0], valid_shape=[r_valid, h])
        pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
        pypto.set_vec_tile_shapes(16, d)
        kv = pypto.matmul(x2, wkv, pypto.DT_BF16, b_trans=True)  # [rows, d] bf16
        kv_v = pypto.view(kv, [rows, d], [0, 0], valid_shape=[r_valid, d])
        # 源码: self.wkv(x) 输出 bf16, 再 self.norm(...)
        pypto.set_vec_tile_shapes(32, d)
        normed = rms_norm(kv_v, weight, eps)
        pypto.assemble(normed, [r_off, 0], out)


@pypto.frontend.jit(
    pass_options={},
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
        "launch_sched_aicpu_num": 3
    },
)
def csa2_compressor_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    wgate: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    kv_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    start_pos_dy: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    kv_state_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    ratio,
    eps,
):
    """ratio == 2 的压缩 kernel (prefill + decode), seq 为动态轴.

    x                 (batch, seq, 5120) bf16
    wkv/wgate         (head_dim, 5120)   fp32   # 源码 ratio>1 时 wkv/wgate 为 fp32
    weight            (head_dim,)        fp32
    kv_state_total    (batch, 2, head_dim) fp32   # 输入旧状态
    score_state_total (batch, 2, head_dim) fp32   # -inf 初值
    start_pos_dy      (1,) int32
    kv_state_out      (batch, 2, head_dim) fp32   # 与 total 同存储, 写当前 token
    score_state_out   (batch, 2, head_dim) fp32
    out               (batch, seq // 2, head_dim) bf16  # 不压缩时不写
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    assert ratio == RATIO, f"ratio is {ratio}, expected {RATIO}"
    bsz, s1, h = x.shape
    d = weight.shape[0]
    n_groups = s1 // ratio          # 动态
    remainder = s1 % ratio          # 动态
    # A/B (candidate C): 展平批量路径下权重仅被 ceil(n_rows/m_L0) 次重读,
    # 逐组重复加载已消除; 保留 L2 让权重在多次 chunk 间命中 (NONE_CACHEABLE 强制
    # 每次访问走 HBM, 长 seq 下反而更慢). 短 seq 只读一次, 差异可忽略.
    # wkv.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    # wgate.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

    b = 64
    b_loop = (bsz + b - 1) // b
    pool_tile = 8        # S-12: 批量池化的 vec tile 行数 (UB 预算内减少 task 数)
    # cast bf16->fp32 tile 行数: 单 tile 字节 = rows*h*(2+4), h=5120 时 4*5120*6=122880 < 196608.
    cast_tile = 4
    dec_cast_tile = 4
    start = start_pos_dy[0]
    if start < 1:
        if remainder < 1:
            # ------- prefill (seq 偶数): 全 token 展平批量投影 + 组池化 -------
            # 根因 (SWIMLANE): 原 per-group matmul 的 M=b*ratio, b=64 batch padding
            # 无法摊薄 → #m-tile = n_groups*b*ratio/m_L0, 权重 wkv/wgate 被重复加载
            # n_groups 次 (长 seq 512 组 → 8GB+, MTE2 占 87%). 展平后 M=rows 无 padding,
            # 权重仅被 ceil(n_rows/m_L0) 次加载; 组内 ratio 归约轴保持整块 (untiled).
            n_rows = bsz * s1
            # decode 形状 (s1==1) 下 n_groups==0; 此分支不被执行但会被编译,
            # 故 reshape 维用 max(1) 兜底避免 0 维 (prefill 时 n_groups>=1 不变).
            n_groups_flat = n_groups.max(1)
            x2d = pypto.reshape(x, [n_rows, h], inplace=True)
            # S-I: x 为 streaming 只读一次 (134MB @2,8192), 不进 L2 可避免驱逐
            # 反复复用的 16MB 权值 (wkv/wgate 靠 L2 命中承载 chunk 间复用).
            x.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
            out2d = pypto.reshape(out, [bsz * n_groups_flat, d], inplace=True)
            rows = 128
            ngrp = rows // ratio          # 每个 row-chunk 覆盖的组数 (静态)
            r_loop = (n_rows + rows - 1) // rows
            for r_idx in pypto.loop(r_loop, submit_before_loop=True, name="LOOP_CSA2_PRE", idx_name="r_idx"):
                r_off = r_idx * rows
                r_valid = (n_rows - r_off).min(rows)
                x_chunk = pypto.view(x2d, [rows, h], [r_off, 0], valid_shape=[r_valid, h])
                pypto.set_vec_tile_shapes(cast_tile, h)
                x2 = pypto.cast(x_chunk, pypto.DT_FP32)   # 源码: x = x.float()
                # S7: nL0 64->128 提升 L0 基本块 / nL1 128->256 减少 N 轴切分次数,
                # 同一 A 操作数在 L1 复用更多次 (实测 (2,8192) 约 -10%, (64,2) 约 -17%).
                pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 256])
                pypto.set_vec_tile_shapes(16, d)
                kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)   # [rows, d]
                sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)
                # reshape/softmax/sum 会丢失 valid_shape, 显式重挂 (尾块 assemble
                # 若按逻辑 shape [ngrp,d] 写会越界覆盖相邻输入 tensor; ratio1 同款).
                vgrp = r_valid // ratio
                kv3 = pypto.view(pypto.reshape(kv, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[vgrp, ratio, d])
                sc3 = pypto.view(pypto.reshape(sc, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[vgrp, ratio, d])
                pypto.set_vec_tile_shapes(pool_tile, ratio, d)
                alpha = softmax(sc3, 1)                                     # [ngrp, ratio, d]
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                # [ngrp, d] fp32
                pypto.set_vec_tile_shapes(pool_tile, d)
                latent = pypto.cast(latent, pypto.DT_BF16)      # 源码: kv.to(dtype)
                normed = rms_norm(latent, weight, eps)                      # [ngrp, d]
                normed = pypto.view(normed, [ngrp, d], [0, 0], valid_shape=[vgrp, d])
                pypto.assemble(normed, [r_off // ratio, 0], out2d)
        else:
            # ------- prefill (seq 奇数): 展平批量投影 (per-batch) + 尾部余数存 state -------
            # 根因: 原 per-group 路径 M=b*ratio 带 batch padding (b=64), 权重按 n_groups
            # 次重载 (长 seq 下 MTE2 占主导, bsz=1/seq=131071 达 1s+).
            # 偶数 seq 的平铺路径把 (bsz*s1) 行当成单一连续流, 依赖 "batch 边界与组边界对齐"
            # —— 该性质仅在 s1 为偶数时成立 (奇 s1 时第 2 个 batch 的组配对整体偏移 1).
            # 奇 s1 下按 batch 循环: 每个 batch 的 [b*s1, (b+1)*s1) 行连续且相对 b*s1 组对齐,
            # 用静态 row-chunk 展平投影 (与偶数路径同款, 权重仅按 chunk 重载).
            # 关键: 每个 batch 拆成 n_full 个完整 128-row chunk + 1 个尾块. 尾块覆盖到
            # batch 末尾、含 remainder token, 其有效行数恒为奇数; 池化按 tail_grp =
            # tail_valid // ratio 取整后, 末尾多余 token 自然不成组 → 不参与池化, 且其投影
            # 复用尾块同一 matmul (无额外权重重载), 只需 view+assemble 写回 state.
            n_rows = bsz * s1
            n_groups_flat = n_groups.max(1)
            x2d = pypto.reshape(x, [n_rows, h], inplace=True)
            out2d = pypto.reshape(out, [bsz * n_groups_flat, d], inplace=True)
            rows = 128
            ngrp = rows // ratio
            # 每个 batch 的完整 128-row chunk 数 + 1 个尾块 (尾块含 remainder, 有效行数恒为奇数)
            cutoff = s1 - remainder
            n_full = cutoff // rows
            tail_off = n_full * rows
            tail_valid = s1 - tail_off
            tail_grp = tail_valid // ratio
            for b_idx in pypto.loop(bsz, name="LOOP_CSA2_ODDB", idx_name="b_idx"):
                b_base = b_idx * s1
                for r_idx in pypto.loop(n_full, name="LOOP_CSA2_ODDR", idx_name="r_idx"):
                    r_off = b_base + r_idx * rows
                    x_chunk = pypto.view(x2d, [rows, h], [r_off, 0], valid_shape=[rows, h])
                    pypto.set_vec_tile_shapes(cast_tile, h)
                    x2 = pypto.cast(x_chunk, pypto.DT_FP32)   # 源码: x = x.float()
                    pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 256])
                    pypto.set_vec_tile_shapes(16, d)
                    kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)   # [rows, d]
                    sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)
                    kv3 = pypto.reshape(kv, [ngrp, ratio, d])
                    sc3 = pypto.reshape(sc, [ngrp, ratio, d])
                    pypto.set_vec_tile_shapes(pool_tile, ratio, d)
                    alpha = softmax(sc3, 1)                                     # [ngrp, ratio, d]
                    latent = pypto.sum(pypto.mul(alpha, kv3), 1)                # [ngrp, d] fp32
                    pypto.set_vec_tile_shapes(pool_tile, d)
                    latent = pypto.cast(latent, pypto.DT_BF16)      # 源码: kv.to(dtype)
                    normed = rms_norm(latent, weight, eps)                      # [ngrp, d]
                    pypto.assemble(normed, [b_idx * n_groups + r_idx * ngrp, 0], out2d)
                # ---- 尾块: 完整 128-row chunk 之后剩余的行 (含 remainder token) ----
                x_tail = pypto.view(x2d, [rows, h], [b_base + tail_off, 0],
                                    valid_shape=[tail_valid, h])
                pypto.set_vec_tile_shapes(cast_tile, h)
                x2 = pypto.cast(x_tail, pypto.DT_FP32)
                pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 256])
                pypto.set_vec_tile_shapes(16, d)
                kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)     # [rows, d]
                sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)
                # 有效行数为奇数: ratio*vgrp 行成组池化, 末尾 remainder 行不成组
                kv3 = pypto.view(pypto.reshape(kv, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[tail_grp, ratio, d])
                sc3 = pypto.view(pypto.reshape(sc, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[tail_grp, ratio, d])
                pypto.set_vec_tile_shapes(pool_tile, ratio, d)
                alpha = softmax(sc3, 1)
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                # [ngrp, d] fp32
                pypto.set_vec_tile_shapes(pool_tile, d)
                latent = pypto.cast(latent, pypto.DT_BF16)
                normed = rms_norm(latent, weight, eps)                      # [ngrp, d]
                normed = pypto.view(normed, [ngrp, d], [0, 0], valid_shape=[tail_grp, d])
                pypto.assemble(normed, [b_idx * n_groups + n_full * ngrp, 0], out2d)
                # 尾部 remainder token 存 state (源码: kv_state[:bsz, :remainder] = ...)
                kv_v = pypto.view(pypto.reshape(kv, [rows, 1, d]), [1, 1, d],
                                  [tail_grp * ratio, 0, 0], valid_shape=[1, 1, d])
                sc_v = pypto.view(pypto.reshape(sc, [rows, 1, d]), [1, 1, d],
                                  [tail_grp * ratio, 0, 0], valid_shape=[1, 1, d])
                pypto.assemble(kv_v, [b_idx, 0, 0], kv_state_out)
                pypto.assemble(sc_v, [b_idx, 0, 0], score_state_out)


    else:
        # ---------------- decode: 跨 batch 批量投影 (M=b), 组满时批量池化 ----------------
        # slot = start % ratio 为标量且全 batch 一致, 故可整块处理:
        # 压缩步 start 恒为奇数 → slot 恒为 1, 另一槽位恒为 0; 直接读旧槽 0 +
        # 当前 token 做两槽 softmax 池化, 无需 scatter_update / 逐 batch 循环.
        for b_idx in pypto.loop(b_loop, submit_before_loop=True, name="LOOP_CSA2_1", idx_name="b_idx"):
            b_valid = (bsz - b_idx * b).min(b)
            slot = start % ratio
            x_tok = pypto.view(x, [b, 1, h], [b_idx * b, 0, 0],
                               valid_shape=[b_valid, 1, h])
            pypto.set_vec_tile_shapes(dec_cast_tile, 1, h)   # h=5120 cast tile (原固定 8 行会溢出 MEM_UB)
            x_tok_fp32 = pypto.cast(x_tok, pypto.DT_FP32)
            x2 = pypto.reshape(x_tok_fp32, [b, h])
            # SWIMLANE: nL0 128->64 (4->8 N 块, 16 个 cube task 填满核), kL0 64->128
            pypto.set_cube_tile_shapes([64, 128], [128, 256], [64, 256])
            pypto.set_vec_tile_shapes(16, d)
            kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)     # [b, d]
            sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)    # [b, d]
            kv3 = pypto.reshape(kv, [b, 1, d])
            sc3 = pypto.reshape(sc, [b, 1, d])
            tok_kv = pypto.view(kv3, [b, 1, d], [0, 0, 0], valid_shape=[b_valid, 1, d])
            tok_sc = pypto.view(sc3, [b, 1, d], [0, 0, 0], valid_shape=[b_valid, 1, d])
            pypto.set_vec_tile_shapes(16, 1, d)
            pypto.assemble(tok_kv, [b_idx * b, slot, 0], kv_state_out)
            pypto.assemble(tok_sc, [b_idx * b, slot, 0], score_state_out)
            if (start + 1) % ratio < 1:
                # 组刚完成: 旧槽 0 + 当前 token (=slot 1) 组成完整组, 手动展开 ratio=2
                # 的 softmax 池化 (与原 softmax+dsum 数值路径一致).
                st0_kv = pypto.view(kv_state_total, [b, 1, d], [b_idx * b, 0, 0],
                                    valid_shape=[b_valid, 1, d])
                st0_sc = pypto.view(score_state_total, [b, 1, d], [b_idx * b, 0, 0],
                                    valid_shape=[b_valid, 1, d])
                pypto.set_vec_tile_shapes(4, 1, d)   # SWIMLANE: 池化 vec tile 16->4 (4->16 task)
                m = pypto.maximum(st0_sc, tok_sc)
                e0 = pypto.exp(pypto.sub(st0_sc, m),
                               precision_type=pypto.PrecisionType.HIGH_PRECISION)
                e1 = pypto.exp(pypto.sub(tok_sc, m),
                               precision_type=pypto.PrecisionType.HIGH_PRECISION)
                denom = pypto.add(e0, e1)
                a0 = pypto.div(e0, denom, precision_type=pypto.PrecisionType.HIGH_PRECISION)
                a1 = pypto.div(e1, denom, precision_type=pypto.PrecisionType.HIGH_PRECISION)
                latent = pypto.add(pypto.mul(a0, st0_kv), pypto.mul(a1, tok_kv))  # [b,1,d] fp32
                pypto.set_vec_tile_shapes(4, 1, d)
                latent = pypto.cast(latent, pypto.DT_BF16)      # 源码: kv.to(dtype)
                normed = rms_norm(latent, weight, eps)
                pypto.assemble(normed, [b_idx * b, 0, 0], out)


@pypto.frontend.jit(
    pass_options={},
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
        "launch_sched_aicpu_num": 3,
    },
)
def csa2_compressor_th_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    wgate: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    out_base: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    kv_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_total: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    start_pos_dy: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    kv_state_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    ratio,
    eps,
):
    """ratio == 2 TH (BS 合轴) kernel: x (T, h) 2D、cu_seqlens 前缀和分段.

    prefill (start_pos < 1): 每 batch 段 [cu[b], cu[b+1]) 独立分组池化 —— 段内
        rows=128 整块 + 尾块 (valid_shape 裁剪), 压缩行 compact 写 out[out_base[b] + ...],
        段尾余数 token 写 state[b, 0]。
    decode  (start_pos >= 1): 每 batch 1 token (行序 = batch 序), 写 state[b, slot],
        组满时按两槽 softmax 池化写 out[b]。
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    assert ratio == RATIO, f"ratio is {ratio}, expected {RATIO}"
    t, h = x.shape
    d = weight.shape[0]
    nb = cu_seqlens.shape[0] - 1
    start = start_pos_dy[0]

    if start < 1:
        # ---------------- prefill: 逐 batch 段分组池化 (动态 L, 动态输出偏移) ----------------
        rows = 128
        ngrp = rows // ratio              # 每 row-chunk 覆盖的组数 (静态)
        cast_tile = 4
        pool_tile = 8
        for b_idx in pypto.loop(nb, name="LOOP_CSA2_TH_B", idx_name="b_idx"):
            tok0 = cu_seqlens[b_idx]
            cur_len = cu_seqlens[b_idx + 1] - tok0
            n_full = cur_len // rows
            rem = cur_len - n_full * rows
            ob = out_base[b_idx]
            for r_idx in pypto.loop(n_full, name="LOOP_CSA2_TH_R", idx_name="r_idx"):
                r_off = tok0 + r_idx * rows
                x_chunk = pypto.view(x, [rows, h], [r_off, 0])
                pypto.set_vec_tile_shapes(cast_tile, h)
                x2 = pypto.cast(x_chunk, pypto.DT_FP32)   # 源码: x = x.float()
                pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 256])
                pypto.set_vec_tile_shapes(16, d)
                kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)   # [rows, d]
                sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)
                kv3 = pypto.reshape(kv, [ngrp, ratio, d])
                sc3 = pypto.reshape(sc, [ngrp, ratio, d])
                pypto.set_vec_tile_shapes(pool_tile, ratio, d)
                alpha = softmax(sc3, 1)                                     # [ngrp, ratio, d]
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                # [ngrp, d] fp32
                pypto.set_vec_tile_shapes(pool_tile, d)
                latent = pypto.cast(latent, pypto.DT_BF16)      # 源码: kv.to(dtype)
                normed = rms_norm(latent, weight, eps)                      # [ngrp, d]
                pypto.assemble(normed, [ob + r_idx * ngrp, 0], out)
            if rem > 0:
                # ---- 尾块: 段内剩余行 (含余数 token), 有效行数动态 ----
                t_off = tok0 + n_full * rows
                x_tail = pypto.view(x, [rows, h], [t_off, 0], valid_shape=[rem, h])
                pypto.set_vec_tile_shapes(cast_tile, h)
                x2 = pypto.cast(x_tail, pypto.DT_FP32)
                pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 256])
                pypto.set_vec_tile_shapes(16, d)
                kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)     # [rows, d]
                sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)
                vgrp = rem // ratio          # 尾块成组的行数 (动态)
                kv3 = pypto.view(pypto.reshape(kv, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[vgrp, ratio, d])
                sc3 = pypto.view(pypto.reshape(sc, [ngrp, ratio, d]), [ngrp, ratio, d],
                                 [0, 0, 0], valid_shape=[vgrp, ratio, d])
                pypto.set_vec_tile_shapes(pool_tile, ratio, d)
                alpha = softmax(sc3, 1)
                latent = pypto.sum(pypto.mul(alpha, kv3), 1)                # [ngrp, d] fp32
                pypto.set_vec_tile_shapes(pool_tile, d)
                latent = pypto.cast(latent, pypto.DT_BF16)
                normed = rms_norm(latent, weight, eps)                      # [ngrp, d]
                normed = pypto.view(normed, [ngrp, d], [0, 0], valid_shape=[vgrp, d])
                pypto.assemble(normed, [ob + n_full * ngrp, 0], out)
                if rem % ratio > 0:
                    # 段尾余数 token 存 state (源码: kv_state[:bsz, :remainder] = ...)
                    kv_v = pypto.view(pypto.reshape(kv, [rows, 1, d]), [1, 1, d],
                                      [vgrp * ratio, 0, 0], valid_shape=[1, 1, d])
                    sc_v = pypto.view(pypto.reshape(sc, [rows, 1, d]), [1, 1, d],
                                      [vgrp * ratio, 0, 0], valid_shape=[1, 1, d])
                    pypto.assemble(kv_v, [b_idx, 0, 0], kv_state_out)
                    pypto.assemble(sc_v, [b_idx, 0, 0], score_state_out)

    else:
        # ---------------- TH decode: 每 batch 1 token (行序 = batch 序) ----------------
        # start_pos 为标量且全 batch 一致, slot = start % ratio 全局相同:
        # 跨 batch 批量投影 (M=64), 写 state[b, slot]; 组满时批量两槽池化写 out[b]。
        b = 64
        d_loop = (nb + b - 1) // b
        slot = start % ratio
        for b_idx in pypto.loop(d_loop, submit_before_loop=True, name="LOOP_CSA2_TH_D", idx_name="b_idx"):
            b_valid = (nb - b_idx * b).min(b)
            x_tok = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
            pypto.set_vec_tile_shapes(4, h)
            x2 = pypto.cast(x_tok, pypto.DT_FP32)
            pypto.set_cube_tile_shapes([64, 128], [128, 256], [64, 256])
            pypto.set_vec_tile_shapes(16, d)
            kv = pypto.matmul(x2, wkv, pypto.DT_FP32, b_trans=True)     # [b, d]
            sc = pypto.matmul(x2, wgate, pypto.DT_FP32, b_trans=True)    # [b, d]
            kv3 = pypto.reshape(kv, [b, 1, d])
            sc3 = pypto.reshape(sc, [b, 1, d])
            tok_kv = pypto.view(kv3, [b, 1, d], [0, 0, 0], valid_shape=[b_valid, 1, d])
            tok_sc = pypto.view(sc3, [b, 1, d], [0, 0, 0], valid_shape=[b_valid, 1, d])
            pypto.set_vec_tile_shapes(16, 1, d)
            pypto.assemble(tok_kv, [b_idx * b, slot, 0], kv_state_out)
            pypto.assemble(tok_sc, [b_idx * b, slot, 0], score_state_out)
            if (start + 1) % ratio < 1:
                # 组刚完成: 旧槽 0 + 当前 token 两槽 softmax 池化
                st0_kv = pypto.view(kv_state_total, [b, 1, d], [b_idx * b, 0, 0],
                                    valid_shape=[b_valid, 1, d])
                st0_sc = pypto.view(score_state_total, [b, 1, d], [b_idx * b, 0, 0],
                                    valid_shape=[b_valid, 1, d])
                pypto.set_vec_tile_shapes(4, 1, d)
                m = pypto.maximum(st0_sc, tok_sc)
                e0 = pypto.exp(pypto.sub(st0_sc, m),
                               precision_type=pypto.PrecisionType.HIGH_PRECISION)
                e1 = pypto.exp(pypto.sub(tok_sc, m),
                               precision_type=pypto.PrecisionType.HIGH_PRECISION)
                denom = pypto.add(e0, e1)
                a0 = pypto.div(e0, denom, precision_type=pypto.PrecisionType.HIGH_PRECISION)
                a1 = pypto.div(e1, denom, precision_type=pypto.PrecisionType.HIGH_PRECISION)
                latent = pypto.add(pypto.mul(a0, st0_kv), pypto.mul(a1, tok_kv))  # [b,1,d] fp32
                pypto.set_vec_tile_shapes(4, 1, d)
                latent = pypto.cast(latent, pypto.DT_BF16)      # 源码: kv.to(dtype)
                normed = rms_norm(latent, weight, eps)          # [b,1,d]
                normed2 = pypto.reshape(normed, [b, d], valid_shape=[b_valid, d])
                pypto.assemble(normed2, [b_idx * b, 0], out)
