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
"""CSA2 Compressor 公共入口：前向 kernel + 反向 kernel 连跑。

对外暴露三种形式（对齐 liger_cross_entropy_loss.py 的公共入口风格）：
    * ``CSA2CompressorFunction`` —— ``torch.autograd.Function``，前向/反向入口；
    * ``csa2_compressor``        —— functional 形式；
    * ``CSA2Compressor``         —— ``nn.Module`` 形式（持有 ratio / eps）。

依赖边界（除 pip 包外）只有同包两个 eager wrapper 入口（相对导入, 直调绕过 torch.ops
派发; 本文件与两个 impl 文件可整体移植到任意包目录）：
    * ``.CSA2_compressor_fwd_impl.npu_csa2_compressor``        —— 前向 kernel
    * ``.CSA2_compressor_bwd_impl.npu_csa2_compressor_bwd``    —— 反向 kernel

接口约定（与两个 kernel 一致）：
    x                              (batch, seq, h)                     bf16
    wkv                            (head_dim, h)                       ratio>1: fp32, ratio==1: bf16
    wgate                          (head_dim, h) 或 None               ratio>1: fp32, ratio==1: None
    weight                         (head_dim,)                         fp32   归一化权重
    kv_state / score_state         (batch, ratio, head_dim) 或 None     ratio>1: fp32 cache, ratio==1: None;
                                   dim0 允许 >= batch (max_batch_size 预分配), 入口内精确切片,
                                   原地写落回调用方 buffer 的 [0, batch) 行
    start_pos                      (1,) int32 设备张量（aclnn 图捕获前创建，勿在捕获内新建）
    cu_seqlens                     TH: (bsz+1,) int32 前缀和; BSH: None   (位于标量参数之前)
    ratio                          1 或 2
    eps                            RMSNorm epsilon（float）

语义说明：
    前向会**原地更新** kv_state/score_state（源码 register_buffer 语义），反向从中读取
    前向后的值；因此 state 作为不可微 cache，不产生梯度（返回 None）；梯度回传到
    x / wkv / wgate / weight。

用法：
    from deepseek_v4_1.CSA2_compressor import csa2_compressor
    out = csa2_compressor(x, wkv, wgate, weight, kv_state, score_state, start_pos,
                          cu_seqlens, 2, 1e-20)
    out.sum().backward()          # dx / dwkv / dwgate / dweight 落到各输入的 .grad
"""

from __future__ import annotations

__all__ = [
    "CSA2CompressorFunction",
    "csa2_compressor",
    "CSA2Compressor",
]

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_npu  # noqa: E402,F401  NPU 设备初始化必需

try:  # 包内布局 (如 deepseek_v4_1 包): 相对导入
    from .CSA2_compressor_fwd_impl import (  # noqa: E402
        npu_csa2_compressor,
        CSA2CompressorArgs,
    )
    from .CSA2_compressor_bwd_impl import (  # noqa: E402
        npu_csa2_compressor_bwd,
        CSA2CompressorBwdArgs,
    )
except ImportError:  # 平铺交付布局: 算子与测试同目录, 作为顶层模块导入
    from CSA2_compressor_fwd_impl import (  # noqa: E402
        npu_csa2_compressor,
        CSA2CompressorArgs,
    )
    from CSA2_compressor_bwd_impl import (  # noqa: E402
        npu_csa2_compressor_bwd,
        CSA2CompressorBwdArgs,
    )


class CSA2CompressorFunction(torch.autograd.Function):
    """前向 kernel + 反向 kernel 的组合入口。"""

    @staticmethod
    def forward(ctx, x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens,
                ratio, eps):
        # state 允许按 max_batch_size 预分配 (dim0 >= 当前 batch, 对齐推理模型 buffer 惯例):
        # 入口内精确切片到 (batch, ratio, d) 以过 kernel 的 shape 契约; dim0 切片连续且共享
        # 存储, 前向的原地写仍落到调用方真实 buffer 的 [0, batch) 行, [batch:) 行不受影响;
        # 反向经 ctx 保存的是切片 (精确 shape, 且读到前向写后的值)。
        # BSH (cu_seqlens 为 None): batch = x.shape[0];
        # TH (BS 合轴, cu_seqlens 给定): batch = cu_seqlens 段数 (x dim0 为 token 总数 T)
        batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else x.shape[0]
        if kv_state is not None and kv_state.shape[0] != batch:
            assert kv_state.shape[0] > batch and score_state.shape[0] > batch, (
                f"state dim0 is ({kv_state.shape[0]}, {score_state.shape[0]}), "
                f"expected == batch {batch} or > it (max_batch_size 预分配)")
            kv_state = kv_state[:batch]
            score_state = score_state[:batch]
        out = npu_csa2_compressor(CSA2CompressorArgs(
            x=x, wkv=wkv, wgate=wgate, weight=weight,
            kv_state=kv_state, score_state=score_state,
            start_pos=start_pos, ratio=ratio, eps=eps, cu_seqlens=cu_seqlens,
        ))
        # 反向需要前向后的 state（前向已原地更新）与全部输入
        if ratio == 1:
            ctx.save_for_backward(x, wkv, weight, start_pos, cu_seqlens)
        else:
            ctx.save_for_backward(x, wkv, wgate, weight, kv_state, score_state, start_pos,
                                  cu_seqlens)
        ctx.ratio = ratio
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.ratio == 1:
            x, wkv, weight, start_pos, cu_seqlens = ctx.saved_tensors
            wgate = kv_state = score_state = None
        else:
            x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens = \
                ctx.saved_tensors

        dx, dwkv, dwgate, dweight = npu_csa2_compressor_bwd(CSA2CompressorBwdArgs(
            grad_out=grad_out, x=x, wkv=wkv, wgate=wgate, weight=weight,
            kv_state=kv_state, score_state=score_state,
            start_pos=start_pos, ratio=ctx.ratio, eps=ctx.eps, cu_seqlens=cu_seqlens,
        ))
        # 顺序对齐 forward 入参: x, wkv, wgate, weight, kv_state, score_state,
        # start_pos, cu_seqlens, ratio, eps
        return dx, dwkv, dwgate, dweight, None, None, None, None, None, None


def csa2_compressor(
    x,
    wkv,
    wgate,
    weight,
    kv_state,
    score_state,
    start_pos,
    cu_seqlens=None,
    ratio=2,
    eps=1e-20,
):
    """functional 入口：前向 +（按需）反向 autograd 连跑。"""
    return CSA2CompressorFunction.apply(
        x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens, ratio, eps
    )


class CSA2Compressor(nn.Module):
    """nn.Module 入口：持有 ratio / eps 配置，权重与 state 由调用方按参数传入。"""

    def __init__(self, ratio: int = 2, eps: float = 1e-20):
        super().__init__()
        assert ratio in (1, 2), f"ratio is {ratio}, expected 1 or 2"
        assert isinstance(eps, float) and eps >= 0.0, (
            f"eps is {eps}, expected non-negative float"
        )
        self.ratio = ratio
        self.eps = eps

    def forward(self, x, wkv, wgate, weight, kv_state, score_state, start_pos,
                cu_seqlens=None):
        return csa2_compressor(
            x, wkv, wgate, weight, kv_state, score_state, start_pos,
            cu_seqlens, self.ratio, self.eps,
        )
