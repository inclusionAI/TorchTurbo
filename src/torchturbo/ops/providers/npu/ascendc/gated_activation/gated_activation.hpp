/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the upstream source repository for the full text of the License.
 */
#pragma once
#include <type_traits>
#include "kernel_operator.h"

// Adapted from ops-transformer/mc2/mega_moe/op_kernel/arch22/utils/gated_activation.hpp.
// Standalone training adds GELU, derivatives and native intermediate rounding.
namespace TorchTurbo {
using namespace AscendC;

__aicore__ inline void ComputeSiluGate(const LocalTensor<float>& output,
                                      const LocalTensor<float>& gate, uint32_t count)
{
    Muls(output, gate, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Exp(output, output, count);
    PipeBarrier<PIPE_V>();
    Adds(output, output, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Div(output, gate, output, count);
    PipeBarrier<PIPE_V>();
}

// a = activation(g), derivative = d activation(g) / dg. Temporary tensors
// are disjoint, including the tanh input and output used by NPU GELU.
__aicore__ inline void Activation(const LocalTensor<float>& a, const LocalTensor<float>& derivative,
                                  const LocalTensor<float>& g, const LocalTensor<float>& t,
                                  const LocalTensor<float>& s, bool gelu, uint32_t count, bool backward)
{
    if (!gelu) {
        Muls(t, g, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(t, t, count);
        PipeBarrier<PIPE_V>();
        Adds(t, t, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(a, g, t, count);
        PipeBarrier<PIPE_V>();
        if (!backward) return;
        Duplicate(s, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(t, s, t, count);
        PipeBarrier<PIPE_V>();
        Sub(s, s, t, count);
        PipeBarrier<PIPE_V>();
        Mul(s, s, g, count);
        PipeBarrier<PIPE_V>();
        Adds(s, s, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Mul(derivative, t, s, count);
    } else {
        // torch_npu GELU uses the tanh approximation in forward and backward.
        Mul(t, g, g, count);
        PipeBarrier<PIPE_V>();
        Muls(t, t, 0.044715f, count);
        PipeBarrier<PIPE_V>();
        Adds(t, t, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Mul(t, t, g, count);
        PipeBarrier<PIPE_V>();
        Muls(t, t, -1.5957691216057308f, count);
        PipeBarrier<PIPE_V>();
        Exp(t, t, count);
        PipeBarrier<PIPE_V>();
        Adds(t, t, 1.0f, count);
        Duplicate(s, 2.0f, count);
        PipeBarrier<PIPE_V>();
        Div(t, s, t, count);
        PipeBarrier<PIPE_V>();
        Adds(t, t, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(a, t, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(a, a, 0.5f, count);
        if (!backward) {
            PipeBarrier<PIPE_V>();
            Mul(a, a, g, count);
            PipeBarrier<PIPE_V>();
            return;
        }
        Mul(s, t, t, count);
        PipeBarrier<PIPE_V>();
        Muls(s, s, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(s, s, 1.0f, count);
        Mul(derivative, g, g, count);
        PipeBarrier<PIPE_V>();
        Muls(derivative, derivative, 0.134145f, count);
        PipeBarrier<PIPE_V>();
        Adds(derivative, derivative, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(derivative, derivative, 0.3989422804014327f, count);
        PipeBarrier<PIPE_V>();
        Mul(derivative, derivative, g, count);
        PipeBarrier<PIPE_V>();
        Mul(derivative, derivative, s, count);
        PipeBarrier<PIPE_V>();
        Add(derivative, derivative, a, count);
        Mul(a, a, g, count);
    }
    PipeBarrier<PIPE_V>();
}
} // namespace TorchTurbo
