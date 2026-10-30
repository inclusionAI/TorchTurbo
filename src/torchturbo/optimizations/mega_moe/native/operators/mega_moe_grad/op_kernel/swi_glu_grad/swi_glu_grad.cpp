/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad.cpp
 * \brief
 */
#include "swi_glu_grad_bf16.hpp"
#include "swi_glu_grad_single.hpp"

using namespace AscendC;
extern "C" inline __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
                                                   GM_ADDR workspace, GM_ADDR tiling, float clamp_limit) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tempTilingGm, tiling);
    if (tempTilingGm.isDoubleBuffer == 1) {
        SwiGluGradSingle<SwiGluGradBF16<DTYPE_DY, float, DTYPE_DY, 2>, DTYPE_DY, DTYPE_DY> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling, clamp_limit);
        op.Process();
    } else {
        SwiGluGradSingle<SwiGluGradBF16<DTYPE_DY, float, DTYPE_DY, 1>, DTYPE_DY, DTYPE_DY> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling, clamp_limit);
        op.Process();
    }
}
