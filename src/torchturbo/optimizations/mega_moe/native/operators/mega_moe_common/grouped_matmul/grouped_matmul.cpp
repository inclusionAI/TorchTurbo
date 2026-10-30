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
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul.h"

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

namespace {
constexpr MatmulConfig matmulCFG = CFG_MDL;
}

template <bool trans = false>
using xType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_DISPATCH_TARGET, trans>;
template <bool trans = false>
using weightType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;
using yType = MatmulType<TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;
using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, &mmTilingData_, &tPipe);                                        \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

extern "C" inline __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling, bool transA, bool transB)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

    if (transA) {
        GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
    } else if (transB) {
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
    } else {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
    }

}
