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
 * \file swi_glu.cpp
 * \brief
 */
#ifndef OPP_MEGA_MOE_SWI_GLU_CPP
#define OPP_MEGA_MOE_SWI_GLU_CPP
#include "swi_glu_bf16.hpp"
#include "swi_glu_single.hpp"

using namespace AscendC;
extern "C" inline __aicore__ void swi_glu(GM_ADDR input_gm, GM_ADDR output_gm,
                                       GM_ADDR workspace, GM_ADDR tiling, float clamp_limit) {
  SwigluSingle<SwigluVectorBF16<DTYPE_Y, float, DTYPE_Y, 2>, DTYPE_Y, DTYPE_Y> op;
  op.Init(input_gm, nullptr, output_gm, tiling, clamp_limit);
  op.Process();
}
#endif  // OPP_MEGA_MOE_SWI_GLU_CPP
