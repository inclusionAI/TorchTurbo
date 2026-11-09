/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SHMEMI_MSTX_REPORT_H__
#define __SHMEMI_MSTX_REPORT_H__

#include <cstdint>

#ifdef __MSTX_DFX_REPORT__

#include "sanitizer_report.h"

#define MSTX_FUSE_SCOPE_START() Sanitizer::SanitizerFuseScopeStart()

#define MSTX_FUSE_SCOPE_END() Sanitizer::SanitizerFuseScopeEnd()

#define MSTX_CROSS_CORE_BARRIER_REPORT(core_num)          \
    do {                                                  \
        Sanitizer::MstxCrossCoreBarrier mstx_barrier__{}; \
        mstx_barrier__.usedCoreNum = (core_num);          \
        Sanitizer::SanitizerReport(mstx_barrier__);       \
    } while (0)

#define MSTX_CROSS_NPU_BARRIER_REPORT(team_size, core_num) \
    do {                                                   \
        Sanitizer::MstxCrossNpuBarrier mstx_barrier__{};   \
        mstx_barrier__.usedDeviceNum = (team_size);        \
        mstx_barrier__.usedCoreNum = (core_num);           \
        Sanitizer::SanitizerReport(mstx_barrier__);        \
    } while (0)

#define MSTX_CROSS_CORE_BARRIER_REPORT_WITH_PIPE(team_size, core_num, pipe_barrier_all) \
    do {                                                                                \
        Sanitizer::MstxCrossNpuBarrier mstx_barrier__{};                                \
        mstx_barrier__.usedDeviceNum = (team_size);                                     \
        mstx_barrier__.usedCoreNum = (core_num);                                        \
        mstx_barrier__.pipeBarrierAll = (pipe_barrier_all);                             \
        Sanitizer::SanitizerReport(mstx_barrier__);                                     \
    } while (0)

#ifdef __CCE_AICORE_ENABLE_MIX__
#define MSTX_BARRIER_NPU_REPORT(team_size, core_num) \
    MSTX_CROSS_CORE_BARRIER_REPORT_WITH_PIPE(team_size, core_num, true)
#else
#define MSTX_BARRIER_NPU_REPORT(team_size, core_num) \
    MSTX_CROSS_CORE_BARRIER_REPORT_WITH_PIPE(team_size, core_num, false)
#endif

#define MSTX_SIGNAL_SET_REPORT(addr_, val_)     \
    do {                                        \
        Sanitizer::MstxSignalSet mstx_set__{};  \
        mstx_set__.addr = (uint64_t)(addr_);    \
        mstx_set__.value = (val_);              \
        Sanitizer::SanitizerReport(mstx_set__); \
    } while (0)

#define MSTX_SIGNAL_WAIT_REPORT(addr_, cmp_, val_)        \
    do {                                                  \
        Sanitizer::MstxSignalWait mstx_wait__{};          \
        mstx_wait__.addr = (uint64_t)(addr_);             \
        mstx_wait__.cmpValue = (val_);                    \
        mstx_wait__.cmpOp = (Sanitizer::CompareOp)(cmp_); \
        Sanitizer::SanitizerReport(mstx_wait__);          \
    } while (0)

#else

#define MSTX_FUSE_SCOPE_START()
#define MSTX_FUSE_SCOPE_END()
#define MSTX_CROSS_CORE_BARRIER_REPORT(core_num)
#define MSTX_CROSS_NPU_BARRIER_REPORT(team_size, core_num)
#define MSTX_CROSS_CORE_BARRIER_REPORT_WITH_PIPE(team_size, core_num, pipe_barrier_all)
#define MSTX_BARRIER_NPU_REPORT(team_size, core_num)
#define MSTX_SIGNAL_SET_REPORT(addr_, val_)
#define MSTX_SIGNAL_WAIT_REPORT(addr_, cmp_, val_)

#endif

#endif
