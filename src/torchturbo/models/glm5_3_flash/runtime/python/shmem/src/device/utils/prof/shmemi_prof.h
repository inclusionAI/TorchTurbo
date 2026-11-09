/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_CYCLE_PROF_H
#define SHMEMI_CYCLE_PROF_H

#include "host_device/shmem_common_types.h"
#include "device/shmemi_device_common.h"

#define SHMEMI_PROF_START(pf_id)                                                                         \
    if ((pf_id) < ACLSHMEM_CYCLE_PROF_FRAME_CNT && AscendC::GetBlockIdx() < ACLSHMEM_CYCLE_PROF_FRAME_CNT) {   \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                      \
        if (device_state->profs != nullptr && device_state->profs->pe_id == shmem_my_pe()) {                    \
            pipe_barrier(PIPE_ALL);                                                                     \
            auto cycle = AscendC::GetSystemCycle();                                                     \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].cycles[pf_id] -= cycle;                   \
        }                                                                                               \
    }

#define SHMEMI_PROF_END(pf_id)                                                                           \
    if ((pf_id) < ACLSHMEM_CYCLE_PROF_FRAME_CNT && AscendC::GetBlockIdx() < ACLSHMEM_CYCLE_PROF_FRAME_CNT) {   \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                      \
        if (device_state->profs != nullptr && device_state->profs->pe_id == shmem_my_pe()) {                    \
            pipe_barrier(PIPE_ALL);                                                                     \
            auto cycle = AscendC::GetSystemCycle();                                                     \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].cycles[pf_id] += cycle;                   \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].ccount[pf_id] += 1;                       \
        }                                                                                               \
    }

#endif

