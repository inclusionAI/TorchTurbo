/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_TEAM_HPP
#define SHMEM_DEVICE_TEAM_HPP

#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE int aclshmem_my_pe(void)
{
    return aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
}

ACLSHMEM_DEVICE int aclshmem_n_pes(void)
{
    return aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->size;
}

ACLSHMEM_DEVICE int aclshmem_team_my_pe(aclshmem_team_t team)
{
    if (team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    } else {
        aclshmemx_team_t *src_team_ptr = aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->mype : -1);
    }
}

ACLSHMEM_DEVICE int aclshmem_team_n_pes(aclshmem_team_t team)
{
    if (team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    } else {
        aclshmemx_team_t *src_team_ptr = aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->size : -1);
    }
}

ACLSHMEM_DEVICE int aclshmem_team_translate_pe(aclshmem_team_t src_team, int src_pe, aclshmem_team_t dest_team)
{
    if (src_team == ACLSHMEM_TEAM_INVALID || dest_team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    aclshmemx_team_t *src_team_ptr = device_state->team_pools[src_team];
    aclshmemx_team_t *dest_team_ptr = device_state->team_pools[dest_team];

    if (src_pe >= src_team_ptr->size) {
        return -1;
    }

    int global_pe = src_team_ptr->pe_mapping[src_pe];
    return dest_team_ptr->pe_mapping[global_pe + ACLSHMEM_MAX_PES];
}

ACLSHMEM_DEVICE int aclshmem_team_pe_mapping(aclshmem_team_t team, int pe)
{
    if (team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    aclshmemx_team_t *team_ptr = device_state->team_pools[team];

    if (pe >= team_ptr->size) {
        return -1;
    }

    return team_ptr->pe_mapping[pe];
}

#ifdef __cplusplus
}
#endif

#endif