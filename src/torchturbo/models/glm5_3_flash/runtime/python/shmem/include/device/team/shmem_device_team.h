/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_TEAM_H
#define SHMEM_DEVICE_TEAM_H

#include "host_device/shmem_common_types.h"
#include "team/shmem_device_team.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
ACLSHMEM_DEVICE int aclshmem_my_pe(void);
#define shmem_my_pe aclshmem_my_pe

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
ACLSHMEM_DEVICE int aclshmem_n_pes(void);
#define shmem_n_pes aclshmem_n_pes

/**
 * @brief Returns the number of the calling PE in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_DEVICE int aclshmem_team_my_pe(aclshmem_team_t team);
#define shmem_team_my_pe aclshmem_team_my_pe

/**
 * @brief Returns the number of PEs in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_DEVICE int aclshmem_team_n_pes(aclshmem_team_t team);
#define shmem_team_n_pes aclshmem_team_n_pes

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 *
 * @param src_team           [in] A ACLSHMEM team handle.
 * @param src_pe             [in] The PE number in src_team.
 * @param dest_team          [in] A ACLSHMEM team handle.
 *
 * @return The corresponding PE number in the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_DEVICE int aclshmem_team_translate_pe(aclshmem_team_t src_team, int src_pe, aclshmem_team_t dest_team);
#define shmem_team_translate_pe aclshmem_team_translate_pe

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in global team.
 *
 * @param team           [in] A ACLSHMEM team handle.
 * @param pe             [in] The PE number in src_team.
 *
 * @return The PE number in the global team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_DEVICE int aclshmem_team_pe_mapping(aclshmem_team_t team, int pe);
#define shmem_team_pe_mapping aclshmem_team_pe_mapping

#ifdef __cplusplus
}
#endif

#endif