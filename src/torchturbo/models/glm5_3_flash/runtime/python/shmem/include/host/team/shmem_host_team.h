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
#ifndef SHMEM_HOST_TEAM_H
#define SHMEM_HOST_TEAM_H

#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Collective Interface. Creates a new ACLSHMEM team from an existing parent team.
 *
 * @param parent_team        [in] A team handle.
 * @param pe_start           [in] The first PE number of the subset of PEs from the parent team.
 * @param pe_stride          [in] The stride between team PE numbers in the parent team.
 * @param pe_size            [in] The total number of PEs in new team.
 * @param new_team           [out] A team handle.
 *
 * @return 0 on successful creation of new_team; otherwise nonzero.
 */
ACLSHMEM_HOST_API int aclshmem_team_split_strided(aclshmem_team_t parent_team, int pe_start, int pe_stride, int pe_size,
                                            aclshmem_team_t *new_team);
#define shmem_team_split_strided aclshmem_team_split_strided


/**
 * @brief Collective Interface. Split team from an existing parent team based on a 2D Cartsian Space.
 *
 * @param parent_team       [in] A team handle.
 * @param x_range           [in] represents the number of elements in the first dimensions
 * @param x_team            [in] A new x-axis team after team split.
 * @param y_team            [in] A new y-axis team after team split.
 *
 * @return 0 on successful creation of new_team; otherwise nonzero.
 */
ACLSHMEM_HOST_API int aclshmem_team_split_2d(aclshmem_team_t parent_team, int x_range, aclshmem_team_t *x_team,
                                       aclshmem_team_t *y_team);
#define shmem_team_split_2d aclshmem_team_split_2d


/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 *
 * @param src_team           [in] A ACLSHMEM team handle.
 * @param src_pe             [in] The PE number in src_team.
 * @param dest_team          [in] A ACLSHMEM team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_HOST_API int aclshmem_team_translate_pe(aclshmem_team_t src_team, int src_pe, aclshmem_team_t dest_team);
#define shmem_team_translate_pe aclshmem_team_translate_pe


/**
 * @brief Collective Interface. Destroys the team referenced by the team handle.
 *
 * @param team              [in] A team handle.
 */
ACLSHMEM_HOST_API void aclshmem_team_destroy(aclshmem_team_t team);
#define shmem_team_destroy aclshmem_team_destroy


/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
ACLSHMEM_HOST_API int aclshmem_my_pe(void);
#define shmem_my_pe aclshmem_my_pe


/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
ACLSHMEM_HOST_API int aclshmem_n_pes(void);
#define shmem_n_pes aclshmem_n_pes


/**
 * @brief Returns the number of the calling PE in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_HOST_API int aclshmem_team_my_pe(aclshmem_team_t team);
#define shmem_team_my_pe aclshmem_team_my_pe


/**
 * @brief Returns the number of PEs in the specified team.
 *
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team.
 *         If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
 */
ACLSHMEM_HOST_API int aclshmem_team_n_pes(aclshmem_team_t team);
#define shmem_team_n_pes aclshmem_team_n_pes


/**
 * @brief return team config which pass in as team created
 *
 * @param team [in] team handle
 * @param config [out] the config associated with team, reserved for future use
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmem_team_get_config(aclshmem_team_t team, aclshmem_team_config_t *config);
#define shmem_team_get_config aclshmem_team_get_config


#ifdef __cplusplus
}
#endif

#endif