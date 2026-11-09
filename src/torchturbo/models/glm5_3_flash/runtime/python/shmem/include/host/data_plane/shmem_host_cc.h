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

/*!
 * \file shmem_host_cc.h
 * \brief shmem host Collective Communication APIs
 */

#ifndef _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_
#define _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_HOST_API void aclshmem_barrier(aclshmem_team_t team)
 * @brief aclshmem_barrier is a collective synchronization routine over a team. Control returns from aclshmem_barrier
 *        after all PEs in the team have called aclshmem_barrier.
 *        aclshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of NPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param team              [in] team to do barrier
 */
ACLSHMEM_HOST_API void aclshmem_barrier(aclshmem_team_t team);

/**
 * @fn ACLSHMEM_HOST_API void aclshmem_barrier_all(void)
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_HOST_API void aclshmem_barrier_all(void);

/**
 * @fn ACLSHMEM_HOST_API void aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream)
 * @brief aclshmem_barrier on the specified stream
 *
 * @param team              [in] team to do barrier
 * @param stream            [in] specified stream to do barrier
 */
ACLSHMEM_HOST_API void aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream);
#define shmemx_barrier_on_stream aclshmemx_barrier_on_stream

/**
 * @fn ACLSHMEM_HOST_API void aclshmemx_barrier_all_on_stream(aclrtStream stream)
 * @brief aclshmemx_barrier_on_stream of all PEs.
 *
 * @param stream            [in] specified stream to do barrier
 */
ACLSHMEM_HOST_API void aclshmemx_barrier_all_on_stream(aclrtStream stream);
#define shmemx_barrier_all_on_stream aclshmemx_barrier_all_on_stream

/**
 * @brief Similar to aclshmem_barrier. In contrast with the aclshmem_barrier routine, aclshmem_sync only ensures
 *        completion and visibility of previously issued memory stores and does not ensure completion of remote memory
 *        updates issued via ACLSHMEM routines.
 *
 * @param team           [in] team to do barrier
 */
ACLSHMEM_HOST_API void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief aclshmem_sync_all of all PEs.
 *
 */
ACLSHMEM_HOST_API void aclshmem_sync_all(void);

#ifdef __cplusplus
}
#endif

#endif // _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_
