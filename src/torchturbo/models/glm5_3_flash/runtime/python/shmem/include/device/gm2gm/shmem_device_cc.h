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

/*
    WARNING：Restrictions of Barrier APIs.

    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if
       it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy).
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.

    3. We provide 2 kinds of barrier:
        a. aclshmem_barrier_xxx
            Barrier of all cores. On systems with only HCCS: All operations of all pes of a team on executing
            stream before the barrier are visible to all pes of the team after the barrier.
        b. aclshmemx_barrier_xxx_vec
            Barrier of all VEC cores. On systems with only HCCS: All operations of ALL VEC CORES of all pes of a
            team on executing stream before the barrier are visible to ALL VEC CORES of all pes of the team after
            the barrier.

    4. The scalar unit of cube core is not affected by aclshmem_barrier_xxx. Make sure don't use that.
*/

/*!
 * \file shmem_device_cc.h
 * \brief shmem device Collective Communication APIs
 */
#ifndef _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
#define _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_

#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);
#define shmemx_set_ffts_config util_set_ffts_config

/**
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
ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Similar to aclshmem_barrier except that only vector cores participate. Useful in communication-over-compute
 *        operators. Cube core may call the api but takes no effect.
 *
 * @param team              [in] team to do barrier
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmem_barrier instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);
#define shmemx_barrier_vec aclshmemx_barrier_vec

/**
 * @brief aclshmemx_barrier_vec of all PEs.
 *
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmem_barrier_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
#define shmemx_barrier_all_vec aclshmemx_barrier_all_vec

/**
 * @brief Similar to aclshmem_barrier. In contrast with the aclshmem_barrier routine, aclshmem_sync only ensures
 *        completion and visibility of previously issued memory stores and does not ensure completion of remote memory
 *        updates issued via ACLSHMEM routines.
 *
 * @param team           [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief aclshmem_sync_all of all PEs.
 *
 */
ACLSHMEM_DEVICE void aclshmem_sync_all(void);

#ifdef __cplusplus
}
#endif

#include "gm2gm/shmemi_device_cc.h"

#endif // _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_