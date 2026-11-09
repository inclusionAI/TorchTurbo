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
#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

constexpr uint64_t ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t UB_ALIGN_SIZE_64 = 64;
constexpr uint32_t ACLSHMEM_NUM_CQE_PER_POLL_CQ = 100;

/**
 * @addtogroup group_structs
 * @{
 */
/**
 * @struct non_contiguous_copy_param
 * @brief Non-Contiguous Datacopy Param.
 *
 * - uint32_t repeat: Data move times
 * - uint32_t length: Data move unit length
 * - uint32_t src_ld: Src data leading dimension. Interval between the head of the repeat and the
 *   head of the following repeat.
 * - uint32_t dst_ld: Dst data leading dimension.
 */
struct non_contiguous_copy_param {
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;
    uint32_t dst_ld;
};

/**
 * @brief State shared by aclshmemx_defer_t and aclshmemx_submit_t actions in one active batch.
 *
 * Value-initialize the object as aclshmemx_submit_state_t state{} before the first operation.
 * The library maintains pending_count and resets it to zero after a successful submit; callers
 * must not modify the fields directly. Use one state object for only one active batch, and keep
 * it alive until every action that references it is no longer used.
 *
 * @note If submit validation or completion polling fails, the calling device kernel is aborted
 *       and pending_count is not reset.
 * @note reserved is retained for ABI-compatible extension and must remain zero.
 */
struct aclshmemx_submit_state_t {
    uint32_t pending_count = 0; ///< Number of operations currently added to the batch.
    uint32_t reserved = 0;      ///< Reserved for future extension; callers must keep it zero.
};

/**
 * @brief Action that adds the current operation to a batch without submitting the batch.
 */
struct aclshmemx_defer_t {
    aclshmemx_submit_state_t& state; ///< Submit state for the active batch.
    /**
     * @param st [in] Submit state that must outlive this action.
     */
    ACLSHMEM_DEVICE explicit aclshmemx_defer_t(aclshmemx_submit_state_t& st) : state(st) {}
};

/**
 * @brief Action that adds the current operation and submits all operations in the batch.
 */
struct aclshmemx_submit_t {
    aclshmemx_submit_state_t& state; ///< Submit state for the active batch.
    /**
     * @param st [in] Submit state that must outlive this action.
     */
    ACLSHMEM_DEVICE explicit aclshmemx_submit_t(aclshmemx_submit_state_t& st) : state(st) {}
};

/**@} */ // end of group_structs

#ifdef __cplusplus
}
#endif

#endif
