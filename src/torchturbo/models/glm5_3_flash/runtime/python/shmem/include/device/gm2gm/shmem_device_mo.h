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

#ifndef SHMEM_DEVICE_MO_H
#define SHMEM_DEVICE_MO_H

#include "host_device/shmem_common_types.h"
#include "gm2gm/shmem_device_mo.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The aclshmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of NPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_quiet(void);
#define shmem_quiet aclshmem_quiet

/**
 * @brief In OpenACLSHMEM specification, aclshmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented aclshmem_fence same as aclshmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of NPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_fence(void);
#define shmem_fence aclshmem_fence

#ifdef __cplusplus
}
#endif

#endif