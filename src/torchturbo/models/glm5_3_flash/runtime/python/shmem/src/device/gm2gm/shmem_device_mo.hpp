/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_MO_HPP
#define SHMEM_DEVICE_MO_HPP

#include "host_device/shmem_common_types.h"
#include "shmemi_device_mo.h"

ACLSHMEM_DEVICE void aclshmemi_quiet()
{
    // clear instruction pipes
    AscendC::PipeBarrier<PIPE_ALL>();

    // flush data cache to GM
    dcci_entire_cache();
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void aclshmem_quiet()
{
    aclshmemi_quiet();
}

ACLSHMEM_DEVICE void aclshmem_fence()
{
    aclshmemi_quiet();
}

#ifdef __cplusplus
}
#endif

#endif