/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEMI_KERNEL_DEBUG_H
#define SHMEMI_KERNEL_DEBUG_H

#if defined(ASCENDC_DEBUG) && defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 1) && defined(DEBUG_MODE)
#define ACLSHMEM_DEBUG_FUNC(func, ...) do { func(__VA_ARGS__); } while(0)
#else
#define ACLSHMEM_DEBUG_FUNC(func, ...) ((void)0)
#endif


template<class... Args>
ACLSHMEM_DEVICE void aclshmemi_kernel_abort(__gm__ const char* fmt, Args&&... args)
{
    AscendC::printf(fmt, args...);
    trap();
}

template<class... Args>
ACLSHMEM_DEVICE void aclshmemi_kernel_printf(__gm__ const char* fmt, Args&&... args)
{
    AscendC::printf(fmt, args...);
}

#endif