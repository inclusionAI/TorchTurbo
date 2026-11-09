/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_KERNEL_H_
#define _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_KERNEL_H_

#include "stdint.h"
#include "acl/acl.h"
#include "host_device/shmem_common_types.h"

// internal kernels
int32_t aclshmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count);

int32_t aclshmemi_call_barrier_on_stream_kernel(aclshmem_team_t team, aclrtStream stream);

void aclshmemi_handle_wait_on_stream(aclshmem_handle_t handle, aclrtStream stream);

void call_aclshmemi_signal_wait_until_on_stream_kernel(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream);

void call_aclshmemi_signal_op_on_stream_kernel(int32_t *sig_addr, int32_t signal, int sig_op, int pe,
    aclrtStream stream);

#endif // _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_KERNEL_H_