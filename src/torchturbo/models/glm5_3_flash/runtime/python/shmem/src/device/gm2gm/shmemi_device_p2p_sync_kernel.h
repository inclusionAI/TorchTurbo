/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_P2P_SYNC_H
#define SHMEMI_DEVICE_P2P_SYNC_H

#include "acl/acl.h"
#include "host_device/shmem_common_types.h"

// internal kernels
void aclshmem_do_signal_wait_until(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream = nullptr,
    size_t block_dim = 1);

#define ACLSHMEM_DO_WAIT_UNTIL(NAME, TYPE)                                                                             \
    void aclshmem_##NAME##_do_wait_until(                                                                              \
        TYPE *ivar, int cmp, TYPE cmp_value, aclrtStream stream = nullptr, size_t block_dim = 1                        \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL);

#define ACLSHMEM_DO_WAIT(NAME, TYPE)                                                                                   \
    void aclshmem_##NAME##_do_wait(TYPE *ivar, TYPE cmp_value, aclrtStream stream = nullptr, size_t block_dim = 1)

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT);

#define ACLSHMEM_DO_WAIT_UNTIL_ALL(NAME, TYPE)                                                                         \
    void aclshmem_##NAME##_do_wait_until_all(                                                                          \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, aclrtStream stream = nullptr,          \
        size_t block_dim = 1                                                                                           \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ALL);

#define ACLSHMEM_DO_WAIT_UNTIL_ANY(NAME, TYPE)                                                                         \
    void aclshmem_##NAME##_do_wait_until_any(                                                                          \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *res_out,                       \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ANY);

#define ACLSHMEM_DO_WAIT_UNTIL_SOME(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_wait_until_some(                                                                         \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, size_t *res_out,      \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_SOME);

#define ACLSHMEM_DO_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                                  \
    void aclshmem_##NAME##_do_wait_until_all_vector(                                                                   \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, aclrtStream stream = nullptr,        \
        size_t block_dim = 1                                                                                           \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ALL_VECTOR);

#define ACLSHMEM_DO_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                  \
    void aclshmem_##NAME##_do_wait_until_any_vector(                                                                   \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, size_t *res_out,                     \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ANY_VECTOR);

#define ACLSHMEM_DO_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                 \
    void aclshmem_##NAME##_do_wait_until_some_vector(                                                                  \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values, size_t *res_out,    \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_SOME_VECTOR);

#define ACLSHMEM_DO_TEST(NAME, TYPE)                                                                                   \
    void aclshmem_##NAME##_do_test(                                                                                    \
        TYPE *ivars, int cmp, TYPE cmp_value, int *res_out, aclrtStream stream = nullptr, size_t block_dim = 1         \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST);

#define ACLSHMEM_DO_TEST_ANY(NAME, TYPE)                                                                               \
    void aclshmem_##NAME##_do_test_any(                                                                                \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *res_out,                       \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ANY);

#define ACLSHMEM_DO_TEST_SOME(NAME, TYPE)                                                                              \
    void aclshmem_##NAME##_do_test_some(                                                                               \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, size_t *res_out,      \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_SOME);

#define ACLSHMEM_DO_TEST_ALL_VECTOR(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_test_all_vector(                                                                         \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, int *res_out,                        \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ALL_VECTOR);

#define ACLSHMEM_DO_TEST_ANY_VECTOR(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_test_any_vector(                                                                         \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, size_t *res_out,                     \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ANY_VECTOR);

#define ACLSHMEM_DO_TEST_SOME_VECTOR(NAME, TYPE)                                                                       \
    void aclshmem_##NAME##_do_test_some_vector(                                                                        \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values, size_t *res_out,    \
        aclrtStream stream = nullptr, size_t block_dim = 1                                                             \
    )

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_SOME_VECTOR);

#endif // _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_KERNEL_H_