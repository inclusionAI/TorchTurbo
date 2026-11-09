/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_P2P_SYNC_HPP
#define SHMEM_DEVICE_P2P_SYNC_HPP

#include <cstdint>

#include "host_device/shmem_common_types.h"
#include "shmemi_device_p2p_sync.h"
#include "utils/mstx/shmemi_mstx_report.h"

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_eq(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr != cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ne(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr == cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_gt(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr <= cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ge(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr < cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_lt(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr >= cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_le(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr > cmp_val);
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_fetch(__gm__ int32_t *sig_addr)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    return *sig_addr;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until(__gm__ volatile T *ivar, int cmp, T cmp_value)
{
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            return aclshmemi_signal_wait_until_eq<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_NE:
            return aclshmemi_signal_wait_until_ne<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_GT:
            return aclshmemi_signal_wait_until_gt<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_GE:
            return aclshmemi_signal_wait_until_ge<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_LT:
            return aclshmemi_signal_wait_until_lt<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_LE:
            return aclshmemi_signal_wait_until_le<T>(ivar, cmp_value);
    }
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test(__gm__ volatile T *sig_addr, int cmp, T cmp_value)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            return *sig_addr == cmp_value;
        case ACLSHMEM_CMP_NE:
            return *sig_addr != cmp_value;
        case ACLSHMEM_CMP_GT:
            return *sig_addr > cmp_value;
        case ACLSHMEM_CMP_GE:
            return *sig_addr >= cmp_value;
        case ACLSHMEM_CMP_LT:
            return *sig_addr < cmp_value;
        case ACLSHMEM_CMP_LE:
            return *sig_addr <= cmp_value;
    }
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value)
{
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            aclshmemi_wait_until<T>(&ivars[idx], cmp, cmp_value);
        }
    }
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_any(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value)
{
    if (nelems == 0) {
        return SIZE_MAX;
    }

    size_t idx = 0;
    bool need_check = false;

    while (true) {
        if (!status || status[idx] == 0) {
            need_check = true;
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_value) == 1) {
                return idx;
            }
        }
        if ((idx + 1) == nelems && !need_check) {
            return SIZE_MAX;
        }
        idx = (idx + 1) % nelems;
    }
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_some(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, T cmp_value)
{
    if (nelems == 0) {
        return 0;
    }

    size_t idx = 0;
    size_t num_matched = 0;
    bool need_check = false;

    while (true) {
        if (!status || status[idx] == 0) {
            need_check = true;
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_value) == 1) {
                indices[num_matched++] = idx;
                break;
            }
        }
        if ((idx + 1) == nelems && !need_check) {
            return 0;
        }
        idx = (idx + 1) % nelems;
    }

    size_t start_idx = idx;
    for (size_t i = 1; i < nelems; i++) {
        idx = (start_idx + i) % nelems;
        if ((!status || status[idx] == 0) && aclshmemi_test<T>(&ivars[idx], cmp, cmp_value) == 1) {
            indices[num_matched++] = idx;
        }
    }
    return num_matched;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values)
{
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            aclshmemi_wait_until<T>(&ivars[idx], cmp, cmp_values[idx]);
        }
    }
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_any_vector(__gm__ volatile T *ivars, size_t nelems,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values)
{
    if (nelems == 0) {
        return SIZE_MAX;
    }

    size_t idx = 0;
    bool need_check = false;

    while (true) {
        if (!status || status[idx] == 0) {
            need_check = true;
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 1) {
                return idx;
            }
        }
        if ((idx + 1) == nelems && !need_check) {
            return SIZE_MAX;
        }
        idx = (idx + 1) % nelems;
    }
}


template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_some_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values)
{
    if (nelems == 0) {
        return 0;
    }

    size_t idx = 0;
    size_t num_matched = 0;
    bool need_check = false;

    while (true) {
        if (!status || status[idx] == 0) {
            need_check = true;
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 1) {
                indices[num_matched++] = idx;
                break;
            }
        }
        if ((idx + 1) == nelems && !need_check) {
            return 0;
        }
        idx = (idx + 1) % nelems;
    }

    size_t start_idx = idx;
    for (size_t i = 1; i < nelems; i++) {
        idx = (start_idx + i) % nelems;
        if ((!status || status[idx] == 0) && aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 1) {
            indices[num_matched++] = idx;
        }
    }
    return num_matched;
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all(__gm__ T *sig_addr, size_t nelems, __gm__ const int *status, int cmp, T cmp_val,
    int stride)
{
    int ret = 1;

    for (size_t i = 0; i < nelems; i++) {
        if (status && status[i] != 0) {
            continue;
        }

        dcci_cacheline((__gm__ uint8_t *)(sig_addr + i * stride));
        switch (cmp) {
            case ACLSHMEM_CMP_EQ:
                ret &= (*(sig_addr + i * stride) == cmp_val);
                break;
            case ACLSHMEM_CMP_NE:
                ret &= (*(sig_addr + i * stride) != cmp_val);
                break;
            case ACLSHMEM_CMP_GT:
                ret &= (*(sig_addr + i * stride) > cmp_val);
                break;
            case ACLSHMEM_CMP_GE:
                ret &= (*(sig_addr + i * stride) >= cmp_val);
                break;
            case ACLSHMEM_CMP_LT:
                ret &= (*(sig_addr + i * stride) < cmp_val);
                break;
            case ACLSHMEM_CMP_LE:
                ret &= (*(sig_addr + i * stride) <= cmp_val);
                break;
        }
    }

    return ret;
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_any(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status, int cmp,
    T cmp_value)
{
    if (nelems == 0) {
        return SIZE_MAX;
    }
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_value) == 1) {
                return idx;
            }
        }
    }

    return SIZE_MAX;
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_some(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, T cmp_value)
{
    if (nelems == 0) {
        return 0;
    }
    size_t num_matched = 0;
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_value) == 1) {
                indices[num_matched++] = idx;
            }
        }
    }

    return num_matched;
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values)
{
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 0) {
                return 0;
            }
        }
    }

    return 1;
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_any_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values)
{
    if (nelems == 0) {
        return SIZE_MAX;
    }
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 1) {
                return idx;
            }
        }
    }

    return SIZE_MAX;
}

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_some_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values)
{
    if (nelems == 0) {
        return 0;
    }
    size_t num_matched = 0;
    for (size_t idx = 0; idx < nelems; idx++) {
        if (!status || status[idx] == 0) {
            if (aclshmemi_test<T>(&ivars[idx], cmp, cmp_values[idx]) == 1) {
                indices[num_matched++] = idx;
            }
        }
    }

    return num_matched;
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    MSTX_FUSE_SCOPE_START();
    aclshmemi_wait_until<int32_t>(sig_addr, cmp, cmp_val);
    MSTX_FUSE_SCOPE_END();
    MSTX_SIGNAL_WAIT_REPORT(sig_addr, cmp, cmp_val);
    return *sig_addr;
}

#define ACLSHMEM_WAIT_UNTIL(NAME, TYPE)                                                                                \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)                      \
    {                                                                                                                  \
        aclshmemi_wait_until<TYPE>(ivar, cmp, cmp_value);                                                              \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL);

#define ACLSHMEM_WAIT(NAME, TYPE)                                                                                      \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait(__gm__ TYPE *ivar, TYPE cmp_value)                                     \
    {                                                                                                                  \
        aclshmemi_wait_until<TYPE>(ivar, ACLSHMEM_CMP_NE, cmp_value);                                                  \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT);

#define ACLSHMEM_WAIT_UNTIL_ALL(NAME, TYPE)                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until_all(                                                             \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )                                                                                                                  \
    {                                                                                                                  \
        aclshmemi_wait_until_all<TYPE>(ivars, nelems, status, cmp, cmp_value);                                         \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL);

#define ACLSHMEM_WAIT_UNTIL_ANY(NAME, TYPE)                                                                            \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any(                                                           \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_wait_until_any<TYPE>(ivars, nelems, status, cmp, cmp_value);                                  \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY);

#define ACLSHMEM_WAIT_UNTIL_SOME(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some(                                                          \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value   \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_wait_until_some<TYPE>(ivars, nelems, indices, status, cmp, cmp_value);                        \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME);

#define ACLSHMEM_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until_all_vector(                                                      \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )                                                                                                                  \
    {                                                                                                                  \
        aclshmemi_wait_until_all_vector<TYPE>(ivars, nelems, status, cmp, cmp_values);                                 \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL_VECTOR);

#define ACLSHMEM_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any_vector(                                                    \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_wait_until_any_vector<TYPE>(ivars, nelems, status, cmp, cmp_values);                          \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY_VECTOR);

#define ACLSHMEM_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                    \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some_vector(                                                   \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp,                  \
        __gm__ TYPE *cmp_values                                                                                        \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_wait_until_some_vector<TYPE>(ivars, nelems, indices, status, cmp, cmp_values);                \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME_VECTOR);

#define ACLSHMEM_TEST(NAME, TYPE)                                                                                      \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test(__gm__ TYPE *ivars, int cmp, TYPE cmp_value)                            \
    {                                                                                                                  \
        return aclshmemi_test<TYPE>(ivars, cmp, cmp_value);                                                            \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST);

#define ACLSHMEM_TEST_ANY(NAME, TYPE)                                                                                  \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any(                                                                 \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_test_any<TYPE>(ivars, nelems, status, cmp, cmp_value);                                        \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY);

#define ACLSHMEM_TEST_SOME(NAME, TYPE)                                                                                 \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some(                                                                \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value   \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_test_some<TYPE>(ivars, nelems, indices, status, cmp, cmp_value);                              \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME);

#define ACLSHMEM_TEST_ALL_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test_all_vector(                                                             \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_test_all_vector<TYPE>(ivars, nelems, status, cmp, cmp_values);                                \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ALL_VECTOR);

#define ACLSHMEM_TEST_ANY_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any_vector(                                                          \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_test_any_vector<TYPE>(ivars, nelems, status, cmp, cmp_values);                                \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY_VECTOR);

#define ACLSHMEM_TEST_SOME_VECTOR(NAME, TYPE)                                                                          \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some_vector(                                                         \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp,                  \
        __gm__ TYPE *cmp_values                                                                                        \
    )                                                                                                                  \
    {                                                                                                                  \
        return aclshmemi_test_some_vector<TYPE>(ivars, nelems, indices, status, cmp, cmp_values);                      \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME_VECTOR);

#ifdef __cplusplus
}
#endif

#endif
