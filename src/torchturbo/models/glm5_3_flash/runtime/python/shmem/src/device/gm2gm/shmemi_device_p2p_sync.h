/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHEMEI_DEVICE_P2P_SYNC_H
#define ACLSHEMEI_DEVICE_P2P_SYNC_H

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int32_t val);

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe, __ubuf__ T* buf, uint32_t sync_id);

ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe);

ACLSHMEM_DEVICE void aclshmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val);

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val);

// Atomicity of ACLSHMEM_SIGNAL_SET not guaranteed
ACLSHMEM_DEVICE void aclshmemi_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_eq(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ne(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_gt(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ge(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_lt(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_le(__gm__ volatile T *sig_addr, T cmp_val);

ACLSHMEM_DEVICE int32_t aclshmemi_signal_fetch(__gm__ int32_t *sig_addr);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until(__gm__ volatile T *ivar, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test(__gm__ volatile T *ivar, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_any(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_some(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_any_vector(__gm__ volatile T *ivars, size_t nelems,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_some_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values);

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all(__gm__ T *sig_addr, size_t nelems, __gm__ const int *status, int cmp,
    T cmp_value, int stride = ACLSHMEM_SYNCBIT_SIZE / sizeof(T));

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_any(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status, int cmp,
    T cmp_value);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_some(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_any_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, __gm__ T *cmp_values);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_test_some_vector(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, __gm__ T *cmp_values);

#endif