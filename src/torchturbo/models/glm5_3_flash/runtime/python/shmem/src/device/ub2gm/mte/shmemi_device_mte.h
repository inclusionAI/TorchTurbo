/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_MTE_H
#define SHMEMI_DEVICE_MTE_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] Pointer on Symmetric memory of the destination data.
 * @param srcUb             [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Byte Size of data in the destination and source arrays.
 * @param toL2Cache         [in] Enable L2Cache or not. False means disable L2Cache.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    uint32_t size, bool toL2Cache = true);

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] GlobalTensor on Symmetric memory of the destination data.
 * @param srcUb             [in] LocalTensor on local UB of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, uint32_t size);

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] Pointer on Symmetric memory of the destination data.
 * @param srcUb             [in] Pointer on local UB of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    AscendC::DataCopyExtParams &copyParams);

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] GlobalTensor on Symmetric memory of the destination data.
 * @param srcUb             [in] LocalTensor on local UB of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, AscendC::DataCopyExtParams &copyParams);

/**
 * @brief Simple Copy interface. Copy contiguous data on symmetric memory from the specified PE to local UB.
 *
 * @param dstUb             [in] Pointer on local UB of the destination data.
 * @param srcGva            [in] Pointer on Symmetric memory of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 * @param toL2Cache         [in] Enable L2Cache or not. False means disable L2Cache.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    uint32_t size, bool toL2Cache = true);

/**
 * @brief Simple Copy interface. Copy contiguous data on symmetric memory from the specified PE to local UB.
 *
 * @param dstUb             [in] LocalTensor on local UB of the destination data.
 * @param srcGva            [in] GlobalTensor on Symmetric memory of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, uint32_t size);

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstUb             [in] Pointer on local UB of the destination data.
 * @param srcGva            [in] Pointer on Symmetric memory of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    AscendC::DataCopyExtParams &copyParams);

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstUb             [in] LocalTensor on local UB of the destination data.
 * @param srcGva            [in] GlobalTensor on Symmetric memory of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, AscendC::DataCopyExtParams &copyParams);

#endif