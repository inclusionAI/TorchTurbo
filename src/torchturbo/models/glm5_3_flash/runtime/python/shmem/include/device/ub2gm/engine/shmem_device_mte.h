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
#ifndef SHMEM_DEVICE_UB2GM_MTE_H
#define SHMEM_DEVICE_UB2GM_MTE_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "ub2gm/mte/shmem_device_mte.hpp"
#include "device/gm2gm/engine/shmem_device_mte.h"

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE
 *        to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, uint32_t elem_size, int pe,
                                           uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           const non_contiguous_copy_param &copy_params, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe,
                                           uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                           uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                           const non_contiguous_copy_param &copy_params, int pe, uint32_t sync_id);

#endif