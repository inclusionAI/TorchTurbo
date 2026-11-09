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
#ifndef SHMEM_DEVICE_SDMA_H
#define SHMEM_DEVICE_SDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_sdma.hpp"

/**
 * @brief Set necessary parameters for SDMA operations.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer (In Bytes), at least 64 bytes and 64-byte aligned.
 * @param sync_id               [in] Sync ID for put or get.
 */
ACLSHMEM_DEVICE void aclshmemx_set_sdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local PE.
 *
 * @param dst               [in] AscendC::GlobalTensor on local device of the destination data.
 * @param src               [in] AscendC::GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(AscendC::GlobalTensor<T> &dst, AscendC::GlobalTensor<T> &src,
                                            AscendC::LocalTensor<T> &buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] AscendC::GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] AscendC::GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(AscendC::GlobalTensor<T> &dst, AscendC::GlobalTensor<T> &src,
                                            AscendC::LocalTensor<T> &buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Performs a l2 cache manager operation on device global memory.
 * WARNING: Currently, cmo_type only supports CMO_TYPE_PREFETCH.
 *
 * @param src               [in] Pointer to device global memory to operate on.
 * @param elem_size         [in] Number of elements (of type T) to operate on.
 * @param cmo_type          [in] Cache operation type.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                                        __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Performs a l2 cache manager operation on device global memory.
 * WARNING: Currently, cmo_type only supports CMO_TYPE_PREFETCH.
 *
 * @param src               [in] AscendC::GlobalTensor of device memory to operate on.
 * @param elem_size         [in] Number of elements.
 * @param cmo_type          [in] Cache operation type.
 * @param buf               [in] LocalTensor on local UB.
 * @param sync_id           [in] ID used to sync.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(AscendC::GlobalTensor<T> &src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                                            AscendC::LocalTensor<T> &buf, uint32_t sync_id);

/**
 * @brief SDMA Quiet function. This synchronous function ensures all previous SDMA SQEs are completed.
 *
 * @param buf                       [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(AscendC::LocalTensor<T> &buf, uint32_t sync_id);

/**
 * @brief SDMA Quiet function. This synchronous function ensures all previous SDMA SQEs are completed.
 *
 * @param buf                       [in] Pointer on local UB.
 * @param ub_size                   [in] The size of temp Buffer on UB. (In Bytes)
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(__ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief AIV direct STARS helper function for notify record.
 *
 * @param buf                       [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(AscendC::LocalTensor<T> &buf, uint32_t sync_id);

/**
 * @brief AIV direct STARS helper function for notify record.
 *
 * @param buf                       [in] Pointer on local UB.
 * @param ub_size                   [in] The size of temp Buffer on UB. (In Bytes)
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(__ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);

#endif
