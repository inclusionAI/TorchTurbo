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

#ifndef SHMEM_DEVICE_SIMT_UB2GM_MTE_H
#define SHMEM_DEVICE_SIMT_UB2GM_MTE_H

#include "device_simt/shmem_simt_common_types.h"
#include "ub2gm/mte/shmem_device_simt_mte.hpp"

namespace simt {

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local
 * ub, executed in thread scope (single thread).
 *
 * @param dst               [in] Pointer on local ub of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi(__ubuf__ T* dst, __gm__ T* src, size_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local
 * ub, executed in block scope.
 *
 * @param dst               [in] Pointer on local ub of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_block(__ubuf__ T* dst, __gm__ T* src, size_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local
 * ub, executed in warp scope (usually 32 threads).
 *
 * @param dst               [in] Pointer on local ub of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_warp(__ubuf__ T* dst, __gm__ T* src, size_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE, executed in
 * thread scope (single thread).
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local ub of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi(__gm__ T* dst, __ubuf__ T* src, size_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE, executed in
 * block scope.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local ub of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_block(__gm__ T* dst, __ubuf__ T* src, size_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE, executed in
 * warp scope (usually 32 threads).
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local ub of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_warp(__gm__ T* dst, __ubuf__ T* src, size_t elem_size, int pe);

} // namespace simt

#endif // !SHMEM_DEVICE_SIMT_MTE_H
