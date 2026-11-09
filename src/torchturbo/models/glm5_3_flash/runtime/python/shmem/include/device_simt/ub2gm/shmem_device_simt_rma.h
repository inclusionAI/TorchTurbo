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

#ifndef SHMEM_DEVICE_SIMT_UB2GM_RMA_H
#define SHMEM_DEVICE_SIMT_UB2GM_RMA_H

#include "device_simt/ub2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/shmem_simt_common_types.h"

namespace simt {

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE            |
 * |-----------|-----------------|
 * |half       | half            |
 * |float      | float           |
 * |int8       | int8_t          |
 * |int16      | int16_t         |
 * |int32      | int32_t         |
 * |int64      | int64_t         |
 * |uint8      | uint8_t         |
 * |uint16     | uint16_t        |
 * |uint32     | uint32_t        |
 * |uint64     | uint64_t        |
 * |char       | signed char     |
 * |char       | unsigned char   |
 * |bfloat16   | bfloat16_t      |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, signed char);     \
    FUNC(char, unsigned char);   \
    FUNC(bfloat16, bfloat16_t)

// ========================================================

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_put(\_\_gm\_\_ TYPE *dst, \_\_ubuf\_\_ TYPE *src, size_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local ub to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local ub of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB(NAME, TYPE)                             \
    __simt_callee__ inline void aclshmem_##NAME##_put(                       \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_block(                \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_warp(                 \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB);

#undef ACLSHMEM_PUT_TYPENAME_MEM_UB

// ========================================================

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_putBITS(\_\_gm\_\_ void *dst, \_\_ubuf\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy a contiguous data on local ub to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local ub of the source data.
 * - **nelems**      - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM_UB(BITS)                                                                               \
    __simt_callee__ inline void aclshmem_put##BITS(__gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS##_block(                                                         \
        __gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe);                                            \
    __simt_callee__ inline void aclshmemx_put##BITS##_warp(                                                          \
        __gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_PUT_SIZE_MEM_UB(8);
ACLSHMEM_PUT_SIZE_MEM_UB(16);
ACLSHMEM_PUT_SIZE_MEM_UB(32);
ACLSHMEM_PUT_SIZE_MEM_UB(64);
ACLSHMEM_PUT_SIZE_MEM_UB(128);

#undef ACLSHMEM_PUT_SIZE_MEM_UB

// ========================================================

/**
 * @brief Synchronous interface. Copy contiguous data on local ub to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local ub of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_putmem(__gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_block(__gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_warp(__gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_get(\_\_ubuf\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local ub.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local ub of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB(NAME, TYPE)                             \
    __simt_callee__ inline void aclshmem_##NAME##_get(                       \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_block(                \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_warp(                 \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB);

#undef ACLSHMEM_GET_TYPENAME_MEM_UB

// ========================================================

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_getBITS(\_\_ubuf\_\_ void *dst, \_\_gm\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local ub.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local ub of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_SIZE_MEM_UB(BITS)                                                                               \
    __simt_callee__ inline void aclshmem_get##BITS(__ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS##_block(                                                         \
        __ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe);                                            \
    __simt_callee__ inline void aclshmemx_get##BITS##_warp(                                                          \
        __ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_GET_SIZE_MEM_UB(8);
ACLSHMEM_GET_SIZE_MEM_UB(16);
ACLSHMEM_GET_SIZE_MEM_UB(32);
ACLSHMEM_GET_SIZE_MEM_UB(64);
ACLSHMEM_GET_SIZE_MEM_UB(128);

#undef ACLSHMEM_GET_SIZE_MEM_UB

// ========================================================

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                       address on the local ub.
 *
 * @param dst               [in] Pointer on local ub of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_getmem(__ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_block(__ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_warp(__ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_ubuf\_\_ TYPE *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 *      Asynchronous interface. Copy a contiguous data on local ub to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local ub of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                         \
    __simt_callee__ inline void aclshmem_##NAME##_put_nbi(                   \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_block(            \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_warp(             \
        __gm__ TYPE* dst, __ubuf__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI);

#undef ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI

// ========================================================

/**
 * @brief  Automatically generates aclshmem put nbi functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_putBITS_nbi(\_\_gm\_\_ void *dst, \_\_ubuf\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy a contiguous data on local ub to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local ub of the source data.
 * - **nelems**      - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM_UB_NBI(BITS)                                \
    __simt_callee__ inline void aclshmem_put##BITS##_nbi(                 \
        __gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_block(          \
        __gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_warp(           \
        __gm__ void* dst, __ubuf__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_PUT_SIZE_MEM_UB_NBI(8);
ACLSHMEM_PUT_SIZE_MEM_UB_NBI(16);
ACLSHMEM_PUT_SIZE_MEM_UB_NBI(32);
ACLSHMEM_PUT_SIZE_MEM_UB_NBI(64);
ACLSHMEM_PUT_SIZE_MEM_UB_NBI(128);

#undef ACLSHMEM_PUT_SIZE_MEM_UB_NBI

// ========================================================

/**
 * @brief Asynchronous interface. Copy a contiguous data on local ub to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local ub of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_putmem_nbi(__gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_nbi_block(
    __gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_nbi_warp(
    __gm__ void* dst, __ubuf__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_get_nbi(\_\_ubuf\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local ub.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local ub of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                         \
    __simt_callee__ inline void aclshmem_##NAME##_get_nbi(                   \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_block(            \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_warp(             \
        __ubuf__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_NBI);

#undef ACLSHMEM_GET_TYPENAME_MEM_UB_NBI

// ========================================================

/**
 * @brief  Automatically generates aclshmem get nbi functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_getBITS_nbi(\_\_ubuf\_\_ void *dst, \_\_gm\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local ub.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local ub of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_SIZE_MEM_UB_NBI(BITS)                                \
    __simt_callee__ inline void aclshmem_get##BITS##_nbi(                 \
        __ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_block(          \
        __ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_warp(           \
        __ubuf__ void* dst, __gm__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_GET_SIZE_MEM_UB_NBI(8);
ACLSHMEM_GET_SIZE_MEM_UB_NBI(16);
ACLSHMEM_GET_SIZE_MEM_UB_NBI(32);
ACLSHMEM_GET_SIZE_MEM_UB_NBI(64);
ACLSHMEM_GET_SIZE_MEM_UB_NBI(128);

#undef ACLSHMEM_GET_SIZE_MEM_UB_NBI

// ========================================================

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                                              address on the local ub.
 *
 * @param dst               [in] Pointer on local ub of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_getmem_nbi(__ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_nbi_block(
    __ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_nbi_warp(
    __ubuf__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

#undef ACLSHMEM_TYPE_FUNC

} // namespace simt

#include "ub2gm/shmem_device_simt_rma.hpp"

#endif // !SHMEM_DEVICE_SIMT_UB2GM_RMA_H
