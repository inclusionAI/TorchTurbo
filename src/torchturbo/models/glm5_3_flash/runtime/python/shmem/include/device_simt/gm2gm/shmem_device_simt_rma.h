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
#ifndef SHMEM_DEVICE_SIMT_GM2GM_RMA_H
#define SHMEM_DEVICE_SIMT_GM2GM_RMA_H

#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"
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

/**
 * @brief Standard test Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |float      | float     |
 * |int8       | int8_t    |
 * |int16      | int16_t   |
 * |int32      | int32_t   |
 * |int64      | int64_t   |
 * |uint8      | uint8_t   |
 * |uint16     | uint16_t  |
 * |uint32     | uint32_t  |
 * |uint64     | uint64_t  |
 * |char       | char      |
 */
#define ACLSHMEM_TEST_TYPE_FUNC(FUNC) \
    FUNC(float, float);               \
    FUNC(int8, int8_t);               \
    FUNC(int16, int16_t);             \
    FUNC(int32, int32_t);             \
    FUNC(int64, int64_t);             \
    FUNC(uint8, uint8_t);             \
    FUNC(uint16, uint16_t);           \
    FUNC(uint32, uint32_t);           \
    FUNC(uint64, uint64_t);           \
    FUNC(char, char)

// ========================================================

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_put(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                              \
    __simt_callee__ inline void aclshmem_##NAME##_put(                     \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_block(              \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_warp(               \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);

#undef ACLSHMEM_PUT_TYPENAME_MEM

// ========================================================

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_putBITS(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM(BITS)                                                                                \
    __simt_callee__ inline void aclshmem_put##BITS(__gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS_block(                                                         \
        __gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe);                                            \
    __simt_callee__ inline void aclshmemx_put##BITS_warp(__gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM);

#undef ACLSHMEM_PUT_SIZE_MEM

// ========================================================

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_putmem(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_block(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_warp(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem p functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_p(\_\_gm\_\_ TYPE *dst, const TYPE value, int pe)
 *
 * @par Function Description
 * Provide a low latency put capability for single element of most basic types.
 *
 * @par Parameters
 * - **dst**    - [in] Symmetric address of the destination data.
 * - **value**  - [in] The element to be put.
 * - **pe**     - [in] The number of the remote PE.
 */
#define ACLSHMEM_TYPENAME_P_AICORE(NAME, TYPE) \
    __simt_callee__ inline void aclshmem_##NAME##_p(__gm__ TYPE* dst, const TYPE value, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);

#undef ACLSHMEM_TYPENAME_P_AICORE

// ========================================================

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_get(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE)                              \
    __simt_callee__ inline void aclshmem_##NAME##_get(                     \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_block(              \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_warp(               \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);

#undef ACLSHMEM_GET_TYPENAME_MEM

// ========================================================

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_getBITS(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, size_t nelems,
 * int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_SIZE_MEM(BITS)                                                                                \
    __simt_callee__ inline void aclshmem_get##BITS(__gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS_block(                                                         \
        __gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe);                                            \
    __simt_callee__ inline void aclshmemx_get##BITS_warp(__gm__ void* dst, __gm__ void* src, size_t nelems, int32_t pe)

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM);

#undef ACLSHMEM_GET_SIZE_MEM

// ========================================================

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                       address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_getmem(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_block(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_warp(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem g functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline TYPE aclshmem_NAME_g(\_\_gm\_\_ TYPE *src, int32_t pe)
 *
 * @par Function Description
 * Provide a low latency get capability for single element of most basic types.
 *
 * @par Parameters
 * - **src**    - [in] Symmetric address of the source data.
 * - **pe**     - [in] The number of the remote PE.
 *
 * @par Returns
 *      A single element of type specified in the input pointer.
 */
#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE) \
    __simt_callee__ inline TYPE aclshmem_##NAME##_g(__gm__ TYPE* src, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);

#undef ACLSHMEM_TYPENAME_G_AICORE

// ========================================================

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 *      Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                          \
    __simt_callee__ inline void aclshmem_##NAME##_put_nbi(                 \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_block(          \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_warp(           \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);

#undef ACLSHMEM_PUT_TYPENAME_MEM_NBI

// ========================================================

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_putBITS_nbi(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM_NBI(BITS)                                    \
    __simt_callee__ inline void aclshmem_put##BITS##_nbi(                  \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_block(           \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_warp(            \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_NBI);

#undef ACLSHMEM_PUT_SIZE_MEM_NBI

// ========================================================

/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_putmem_nbi(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_nbi_block(
    __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_putmem_nbi_warp(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

// ========================================================

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline void aclshmem_NAME_get_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                          \
    __simt_callee__ inline void aclshmem_##NAME##_get_nbi(                 \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_block(          \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_warp(           \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);

#undef ACLSHMEM_GET_TYPENAME_MEM_NBI

// ========================================================

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark __simt_callee__ inline void aclshmem_getBITS_nbi(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, size_t
 * elem_size, int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_SIZE_MEM_NBI(BITS)                                    \
    __simt_callee__ inline void aclshmem_get##BITS##_nbi(                  \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_block(           \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe); \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_warp(            \
        __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM_NBI);

#undef ACLSHMEM_GET_SIZE_MEM_NBI

// ========================================================

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                                              address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
__simt_callee__ inline void aclshmem_getmem_nbi(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_nbi_block(
    __gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);
__simt_callee__ inline void aclshmemx_getmem_nbi_warp(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe);

} // namespace simt

#include "gm2gm/shmem_device_simt_rma.hpp"
#endif // ACLSHMEM_DEVICE_SIMT_RMA_H
