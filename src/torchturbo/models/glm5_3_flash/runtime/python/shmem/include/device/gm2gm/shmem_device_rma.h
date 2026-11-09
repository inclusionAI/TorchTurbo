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
#ifndef SHMEM_DEVICE_GM2GM_RMA_H
#define SHMEM_DEVICE_GM2GM_RMA_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_udma.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "host/shmem_host_def.h"
#include "device/shmem_def.h"

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(double, double);        \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, char);            \
    FUNC(bfloat16, bfloat16_t)

/**
 * @brief Standard test Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |float      | float     |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
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

/**
 * @brief  Automatically generates aclshmem p functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_p(\_\_gm\_\_ TYPE *dst, const TYPE value, int pe)
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
    ACLSHMEM_DEVICE void aclshmem_##NAME##_p(__gm__ TYPE* dst, const TYPE value, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);
/** \endcond */
#define shmem_half_p aclshmem_half_p
#define shmem_float_p aclshmem_float_p
#define shmem_double_p aclshmem_double_p
#define shmem_int8_p aclshmem_int8_p
#define shmem_int16_p aclshmem_int16_p
#define shmem_int32_p aclshmem_int32_p
#define shmem_int64_p aclshmem_int64_p
#define shmem_uint8_p aclshmem_uint8_p
#define shmem_uint16_p aclshmem_uint16_p
#define shmem_uint32_p aclshmem_uint32_p
#define shmem_uint64_p aclshmem_uint64_p
#define shmem_char_p aclshmem_char_p
#define shmem_bfloat16_p aclshmem_bfloat16_p

/**
 * @brief  Automatically generates aclshmem g functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_g(\_\_gm\_\_ TYPE *src, int32_t pe)
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
#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE) ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_g(__gm__ TYPE* src, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);
/** \endcond */
#define shmem_half_g aclshmem_half_g
#define shmem_float_g aclshmem_float_g
#define shmem_double_g aclshmem_double_g
#define shmem_int8_g aclshmem_int8_g
#define shmem_int16_g aclshmem_int16_g
#define shmem_int32_g aclshmem_int32_g
#define shmem_int64_g aclshmem_int64_g
#define shmem_uint8_g aclshmem_uint8_g
#define shmem_uint16_g aclshmem_uint16_g
#define shmem_uint32_g aclshmem_uint32_g
#define shmem_uint64_g aclshmem_uint64_g
#define shmem_char_g aclshmem_char_g
#define shmem_bfloat16_g aclshmem_bfloat16_g

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local PE. Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
ACLSHMEM_DEVICE void aclshmem_getmem(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe);
#define shmem_getmem aclshmem_getmem

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);
/** \endcond */
#define shmem_get_half_mem aclshmem_half_get
#define shmem_get_float_mem aclshmem_float_get
#define shmem_get_double_mem aclshmem_double_get
#define shmem_get_int8_mem aclshmem_int8_get
#define shmem_get_int16_mem aclshmem_int16_get
#define shmem_get_int32_mem aclshmem_int32_get
#define shmem_get_int64_mem aclshmem_int64_get
#define shmem_get_uint8_mem aclshmem_uint8_get
#define shmem_get_uint16_mem aclshmem_uint16_get
#define shmem_get_uint32_mem aclshmem_uint32_get
#define shmem_get_uint64_mem aclshmem_uint64_get
#define shmem_get_char_mem aclshmem_char_get
#define shmem_get_bfloat16_mem aclshmem_bfloat16_get

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_iget(\_\_gm\_\_ TYPE *dest, \_\_gm\_\_ TYPE *source, ptrdiff_t dst,
 * ptrdiff_t sst, size_t nelems, int pe)
 *
 * @par Function Description
 * Synchronous interface. Copy strided data elements from a symmetric array from a specified remote PE to strided
 * locations on a local array.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on local device of the destination data.
 * - **source**   - [in] Pointer on Symmetric memory of the source data.
 * - **dst**      - [in] The stride between consecutive elements of the dest array.
 * - **sst**      - [in] The stride between consecutive elements of the source array.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 */
#define ACLSHMEM_IGET_TYPENAME_MEM(NAME, TYPE)   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_iget( \
        __gm__ TYPE* dest, __gm__ TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_IGET_TYPENAME_MEM);
/** \endcond */
#undef ACLSHMEM_IGET_TYPENAME_MEM

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_getBITS(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, uint32_t elem_size, int32_t
 * pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *    Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_GET_SIZE_MEM(BITS) \
    ACLSHMEM_DEVICE void aclshmem_get##BITS(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM);
/** \endcond */
#undef ACLSHMEM_GET_SIZE_MEM

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_igetBITS(\_\_gm\_\_ void *dest, \_\_gm\_\_ void *source, ptrdiff_t dst,
 * ptrdiff_t sst, size_t nelems, int pe)
 *
 * @par Function Description
 * Synchronous interface. Copy strided data elements from a symmetric array from a specified remote PE to strided
 * locations on a local array.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on local device of the destination data.
 * - **source**   - [in] Pointer on Symmetric memory of the source data.
 * - **dst**      - [in] The stride between consecutive elements of the dest array.
 * - **sst**      - [in] The stride between consecutive elements of the source array.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 */
#define ACLSHMEM_IGET_SIZE_MEM(BITS)          \
    ACLSHMEM_DEVICE void aclshmem_iget##BITS( \
        __gm__ void* dest, __gm__ void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_IGET_SIZE_MEM);
/** \endcond */
#undef ACLSHMEM_IGET_SIZE_MEM

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
ACLSHMEM_DEVICE void aclshmem_putmem(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe);
#define shmem_putmem aclshmem_putmem

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *      Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);
/** \endcond */
#define shmem_put_half_mem aclshmem_half_put
#define shmem_put_float_mem aclshmem_float_put
#define shmem_put_double_mem aclshmem_double_put
#define shmem_put_int8_mem aclshmem_int8_put
#define shmem_put_int16_mem aclshmem_int16_put
#define shmem_put_int32_mem aclshmem_int32_put
#define shmem_put_int64_mem aclshmem_int64_put
#define shmem_put_uint8_mem aclshmem_uint8_put
#define shmem_put_uint16_mem aclshmem_uint16_put
#define shmem_put_uint32_mem aclshmem_uint32_put
#define shmem_put_uint64_mem aclshmem_uint64_put
#define shmem_put_char_mem aclshmem_char_put
#define shmem_put_bfloat16_mem aclshmem_bfloat16_put

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_iput(\_\_gm\_\_ TYPE *dest, \_\_gm\_\_ TYPE *source, ptrdiff_t dst,
 * ptrdiff_t sst, size_t nelems, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy strided data elements (specified by sst) of an array from a source array on the
 *      local PE to locations specified by stride dst on a dest array on specified remote PE.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on Symmetric memory of the destination data.
 * - **source**   - [in] Pointer on local device of the source data.
 * - **dst**      - [in] The stride between consecutive elements of the dest array.
 * - **sst**      - [in] The stride between consecutive elements of the source array.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 */
#define ACLSHMEM_IPUT_TYPENAME_MEM(NAME, TYPE)   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_iput( \
        __gm__ TYPE* dest, __gm__ TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_IPUT_TYPENAME_MEM);
/** \endcond */
#undef ACLSHMEM_IPUT_TYPENAME_MEM

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_putBITS(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, uint32_t elem_size, int32_t
 * pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *    Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_SIZE_MEM(BITS) \
    ACLSHMEM_DEVICE void aclshmem_put##BITS(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE_MEM

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_iputBITS(\_\_gm\_\_ void *dest, \_\_gm\_\_ void *source, ptrdiff_t dst,
 * ptrdiff_t sst, size_t nelems, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy strided data elements (specified by sst) of an array from a source array on the
 *      local PE to locations specified by stride dst on a dest array on specified remote PE.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on Symmetric memory of the destination data.
 * - **source**   - [in] Pointer on local device of the source data.
 * - **dst**      - [in] The stride between consecutive elements of the dest array.
 * - **sst**      - [in] The stride between consecutive elements of the source array.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 */
#define ACLSHMEM_IPUT_SIZE_MEM(BITS)          \
    ACLSHMEM_DEVICE void aclshmem_iput##BITS( \
        __gm__ void* dest, __gm__ void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_IPUT_SIZE_MEM);
/** \endcond */
#undef ACLSHMEM_IPUT_SIZE_MEM

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local PE. Supports MTE, RDMA, or SDMA as the underlying transport.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
ACLSHMEM_DEVICE void aclshmem_getmem_nbi(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe);
#define shmem_getmem_nbi aclshmem_getmem_nbi

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);
/** \endcond */
#define shmem_get_half_mem_nbi aclshmem_half_get_nbi
#define shmem_get_float_mem_nbi aclshmem_float_get_nbi
#define shmem_get_double_mem_nbi aclshmem_double_get_nbi
#define shmem_get_int8_mem_nbi aclshmem_int8_get_nbi
#define shmem_get_int16_mem_nbi aclshmem_int16_get_nbi
#define shmem_get_int32_mem_nbi aclshmem_int32_get_nbi
#define shmem_get_int64_mem_nbi aclshmem_int64_get_nbi
#define shmem_get_uint8_mem_nbi aclshmem_uint8_get_nbi
#define shmem_get_uint16_mem_nbi aclshmem_uint16_get_nbi
#define shmem_get_uint32_mem_nbi aclshmem_uint32_get_nbi
#define shmem_get_uint64_mem_nbi aclshmem_uint64_get_nbi
#define shmem_get_char_mem_nbi aclshmem_char_get_nbi
#define shmem_get_bfloat16_mem_nbi aclshmem_bfloat16_get_nbi

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_getBITS_nbi(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *    Supports MTE, RDMA, or SDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_GET_SIZE_MEM_NBI(BITS) \
    ACLSHMEM_DEVICE void aclshmem_get##BITS##_nbi(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM_NBI);
/** \endcond */
#undef ACLSHMEM_GET_SIZE_MEM_NBI

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, const
 * non_contiguous_copy_param &copy_params, int32_t pe)
 *
 * @par Function Description
 *      Asynchronous interface. Provide a high-performance way to copy non-contiguous data on symmetric memory from
 *      the specified PE to address on the local device.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(        \
        __gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,
 * uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on local device of the destination data.
 * - **src**         - [in] GlobalTensor on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may reference any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(      \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,
 * uint32_t elem_size, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Provide a high-performance way to copy non-contiguous data on symmetric memory from
 *      the specified PE to address on the local device.
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on local device of the destination data.
 * - **src**         - [in] GlobalTensor on Symmetric memory of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)         \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(                       \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
        const non_contiguous_copy_param& copy_params, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);
/** \endcond */
#define shmem_put_half_mem_nbi aclshmem_half_put_nbi
#define shmem_put_float_mem_nbi aclshmem_float_put_nbi
#define shmem_put_double_mem_nbi aclshmem_double_put_nbi
#define shmem_put_int8_mem_nbi aclshmem_int8_put_nbi
#define shmem_put_int16_mem_nbi aclshmem_int16_put_nbi
#define shmem_put_int32_mem_nbi aclshmem_int32_put_nbi
#define shmem_put_int64_mem_nbi aclshmem_int64_put_nbi
#define shmem_put_uint8_mem_nbi aclshmem_uint8_put_nbi
#define shmem_put_uint16_mem_nbi aclshmem_uint16_put_nbi
#define shmem_put_uint32_mem_nbi aclshmem_uint32_put_nbi
#define shmem_put_uint64_mem_nbi aclshmem_uint64_put_nbi
#define shmem_put_char_mem_nbi aclshmem_char_put_nbi
#define shmem_put_bfloat16_mem_nbi aclshmem_bfloat16_put_nbi

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_putBITS_nbi(\_\_gm\_\_ void *dst, \_\_gm\_\_ void *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *    Supports MTE, RDMA, or SDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_SIZE_MEM_NBI(BITS) \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_nbi(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_NBI);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE_MEM_NBI

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, const
 * non_contiguous_copy_param &copy_params, int32_t pe)
 *
 * @par Function Description
 *      Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *      on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(        \
        __gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,
 * uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on Symmetric memory of the destination data.
 * - **src**         - [in] GlobalTensor on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may reference any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(      \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,
 * const non_contiguous_copy_param &copy_params, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data on local PE to symmetric address
 * on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on Symmetric memory of the destination data.
 * - **src**         - [in] GlobalTensor on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)         \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(                       \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
        const non_contiguous_copy_param& copy_params, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);
/** \endcond */

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        Supports MTE, RDMA, or SDMA as the underlying transport.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_nbi(__gm__ void* dst, __gm__ void* src, uint32_t elem_size, int32_t pe);
#define shmem_putmem_nbi aclshmem_putmem_nbi

/**
 * @brief Set necessary parameters for put or get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param sync_id               [in] Sync ID for put or get.
 */
ACLSHMEM_DEVICE void aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);
#define shmem_mte_set_ub_params aclshmemx_set_mte_config

/**
 * @brief Set necessary parameters for UDMA MTE staging operations.
 *        High-level UDMA RMA APIs use this UB scratch to stage one full WQE
 *        block on the default PIPE_MTE3 path.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer in bytes. It
 *                              must be at least ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE
 *                              bytes for UDMA MTE staging.
 * @param sync_id               [in] Sync ID for the MTE3->S synchronization.
 */
ACLSHMEM_DEVICE void aclshmemx_set_udma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);

#include "gm2gm/shmem_device_rma.hpp"
#endif
