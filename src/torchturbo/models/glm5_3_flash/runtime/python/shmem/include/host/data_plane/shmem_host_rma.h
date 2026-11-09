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
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Standard RMA Types and Names valid on Host
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
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
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
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
    FUNC(char, char)

/**
 * @addtogroup group_enums
 * @{
 */
/// \brief Enumeration indicating whether the method is blocking or non-blocking
enum aclshmem_block_mode_t {
    NO_NBI = 0, ///< Non-blocking mode disabled (method is blocking)
    NBI         ///< Non-blocking mode enabled (method is non-blocking, NBI = Non-Blocking Interface)
};
/**@} */ // end of group_enums

/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE.
 *        First check whether the input address and PE are valid, then translate the address. Return a null pointer
 *        for invalid input.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE, or a null pointer if ptr or pe is invalid.
 * @note The supported access method for the returned address depends on the transport and topology.
 */
ACLSHMEM_HOST_API void* aclshmem_ptr(void* ptr, int pe);
#define shmem_ptr aclshmem_ptr

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_put(TYPE *dest, TYPE *source, size_t nelems, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on Symmetric memory of the destination data.
 * - **source**   - [in] Pointer on local device of the source data.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 *
 * @note Address requirements: dest must point to symmetric memory because it is translated to the corresponding
 *       address on pe. source is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
#define ACLSHMEM_TYPE_PUT(NAME, TYPE) \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put(TYPE* dest, TYPE* source, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT);
/** \endcond */
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
#undef ACLSHMEM_TYPE_PUT

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_put_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dest**     - [in] Pointer on Symmetric memory of the destination data.
 * - **source**   - [in] Pointer on local device of the source data.
 * - **nelems**   - [in] Number of elements in the destination and source arrays.
 * - **pe**       - [in] PE number of the remote PE.
 *
 * @note Address requirements: dest must point to symmetric memory because it is translated to the corresponding
 *       address on pe. source is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
#define ACLSHMEM_TYPE_PUT_NBI(NAME, TYPE) \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_nbi(TYPE* dest, TYPE* source, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT_NBI);
/** \endcond */
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
#undef ACLSHMEM_TYPE_PUT_NBI

/**
 * @brief  Automatically generates aclshmem put functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_iput(TYPE *dest, TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t
 * nelems, int pe)
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
#define ACLSHMEM_TYPE_IPUT(NAME, TYPE)             \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_iput( \
        TYPE* dest, TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_IPUT);
/** \endcond */
#undef ACLSHMEM_TYPE_IPUT

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_putBITS(void *dst, void *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
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
 */
#define ACLSHMEM_PUT_SIZE(BITS) \
    ACLSHMEM_HOST_API void aclshmem_put##BITS(void* dst, void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_putBITS_nbi(void *dst, void *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
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
 */
#define ACLSHMEM_PUT_SIZE_NBI(BITS) \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_nbi(void* dst, void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_NBI);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE_NBI

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_iputBITS(void *dest, void *source, ptrdiff_t dst, ptrdiff_t sst, size_t
 * nelems, int pe)
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
#define ACLSHMEM_IPUT_SIZE(BITS)                \
    ACLSHMEM_HOST_API void aclshmem_iput##BITS( \
        void* dest, void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_IPUT_SIZE);
/** \endcond */
#undef ACLSHMEM_IPUT_SIZE

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_get(TYPE *dest, TYPE *source, size_t nelems, int pe)
 *
 * @par Function Description
 * Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dest**        - [in] Pointer on local device of the destination data.
 * - **source**      - [in] Pointer on Symmetric memory of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: source must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dest is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
#define ACLSHMEM_TYPE_GET(NAME, TYPE) \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get(TYPE* dest, TYPE* source, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET);
/** \endcond */
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
#undef ACLSHMEM_TYPE_GET

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_get_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dest**        - [in] Pointer on local device of the destination data.
 * - **source**      - [in] Pointer on Symmetric memory of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: source must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dest is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
#define ACLSHMEM_TYPE_GET_NBI(NAME, TYPE) \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get_nbi(TYPE* dest, TYPE* source, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET_NBI);
/** \endcond */
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
#undef ACLSHMEM_TYPE_GET_NBI

/**
 * @brief  Automatically generates aclshmem get functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_iget(TYPE *dest, TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t
 * nelems, int pe)
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
#define ACLSHMEM_TYPE_IGET(NAME, TYPE)             \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_iget( \
        TYPE* dest, TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_IGET);
/** \endcond */
#undef ACLSHMEM_TYPE_IGET

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_getBITS(void *dst, void *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
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
 */
#define ACLSHMEM_GET_SIZE(BITS) \
    ACLSHMEM_HOST_API void aclshmem_get##BITS(void* dst, void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE);
/** \endcond */
#undef ACLSHMEM_GET_SIZE

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_getBITS_nbi(void *dst, void *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 *   Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
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
 */
#define ACLSHMEM_GET_SIZE_NBI(BITS) \
    ACLSHMEM_HOST_API void aclshmem_get##BITS##_nbi(void* dst, void* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_NBI);
/** \endcond */
#undef ACLSHMEM_GET_SIZE_NBI

/**
 * @brief  Automatically generates aclshmem get functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_igetBITS(void *dest, void *source, ptrdiff_t dst, ptrdiff_t sst, size_t
 * nelems, int pe)
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
#define ACLSHMEM_IGET_SIZE(BITS)                \
    ACLSHMEM_HOST_API void aclshmem_iget##BITS( \
        void* dest, void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_IGET_SIZE);
/** \endcond */
#undef ACLSHMEM_IGET_SIZE

/**
 * @brief  Automatically generates aclshmem p functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_p(TYPE* dst, const TYPE value, int pe)
 *
 * @par Function Description
 * Provide a low latency put capability for single element of most basic types.
 *
 * @par Parameters
 * - **dst**    - [in] Symmetric address of the destination data on local PE.
 * - **value**  - [in] The element to be put.
 * - **pe**     - [in] The number of the remote PE.
 */
#define ACLSHMEM_TYPENAME_P(NAME, TYPE) ACLSHMEM_HOST_API void aclshmem_##NAME##_p(TYPE* dst, const TYPE value, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P);
/** \endcond */
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
#undef ACLSHMEM_TYPENAME_P

/**
 * @brief  Automatically generates aclshmem g functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_g(TYPE* src, int32_t pe)
 *
 * @par Function Description
 * Provide a low latency get capability for single element of most basic types.
 *
 * @par Parameters
 * - **src**    - [in] Symmetric address of the source data on local PE.
 * - **pe**     - [in] The number of the remote PE.
 *
 * @par Returns
 *      A single element of type specified in the input pointer.
 */
#define ACLSHMEM_TYPENAME_G(NAME, TYPE) ACLSHMEM_HOST_API TYPE aclshmem_##NAME##_g(TYPE* src, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G);
/** \endcond */
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
#undef ACLSHMEM_TYPENAME_G

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst                [in] Pointer on Symmetric memory of the destination data.
 * @param src                [in] Pointer on local memory of the source data.
 * @param elem_size          [in] size of elements in the destination and source addr.
 * @param pe                 [in] PE number of the remote PE.
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
ACLSHMEM_HOST_API void aclshmem_putmem(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_putmem aclshmem_putmem

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local PE.
 *
 * @param dst                [in] Pointer on local device of the destination data.
 * @param src                [in] Pointer on Symmetric memory of the source data.
 * @param elem_size          [in] size of elements in the destination and source addr.
 * @param pe                 [in] PE number of the remote PE.
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
ACLSHMEM_HOST_API void aclshmem_getmem(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_getmem aclshmem_getmem

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst                [in] Pointer on Symmetric memory of the destination data.
 * @param src                [in] Pointer on local memory of the source data.
 * @param elem_size          [in] size of elements in the destination and source addr.
 * @param pe                 [in] PE number of the remote PE.
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
ACLSHMEM_HOST_API void aclshmem_putmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_putmem_nbi aclshmem_putmem_nbi

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local PE.
 *
 * @param dst                [in] Pointer on local device of the destination data.
 * @param src                [in] Pointer on Symmetric memory of the source data.
 * @param elem_size          [in] size of elements in the destination and source addr.
 * @param pe                 [in] PE number of the remote PE.
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 */
ACLSHMEM_HOST_API void aclshmem_getmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_getmem_nbi aclshmem_getmem_nbi

/**
 * @brief Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param stream            [in] copy used stream(use default stream if stream == NULL).
 * @note Address requirements: src must point to symmetric memory because it is translated to the corresponding
 *       address on pe. dst is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @note Cross-machine support: Supports cross-machine. Uses MTE when HCCS is connected, otherwise uses RDMA if
 * available.
 */
ACLSHMEM_HOST_API void aclshmemx_getmem_on_stream(
    void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
#define shmemx_getmem_on_stream aclshmemx_getmem_on_stream

/**
 * @brief Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param stream            [in] copy used stream(use default stream if stream == NULL).
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @note Cross-machine support: Supports cross-machine. Uses MTE when HCCS is connected, otherwise uses RDMA if
 * available.
 */
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(
    void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
#define shmemx_putmem_on_stream aclshmemx_putmem_on_stream
/**
 * @brief Set necessary parameters for put or get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param sync_id               [in] Sync ID for put or get.
 * @return Returns 0 on success or an error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);
#define shmem_mte_set_ub_params aclshmemx_set_mte_config

/**
 * @brief Set necessary parameters for SDMA operations.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param sync_id               [in] Sync ID for put or get.
 * @return Returns 0 on success or an error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_set_sdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Set necessary parameters for RDMA operations.
 *        RDMA RMA APIs use this UB scratch to stage a full WQE via MTE3.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer in bytes. It
 *                              must be at least ACLSHMEM_RDMA_MTE_STAGING_UB_SIZE
 *                              bytes (128) for RDMA MTE3 WQE staging.
 * @param sync_id               [in] Sync ID for MTE3 pipeline synchronization.
 * @return Returns 0 on success or an error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_set_rdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);
#define shmem_rdma_set_ub_params aclshmemx_set_rdma_config

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
 * @return Returns 0 on success or an error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_set_udma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Ensures completion of all previously issued operations on symmetric data objects
 *        issued by the calling PE. The quiet is queued on the given stream; the caller must
 *        synchronize the stream to observe completion from the host.
 *
 * @param stream            [in] ACL stream on which to queue the quiet operation. Passing NULL uses the default stream.
 *
 * @note Quiet operations are scoped to the issuing context: quiet called from the host ensures completion
 *       of host-issued operations; quiet called from a device kernel ensures completion of operations
 *       issued by that kernel.
 */
ACLSHMEM_HOST_API void aclshmemx_quiet_on_stream(aclrtStream stream);
#define shmemx_quiet_on_stream aclshmemx_quiet_on_stream

#ifdef __cplusplus
}
#endif

#endif
