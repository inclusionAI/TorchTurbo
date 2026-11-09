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
#ifndef SHMEM_DEVICE_UB2GM_RMA_H
#define SHMEM_DEVICE_UB2GM_RMA_H

#include "kernel_operator.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
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
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(\_\_ubuf\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.  
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local UB of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                               \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.
 *
 * @par Parameters
 * - **dst**         - [in] LocalTensor on local UB of the destination data.
 * - **src**         - [in] GlobalTensor on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(\_\_ubuf\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, const non_contiguous_copy_param &copy_params, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data on symmetric memory from the 
 * specified PE to address on the local UB.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local UB of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src,                   \
                                                 const non_contiguous_copy_param &copy_params, int pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem get nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param &copy_params, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data on symmetric memory from the 
 * specified PE to address on the local UB.
 *
 * @par Parameters
 * - **dst**         - [in] LocalTensor on local UB of the destination data.
 * - **src**         - [in] GlobalTensor on Symmetric memory of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is managed in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                    \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_ubuf\_\_ TYPE *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.  
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local UB of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.  
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on Symmetric memory of the destination data.
 * - **src**         - [in] LocalTensor on local UB of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_ubuf\_\_ TYPE *src, const non_contiguous_copy_param &copy_params, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local UB of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src,                     \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, const non_contiguous_copy_param &copy_params, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE. 
 *
 * @par Parameters
 * - **dst**         - [in] GlobalTensor on Symmetric memory of the destination data.
 * - **src**         - [in] LocalTensor on local UB of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                    \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);
/** \endcond */

#include "ub2gm/shmem_device_rma.hpp"
#endif