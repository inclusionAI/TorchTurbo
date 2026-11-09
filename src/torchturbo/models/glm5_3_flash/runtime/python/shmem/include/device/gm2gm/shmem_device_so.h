/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SO_H
#define SHMEM_DEVICE_SO_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "host/shmem_host_def.h"
#include "device/shmem_def.h"
#include "gm2gm/shmem_device_so.hpp"

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
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE
 *       then update sig_addr
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal aclshmem_putmem_signal

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t elem_size,\
 * \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,       \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);
/** \endcond */
#define shmem_put_half_mem_signal aclshmem_half_put_signal
#define shmem_put_float_mem_signal aclshmem_float_put_signal
#define shmem_put_double_mem_signal aclshmem_doubl_put_signal
#define shmem_put_int8_mem_signal aclshmem_int8_put_signal
#define shmem_put_int16_mem_signal aclshmem_int16_put_signal
#define shmem_put_int32_mem_signal aclshmem_int32_put_signal
#define shmem_put_int64_mem_signal aclshmem_int64_put_signal
#define shmem_put_uint8_mem_signal aclshmem_uint8_put_signal
#define shmem_put_uint16_mem_signal aclshmem_uint16_put_signal
#define shmem_put_uint32_mem_signal aclshmem_uint32_put_signal
#define shmem_put_uint64_mem_signal aclshmem_uint64_put_signal
#define shmem_put_char_mem_signal aclshmem_char_put_signal
#define shmem_put_bfloat16_mem_signal aclshmem_bfloat16_put_signal

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE>\
 * src, size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                    size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal,             \
                                                    int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, const\
 * non_contiguous_copy_param &copy_params, \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE then update sig_addr
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED(NAME, TYPE)                                                      \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src,                          \
                                                    const non_contiguous_copy_param &copy_params,                  \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE>\
 * src，const non_contiguous_copy_param &copy_params, \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                    const non_contiguous_copy_param &copy_params,                           \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_putBITS_signal(void *dst, void *src, size_t nelems, int32_t *sig_addr,\
 * int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
 *    updating a remote signal flag on completion.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAIL(BITS)                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_signal(__gm__ void *dst, __gm__ void *src, size_t nelems,                         \
                                                     __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAIL);
/** \endcond */

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE then update
 * sig_addr
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size,
                                          __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

/**
 * @brief  Automatically generates aclshmem put signal nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t\
 * elem_size, \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,       \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI);
/** \endcond */
#define shmem_put_half_mem_signal_nbi aclshmem_half_put_signal_nbi
#define shmem_put_float_mem_signal_nbi aclshmem_float_put_signal_nbi
#define shmem_put_double_mem_signal_nbi aclshmem_double_put_signal_nbi
#define shmem_put_int8_mem_signal_nbi aclshmem_int8_put_signal_nbi
#define shmem_put_int16_mem_signal_nbi aclshmem_int16_put_signal_nbi
#define shmem_put_int32_mem_signal_nbi aclshmem_int32_put_signal_nbi
#define shmem_put_int64_mem_signal_nbi aclshmem_int64_put_signal_nbi
#define shmem_put_uint8_mem_signal_nbi aclshmem_uint8_put_signal_nbi
#define shmem_put_uint16_mem_signal_nbi aclshmem_uint16_put_signal_nbi
#define shmem_put_uint32_mem_signal_nbi aclshmem_uint32_put_signal_nbi
#define shmem_put_uint64_mem_signal_nbi aclshmem_uint64_put_signal_nbi
#define shmem_put_char_mem_signal_nbi aclshmem_char_put_signal_nbi
#define shmem_put_bfloat16_mem_signal_nbi aclshmem_bfloat16_put_signal_nbi

/**
 * @brief  Automatically generates aclshmem put signal nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal_nbi(AscendC::GlobalTensor<TYPE> dst,\
 * AscendC::GlobalTensor<TYPE> src, size_t elem_size, size_t elem_size, \_\_gm\_\_ int32_t *sig_addr, int32_t signal,\
 * int sig_op, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI(NAME, TYPE)                                                       \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(AscendC::GlobalTensor<TYPE> dst,                            \
                                                        AscendC::GlobalTensor<TYPE> src, size_t elem_size,            \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, const\
 * non_contiguous_copy_param &copy_params, \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE then update sig_addr
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI(NAME, TYPE)                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                         \
                                                        const non_contiguous_copy_param &copy_params,                 \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE>\
 * src，const non_contiguous_copy_param &copy_params, \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **copy_params** - [in] Params to describe how non-contiguous data is organized in src and dst.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI(NAME, TYPE)                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(                                                          \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,                                           \
        const non_contiguous_copy_param &copy_params, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_putBITS_signal_nbi(void *dst, void *src, size_t nelems, int32_t \
 * *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
 *    updating a remote signal flag on completion.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **nelems**      - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAILED_NBI(BITS)                                                               \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t nelems,           \
                                                       __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAILED_NBI);
/** \endcond */

#endif