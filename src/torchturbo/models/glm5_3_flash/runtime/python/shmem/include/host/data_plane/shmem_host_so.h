/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_SO_H
#define SHMEM_HOST_SO_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"
#include "host/data_plane/shmem_host_rma.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_put_signal(TYPE* dst, TYPE* src, size_t elem_size, uint8_t *sig_addr,
 * int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy contiguous data from local device memory to symmetric memory on the specified PE
 *      and update a remote signal variable.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer to the destination data in symmetric memory.
 * - **src**         - [in] Pointer to the source data on the local device.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note The data transfer path uses MTE only.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_signal( \
        TYPE* dst, TYPE* src, size_t elem_size, uint8_t* sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);
/** \endcond */
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
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL

/**
 * @brief  Automatically generates aclshmem put signal nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_put_signal_nbi(TYPE* dst, TYPE* src, size_t elem_size, uint8_t
 * *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Asynchronous interface. Copy contiguous data from local device memory to symmetric memory on the specified PE
 *      and update a remote signal variable.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer to the destination data in symmetric memory.
 * - **src**         - [in] Pointer to the source data on the local device.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note The data transfer path uses MTE only.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_signal_nbi( \
        TYPE* dst, TYPE* src, size_t elem_size, uint8_t* sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI);
/** \endcond */
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
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_putBITS_signal(void *dst, void *src, size_t nelems, int32_t *sig_addr,
 * int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *    Synchronous interface. Copy contiguous data from local device memory to symmetric memory on the specified PE
 *    and update a remote signal variable.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer to the destination data in symmetric memory.
 * - **src**         - [in] Pointer to the source data on the local device.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note The data transfer path uses MTE only.
 */
#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL(BITS)              \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_signal( \
        void* dst, void* src, size_t nelems, int32_t* sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE_MEM_SIGNAL

/**
 * @brief  Automatically generates aclshmem put functions for different bits (e.g., 8, 16).
 *         The macro parameters: BITS is the bits.
 *
 * \remark ACLSHMEM_HOST_API void aclshmem_putBITS_signal_nbi(void *dst, void *src, size_t nelems, int32_t *sig_addr,
 * int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *    Asynchronous interface. Copy contiguous data from local device memory to symmetric memory on the specified PE
 *    and update a remote signal variable.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer to the destination data in symmetric memory.
 * - **src**         - [in] Pointer to the source data on the local device.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note The data transfer path uses MTE only.
 */
#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI(BITS)              \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_signal_nbi( \
        void* dst, void* src, size_t nelems, int32_t* sig_addr, int32_t signal, int sig_op, int pe)

/** \cond */
ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI);
/** \endcond */
#undef ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE and
 *        update a remote signal variable.
 *
 * @param dst               [in] Pointer to the destination data in symmetric memory.
 * @param src               [in] Pointer to the source data on the local device.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal.
 *                               Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 * @note The data transfer path uses MTE only.
 */
ACLSHMEM_HOST_API void aclshmemx_putmem_signal_nbi(
    void* dst, void* src, size_t elem_size, void* sig_addr, int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal_nbi aclshmemx_putmem_signal_nbi

/**
 * @brief Synchronously copy contiguous data from local device memory to symmetric memory on the specified PE and
 *        update a remote signal variable.
 *
 * @param dst               [in] Pointer to the destination data in symmetric memory.
 * @param src               [in] Pointer to the source data on the local device.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal.
 *                               Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 * @note The data transfer path uses MTE only.
 */
ACLSHMEM_HOST_API void aclshmemx_putmem_signal(
    void* dst, void* src, size_t elem_size, void* sig_addr, int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal aclshmemx_putmem_signal

/**
 * @brief This routine performs an atomic operation on a remote signal variable at the specified PE, with the operation
 *        executed on the given stream. It is used to modify a remote signal and is typically paired with wait routines
 *        like aclshmemx_signal_wait_until_on_stream.
 *
 * @param sig_addr              [in] Symmetric address of the signal variable to update on the target PE.
 * @param signal                [in] The value to be used in the atomic operation.
 * @param sig_op                [in] The operation to perform on the remote signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD.
 * @param pe                    [in] The PE number on which the remote signal variable is to be updated.
 * @param stream                [in] Used stream(use default stream if stream == NULL).
 *
 * @note Cross-machine support:
 *       - ACLSHMEM_SIGNAL_SET: Supports RDMA cross-machine.
 *       - ACLSHMEM_SIGNAL_ADD: Does not support RDMA cross-machine. Supports MTE cross-machine when HCCS is available.
 */
ACLSHMEM_HOST_API void aclshmemx_signal_op_on_stream(
    int32_t* sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream);
#define shmem_signal_op_on_stream aclshmemx_signal_op_on_stream

#ifdef __cplusplus
}
#endif

#endif
