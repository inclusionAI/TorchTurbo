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

/*
    WARNING：

    Our barrier implementation ensures that:
        On systems with only HCCS: All operations of all pes of a team ON EXECUTING/INTERNAL STREAMs
        before the barrier are visible to all pes of the team after the barrier.

    Refer to shmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_HOST_API uint64_t util_get_ffts_config(void)
 * @brief Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel
 * using aclshmemx_set_ffts. Refer to aclshmemx_set_ffts for more details.
 *
 * @return ffts config
 */
ACLSHMEM_HOST_API uint64_t util_get_ffts_config(void);
#define shmemx_get_ffts_config util_get_ffts_config

/**
 * @fn ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream)
 * @brief Wait asynchronous RMA operations to finish.
 *
 * @param handle              [in] handle use to wait asynchronous RMA operations to finish
 * @param stream              [in] specified stream to do wait
 */
ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream);
#define shmem_handle_wait aclshmemx_handle_wait

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @param stream                [in] Used stream(use default stream if stream == NULL).
 */
ACLSHMEM_HOST_API void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream);
#define shmem_signal_wait_until_on_stream aclshmemx_signal_wait_until_on_stream

/**
 * @brief Implements point-to-point synchronization by blocking until the value of sig_addr at the calling PE
 *        satisfies the condition defined by the comparison operator, cmp, and comparison value, cmp_value.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 */
ACLSHMEM_HOST_API void aclshmem_signal_wait_until(int32_t *sig_addr, int cmp, int32_t cmp_val);

/**
 * @brief  Automatically generates aclshmem wait until functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_wait_until(TYPE *ivar, int cmp, TYPE cmp_value)
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until the value at ivar
 *      satisfies the condition defined by the comparison operator, cmp, and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of a remotely accessible data object. 
 *                         The type of ivar should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC. 
 * - **cmp**        - [in] The comparison operator that compares ivar with cmp_val.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivar. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL(NAME, TYPE)                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_wait(TYPE *ivar, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until the value of ivar is not equal to
 *      comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of a remotely accessible data object. 
 *                         The type of ivar should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC. 
 * - **cmp_value**  - [in] The value to be compared with ivar. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT(NAME, TYPE)                                                                                      \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait(TYPE *ivar, TYPE cmp_value)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until all functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_wait_until_all(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until all entries in the wait set specified
 *      by ivars and status satisfy the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of a remotely accessible data object. 
 *                         The type of ivar should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivar with cmp_val.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivar. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL_ALL(NAME, TYPE)                                                                            \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_all(                                                           \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value                                         \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until any functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API void aclshmem_NAME_wait_until_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until any one entry in the wait set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivar with cmp_val.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivar. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL_ANY(NAME, TYPE)                                                                            \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_any(                                                           \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *res_out                        \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until some functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until at least one entry in the wait set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **indices**    - [out] Local address of an array of indices of length at least nelems into ivars
 *                          that satisfied the wait condition.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivar. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL_SOME(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_some(                                                          \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, size_t *res_out       \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until all vector functions for different data types (e.g., float,
 * int8_t). The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until all entries in the wait set specified
 *      by ivars and status satisfy the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values. 
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * 
 */
#define ACLSHMEM_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_all_vector(                                                    \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values                                       \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until any vector functions for different data types (e.g., float,
 *         int8_t). The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until any one entry in the wait set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_any_vector(                                                    \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, size_t *res_out                      \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until some vector functions for different data types (e.g., float,
 *         int8_t). The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by blocking until at least one entry in the wait set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **indices**    - [out] Local address of an array of indices of length at least nelems into ivars
 *                          that satisfied the wait condition.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                    \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_some_vector(                                                   \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values, size_t *res_out     \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API int aclshmem_NAME_test(TYPE *ivars, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether the value of ivar satisfies the condition
 *      defined by the comparison operator, cmp, and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value against which the object pointed to by ivar will be compared.
 *                         The type of cmp_value should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC 
 */
#define ACLSHMEM_TEST(NAME, TYPE)                                                                                      \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test(TYPE *ivars, int cmp, TYPE cmp_value, int *res_out)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test any functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_test_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether any one entry in the test set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the wait set;
 *                         If status[i] != 0, then ivars[i] is excluded from the wait set;
 *                         If status is NULL, all elements of ivars are included in the wait set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivars. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * 
 */
#define ACLSHMEM_TEST_ANY(NAME, TYPE)                                                                                  \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_any(                                                                 \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *res_out                        \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test some functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_test_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether at least one entry in the test set
 *      specified by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_value.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **indices**    - [out] Local address of an array of indices of length at least nelems into ivars
 *                          that satisfied the test condition.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the test set;
 *                         If status[i] != 0, then ivars[i] is excluded from the test set;
 *                         If status is NULL, all elements of ivars are included in the test set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_value**  - [in] The value to be compared with ivars. The type of cmp_value should match that
 *                         implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_TEST_SOME(NAME, TYPE)                                                                                 \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_some(                                                                \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, size_t *res_out       \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test all vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether all entries in the test set specified
 *      by ivars and status satisfy the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the test set;
 *                         If status[i] != 0, then ivars[i] is excluded from the test set;
 *                         If status is NULL, all elements of ivars are included in the test set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_TEST_ALL_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_all_vector(                                                          \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, int *res_out                         \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ALL_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test any vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether any one entry in the test set specified
 *      by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the test set;
 *                         If status[i] != 0, then ivars[i] is excluded from the test set;
 *                         If status is NULL, all elements of ivars are included in the test set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_TEST_ANY_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_any_vector(                                                          \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values, size_t *res_out                      \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test some vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_HOST_API size_t aclshmem_NAME_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values)
 *
 * @par Function Description
 *      Implements point-to-point synchronization by testing whether at least one entry in the test set
 *      specified by ivars and status satisfies the condition defined by the comparison operator, cmp,
 *      and comparison value, cmp_values.
 *
 * @par Parameters
 * - **ivar**       - [in] Symmetric address of an array of remotely accessible data objects.
 *                         The type of ivars should match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 * - **nelems**     - [in] The number of elements in the ivars array.
 * - **indices**    - [out] Local address of an array of indices of length at least nelems into ivars
 *                          that satisfied the test condition.
 * - **status**     - [in] Local address of an optional mask array of length nelems.
 *                         If status[i] == 0, then ivars[i] is included in the test set;
 *                         If status[i] != 0, then ivars[i] is excluded from the test set;
 *                         If status is NULL, all elements of ivars are included in the test set.
 * - **cmp**        - [in] The comparison operator that compares ivars with cmp_value.
 *                         Supported operators:
 *                         ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                         ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * - **cmp_values** - [in] Local address of an array of length nelems containing values to be
 *                         compared with the respective value in ivars. The type of cmp_values should
 *                         match that implied in the ACLSHMEM_P2P_SYNC_TYPE_FUNC.
 */
#define ACLSHMEM_TEST_SOME_VECTOR(NAME, TYPE)                                                                          \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_some_vector(                                                         \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values, size_t *res_out     \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME_VECTOR);
/** \endcond */

#ifdef __cplusplus
}
#endif

#endif