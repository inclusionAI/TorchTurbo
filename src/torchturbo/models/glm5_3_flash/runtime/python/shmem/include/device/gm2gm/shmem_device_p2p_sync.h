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

#ifndef SHMEM_DEVICE_P2P_SYNC_H
#define SHMEM_DEVICE_P2P_SYNC_H

#include "host_device/shmem_common_types.h"
#include "gm2gm/shmem_device_p2p_sync.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 *        On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

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
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until

/**
 * @brief  Automatically generates aclshmem wait until functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_wait_until(\_\_gm\_\_ TYPE *ivar, int cmp, TYPE cmp_value)
 *
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
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_wait(\_\_gm\_\_ TYPE *ivar, TYPE cmp_value)
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
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait(__gm__ TYPE *ivar, TYPE cmp_value)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until all functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_wait_until_all(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, TYPE cmp_value)
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
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until_all(                                                             \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until any functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_wait_until_any(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, TYPE cmp_value)
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
 * @par Returns
 *      Return the index of an element in the ivars array that satisfies the wait condition.
 *      If the wait set is empty, this routine returns SIZE_MAX.
 * 
 */
#define ACLSHMEM_WAIT_UNTIL_ANY(NAME, TYPE)                                                                            \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any(                                                           \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until some functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_wait_until_some(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ size_t *indices, \_\_gm\_\_ const int *status, int cmp, TYPE cmp_value)
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
 * @par Returns
 *      Return the number of indices returned in the indices array.
 *      If the wait set is empty, this routine returns 0. 
 * 
 */
#define ACLSHMEM_WAIT_UNTIL_SOME(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some(                                                          \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value   \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until all vector functions for different data types (e.g., float, 
 *         int8_t). The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_wait_until_all_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
    ACLSHMEM_DEVICE void aclshmem_##NAME##_wait_until_all_vector(                                                      \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until any vector functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_wait_until_any_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
 * 
 * @par Returns
 *      Return the index of an element in the ivars array that satisfies the wait condition.
 *      If the wait set is empty, this routine returns SIZE_MAX.
 */
#define ACLSHMEM_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any_vector(                                                    \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem wait until some vector functions for different data types (e.g., float,
 *         int8_t). The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_wait_until_some_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ size_t *indices, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
 * 
 * @par Returns
 *      Return the number of indices returned in the indices array.
 *      If the wait set is empty, this routine returns 0. 
 */
#define ACLSHMEM_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                    \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some_vector(                                                   \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp,                  \
        __gm__ TYPE *cmp_values                                                                                        \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE int aclshmem_NAME_test(\_\_gm\_\_ TYPE *ivars, int cmp, TYPE cmp_value)
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
 * 
 * @par Returns
 *      Return 1 if the comparison (via the operator cmp) between the ivar and cmp_value results in true;
 *      otherwise, return 0. 
 */
#define ACLSHMEM_TEST(NAME, TYPE)                                                                                      \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test(__gm__ TYPE *ivars, int cmp, TYPE cmp_value)

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test any functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_test_any(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, TYPE cmp_value)
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
 * @par Returns
 *      Return the index of an element in the ivars array that satisfies the test condition.
 *      If the test set is empty or no conditions in the test set are satisfied,
 *      this routine returns SIZE_MAX. 
 */
#define ACLSHMEM_TEST_ANY(NAME, TYPE)                                                                                  \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any(                                                                 \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value                           \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test some functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_test_some(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ size_t *indices, \_\_gm\_\_ const int *status, int cmp, TYPE cmp_value)
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
 * 
 * @par Returns
 *      Return the number of indices returned in the indices array.
 *      If the test set is empty, this routine returns 0.
 */
#define ACLSHMEM_TEST_SOME(NAME, TYPE)                                                                                 \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some(                                                                \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value   \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test all vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_test_all_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
 * 
 * @par Returns
 *      Return 1 if all elements in ivars satisfy the test conditions or if nelems is 0,
 *      otherwise this routine returns 0.
 */
#define ACLSHMEM_TEST_ALL_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test_all_vector(                                                             \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ALL_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test any vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_test_any_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
 * 
 * @par Returns
 *      Return the index of an element in the ivars array that satisfies the test condition.
 *      If the test set is empty or no conditions in the test set are satisfied, 
 *      this routine returns SIZE_MAX.
 */
#define ACLSHMEM_TEST_ANY_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any_vector(                                                          \
        __gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values                  \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY_VECTOR);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem test some vector functions for different data types (e.g., float, int8_t).
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE size_t aclshmem_NAME_test_some_vector(\_\_gm\_\_ TYPE *ivars, size_t nelems, \_\_gm\_\_ size_t *indices, \_\_gm\_\_ const int *status, int cmp, \_\_gm\_\_ TYPE *cmp_values)
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
 * 
 * @par Returns
 *      Return the number of indices returned in the indices array.
 *      If the test set is empty, this routine returns 0.
 */
#define ACLSHMEM_TEST_SOME_VECTOR(NAME, TYPE)                                                                          \
    ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some_vector(                                                         \
        __gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp,                  \
        __gm__ TYPE *cmp_values                                                                                        \
    )

/** \cond */
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME_VECTOR);
/** \endcond */

#ifdef __cplusplus
}
#endif

#endif