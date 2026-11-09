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
#ifndef SHMEM_DEVICE_ATOMIC_H
#define SHMEM_DEVICE_ATOMIC_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"

/**
 * @brief Type Function Macros for Atomic Operations
 *
 * Each macro is used to generate atomic operation functions for specific data types.
 *
 * | Macro Name                         | Used by Atomic Interfaces                                         |
 * |------------------------------------|-------------------------------------------------------------------|
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_ADD      | atomic_add                                                        |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_910  | atomic_add                                                        |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_EXT  | atomic_add                                                        |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP     | atomic_set, atomic_swap, atomic_compare_swap                      |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST| atomic_set, atomic_swap, atomic_compare_swap (CAST types)         |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST | atomic_compare_swap (CAST types)                                  |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950  | atomic_fetch_add, atomic_inc, atomic_fetch_inc, atomic_fetch      |
 * | ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC    | atomic_and, atomic_or, atomic_xor, atomic_fetch_and,              |
 * |                                    | atomic_fetch_or, atomic_fetch_xor                                 |
 */

/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int8       | int8_t    |
 * |int16      | int16_t   |
 * |int32      | int32_t   |
 * |bfloat16   | bfloat16_t|
 * |half       | half      |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(FUNC) \
    FUNC(int8, int8_t);                     \
    FUNC(int16, int16_t);                   \
    FUNC(bfloat16, bfloat16_t);             \
    FUNC(half, half)
/**
 * @brief Ascend_910 operations - support types for atomic_add
 * |NAME   | TYPE     |
 * |-------|----------|
 * |int32  | int32_t  |
 * |float  | float    |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_910(FUNC) \
    FUNC(int32, int32_t);                       \
    FUNC(float, float)

/**
 * @brief Ascend950 operations - support types for atomic_add
 * |NAME   | TYPE     |
 * |-------|----------|
 * |uint32 | uint32_t |
 * |uint64 | uint64_t |
 * |int64  | int64_t  |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_EXT(FUNC) \
    FUNC(uint32, uint32_t);                     \
    FUNC(uint64, uint64_t);                     \
    FUNC(int64, int64_t)

/** @brief Integer-only operations - direct support types
 * |NAME   | TYPE     |
 * |-------|----------|
 * |uint32 | uint32_t |
 * |uint64 | uint64_t |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(FUNC) \
    FUNC(uint32, uint32_t);                  \
    FUNC(uint64, uint64_t)

/** @brief Integer-only operations - CAST support types
 * |NAME   | TYPE     | CAST TYPE |
 * |-------|----------|-----------|
 * |int32  | int32_t  | uint32_t  |
 * |int64  | int64_t  | uint64_t  |
 * |float  | float    | uint32_t  |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST(FUNC) \
    FUNC(int32, int32_t, uint32_t);               \
    FUNC(int64, int64_t, uint64_t);               \
    FUNC(float, float, uint32_t)

/** @brief Integer-only operations - CAST support types
 * |NAME   | TYPE     | CAST TYPE |
 * |-------|----------|-----------|
 * |int32  | int32_t  | uint32_t  |
 * |int64  | int64_t  | uint64_t  |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST(FUNC) \
    FUNC(int32, int32_t, uint32_t);              \
    FUNC(int64, int64_t, uint64_t)

/** @brief Ascend950 operations - support types for atomic_add
 * |NAME   | TYPE     |
 * |-------|----------|
 * |uint32 | uint32_t |
 * |uint64 | uint64_t |
 * |int32  | int32_t  |
 * |int64  | int64_t  |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(FUNC) \
    FUNC(uint32, uint32_t);                     \
    FUNC(uint64, uint64_t);                     \
    FUNC(int32, int32_t);                       \
    FUNC(int64, int64_t)

/** @brief Logic operations - support types for atomic_and, atomic_or, atomic_xor, atomic_fetch_and,
 * |NAME   | TYPE     |
 * |-------|----------|
 * |uint32 | uint32_t |
 * |uint64 | uint64_t |
 * |int32  | int32_t  |
 * |int64  | int64_t  |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(FUNC) \
    FUNC(uint32, uint32_t);                   \
    FUNC(uint64, uint64_t);                   \
    FUNC(int32, int32_t);                     \
    FUNC(int64, int64_t)

/**
 * @brief  Automatically generates aclshmem atomic add functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform                |
 * |-------|---------------------------------------|----------------------------------|
 * | MTE   | int8, int16, bf16, half, int32, float | Ascend910B/Ascend910C/Ascend950  |
 * | MTE   | uint32, uint64, int64                 | Ascend950                        |
 * | RDMA  | int32, uint32, int64, uint64,         | Ascend950                        |
 * | UDMA  | int32, uint32, int64, uint64, float   | Ascend950                        |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Perform contiguous data atomic add operation on
 * symmetric memory from the specified PE to address on the local PE.
 *
 * The implementation dispatches to MTE, RDMA, or UDMA based on the transport topology.
 * Pipeline synchronization requirements differ by transport and data type, and must
 * be ensured externally by the caller.
 *
 * @par MTE Path Synchronization
 * MTE path behavior differs by data type:
 * - **int8, int16, bf16, half, int32, float** (UB + MTE3):
 *   The Scalar unit writes operands to UB, then MTE3 reads UB and performs the
 *   remote atomic add to GM via the MTE3 atomic add hardware.
 *   - **Before calling**: if another unit (e.g. MTE2) is also writing to the same UB
 *     region, the caller must fence those writes before Scalar writes to UB
 *     (e.g. SetFlag/WaitFlag on MTE2_S event), otherwise UB data may be corrupted.
 *   - **After calling**: if there is a data dependency on the atomic add result, the
 *     caller must fence MTE3 before reading GM (e.g. SetFlag/WaitFlag on
 *     MTE3_MTE2 event). Likewise, if new values will be written to the same UB
 *     region, the caller must fence MTE3 before overwriting UB (e.g. SetFlag/
 *     WaitFlag on MTE3_S event), otherwise UB data may be overwritten before
 *     MTE3 has finished reading.
 * - **uint32, uint64, int64** (Scalar AtomicAdd):
 *   The atomic add is performed directly by the Scalar unit (Scalar AtomicAdd) on
 *   Ascend950, without going through UB and MTE3. Since the Scalar unit writes
 *   directly to GM, the synchronization model differs from the UB+MTE3 path.
 *   The caller must follow the general MTE synchronization rules
 *   (see the "Atomic Interface Synchronization Model" section below):
 *   if there are data dependencies between the Scalar computation unit and the
 *   transfer units (MTE2/MTE3) when reading/writing GM, insert appropriate
 *   SetFlag/WaitFlag synchronization based on the actual data flow.
 *
 * The MTE UB buffer offset defaults to 0 and can be adjusted via
 * aclshmemx_set_mte_config(offset, ub_size, sync_id).
 *
 * @par RDMA Path Synchronization
 * The atomic add is issued asynchronously over RDMA. The caller must call
 * aclshmemx_roce_quiet before reading the result to guarantee the operation
 * has completed on the target PE.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)

#define ACLSHMEM_ATOMIC_ADD_910_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)

#define ACLSHMEM_ATOMIC_ADD_EXT_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_910(ACLSHMEM_ATOMIC_ADD_910_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_EXT(ACLSHMEM_ATOMIC_ADD_EXT_TYPENAME);
/** \endcond */

#define shmem_int8_atomic_add aclshmem_int8_atomic_add
#define shmem_int16_atomic_add aclshmem_int16_atomic_add
#define shmem_int32_atomic_add aclshmem_int32_atomic_add
#define shmem_half_atomic_add aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add aclshmem_float_atomic_add

/** ============================================================================
 * @section amo_sync_model Atomic Interface Synchronization Model
 *
 * - **Return value** (fetch_add, fetch_inc, fetch_and, fetch_or, fetch_xor,
 *   fetch, swap, compare_swap): **Synchronous**. Completes before returning.
 *   Caller needs no additional synchronization.
 *
 * - **Void return** (add, inc, and, or, xor, set): **Asynchronous**. May not
 *   have completed on return. Caller must synchronize:
 *   - **RDMA**: aclshmemx_roce_quiet
 *   - **UDMA**: aclshmemx_udma_quiet
 *   - **MTE**: Insert SetFlag/WaitFlag per data dependency (see below)
 *
 * @par MTE Path Synchronization
 * MTE atomic ops use Scalar→UB→MTE2/MTE3→GM path. When data dependencies exist,
 * insert SetFlag/WaitFlag at Kernel level:
 * - **Pre-call**: Fence prior UB writes (SetFlag/WaitFlag on MTE2_S)
 * - **Post-call**: Fence MTE3 before reading GM (MTE3_MTE2) or reusing UB (MTE3_S)
 * MTE UB buffer offset defaults to 0; adjust via aclshmemx_set_mte_config().
 *
 * @par RDMA Path Synchronization
 * Synchronous interfaces auto-quiet. Asynchronous interfaces require an explicit
 * quiet call to ensure completion.
 *
 * @warning Concurrent RMA/AMO to the same PE over RDMA is not supported.
 *          Use sync_id in device_state.rdma_config for pipeline synchronization.
 * ============================================================================ */

/**
 * @brief  Automatically generates aclshmem atomic fetch add functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | int32, uint32, uint64, int64, float   | Ascend950         |
 * | RDMA  | int32, uint32, int64, uint64          | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_add(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically adds value to the value at dest and returns the old value.
 * The function returns after the remote atomic operation has completed and the result
 * is visible on the remote PE. An internal quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the addition.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_add(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic inc functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | int32, uint32, uint64, int64,         | Ascend950         |
 * | RDMA  | int32, uint32, int64, uint64, float   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64, float   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_inc(\_\_gm\_\_ TYPE *dst, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Perform atomic increment operation on symmetric memory
 * from the specified PE to address on the local PE. Increments the value at the
 * destination by 1.
 * This is an asynchronous operation. The caller must explicitly synchronize to
 * ensure the operation has completed and the result is visible on the remote PE.
 *
 * @par Synchronization
 * - **MTE path**: The operation is issued via Scalar→UB→MTE3 and returns immediately.
 *   If there are data dependencies between the Scalar computation unit and MTE2/MTE3
 *   transfer units when reading/writing GM or sharing UB, the caller must insert
 *   appropriate SetFlag/WaitFlag synchronization at the Kernel Runtime level based
 *   on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer to the target data in symmetric memory.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_INC_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_inc(__gm__ TYPE* dst, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_INC_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch inc functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | int32, uint32, uint64, int64, float   | Ascend950         |
 * | RDMA  | int32, uint32, int64, uint64          | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_inc(\_\_gm\_\_ TYPE *dest, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically increments the value at dest by 1 and returns the old value.
 * The function returns after the remote atomic operation has completed and the result
 * is visible on the remote PE. An internal quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the increment.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_inc(__gm__ TYPE* dest, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic and functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_and(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Atomically performs bitwise AND operation between the value at dest and the specified value.
 *
 * @par Synchronization
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise AND with.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_AND_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_and(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_AND_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic or functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_or(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Atomically performs bitwise OR operation between the value at dest and the specified value.
 *
 * @par Synchronization
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise OR with.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_OR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_or(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_OR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic xor functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_xor(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Atomically performs bitwise XOR operation between the value at dest and the specified value.
 *
 * @par Synchronization
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise XOR with.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_XOR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_xor(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_XOR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch and functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_and(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically performs bitwise AND operation between the value at dest
 * and the specified value, and returns the old value. The function returns after the remote
 * atomic operation has completed and the result is visible on the remote PE. An internal
 * quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise AND with.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the operation.
 */
#define ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_and(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch or functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_or(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically performs bitwise OR operation between the value at dest
 * and the specified value, and returns the old value. The function returns after the remote
 * atomic operation has completed and the result is visible on the remote PE. An internal
 * quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise OR with.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the operation.
 */
#define ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_or(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch xor functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                | Hardware Platform |
 * |-------|--------------------------------|-------------------|
 * | RDMA  | int32, uint32, int64, uint64   | Ascend950         |
 * | UDMA  | int32, uint32, int64, uint64   | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_xor(\_\_gm\_\_ TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically performs bitwise XOR operation between the value at dest
 * and the specified value, and returns the old value. The function returns after the remote
 * atomic operation has completed and the result is visible on the remote PE. An internal
 * quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to perform bitwise XOR with.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the operation.
 */
#define ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_xor(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64, float   | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64, float   | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch(const TYPE *source, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically reads the value from source and returns it.
 * This operation does not modify the data at source.
 * The function returns after the remote atomic operation has completed and the result
 * is visible on the remote PE. An internal quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **source**  - [in] Pointer to the source data in symmetric memory.
 * - **pe**      - [in] PE number of the remote PE.
 *
 * @par Return
 * The value at source.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_FETCH_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch(__gm__ const TYPE* source, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic set functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64, float   | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64, float   | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_set(TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Atomically sets the value at dest to the specified value.
 *
 * @par Synchronization
 * - **MTE path**: The operation is issued via Scalar→UB→MTE3 and returns immediately.
 *   If there are data dependencies between the Scalar computation unit and MTE2/MTE3
 *   transfer units when reading/writing GM or sharing UB, the caller must insert
 *   appropriate SetFlag/WaitFlag synchronization at the Kernel Runtime level based
 *   on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to set.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_SET_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_set(__gm__ TYPE* dest, TYPE value, int32_t pe)

/**
 * @brief  Automatically generates aclshmem atomic set functions for types requiring CAST.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type,
 *        SUBNAME is the underlying type name, SUBTYPE is the underlying type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64, float   | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64, float   | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_set(TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Atomically sets the value at dest to the specified value.
 * Types are CAST to underlying unsigned integer types for the atomic operation.
 *
 * @par Synchronization
 * - **MTE path**: The operation is issued via Scalar→UB→MTE3 and returns immediately.
 *   If there are data dependencies between the Scalar computation unit and MTE2/MTE3
 *   transfer units when reading/writing GM or sharing UB, the caller must insert
 *   appropriate SetFlag/WaitFlag synchronization at the Kernel Runtime level based
 *   on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: This is an asynchronous operation. The caller must invoke
 *   aclshmemx_roce_quiet to guarantee the operation has completed and the
 *   data is visible on the remote PE.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to set.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_SET_TYPENAME_CAST(NAME, TYPE, SUBTYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_set(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_SET_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST(ACLSHMEM_ATOMIC_SET_TYPENAME_CAST);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic swap functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64, float   | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64, float   | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_swap(TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically swaps the value at dest with the specified value
 * and returns the old value. The function returns after the remote atomic operation
 * has completed and the result is visible on the remote PE. An internal quiet
 * operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to swap.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the swap.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_SWAP_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_swap(__gm__ TYPE* dest, TYPE value, int32_t pe)

/**
 * @brief  Automatically generates aclshmem atomic swap functions for types requiring CAST.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type,
 *        SUBNAME is the underlying type name, SUBTYPE is the underlying type.
 *
 * | Path  | Supported Types                       | Hardware Platform |
 * |-------|---------------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64, float   | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64, float   | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64          | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_swap(TYPE *dest, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically swaps the value at dest with the specified value
 * and returns the old value. Types are CAST to underlying unsigned integer types
 * for the atomic operation. The function returns after the remote atomic operation
 * has completed and the result is visible on the remote PE. An internal quiet
 * operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **value**  - [in] Value to swap.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the swap.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_SWAP_TYPENAME_CAST(NAME, TYPE, SUBTYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_swap(__gm__ TYPE* dest, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_SWAP_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST(ACLSHMEM_ATOMIC_SWAP_TYPENAME_CAST);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic compare swap functions for different data types.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * | Path  | Supported Types               | Hardware Platform |
 * |-------|-------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64  | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64  | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64  | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically compares the value at dest with cond.
 * If they are equal, the value at dest is set to value. Returns the old value at dest.
 * The function returns after the remote atomic operation has completed and the result
 * is visible on the remote PE. An internal quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **cond**   - [in] Value to compare against.
 * - **value**  - [in] Value to set if comparison succeeds.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the operation.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe)

/**
 * @brief  Automatically generates aclshmem atomic compare swap functions for types requiring CAST.
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type,
 *        SUBNAME is the underlying type name, SUBTYPE is the underlying type.
 *
 * | Path  | Supported Types               | Hardware Platform |
 * |-------|-------------------------------|-------------------|
 * | MTE   | uint32, uint64, int32, int64  | Ascend950         |
 * | RDMA  | uint32, uint64, int32, int64  | Ascend950         |
 * | UDMA  | uint32, uint64, int32, int64  | Ascend950         |
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Atomically compares the value at dest with cond.
 * If they are equal, the value at dest is set to value. Returns the old value at dest.
 * Types are CAST to underlying unsigned integer types for the atomic operation.
 * The function returns after the remote atomic operation has completed and the result
 * is visible on the remote PE. An internal quiet operation is performed before returning.
 *
 * @par Synchronization
 * - **MTE path**: Internally synchronized. However, if there are data dependencies between
 *   the Scalar computation unit and MTE2/MTE3 transfer units when reading/writing GM or
 *   sharing UB, the caller must insert appropriate SetFlag/WaitFlag synchronization at the
 *   Kernel Runtime level based on the actual data flow. See @ref amo_sync_model for details.
 * - **RDMA path**: Internally synchronized via built-in quiet. No additional
 *   aclshmemx_roce_quiet call is required.
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory.
 * - **cond**   - [in] Value to compare against.
 * - **value**  - [in] Value to set if comparison succeeds.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 * The old value at dest before the operation.
 *
 * @note The MTE transport for this operation does not support cross-PCIe (inter-node)
 *       communication. Use the RDMA or UDMA transport paths for cross-PCIe scenarios.
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_CAST(NAME, TYPE, SUBTYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_CAST);
/** \endcond */

#include "gm2gm/shmem_device_amo.hpp"
#endif
