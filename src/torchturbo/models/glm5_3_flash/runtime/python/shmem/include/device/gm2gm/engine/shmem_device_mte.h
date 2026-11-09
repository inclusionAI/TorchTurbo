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
#ifndef SHMEM_DEVICE_MTE_H
#define SHMEM_DEVICE_MTE_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_mte.hpp"

/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE.
 * @note The supported access method for the returned address depends on the transport and topology.
 */
ACLSHMEM_DEVICE __gm__ void* aclshmem_ptr(__gm__ void* ptr, int pe);
#define shmem_ptr aclshmem_ptr

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local
 * device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 *        PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf,
    const non_contiguous_copy_param& copy_params, int pe, uint32_t sync_id);
#define shmem_mte_get_mem_nbi aclshmemx_mte_get_nbi

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf,
    const non_contiguous_copy_param& copy_params, int pe, uint32_t sync_id);
#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi

/**
 * @brief Asynchronous interface. Clear instruction pipes and flush data cache to GM.
 */
ACLSHMEM_DEVICE void aclshmemx_mte_quiet();

/**
 * @brief Atomic fetch operation. Returns the value at the source address on the specified PE.
 *        Supported hardware platform: Ascend950.
 *        WARNING: Use sync_id in device_state.mte_config for pipeline synchronization.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports int32_t/uint32_t/float/int64_t/uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param src               [in] Symmetric address of the source data.
 * @param pe                [in] PE number of the remote PE.
 * @return The value at the source address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch(__gm__ T* src, int32_t pe);

/**
 * @brief Atomic set operation. Sets the value at the destination address on the specified PE.
 *        Supported hardware platform: Ascend950.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports uint32_t and uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be set.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic compare and swap operation. Conditionally updates the value at the destination address.
 *        Supported hardware platform: Ascend950.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports uint32_t and uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param cond              [in] Value to compare against.
 * @param value             [in] Value to be written if comparison succeeds.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Atomic swap operation. Swaps the value at the destination address.
 *        Supported hardware platform: Ascend950.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports uint32_t and uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be swapped.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic increment operation. Increments the value at the destination address by 1.
 *        Supported hardware platform: Ascend950.
 *        WARNING: Use sync_id in device_state.mte_config for pipeline synchronization.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports int32_t/uint32_t/float/int64_t/uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic add operation. Adds the value to the destination address.
 *        WARNING: Use sync_id in device_state.mte_config for pipeline synchronization.
 *        NOTE: This is an asynchronous interface. The final operation differs by platform and type:
 *              - On Ascend910B/Ascend910C: The atomic add is performed by MTE2 unit.
 *              - On Ascend950:
 *                - uint32_t/int64_t/uint64_t: scalar AtomicAdd (Scalar unit).
 *                - Other types: UB + MTE3 (MTE3 atomic add).
 *              If there are data dependencies, developers need to insert synchronization according
 *              to actual situations. Scalar write to UB before MTE3 read requires S_MTE3 event
 *              synchronization; MTE3 write to GM before subsequent reads requires MTE3_MTE2
 *              event synchronization.
 *
 * | Data Type                      | Ascend910B/Ascend910C | Ascend950 |
 * |--------------------------------|-----------------------|-----------|
 * | int8_t/int16_t/half/bfloat16_t | ✓                     | ✓         |
 * | int32_t/float                  | ✓                     | ✓         |
 * | uint32_t/int64_t/uint64_t      | ✗                     | ✓         |
 *
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic fetch increment operation. Increments the value at the destination address by 1 and returns the old
 *        value.
 *
 * @note Supported hardware platform: Ascend950.
 * @warning Use sync_id in device_state.mte_config for pipeline synchronization.
 * @note This is an asynchronous interface involving scalar computation (Scalar). If there are data dependencies
 *       between the scalar computation unit and the move unit (MTE2/MTE3) when reading or writing GM, developers must
 *       insert synchronization as appropriate.
 * @note T only supports int32_t/uint32_t/float/int64_t/uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before increment.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic fetch add operation. Adds the value to the destination address and returns the old value.
 *
 * @note Supported hardware platform: Ascend950.
 * @warning Use sync_id in device_state.mte_config for pipeline synchronization.
 * @note This is an asynchronous interface involving scalar computation (Scalar). If there are data dependencies
 *       between the scalar computation unit and the move unit (MTE2/MTE3) when reading or writing GM, developers must
 *       insert synchronization as appropriate.
 * @note T only supports int32_t/uint32_t/float/int64_t/uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before addition.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

#endif
