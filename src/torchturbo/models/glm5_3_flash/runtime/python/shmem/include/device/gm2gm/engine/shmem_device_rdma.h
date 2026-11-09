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
#ifndef SHMEM_DEVICE_RDMA_H
#define SHMEM_DEVICE_RDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_rdma.hpp"

/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE for RDMA
 *        operations.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE for use by RDMA operations.
 */
ACLSHMEM_DEVICE __gm__ void* aclshmem_roce_ptr(__gm__ void* ptr, int pe);
#define shmem_roce_ptr aclshmem_roce_ptr

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data on the local PE.
 * @param src               [in] Symmetric address of the source data.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data on the local PE.
 * @param src               [in] Symmetric address of the source data.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data on the local PE.
 * @param src               [in] GlobalTensor at the symmetric address of the source data.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size,
    int pe);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data on the local PE.
 * @param src               [in] GlobalTensor at the symmetric address of the source data.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);
#define shmem_roce_get_mem_nbi aclshmemx_roce_get_nbi

/**
 * @brief Adds the current nonblocking RoCE Get operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the local PE.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the local PE.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Local GlobalTensor for the destination data on the local PE.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Local GlobalTensor for the destination data on the local PE.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data.
 * @param src               [in] Symmetric address of the source data on the local PE.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data.
 * @param src               [in] Symmetric address of the source data on the local PE.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id);
/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data.
 * @param src               [in] GlobalTensor at the symmetric address of the source data on the local PE.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size,
    int pe);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data.
 * @param src               [in] GlobalTensor at the symmetric address of the source data on the local PE.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);
#define shmem_roce_put_mem_nbi aclshmemx_roce_put_nbi

/**
 * @brief Adds the current nonblocking RoCE Put operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Symmetric source address on the local PE.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it
 *       returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Symmetric source address on the local PE.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data on the local PE.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is currently supported only on XSCALE, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it
 *       returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data on the local PE.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief RDMA Quiet function. This synchronous function ensures all previous RDMA WQEs are completed
 * (data has arrived at the destination NIC).
 *
 * @param pe                [in] PE number of the remote PE.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_quiet(uint32_t pe, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Team Sync function. Performs a synchronization operation on the specified team,
 * ensuring all PEs in the team reach the sync point before proceeding.
 * This is a collective operation that uses RDMA-based dissemination algorithm.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 *
 * @param team              [in] Pointer to the team on which to perform synchronization.
 */
ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team);
#define aclshmemx_roce_sync aclshmemx_roce_team_sync

/**
 * @brief RDMA Team Sync function with explicit UB buffer and sync_id. Performs a synchronization
 * operation on the specified team, ensuring all PEs in the team reach the sync point before proceeding.
 * This is a collective operation that uses RDMA-based dissemination algorithm.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param team              [in] Pointer to the team on which to perform synchronization.
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 * @return 0 on success, non-zero on failure.
 */
template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Sync All function. Performs a synchronization operation on all PEs
 * (ACLSHMEM_TEAM_WORLD), ensuring all PEs reach the sync point before proceeding.
 * Equivalent to aclshmemx_roce_team_sync with ACLSHMEM_TEAM_WORLD team.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 */
ACLSHMEM_DEVICE void aclshmemx_roce_sync_all();

/**
 * @brief RDMA Sync All function with explicit UB buffer and sync_id. Performs a synchronization
 * operation on all PEs (ACLSHMEM_TEAM_WORLD), ensuring all PEs reach the sync point before proceeding.
 * Equivalent to aclshmemx_roce_team_sync with ACLSHMEM_TEAM_WORLD team.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_sync_all(__ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Barrier function. Performs a barrier operation on the specified team,
 * ensuring all previous RDMA operations are completed and all PEs in the team
 * reach the barrier point before proceeding. This function first performs a quiet
 * operation on all QPs for all PEs in the team, then performs a sync operation.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 *
 * @param team              [in] Pointer to the team on which to perform barrier.
 */
ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team);

/**
 * @brief RDMA Barrier function with explicit UB buffer and sync_id. Performs a barrier operation
 * on the specified team, ensuring all previous RDMA operations are completed and all PEs in the team
 * reach the barrier point before proceeding. This function first performs a quiet operation on all QPs
 * for all PEs in the team, then performs a sync operation.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param team              [in] Pointer to the team on which to perform barrier.
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Barrier All function. Performs a barrier operation on all PEs
 * (ACLSHMEM_TEAM_WORLD), ensuring all previous RDMA operations are completed and
 * all PEs reach the barrier point before proceeding.
 * Equivalent to aclshmemx_roce_barrier with ACLSHMEM_TEAM_WORLD team.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 */
ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all();

/**
 * @brief RDMA Barrier All function with explicit UB buffer and sync_id. Performs a barrier operation
 * on all PEs (ACLSHMEM_TEAM_WORLD), ensuring all previous RDMA operations are completed and
 * all PEs reach the barrier point before proceeding.
 * Equivalent to aclshmemx_roce_barrier with ACLSHMEM_TEAM_WORLD team.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all(__ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief Synchronous interface. Returns the value at the source address on the specified PE.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit data types.
 *
 * @param src               [in] Symmetric address of the source data.
 * @param pe                [in] PE number of the remote PE.
 * @return The value at the source address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch(__gm__ T* src, int32_t pe);

/**
 * @brief Asynchronous interface. Sets the value at the destination address on the specified PE.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit data types.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be set.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Conditionally updates the value at the destination address.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param cond              [in] Value to compare against.
 * @param value             [in] Value to be written if comparison succeeds.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Synchronous interface. Swaps the value at the destination address. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be swapped.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Increments the value at the destination address by 1.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Asynchronous interface. Adds the value to the destination address. Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_xor(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Increments the value at the destination address by 1 and returns the old
 * value. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before increment.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_inc(__gm__ T* dst, int32_t pe);
/**
 * @brief Synchronous interface. Adds the value to the destination address and returns the old value.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before addition.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe);

#endif
