/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H

#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"

/*
 *  =====================================================================================================
 *  SHMEM RDMA Architecture Structure —— Base Header
 *  =====================================================================================================
 *  We provide two categories of APIs:
 *  1. Fine-grained primitives (ibverbs-style)
 *      - aclshmemi_roce_fill_wqe
 *      - aclshmemi_roce_ring_sq_doorbell
 *      - aclshmemi_roce_post_send
 *      - aclshmemi_roce_ring_cq_doorbell
 *      - aclshmemi_roce_poll_cq
 *
 *  2. Coarse-grained operations
 *      - aclshmemi_roce_write
 *      - aclshmemi_roce_read
 *      - aclshmemi_roce_atomic_fetch_and_add
 *      - aclshmemi_roce_atomic_compare_and_swap
 * =====================================================================================================
 *  Internal call chain (base -> backend-specialized):
 *      aclshmemi_roce_post_send<B, OP_CODE>()                  // compile-time dispatched entry
 *          -> aclshmemi_roce_post_send_read_write()            // shared post-send helper for READ/WRITE
 *              -> aclshmemi_roce_fill_wqe<B, OP_CODE>()        // compile-time dispatched fill_wqe
 *                  -> aclshmemi_rdma_fill_wqe_write_read()     // shared fill_wqe helper for READ/WRITE
 *          -> aclshmemi_roce_ring_sq_doorbell<B>()             // backend-dispatched doorbell
 * =====================================================================================================
 */

/**
 * @brief Fill WQE for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param sq_context             [in] Current QP's Send queue Context
 * @param wqe_addr               [in] WQE address to fill
 * @param cur_head               [in] Current head of WQE queue
 * @return uint32_t              [out] total size of WQE in bytes
 */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head);

/**
 * @brief Ring SQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param sq_context             [in] Current QP's Send queue Context
 * @param cur_head               [in] Current head of SQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief AIV direct RDMA helper function for post send, prepare WQE and ring doorbell.
 * Directly calls the underlying implementation through template parameters
 * without using switch, resulting in higher performance.
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Ring CQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param cur_tail               [in] Current tail of CQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell(
    uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief RDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param target_idx             [in] expect completion queue consumer index after polling
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(
    uint32_t pe, uint32_t qp_idx, uint32_t target_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_write(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)dst;
    wr.local_addr = (__gm__ uint8_t*)src;
    wr.message_len = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_read(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    // Attention: Read need src to be wr.remote_addr.
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)src;
    wr.local_addr = (__gm__ uint8_t*)dst;
    wr.message_len = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/**
 * @brief Asynchronous RDMA Atomic Fetch and Add function.
 *
 * @tparam T                     Data type for the atomic operation.
 * @tparam IS_MASKED             Flag indicating whether this is a masked atomic operation.
 * @tparam B                     RDMA Backend type
 *
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] reserved field, input could be invalid address, implementation should not use it
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param add_val                [in] value to be added
 * @param boundary               [in] boundary value for masked FA operation
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_fetch_and_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Atomic Compare and Swap function.
 *
 * @tparam T                     Data type for the atomic operation.
 * @tparam IS_MASKED             Flag indicating whether this is a masked atomic operation.
 * @tparam B                     RDMA Backend type
 *
 * @param dst                    [in] Destination address in remote HBM
 * @param src                    [in] reserved field, input could be invalid address, implementation should not use it
 * @param pe                     [in] PE number of the remote PE
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param swap_val               [in] Value to be swapped
 * @param comp_val               [in] Value to be compared
 * @param swap_mask              [in] Mask to apply to swap_val
 * @param comp_mask              [in] Mask to apply to comp_val
 * @param ub_local64             [in] Temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] Temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event
 */
template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_compare_and_swap(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/*
 * Backend Atomic Operations Traits System
 *
 * Static polymorphism design for RDMA atomic operations across different backends.
 *
 * Naming Convention:
 *   - aclshmemi_backend_traits<B>: Primary template (undefined, forces specialization)
 *   - atomic_op_traits<T, IS_MASKED>: Nested traits for type-specific operations
 *
 * Structure:
 *   template <aclshmemi_rdma_backend_t B>
 *   struct aclshmemi_backend_traits {
 *       template <typename T, bool IS_MASKED>
 *       struct atomic_op_traits {
 *           static uint32_t fill_wqe(...);   // Construct WQE
 *           static void post_send(...);      // Submit and ring doorbell
 *       };
 *   };
 *
 * Usage:
 *   Each backend (e.g., IN_DIE, XSCALE) must provide a full specialization:
 *
 *   template <>
 *   struct aclshmemi_backend_traits<aclshmemi_rdma_backend_t::XSCALE> {
 *       template <typename T, bool IS_MASKED>
 *       struct atomic_op_traits {
 *           template <aclshmemi_rdma_atomic_op_t OP>
 *           static uint32_t fill_wqe(...) { ... }
 *           template <aclshmemi_rdma_atomic_op_t OP>
 *           static void post_send(...) { ... }
 *       };
 *   };
 *
 * Features:
 *   - Zero runtime overhead through compile-time dispatch
 *   - Each backend implements ONE specialization for all T and IS_MASKED combinations
 */

template <aclshmemi_rdma_backend_t B>
struct aclshmemi_backend_traits {
    template <typename T, bool IS_MASKED>
    struct atomic_op_traits {
        /**
         * @brief Construct Work Queue Entry (WQE) for atomic operation.
         *
         * @tparam ATOMIC_OP_CODE Atomic operation code (OP_ATOMIC_FA or OP_ATOMIC_CAS)
         *
         * @param wr Work request with atomic parameters. Must include:
         *           - remote_addr: Target address in remote memory
         *           - local_addr: Local buffer for result data
         *           - lkey: Local memory key
         *           - rkey: Remote memory key
         *           - atomic.masked_common: Atomic operation parameters
         * @param sq_context Send Queue context for backend configuration
         * @param wqe_addr HBM address for WQE
         * @param cur_head Current SQ producer index
         *
         * @return uint32_t WQE size in bytes for cache flush operations
         */
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE uint32_t fill_wqe(
            aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
            uint32_t cur_head);

        /**
         * @brief Submit atomic operation WQE and ring doorbell.
         *
         * Main entry point for atomic operations. Handles QP context retrieval,
         * queue capacity management, WQE construction, cache flush, and doorbell ring.
         *
         * @tparam ATOMIC_OP_CODE Atomic operation code determining operation type
         *
         * @param wr Work request with atomic parameters. Must include:
         *           - remote_addr: Target address in remote memory
         *           - local_addr: Local buffer for result data
         *           - lkey: Local memory key
         *           - rkey: Remote memory key
         *           - atomic.masked_common: Atomic operation parameters
         * @param pe Remote PE number
         * @param qp_idx Queue Pair index (0 for single-QP configurations)
         * @param ub_local64 64-bit workspace for doorbell operations
         * @param ub_local32 32-bit workspace for index updates
         * @param sync_id Sync ID for S/MTE3 pipeline coordination
         *
         */
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE void post_send(
            aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
            AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);
    };
};

// The following structure is used to provide compile-time errors when calling unimplemented atomic operations
template <aclshmemi_rdma_atomic_op_t>
struct aclshmemi_atomic_op_dependent_false : std::false_type {};

// The following structure is used to provide compile-time errors when calling an unimplemented backend
template <aclshmemi_rdma_backend_t>
struct aclshmemi_rdma_backend_dependent_false : std::false_type {};

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H
