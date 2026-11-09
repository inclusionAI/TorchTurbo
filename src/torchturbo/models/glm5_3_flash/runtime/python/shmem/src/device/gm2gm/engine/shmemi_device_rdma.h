/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_DEVICE_RDMA_H
#define ACLSHMEMI_DEVICE_RDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "utils/shmemi_kernel_debug.h"

enum class aclshmemi_rdma_backend_t : uint32_t { IN_DIE = 0, XSCALE, HNS_1825 };

enum class aclshmemi_rdma_opcode_t : uint32_t { OP_RDMA_READ = 0, OP_RDMA_WRITE, OP_RDMA_WRITE_WITH_IMM };

enum class aclshmemi_rdma_atomic_op_t : uint32_t { OP_ATOMIC_FA = 0, OP_ATOMIC_CAS };

enum class aclshmemi_rdma_db_mode_t : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };
struct aclshmemi_rdma_info {
    uint32_t qp_num;  // number of QP per connection
    uint64_t sq_ptr;  // pointer to send queue address array of size [PE_NUM][qp_num]
    uint64_t rq_ptr;  // pointer to receive queue address array of size [PE_NUM][qp_num]
    uint64_t scq_ptr; // pointer to send completion queue address array of size [PE_NUM][qp_num]
    uint64_t rcq_ptr; // pointer to receive completion queue address array of size [PE_NUM][qp_num]
    uint64_t mem_ptr; // pointer to memory region array of size [MAX_PE_NUM]
};

struct aclshmemi_rdma_mem_info {
    uint64_t size; // size of the memory region
    uint64_t addr; // start address of the memory region
    uint32_t lkey; // local key of the memory region
    uint32_t rkey; // remote key of the memory region
};

struct aclshmemi_rdma_sq_ctx {
    uint32_t wqn;       // work queue number
    uint64_t buf_addr;  // start address of ring buffer
    uint32_t wqe_size;  // size of each WQE
    uint32_t depth;     // depth of ring buffer
    uint64_t head_addr; // work queue head (Producer Index) address
    uint64_t tail_addr; // work queue tail (Consumer Index) address
    aclshmemi_rdma_db_mode_t db_mode;
    uint64_t db_addr;  // doorbell address (legacy, kept for backward compatibility)
    uint32_t sl;       // service level
    uint64_t amo_addr; // addr for atomic operation
    uint32_t amo_lkey; // lkey for amo_addr
    // hns1825 (Ascend950) specific fields
    uint64_t db_sw_addr; // software shadow doorbell address (used by hns_1825)
    uint8_t mtu_shift;   // MTU shift for WQE size calculation (used by hns_1825)
    uint8_t reserved[7]; // padding
};

struct aclshmemi_rdma_cq_ctx {
    uint32_t cqn;       // completion queue number
    uint64_t buf_addr;  // start address of ring buffer
    uint32_t cqe_size;  // size of each CQE
    uint32_t depth;     // depth of ring buffer
    uint64_t head_addr; // work queue head (Producer Index) address
    uint64_t tail_addr; // work queue tail (Consumer Index) address
    aclshmemi_rdma_db_mode_t db_mode;
    uint64_t db_addr; // doorbell address (used by XSCALE/indie/hns_1825)
    // hns1825 (Ascend950) specific fields
    uint64_t db_sw_addr; // software shadow doorbell address
};

struct aclshmemi_rdma_sge {
    __gm__ uint8_t* addr;
    uint64_t length;
    uint32_t lkey;
};

struct aclshmemi_rdma_atomic_params {
    union {
        struct {
            uint64_t compare_add;
            uint64_t swap;
        };
        struct {
            uint64_t swap_add_data;
            uint64_t compare_data;
            uint64_t swap_add_mask;
            uint64_t compare_mask;
        } masked_common;
        struct {
            uint64_t swap_data;
            uint64_t compare_data;
            uint64_t swap_mask;
            uint64_t compare_mask;
        } masked_cas64;
        struct {
            uint64_t add_data;
            uint64_t field_boundary;
        } masked_fa64;
        struct {
            uint32_t swap_data;
            uint32_t compare_data;
            uint32_t swap_mask;
            uint32_t compare_mask;
        } masked_cas32;
        struct {
            uint32_t add_data;
            uint32_t field_boundary;
            uint64_t _reserved;
        } masked_fa32;
    };
};

struct aclshmemi_rdma_send_wr {
    // ---- Operation ----
    uint32_t send_flags; // bitmask of send_flags
    uint32_t imm_data;

    // ---- Remote side ----
    __gm__ uint8_t* remote_addr;
    uint32_t rkey;

    // ---- Local side ----
    __gm__ uint8_t* local_addr;
    uint64_t message_len;
    uint32_t lkey;

    // ---- Local side(Multi-sge) ----
    uint32_t num_sge;
    aclshmemi_rdma_sge* sg_list;

    // ---- Atomic ----
    aclshmemi_rdma_atomic_params atomic;
};

ACLSHMEM_DEVICE __gm__ aclshmemi_rdma_info* aclshmemi_qp_info_fetch();

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id);

/**
 * @brief RDMA Quiet function for a single QP. This synchronous function ensures all previous RDMA WQEs
 * on the specified QP are completed (data has arrived at the destination NIC).
 *
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
ACLSHMEM_DEVICE void aclshmemi_roce_quiet(
    uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32,
    uint32_t sync_id);

/**
 * @brief RDMA Quiet function for all QPs of a PE. This synchronous function ensures all previous RDMA WQEs
 * on all QPs are completed (data has arrived at the destination NIC).
 *
 * @param pe                     [in] PE number of the remote PE.
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
ACLSHMEM_DEVICE void aclshmemi_roce_quiet(
    uint32_t pe, AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32,
    uint32_t sync_id);

/**
 * @brief RDMA Sync function. Performs a synchronization operation on the specified team,
 * ensuring all PEs in the team reach the sync point before proceeding.
 * Uses RDMA-based dissemination algorithm with highlevel signal operations.
 *
 * @param team              [in] Pointer to the team on which to perform synchronization.
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_team_sync(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Barrier function with explicit UB buffer and sync_id. Performs a barrier operation
 * on the specified team, ensuring all previous RDMA operations are completed and all PEs in the team
 * reach the barrier point before proceeding. First performs quiet on all QPs for all PEs in the team,
 * then performs a sync operation.
 * This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 * conflicts with the default rdma_config in device_state.
 *
 * @param team                   [in] Pointer to the team on which to perform barrier.
 * @param buf                    [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_barrier(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Atomic Fetch and Add function.
 *
 * @tparam T                     [in] data type of the atomic fetch and add operation
 * @tparam IS_MASKED             [in] whether the atomic fetch and add operation is masked
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] Reserved field, not used in atomic fetch and add operation. It is recommended to
 * pass nullptr when called externally.
 * @param src                    [in] reserved field, not used in atomic fetch and add operation
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param add_val                [in] add val for atomic fetch and add operation
 * @param boundary               [in] boundary value for masked atomic fetch and add operation
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 * @return T 0 for success, non-zero for failure
 */
template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE T aclshmemi_roce_amo_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Atomic Compare and Swap function.
 *
 * @tparam T                     [in] data type of the atomic compare and swap operation
 * @tparam IS_MASKED             [in] whether the atomic compare and swap operation is masked
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] Reserved field, not used in atomic compare and swap operation. It is recommended
 * to pass nullptr when called externally.
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param swap_val               [in] swap val for atomic compare and swap operation
 * @param comp_val               [in] compare val for atomic compare and swap operation
 * @param swap_mask              [in] swap mask for masked atomic compare and swap operation
 * @param comp_mask              [in] compare mask for masked atomic compare and swap operation
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 * @return T 0 for success, non-zero for failure
 */
template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE T aclshmemi_roce_amo_cas(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t> ub_local64,
    AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id);

#endif // ACLSHMEMI_DEVICE_RDMA_H
