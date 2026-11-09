/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_XSCALE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_XSCALE_HPP

#include <type_traits>
#include "rdma_device_backend_base.h"

constexpr int ACLSHMEMI_XSCALE_API_VERSION_VAR = 2;

static_assert(
    ACLSHMEMI_XSCALE_API_VERSION_VAR == 1 || ACLSHMEMI_XSCALE_API_VERSION_VAR == 2,
    "ACLSHMEMI_XSCALE_API_VERSION must be 1 or 2");

struct aclshmemi_xscdv_wqe_ctrl_seg_v1 {
    uint8_t msg_opcode;
    uint8_t with_imm : 1;
    uint8_t csum_en : 2;
    uint8_t ds_data_num : 5;
    uint16_t wqe_id; // wqe ds index = wqe_idx << 3 (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT)
    uint32_t msg_len;
    /**** 8 bytes ****/
    uint32_t opcode_data;
    /**** 12 bytes ****/
    uint32_t se : 1;
    uint32_t ce : 1; // If set to 1, a CQE will be generated
    uint32_t in_line : 1;
    uint32_t fence_mode : 2;
    uint32_t mask : 2;
    uint32_t rsv : 25;
    /**** 16 bytes ****/
};

struct aclshmemi_xscdv_wqe_ctrl_seg_v2 {
    uint8_t msg_opcode;
    uint8_t with_imm : 1;
    uint8_t csum_en : 2;
    uint8_t ds_data_num : 5;
    uint16_t rsv1;
    uint32_t msg_len;
    /**** 8 bytes ****/
    uint32_t opcode_data;
    /**** 12 bytes ****/
    uint32_t se : 1;
    uint32_t ce : 1; // If set to 1, a CQE will be generated
    uint32_t in_line : 1;
    uint32_t fence_mode : 2;
    uint32_t mask : 2;
    uint32_t rsv : 5;
    uint32_t wqe_id : 20; // wqe ds index = wqe_idx << 3
                          // (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT)
    /**** 16 bytes ****/
};

using aclshmemi_xscdv_wqe_ctrl_seg_t = std::conditional_t<
    ACLSHMEMI_XSCALE_API_VERSION_VAR == 1, aclshmemi_xscdv_wqe_ctrl_seg_v1, aclshmemi_xscdv_wqe_ctrl_seg_v2>;

struct aclshmemi_xscdv_wqe_data_seg_t {
    union {
        struct {
            uint32_t rsv : 1;
            uint32_t seg_len : 31;
            uint32_t m_key; // Use lkey for local address, rkey for remote address
            /**** 8 bytes ****/
            uint64_t va;
            /**** 16 bytes ****/
        };
        struct {
            uint8_t inline_data[16];
        };
    };
    /**** 16 bytes ****/
};

struct aclshmemi_xscdv_diamond_cqe_v1 {
    uint32_t error_code : 8;
    uint32_t qp_id : 15; // Corresponds to the QP's qpn
    uint32_t rsv : 1;
    uint32_t se : 1;
    uint32_t has_pph : 1;
    uint32_t type : 1;
    uint32_t with_imm : 1;
    uint32_t csum_err : 4;
    /**** 4 bytes ****/
    uint32_t imm_data;
    /**** 8 bytes ****/
    uint32_t msg_len;
    uint32_t vni;
    /**** 16 bytes ****/
    uint64_t ts : 48;
    uint64_t wqe_id : 16; // Corresponds to wqe_id << 3 in the WQE
                          // (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT)
    /**** 24 bytes ****/
    uint8_t msg_opcode;
    uint8_t rsv0;
    uint16_t rsv1[2];
    uint16_t rsv2 : 15;
    uint16_t owner : 1; // Checking the owner bit confirms whether the current CQE can be parsed by software
    /**** 32 bytes ****/
};

struct aclshmemi_xscdv_diamond_cqe_v2 {
    union {
        struct {
            uint32_t error_code : 8; // [0:7]
            uint32_t qp_id : 15;     // [8:22] Corresponds to the QP's qpn
            // [23:31] flags
            uint32_t rsv : 1;
            uint32_t se : 1;
            uint32_t has_pph : 1;
            uint32_t type : 1;
            uint32_t with_imm : 1;
            uint32_t csum_err : 4;
        };
        uint32_t flags_qp_id_err_code;
    };
    /**** 4 bytes ****/
    uint32_t imm_data; // immediate value
    /**** 8 bytes ****/
    uint32_t msg_len; // message length
    /**** 12 bytes ****/
    uint32_t vni;
    /**** 16 bytes ****/
    uint32_t ts_l;
    /**** 20 bytes ****/
    uint32_t ts_h;
    /**** 24 bytes ****/
    union {
        struct {
            uint32_t msg_opcode : 8; // [0:7] msg_opcode of corresponding finished wqe, check aclshmemi_xscdv_msg_type_t
            uint32_t rsv1 : 4;       // [8:11]
            uint32_t wqe_id : 20;    // [12:31] Corresponds to wqe_id << 3 in the WQE
        };
        uint32_t wqe_id_rsv_opcode;
    };
    /**** 28 bytes ****/
    union {
        struct {
            uint32_t rsv2 : 31;
            uint32_t owner : 1; // Owner bit, checking the owner bit confirms whether the current CQE can be parsed by
                                // software
        };
        uint32_t owner_rsv;
    };
    /**** 32 bytes ****/
};

using aclshmemi_xscdv_diamond_cqe_t = std::conditional_t<
    ACLSHMEMI_XSCALE_API_VERSION_VAR == 1, aclshmemi_xscdv_diamond_cqe_v1, aclshmemi_xscdv_diamond_cqe_v2>;

struct aclshmemi_xscdv_cqe64_t {
    aclshmemi_xscdv_diamond_cqe_t cqe;
    /**** 32 bytes ****/
    uint8_t padding[32];
    /**** 64 bytes ****/
};

union aclshmemi_xscdv_diamond_cq_doorbell_t {
    struct {
        uint64_t cq_next_cid : 23; // ID of the next CQE to be processed in the CQ, e.g., if 4 CQEs have been processed,
                                   // cq_next_cid is 4
        uint64_t cq_id : 16;       // Corresponds to the CQ's cqn
        uint64_t cq_sta : 2;
        /**** 8 bytes ****/
    };
    uint64_t raw;
};

union aclshmemi_xscdv_diamond_recv_doorbell_t {
    struct {
        uint64_t next_pid : 17; // ID of the next WQE to be processed in the RQ, e.g., if WQEs 0-3 are to be sent,
                                // next_pid is 4
        uint64_t qp_id : 16;    // Corresponds to the RQ's qpn
        /**** 8 bytes ****/
    };
    uint64_t raw;
};

union aclshmemi_xscdv_diamond_send_doorbell_v1 {
    struct {
        uint64_t next_pid : 17; // ID of the next WQE to be processed in the SQ, e.g., if WQEs 0-3 are to be sent,
                                // next_pid is 4
        uint64_t qp_id : 16;    // Corresponds to the SQ's qpn
        /**** 8 bytes ****/
    };
    uint64_t raw;
};

union aclshmemi_xscdv_diamond_send_doorbell_v2 {
    struct {
        uint64_t next_pid : 21; // ID of the next WQE to be processed in the SQ, e.g., if WQEs 0-3 are to be sent,
                                // next_pid is 4
        uint64_t qp_id : 16;    // Corresponds to the SQ's qpn
        /**** 8 bytes ****/
    };
    uint64_t raw;
};

using aclshmemi_xscdv_diamond_send_doorbell_t = std::conditional_t<
    ACLSHMEMI_XSCALE_API_VERSION_VAR == 1, aclshmemi_xscdv_diamond_send_doorbell_v1,
    aclshmemi_xscdv_diamond_send_doorbell_v2>;

struct aclshmemi_xscdv_diamond_data_seg_t {
    uint32_t length;
    uint32_t key;
    /**** 8 bytes ****/
    uint64_t addr;
    /**** 16 bytes ****/
};

struct aclshmemi_xsc_wqe_atomic_seg_t {
    uint64_t swap_add;
    /**** 8 bytes ****/
    uint64_t compare;
    /**** 16 bytes ****/
};

struct aclshmemi_xsc_wqe_atomic_64_masked_fa_seg_t {
    uint64_t add_data;
    /**** 8 bytes ****/
    uint64_t field_boundary;
    /**** 16 bytes ****/
};

struct aclshmemi_xsc_wqe_atomic_32_masked_fa_seg_t {
    uint32_t add_data;
    /**** 4 bytes ****/
    uint32_t field_boundary;
    /**** 8 bytes ****/
    uint64_t reserved;
    /**** 16 bytes ****/
};

struct aclshmemi_xsc_wqe_atomic_64_masked_cas_seg_t {
    uint64_t swap_add;
    /**** 8 bytes ****/
    uint64_t compare;
    /**** 16 bytes ****/
};

struct aclshmemi_xsc_wqe_atomic_32_masked_cas_seg_t {
    uint32_t swap_data;
    /**** 4 bytes ****/
    uint32_t compare_data;
    /**** 8 bytes ****/
    uint32_t swap_mask;
    /**** 12 bytes ****/
    uint32_t compare_mask;
    /**** 16 bytes ****/
};

enum class aclshmemi_xscdv_msg_type_t : uint32_t {
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_WRITE = 1,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_READ = 2,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_CMP_AND_SWAP = 26,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_FETCH_AND_ADD = 27,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_8B_MSK_CMP_AND_SWAP = 31,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_8B_MSK_FETCH_AND_ADD = 32,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_4B_MSK_CMP_AND_SWAP = 33,
    ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_4B_MSK_FETCH_AND_ADD = 34,
};

enum class aclshmemi_xsc_opcode_t : uint32_t {
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_SEND = 0,
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_SEND_IMMDT = 1,
    ACLSHMEMI_XSC_OPCODE_RDMA_RSP_RECV = 2,
    ACLSHMEMI_XSC_OPCODE_RDMA_RSP_RECV_IMMDT = 3,
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_WRITE = 4,
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_WRITE_IMMDT = 5,
    ACLSHMEMI_XSC_OPCODE_RDMA_RSP_WRITE_IMMDT = 6,
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_READ = 7,
    ACLSHMEMI_XSC_OPCODE_RDMA_REQ_ERROR = 8,
    ACLSHMEMI_XSC_OPCODE_RDMA_RSP_ERROR = 9,
    ACLSHMEMI_XSC_OPCODE_RDMA_CQE_ERROR = 10,
    ACLSHMEMI_XSC_OPCODE_RDMA_MAD_REQ_SEND = 11,
    ACLSHMEMI_XSC_OPCODE_RDMA_MAD_RSP_RECV = 12,
    ACLSHMEMI_XSC_OPCODE_RDMA_CQE_RAW_SNF = 13,
};

constexpr int ACLSHMEMI_XSCALE_BASE_WQE_SHIFT = 4;
constexpr int ACLSHMEMI_XSCALE_SND_WQE_SHIFT = 7;
constexpr uint32_t ACLSHMEMI_XSCALE_SND_WQE_SIZE = 1 << ACLSHMEMI_XSCALE_SND_WQE_SHIFT;

constexpr uint32_t ACLSHMEMI_XSC_CQE_OWNER_MASK = 1;

constexpr uint64_t ACLSHMEMI_BYTE_WIDTH = 8;
constexpr uint64_t ACLSHMEMI_BYTE_MASK = 0xFF;

constexpr uint32_t BYTES_32 = sizeof(uint32_t);
constexpr uint32_t BYTES_64 = sizeof(uint64_t);

constexpr uint64_t ACLSHMEMI_HOST_BYTE_7_SHIFT = 0;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_6_SHIFT = ACLSHMEMI_BYTE_WIDTH * 1;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_5_SHIFT = ACLSHMEMI_BYTE_WIDTH * 2;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_4_SHIFT = ACLSHMEMI_BYTE_WIDTH * 3;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_3_SHIFT = ACLSHMEMI_BYTE_WIDTH * 4;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_2_SHIFT = ACLSHMEMI_BYTE_WIDTH * 5;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_1_SHIFT = ACLSHMEMI_BYTE_WIDTH * 6;
constexpr uint64_t ACLSHMEMI_HOST_BYTE_0_SHIFT = ACLSHMEMI_BYTE_WIDTH * 7;

constexpr uint64_t ACLSHMEMI_BE_BYTE_0_SHIFT = ACLSHMEMI_BYTE_WIDTH * 7;
constexpr uint64_t ACLSHMEMI_BE_BYTE_1_SHIFT = ACLSHMEMI_BYTE_WIDTH * 6;
constexpr uint64_t ACLSHMEMI_BE_BYTE_2_SHIFT = ACLSHMEMI_BYTE_WIDTH * 5;
constexpr uint64_t ACLSHMEMI_BE_BYTE_3_SHIFT = ACLSHMEMI_BYTE_WIDTH * 4;
constexpr uint64_t ACLSHMEMI_BE_BYTE_4_SHIFT = ACLSHMEMI_BYTE_WIDTH * 3;
constexpr uint64_t ACLSHMEMI_BE_BYTE_5_SHIFT = ACLSHMEMI_BYTE_WIDTH * 2;
constexpr uint64_t ACLSHMEMI_BE_BYTE_6_SHIFT = ACLSHMEMI_BYTE_WIDTH * 1;
constexpr uint64_t ACLSHMEMI_BE_BYTE_7_SHIFT = 0;

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
constexpr uint32_t ACLSHMEMI_XSC_CYCLE_TO_TIME_BASE = 50;
#else
constexpr uint32_t ACLSHMEMI_XSC_CYCLE_TO_TIME_BASE = 1000;
#endif
constexpr uint32_t ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_DURATION = 5 * 60 * 1000000; // 5 minutes in microseconds
constexpr uint64_t ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_CYCLES =
    ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_DURATION *
    ACLSHMEMI_XSC_CYCLE_TO_TIME_BASE; // the maximum cycles to wait for a single CQE
// Set the error code to exceed the maximum value of cqe->error_code to ensure it can be used for judgment
constexpr uint32_t ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_ERROR = 0x10000;

ACLSHMEM_DEVICE uint64_t aclshmemi_htobe64(uint64_t host_val)
{
    return ((host_val >> ACLSHMEMI_HOST_BYTE_7_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_0_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_6_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_1_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_5_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_2_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_4_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_3_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_3_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_4_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_2_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_5_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_1_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_6_SHIFT |
           ((host_val >> ACLSHMEMI_HOST_BYTE_0_SHIFT) & ACLSHMEMI_BYTE_MASK) << ACLSHMEMI_BE_BYTE_7_SHIFT;
}

ACLSHMEM_DEVICE uint32_t aclshmemi_htobe32(uint32_t host_val)
{
    return ((host_val >> 24) & 0xFF) | ((host_val >> 8) & 0xFF00) | ((host_val << 8) & 0xFF0000) |
           ((host_val << 24) & 0xFF000000);
}

/**
 * @brief Write data from UB local buffer to GM global memory with synchronization
 *
 * This function writes data from local UB (Uniform Buffer) to specified GM (Global Memory) address
 * with complete hardware synchronization sequence to ensure data write completion.
 *
 * @tparam T Data type template parameter
 * @param addr GM target address where data will be written
 * @param ub_local Local UB buffer containing data to write
 * @param size Data size in bytes to write
 * @param sync_id Synchronization event ID for hardware event synchronization
 *
 * @note Synchronization sequence:
 *   1. S_MTE3: Scalar to MTE3 synchronization before data copy
 *   2. DataCopyPad: Execute data copy from UB to GM
 *   3. PIPE_MTE3: MTE3 pipeline barrier to ensure all MTE3 operations complete
 *   4. MTE3_S: MTE3 to Scalar synchronization after data copy
 *
 * @note Usage: Used in RDMA operations to update critical data structures in global memory
 *       (e.g., CQ tail pointer, SQ head pointer, doorbell registers) with strict synchronization.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write_ub_to_gm_with_sync(
    uint64_t addr, AscendC::LocalTensor<T>& ub_local, uint32_t size, uint32_t sync_id)
{
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<T> tmp_global_tensor;
    tmp_global_tensor.SetGlobalBuffer((__gm__ T*)addr);
    AscendC::DataCopyExtParams copy_params{1, size, 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(tmp_global_tensor, ub_local, copy_params);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

ACLSHMEM_DEVICE bool aclshmemi_roce_xscale_check_cqe_owner(
    __gm__ aclshmemi_xscdv_cqe64_t* cqe64, uint32_t cur_tail, uint32_t depth)
{
    bool owner_bit = (cqe64->cqe.owner & ACLSHMEMI_XSC_CQE_OWNER_MASK);
    // ! For performance reasons, depth should be a power of two, which will be optimized to bitwise operations by the
    // ! compiler at compile time
    bool expect_bit = ((cur_tail / depth) & 1); // same as (cur_tail >> log2(depth)) & 1
    return owner_bit == expect_bit;
}
template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::XSCALE>(
    uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();

    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));
    // Update cur_tail in global memory
    ub_local32.SetValue(0, cur_tail);
    aclshmemi_roce_write_ub_to_gm_with_sync(cq_context->tail_addr, ub_local32, sizeof(uint32_t), sync_id);

    aclshmemi_xscdv_diamond_cq_doorbell_t cq_db_buf;
    cq_db_buf.raw = 0;
    cq_db_buf.cq_next_cid = cur_tail;
    cq_db_buf.cq_id = cq_context->cqn;
    cq_db_buf.cq_sta = 0;
    ub_local64.SetValue(0, cq_db_buf.raw);

    aclshmemi_roce_write_ub_to_gm_with_sync(
        cq_context->db_addr, ub_local64, sizeof(aclshmemi_xscdv_diamond_cq_doorbell_t), sync_id);
}

// This function expects cur_tail to reach target_idx within the internally set timeout period. If this requirement is
// not met when exiting, the function is considered to have an error.
template <>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::XSCALE>(
    uint32_t pe, uint32_t qp_idx, uint32_t target_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));
    auto cq_base_addr = cq_context->buf_addr;
    auto cqe_size = cq_context->cqe_size;
    auto depth = cq_context->depth;
    auto cur_hardware_tail_addr = cq_context->tail_addr;
    dcci_cachelines((__gm__ uint8_t*)cur_hardware_tail_addr, 8);
    uint32_t cur_tail = *(__gm__ uint32_t*)(cur_hardware_tail_addr);
    uint32_t original_cur_tail = cur_tail;
    uint64_t run_cycles = 0;
    uint32_t status = 0;
    uint32_t wqn = 0;

    while (cur_tail != target_idx) {
        run_cycles = 0;
        __gm__ aclshmemi_xscdv_cqe64_t* cqe_addr =
            (__gm__ aclshmemi_xscdv_cqe64_t*)(cq_base_addr + cqe_size * (cur_tail % depth));
        while (!aclshmemi_roce_xscale_check_cqe_owner(cqe_addr, cur_tail, depth) &&
               run_cycles < ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_CYCLES) {
            run_cycles++;
            dcci_cachelines((__gm__ uint8_t*)cqe_addr, sizeof(aclshmemi_xscdv_cqe64_t));
        }
        if (run_cycles >= ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_CYCLES) {
            // timeout and not received CQE with owner bit set
            status = ACLSHMEMI_XSC_POLL_CQ_TIMEOUT_ERROR;
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_printf,
                "Poll CQE timeout: pe=%u, qp_idx=%u, cur_tail=%u, target_idx=%u, original_tail=%u, backend=%u\n", pe,
                qp_idx, cur_tail, target_idx, original_cur_tail, (uint32_t)aclshmemi_rdma_backend_t::XSCALE);
            break;
        }
        cur_tail++;
        wqn = cqe_addr->cqe.qp_id & 0x7FFF; // reserved for multi WQ share the same CQ
        // Check CQE status
        status = cqe_addr->cqe.error_code;
        if (status) {
            // when we receive CQE with error, return
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_printf,
                "Receive CQE with error: %d in pe %u, cur_tail: %u, wqn: %u, qp_idx: %u, backend %u\n", status, pe,
                cur_tail, wqn, qp_idx, (uint32_t)aclshmemi_rdma_backend_t::XSCALE);
            break;
        }
    }

    aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::XSCALE>(
        pe, qp_idx, cur_tail, ub_local64, ub_local32, sync_id);

    return status;
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_roce_xscale_fill_wqe_ctrl_seg(
    __gm__ uint8_t* wqe_addr, uint32_t ds_data_num, uint32_t cur_head, aclshmemi_xscdv_msg_type_t opcode,
    uint32_t msg_len, uint32_t inline_mode)
{
    __gm__ aclshmemi_xscdv_wqe_ctrl_seg_t* ctrl_seg = (__gm__ aclshmemi_xscdv_wqe_ctrl_seg_t*)wqe_addr;
    uint32_t wqe_id = cur_head << (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT);
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        ctrl_seg->wqe_id = wqe_id & 0xFFFF;
    } else {
        ctrl_seg->wqe_id = wqe_id & 0xFFFFF;
    }
    ctrl_seg->with_imm = 0;
    ctrl_seg->ds_data_num = ds_data_num;
    ctrl_seg->ce = 1;
    ctrl_seg->msg_opcode = (uint8_t)opcode;
    ctrl_seg->msg_len = msg_len;
    ctrl_seg->in_line = inline_mode;

    return wqe_addr + sizeof(aclshmemi_xscdv_wqe_ctrl_seg_t);
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_roce_xscale_fill_wqe_data_seg(
    __gm__ uint8_t* wqe_addr, uint32_t rkey, __gm__ uint8_t* remote_addr, uint32_t lkey, __gm__ uint8_t* local_addr,
    uint32_t data_len)
{
    __gm__ aclshmemi_xscdv_diamond_data_seg_t* rdata_seg = (__gm__ aclshmemi_xscdv_diamond_data_seg_t*)wqe_addr;
    rdata_seg->length = data_len;
    rdata_seg->key = rkey;
    rdata_seg->addr = (uint64_t)remote_addr;

    __gm__ aclshmemi_xscdv_diamond_data_seg_t* ldata_seg = (__gm__ aclshmemi_xscdv_diamond_data_seg_t*)(rdata_seg + 1);
    ldata_seg->length = data_len;
    ldata_seg->key = lkey;
    ldata_seg->addr = (uint64_t)local_addr;
    return wqe_addr + sizeof(aclshmemi_xscdv_diamond_data_seg_t) + sizeof(aclshmemi_xscdv_diamond_data_seg_t);
}

ACLSHMEM_DEVICE uint32_t aclshmemi_roce_xscale_fill_wqe_write_read(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head,
    aclshmemi_xscdv_msg_type_t opcode)
{
    constexpr uint32_t ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM = 2;

    __gm__ uint8_t* cur_wqe_addr = wqe_addr;

    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_ctrl_seg(
        cur_wqe_addr, ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM, cur_head, opcode, wr.message_len, 0);
    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_data_seg(
        cur_wqe_addr, wr.rkey, wr.remote_addr, wr.lkey, wr.local_addr, wr.message_len);

    return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
}

template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::XSCALE, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    return aclshmemi_roce_xscale_fill_wqe_write_read(
        wr, sq_context, wqe_addr, cur_head, aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_WRITE);
}

template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::XSCALE, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    return aclshmemi_roce_xscale_fill_wqe_write_read(
        wr, sq_context, wqe_addr, cur_head, aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_READ);
}

/**
 * @brief Ring the Send Queue (SQ) doorbell to notify hardware of new work queue entries (WQEs)
 *
 * This function updates the SQ head pointer in global memory and programs the doorbell register
 * to signal the hardware that new WQEs are available for processing. It handles both the head pointer
 * update and doorbell programming with proper synchronization.
 *
 * @param sq_context Pointer to the SQ context structure containing queue configuration and addresses
 * @param cur_head The index of the next WQE to be sent, equal to the total number of WQEs sent so far
 * @param ub_local64 Local UB tensor for 64-bit data operations (used for doorbell data)
 * @param ub_local32 Local UB tensor for 32-bit data operations (used for head pointer update)
 * @param sync_id Synchronization event ID for hardware event synchronization
 */
template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::XSCALE>(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    // Update cur_head in global memory
    auto cur_hardware_head_addr = sq_context->head_addr;
    ub_local32.SetValue(0, cur_head);
    aclshmemi_roce_write_ub_to_gm_with_sync(cur_hardware_head_addr, ub_local32, sizeof(uint32_t), sync_id);

    aclshmemi_xscdv_diamond_send_doorbell_t door_bell_info;
    door_bell_info.raw = 0;
    door_bell_info.qp_id = sq_context->wqn;
    door_bell_info.next_pid = cur_head << (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT);
    ub_local64.SetValue(0, door_bell_info.raw);

    auto door_bell_addr = sq_context->db_addr;
    aclshmemi_roce_write_ub_to_gm_with_sync(
        door_bell_addr, ub_local64, sizeof(aclshmemi_xscdv_diamond_send_doorbell_t), sync_id);
}
/**
 * @brief RDMA WRITE/READ post-send for the XSCALE backend.
 *
 * Assembles a single 128B WQE in UB and copies it to the HBM SQ ring slot
 * via DataCopyPad (MTE3 pipeline).  The WQE staging area reuses the UB
 * memory that backs ub_local32: although the tensor is declared with
 * dataLen = 32, the caller guarantees at least 128 contiguous bytes
 * starting from ub_local32.GetPhyAddr().
 *
 * @tparam OP_CODE RDMA operation code (OP_RDMA_READ or OP_RDMA_WRITE)
 * @param wr [in/out] RDMA send work request. The caller must fill:
 *           - remote_addr  Remote GM address
 *           - local_addr   Local GM address
 *           - message_len  Transfer size in bytes
 *           The function fills rkey / lkey from the memory-info table.
 * @param pe        Target PE number
 * @param qp_idx    QP index
 * @param ub_local64 64-bit UB workspace for doorbell register I/O
 * @param ub_local32 32-bit UB workspace for queue-pointer update *and*
 *                   WQE staging scratch.  The actual UB backing must be
 *                   at least 128 B, though the tensor itself is 32 B.
 * @param sync_id   Hardware event ID for MTE3 pipeline synchronization
 */
template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_xscale_post_send_read_write(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    constexpr int XSCALE_POLL_CQ_THRESHOLD = 16;
    constexpr uint32_t ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM = 2;
    constexpr uint32_t XSCDV_WRITE_READ_WQE_SIZE = 128;

    // Step 1: Fetch RDMA global info and locate SQ context for target PE/QP
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    auto mem_info_table = rdma_info->mem_ptr;
    auto sq_base_addr = sq_context->buf_addr;
    auto wqe_size = sq_context->wqe_size;
    auto depth = sq_context->depth;

    // Step 2: Read SQ producer index (head) with cache invalidation
    auto sq_pi_addr = sq_context->head_addr;
    dcci_cachelines((__gm__ uint8_t*)sq_pi_addr, 8);
    uint32_t cur_head = *(__gm__ uint32_t*)(sq_pi_addr);

    // Step 3: Read SQ consumer index (tail) and poll CQ if SQ is nearly full
    auto sq_ci_addr = sq_context->tail_addr;
    dcci_cachelines((__gm__ uint8_t*)sq_ci_addr, 8);
    uint32_t cur_tail = (*(__gm__ uint32_t*)(sq_ci_addr));
    if ((cur_head + XSCALE_POLL_CQ_THRESHOLD) % depth == (depth + cur_tail) % depth) {
        uint32_t ret = aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::XSCALE>(
            pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (ret) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_printf,
                "Poll CQE error in aclshmemi_xscale_post_send_read_write: pe=%u, qp_idx=%u, cur_tail=%u, "
                "cur_head=%u\n",
                pe, qp_idx, cur_tail, cur_head);
            return;
        }
    }

    // Step 4: Retrieve remote/local memory info and fill rkey/lkey into wr
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);
    __gm__ aclshmemi_rdma_mem_info* local_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * aclshmemi_get_my_pe());
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = local_mem_info->lkey;

    // Step 5: Assemble WQE in UB scratch
    //   Layout: [ctrl_seg (16B)] [rdata_seg (16B)] [ldata_seg (16B)] [padding -> 128B]
    //
    //   Fill ctrl_seg (wqe_id, opcode, msg_len, ds_data_num, ce=1, inline=0)
    //   Using the same ubuf of doorbell
    __ubuf__ uint8_t* ub_scratch = (__ubuf__ uint8_t*)ub_local32.GetPhyAddr();
    __ubuf__ aclshmemi_xscdv_wqe_ctrl_seg_t* ctrl_seg = (__ubuf__ aclshmemi_xscdv_wqe_ctrl_seg_t*)ub_scratch;
    uint32_t wqe_id = cur_head << (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT);
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        ctrl_seg->wqe_id = wqe_id & 0xFFFF;
    } else {
        ctrl_seg->wqe_id = wqe_id & 0xFFFFF;
    }
    ctrl_seg->with_imm = 0;
    ctrl_seg->ds_data_num = ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM;
    ctrl_seg->ce = 1;
    if constexpr (OP_CODE == aclshmemi_rdma_opcode_t::OP_RDMA_WRITE) {
        ctrl_seg->msg_opcode = static_cast<uint8_t>(aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_WRITE);
    } else {
        ctrl_seg->msg_opcode = static_cast<uint8_t>(aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_READ);
    }
    const uint32_t wr_msg_len = static_cast<uint32_t>(wr.message_len);
    ctrl_seg->msg_len = wr_msg_len;
    ctrl_seg->in_line = 0;

    //   Fill remote data segment (rkey + remote_addr + message_len)
    __ubuf__ aclshmemi_xscdv_diamond_data_seg_t* rdata_seg =
        (__ubuf__ aclshmemi_xscdv_diamond_data_seg_t*)(ub_scratch + sizeof(aclshmemi_xscdv_wqe_ctrl_seg_t));
    rdata_seg->length = wr_msg_len;
    rdata_seg->key = wr.rkey;
    rdata_seg->addr = reinterpret_cast<uint64_t>(wr.remote_addr);

    //   Fill local data segment (lkey + local_addr + message_len)
    __ubuf__ aclshmemi_xscdv_diamond_data_seg_t* ldata_seg =
        (__ubuf__ aclshmemi_xscdv_diamond_data_seg_t*)(rdata_seg + 1);
    ldata_seg->length = wr_msg_len;
    ldata_seg->key = wr.lkey;
    ldata_seg->addr = reinterpret_cast<uint64_t>(wr.local_addr);

    // Step 6: Calculate HBM SQ ring slot address for this WQE
    __gm__ uint8_t* wqe_addr = (__gm__ uint8_t*)(sq_base_addr + (uint64_t)wqe_size * (cur_head % depth));

    // Step 7: Construct LocalTensor mapped to ub_scratch and copy WQE UB -> HBM
    //         Uses DataCopyPad with S_MTE3 / MTE3_S / PipeBarrier synchronization
    AscendC::LocalTensor<uint8_t> ub_local_wqe;
    ub_local_wqe.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_local_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_scratch);
    aclshmemi_roce_write_ub_to_gm_with_sync(
        reinterpret_cast<uint64_t>(wqe_addr), ub_local_wqe, XSCDV_WRITE_READ_WQE_SIZE, sync_id);

    // Step 8: Advance producer index and ring SQ doorbell to notify hardware
    cur_head++;
    aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::XSCALE>(
        sq_context, cur_head, ub_local64, ub_local32, sync_id);
}

/**
 * @brief UB version of fill_wqe_ctrl_seg. Fills the control segment of a WQE in UB memory.
 *
 * @param wqe_addr           [in] Start address of the WQE in UB buffer.
 * @param ds_data_num        [in] Number of data segments.
 * @param wqe_head           [in] WQE head index (cur_head + staged offset).
 * @param opcode             [in] RDMA message opcode (WRITE / READ).
 * @param msg_len            [in] Message length in bytes.
 * @return Advanced UB pointer past the control segment.
 */
ACLSHMEM_DEVICE __ubuf__ uint8_t* aclshmemi_roce_xscale_fill_wqe_ctrl_seg_ub(
    __ubuf__ uint8_t* wqe_addr, uint32_t ds_data_num, uint32_t wqe_head, aclshmemi_xscdv_msg_type_t opcode,
    uint32_t msg_len)
{
    __ubuf__ aclshmemi_xscdv_wqe_ctrl_seg_t* ctrl_seg = (__ubuf__ aclshmemi_xscdv_wqe_ctrl_seg_t*)wqe_addr;
    uint32_t wqe_id = wqe_head << (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT);
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        ctrl_seg->wqe_id = wqe_id & 0xFFFF;
    } else {
        ctrl_seg->wqe_id = wqe_id & 0xFFFFF;
    }
    ctrl_seg->with_imm = 0;
    ctrl_seg->ds_data_num = ds_data_num;
    ctrl_seg->ce = 1;
    ctrl_seg->msg_opcode = (uint8_t)opcode;
    ctrl_seg->msg_len = msg_len;
    ctrl_seg->in_line = 0;

    return wqe_addr + sizeof(aclshmemi_xscdv_wqe_ctrl_seg_t);
}

/**
 * @brief UB version of fill_wqe_data_seg. Fills remote and local data segments of a WQE in UB memory.
 *
 * @param wqe_addr           [in] Start address of the WQE in UB buffer (past ctrl_seg).
 * @param rkey               [in] Remote memory key.
 * @param remote_addr        [in] Remote GM address.
 * @param lkey               [in] Local memory key.
 * @param local_addr         [in] Local GM address.
 * @param data_len           [in] Data length in bytes.
 * @return Advanced UB pointer past both data segments.
 */
ACLSHMEM_DEVICE __ubuf__ uint8_t* aclshmemi_roce_xscale_fill_wqe_data_seg_ub(
    __ubuf__ uint8_t* wqe_addr, uint32_t rkey, __gm__ uint8_t* remote_addr, uint32_t lkey, __gm__ uint8_t* local_addr,
    uint32_t data_len)
{
    __ubuf__ aclshmemi_xscdv_diamond_data_seg_t* rdata_seg = (__ubuf__ aclshmemi_xscdv_diamond_data_seg_t*)wqe_addr;
    rdata_seg->length = data_len;
    rdata_seg->key = rkey;
    rdata_seg->addr = (uint64_t)remote_addr;

    __ubuf__ aclshmemi_xscdv_diamond_data_seg_t* ldata_seg =
        (__ubuf__ aclshmemi_xscdv_diamond_data_seg_t*)(rdata_seg + 1);
    ldata_seg->length = data_len;
    ldata_seg->key = lkey;
    ldata_seg->addr = (uint64_t)local_addr;

    return wqe_addr + sizeof(aclshmemi_xscdv_diamond_data_seg_t) + sizeof(aclshmemi_xscdv_diamond_data_seg_t);
}

/**
 * @brief Fill a complete RDMA WQE into UB buffer via MTE3 path.
 *        Wraps the two UB fill helpers (ctrl_seg + data_seg) into a single call.
 *
 * @param wqe_addr           [in] Start address of the WQE in UB buffer.
 * @param wqe_head           [in] WQE head index for wqe_id calculation.
 * @param message_len        [in] Message length in bytes.
 * @param opcode             [in] RDMA message opcode (WRITE / READ).
 * @param rkey               [in] Remote memory key.
 * @param remote_addr        [in] Remote GM address.
 * @param lkey               [in] Local memory key.
 * @param local_addr         [in] Local GM address.
 * @return WQE size in bytes (128 for XSCALE).
 */
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_xscale_fill_wqe_ub_write_read(
    __ubuf__ uint8_t* wqe_addr, uint32_t wqe_head, uint32_t message_len, aclshmemi_xscdv_msg_type_t opcode,
    uint32_t rkey, __gm__ uint8_t* remote_addr, uint32_t lkey, __gm__ uint8_t* local_addr)
{
    constexpr uint32_t ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM = 2;

    __ubuf__ uint8_t* cur_wqe_addr = wqe_addr;

    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_ctrl_seg_ub(
        cur_wqe_addr, ACLSHMEMI_XSCDV_WRITE_READ_DS_DATA_NUM, wqe_head, opcode, message_len);
    cur_wqe_addr =
        aclshmemi_roce_xscale_fill_wqe_data_seg_ub(cur_wqe_addr, rkey, remote_addr, lkey, local_addr, message_len);

    return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
}

template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_ub(
    __ubuf__ uint8_t* wqe_addr, uint32_t wqe_head, uint32_t message_len, uint32_t rkey, __gm__ uint8_t* remote_addr,
    uint32_t lkey, __gm__ uint8_t* local_addr);

struct aclshmemi_roce_xscale_aggregate_cache_t {
    uint64_t sq_base_addr;
    uint64_t sq_head_addr;
    uint64_t sq_tail_addr;
    uint64_t sq_db_addr;
    uint32_t sq_depth;
    uint32_t sq_wqn;
    uint32_t base_head;
    uint32_t remote_rkey;
    uint32_t local_lkey;
    uint32_t reserved[3];
};

static_assert(sizeof(aclshmemi_roce_xscale_aggregate_cache_t) <= UB_ALIGN_SIZE * 2);

ACLSHMEM_DEVICE __ubuf__ aclshmemi_roce_xscale_aggregate_cache_t* aclshmemi_roce_xscale_aggregate_cache(
    __ubuf__ uint8_t* buf)
{
    return (__ubuf__ aclshmemi_roce_xscale_aggregate_cache_t*)(buf - UB_ALIGN_SIZE * 2);
}

ACLSHMEM_DEVICE void aclshmemi_roce_xscale_fill_aggregate_cache(
    __ubuf__ aclshmemi_roce_xscale_aggregate_cache_t* cache, __gm__ aclshmemi_rdma_info* rdma_info, uint32_t pe,
    uint32_t qp_idx)
{
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));

    cache->sq_base_addr = sq_context->buf_addr;
    cache->sq_depth = sq_context->depth;
    cache->sq_head_addr = sq_context->head_addr;
    cache->sq_tail_addr = sq_context->tail_addr;
    cache->sq_db_addr = sq_context->db_addr;
    cache->sq_wqn = sq_context->wqn;

    dcci_cachelines((__gm__ uint8_t*)cache->sq_head_addr, 8);
    cache->base_head = *(__gm__ uint32_t*)(cache->sq_head_addr);

    auto mem_info_table = rdma_info->mem_ptr;
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);
    __gm__ aclshmemi_rdma_mem_info* local_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * aclshmemi_get_my_pe());
    cache->remote_rkey = remote_mem_info->rkey;
    cache->local_lkey = local_mem_info->lkey;
}

ACLSHMEM_DEVICE void aclshmemi_roce_xscale_ring_sq_doorbell_from_cache(
    __ubuf__ aclshmemi_roce_xscale_aggregate_cache_t* cache, uint32_t cur_head,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    uint64_t sq_head_addr = cache->sq_head_addr;
    uint64_t sq_db_addr = cache->sq_db_addr;
    uint32_t sq_wqn = cache->sq_wqn;

    ub_local32.SetValue(0, cur_head);
    aclshmemi_roce_write_ub_to_gm_with_sync(sq_head_addr, ub_local32, sizeof(uint32_t), sync_id);

    aclshmemi_xscdv_diamond_send_doorbell_t door_bell_info;
    door_bell_info.raw = 0;
    door_bell_info.qp_id = sq_wqn;
    door_bell_info.next_pid = cur_head << (ACLSHMEMI_XSCALE_SND_WQE_SHIFT - ACLSHMEMI_XSCALE_BASE_WQE_SHIFT);
    ub_local64.SetValue(0, door_bell_info.raw);

    aclshmemi_roce_write_ub_to_gm_with_sync(
        sq_db_addr, ub_local64, sizeof(aclshmemi_xscdv_diamond_send_doorbell_t), sync_id);
}

template <>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_ub<aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    __ubuf__ uint8_t* wqe_addr, uint32_t wqe_head, uint32_t message_len, uint32_t rkey, __gm__ uint8_t* remote_addr,
    uint32_t lkey, __gm__ uint8_t* local_addr)
{
    return aclshmemi_roce_xscale_fill_wqe_ub_write_read(
        wqe_addr, wqe_head, message_len, aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_WRITE, rkey,
        remote_addr, lkey, local_addr);
}

template <>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_ub<aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    __ubuf__ uint8_t* wqe_addr, uint32_t wqe_head, uint32_t message_len, uint32_t rkey, __gm__ uint8_t* remote_addr,
    uint32_t lkey, __gm__ uint8_t* local_addr)
{
    return aclshmemi_roce_xscale_fill_wqe_ub_write_read(
        wqe_addr, wqe_head, message_len, aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_READ, rkey,
        remote_addr, lkey, local_addr);
}

template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_stage_rma_wqe_ub_xscale(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* buf, uint32_t sync_id, aclshmemx_defer_t action)
{
    __ubuf__ aclshmemi_roce_xscale_aggregate_cache_t* cache = aclshmemi_roce_xscale_aggregate_cache(buf);
    uint32_t cnt = action.state.pending_count;
    if (cnt == 0) {
        __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
        aclshmemi_roce_xscale_fill_aggregate_cache(cache, rdma_info, pe, qp_idx);
    }

    // Build WQE directly in the UB buffer. Each staged WQE occupies 128B.
    __ubuf__ uint8_t* wqe_addr = buf + (uint64_t)cnt * ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    uint32_t wqe_head = cache->base_head + cnt;

    aclshmemi_roce_fill_wqe_ub<OP_CODE>(
        wqe_addr, wqe_head, (uint32_t)message_len, cache->remote_rkey, remote_addr, cache->local_lkey, local_addr);

    // Track staged WQE count for commit
    action.state.pending_count++;
}

template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_commit_rma_wqes_xscale(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, __ubuf__ uint8_t* buf,
    uint32_t sync_id, aclshmemx_submit_t action)
{
    constexpr int XSCALE_POLL_CQ_THRESHOLD = 16;
    __ubuf__ aclshmemi_roce_xscale_aggregate_cache_t* cache = aclshmemi_roce_xscale_aggregate_cache(buf);

    // Step 1: Read SQ head. Staged batches capture the base head on the first stage call.
    uint32_t pending_count = action.state.pending_count;
    if (pending_count == 0) {
        __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
        aclshmemi_roce_xscale_fill_aggregate_cache(cache, rdma_info, pe, qp_idx);
    }
    uint32_t cur_head = cache->base_head;
    uint32_t depth = cache->sq_depth;
    uint32_t total_wqe_cnt = pending_count + 1; // +1 for the commit call's own WQE

    if (total_wqe_cnt >= depth) {
        aclshmemi_kernel_abort(
            "XSCALE aggregate commit batch too large: pe=%u, qp_idx=%u, pending_count=%u, total_wqe_cnt=%u, "
            "depth=%u\n",
            pe, qp_idx, pending_count, total_wqe_cnt, depth);
        return;
    }

    // Step 2: Check SQ capacity with total_wqe_cnt, poll CQ once if full
    dcci_cachelines((__gm__ uint8_t*)cache->sq_tail_addr, 8);
    uint32_t cur_tail = *(__gm__ uint32_t*)(cache->sq_tail_addr);
    uint32_t outstanding_after_post = cur_head + total_wqe_cnt - cur_tail;
    if (outstanding_after_post + XSCALE_POLL_CQ_THRESHOLD >= depth) {
        uint32_t ret = aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::XSCALE>(
            pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (ret) {
            aclshmemi_kernel_abort(
                "XSCALE aggregate commit poll CQ failed: pe=%u, qp_idx=%u, pending_count=%u, total_wqe_cnt=%u, "
                "cur_head=%u, cur_tail=%u, depth=%u, ret=%u\n",
                pe, qp_idx, pending_count, total_wqe_cnt, cur_head, cur_tail, depth, ret);
            return;
        }
        __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
        aclshmemi_roce_xscale_fill_aggregate_cache(cache, rdma_info, pe, qp_idx);
        cache->base_head = cur_head;
    }

    // Step 3: Assemble the commit call's own WQE at position pending_count (0-indexed).
    //         Earlier WQEs already assembled in stage.
    __ubuf__ uint8_t* commit_wqe_addr = buf + (uint64_t)pending_count * ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    aclshmemi_roce_fill_wqe_ub<OP_CODE>(
        commit_wqe_addr, cur_head + pending_count, (uint32_t)message_len, cache->remote_rkey, remote_addr,
        cache->local_lkey, local_addr);

    // Step 4: Bulk copy all WQEs from UB buffer to GM SQ buffer, handling ring wrap
    uint32_t ring_offset = cur_head % depth;
    uint32_t first_seg_cnt = depth - ring_offset;
    uint32_t copy_total_size = total_wqe_cnt * ACLSHMEMI_XSCALE_SND_WQE_SIZE;

    AscendC::LocalTensor<uint8_t> ub_local_wqe;
    ub_local_wqe.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);

    if (total_wqe_cnt <= first_seg_cnt) {
        // No wrap: single contiguous copy
        __gm__ uint8_t* sq_dst =
            (__gm__ uint8_t*)(cache->sq_base_addr + (uint64_t)ACLSHMEMI_XSCALE_SND_WQE_SIZE * ring_offset);
        ub_local_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
        aclshmemi_roce_write_ub_to_gm_with_sync(
            reinterpret_cast<uint64_t>(sq_dst), ub_local_wqe, copy_total_size, sync_id);
    } else {
        // Ring wrap: first segment to ring end, second segment from ring start
        uint32_t first_seg_size = first_seg_cnt * ACLSHMEMI_XSCALE_SND_WQE_SIZE;
        uint32_t second_seg_size = copy_total_size - first_seg_size;

        __gm__ uint8_t* sq_dst =
            (__gm__ uint8_t*)(cache->sq_base_addr + (uint64_t)ACLSHMEMI_XSCALE_SND_WQE_SIZE * ring_offset);
        ub_local_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
        aclshmemi_roce_write_ub_to_gm_with_sync(
            reinterpret_cast<uint64_t>(sq_dst), ub_local_wqe, first_seg_size, sync_id);

        ub_local_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + first_seg_size;
        aclshmemi_roce_write_ub_to_gm_with_sync(cache->sq_base_addr, ub_local_wqe, second_seg_size, sync_id);
    }

    // Step 5: Ring SQ doorbell once to post all WQEs
    uint32_t next_head = cur_head + total_wqe_cnt;
    aclshmemi_roce_xscale_ring_sq_doorbell_from_cache(cache, next_head, ub_local64, ub_local32, sync_id);

    action.state.pending_count = 0;
}

// Currently only support 8B and 4B
template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE uint32_t aclshmemi_rdma_xscdv_fill_wqe_fa(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    constexpr uint32_t XSCDV_FA_MASKED_DS_DATA_NUM = 3;
    constexpr uint8_t TYPE_BYTES = sizeof(T);
    constexpr uint32_t MSG_LEN = []() {
        if constexpr (TYPE_BYTES == BYTES_32 && IS_MASKED) {
            constexpr uint32_t FA_MASKED_4B_DATA_LEN = 8;
            return FA_MASKED_4B_DATA_LEN;
        } else {
            constexpr uint32_t FA_DEFAULT_DATA_LEN = 16;
            return FA_DEFAULT_DATA_LEN;
        }
    }();
    constexpr uint32_t DATA_LEN = []() {
        if constexpr (TYPE_BYTES <= BYTES_32) {
            return BYTES_32;
        } else if constexpr (TYPE_BYTES <= BYTES_64) {
            return BYTES_64;
        } else {
            return 0;
        }
    }();
    constexpr aclshmemi_xscdv_msg_type_t OPCODE = []() {
        if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_8B_MSK_FETCH_AND_ADD;
        } else if constexpr (TYPE_BYTES == BYTES_64 && !IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_FETCH_AND_ADD;
        } else if constexpr (TYPE_BYTES <= BYTES_32 && IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_4B_MSK_FETCH_AND_ADD;
        } else {
            return static_cast<aclshmemi_xscdv_msg_type_t>(0);
        }
    }();

    static_assert(
        TYPE_BYTES == BYTES_64 || TYPE_BYTES == BYTES_32,
        "XSCALE backend only support fetch and add operation with type size of 8B or 4B\n");

    static_assert(
        !(TYPE_BYTES <= BYTES_32 && !IS_MASKED),
        "XSCALE backend doesn't support fetch and add operation with type size of 4B without mask\n");

    __gm__ uint8_t* cur_wqe_addr = wqe_addr;
    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_ctrl_seg(
        cur_wqe_addr, XSCDV_FA_MASKED_DS_DATA_NUM, cur_head, OPCODE, MSG_LEN, 1);
    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_data_seg(
        cur_wqe_addr, wr.rkey, wr.remote_addr, wr.lkey, wr.local_addr, DATA_LEN);
    // fill atomic segmentations
    if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_64_masked_fa_seg_t* fa_seg =
            (__gm__ aclshmemi_xsc_wqe_atomic_64_masked_fa_seg_t*)cur_wqe_addr;
        fa_seg->add_data = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_data);
        fa_seg->field_boundary = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_mask);

        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else if constexpr (TYPE_BYTES == BYTES_64 && !IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_seg_t* atomic_seg = (__gm__ aclshmemi_xsc_wqe_atomic_seg_t*)cur_wqe_addr;
        atomic_seg->swap_add = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_data);
        atomic_seg->compare = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_mask);

        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else if constexpr (TYPE_BYTES <= BYTES_32 && IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_32_masked_fa_seg_t* fa_seg =
            (__gm__ aclshmemi_xsc_wqe_atomic_32_masked_fa_seg_t*)cur_wqe_addr;
        fa_seg->add_data = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.swap_add_data & UINT32_MAX);
        fa_seg->field_boundary = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.swap_add_mask & UINT32_MAX);

        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else {
        return 0;
    }
}
// Currently only support 8B and 4B
template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE uint32_t aclshmemi_rdma_xscdv_fill_wqe_cas(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    constexpr uint8_t TYPE_BYTES = sizeof(T);
    constexpr uint32_t XSCDV_CAS_DS_DATA_NUM = []() {
        if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
            constexpr uint32_t XSCALE_CAS_DS_MASK_64 = 4;
            return XSCALE_CAS_DS_MASK_64;
        } else {
            constexpr uint32_t XSCALE_CAS_DS_MASK_OTHER = 3;
            return XSCALE_CAS_DS_MASK_OTHER;
        }
    }();
    constexpr uint32_t MSG_LEN = []() {
        if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
            constexpr uint32_t CAS_MASKED_8B_DATA_LEN = 32;
            return CAS_MASKED_8B_DATA_LEN;
        } else {
            constexpr uint32_t CAS_DEFAULT_DATA_LEN = 16;
            return CAS_DEFAULT_DATA_LEN;
        }
    }();
    constexpr uint32_t DATA_LEN = []() {
        if constexpr (TYPE_BYTES <= BYTES_32) {
            return BYTES_32;
        } else if constexpr (TYPE_BYTES <= BYTES_64) {
            return BYTES_64;
        } else {
            return 0;
        }
    }();
    constexpr aclshmemi_xscdv_msg_type_t OPCODE = []() {
        if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_8B_MSK_CMP_AND_SWAP;
        } else if constexpr (TYPE_BYTES == BYTES_64 && !IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_CMP_AND_SWAP;
        } else if constexpr (TYPE_BYTES <= BYTES_32 && IS_MASKED) {
            return aclshmemi_xscdv_msg_type_t::ACLSHMEMI_XSCALE_MSG_OPCODE_RDMA_ATOMIC_4B_MSK_CMP_AND_SWAP;
        } else {
            return static_cast<aclshmemi_xscdv_msg_type_t>(0);
        }
    }();

    static_assert(
        TYPE_BYTES == BYTES_64 || TYPE_BYTES == BYTES_32,
        "XSCALE backend only support compare and swap operation with type size of 8B or 4B\n");

    static_assert(
        !(TYPE_BYTES <= BYTES_32 && !IS_MASKED),
        "XSCALE backend doesn't support compare and swap operation with type size of 4B without mask\n");

    __gm__ uint8_t* cur_wqe_addr = wqe_addr;
    cur_wqe_addr =
        aclshmemi_roce_xscale_fill_wqe_ctrl_seg(cur_wqe_addr, XSCDV_CAS_DS_DATA_NUM, cur_head, OPCODE, MSG_LEN, 1);
    cur_wqe_addr = aclshmemi_roce_xscale_fill_wqe_data_seg(
        cur_wqe_addr, wr.rkey, wr.remote_addr, wr.lkey, wr.local_addr, DATA_LEN);
    // fill atomic segmentations
    if constexpr (TYPE_BYTES == BYTES_64 && IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_64_masked_cas_seg_t* atomic_seg0 =
            (__gm__ aclshmemi_xsc_wqe_atomic_64_masked_cas_seg_t*)cur_wqe_addr;
        atomic_seg0->swap_add = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_data);
        atomic_seg0->compare = aclshmemi_htobe64(wr.atomic.masked_common.compare_data);

        __gm__ aclshmemi_xsc_wqe_atomic_64_masked_cas_seg_t* atomic_seg1 =
            (__gm__ aclshmemi_xsc_wqe_atomic_64_masked_cas_seg_t*)(atomic_seg0 + 1);
        atomic_seg1->swap_add = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_mask);
        atomic_seg1->compare = aclshmemi_htobe64(wr.atomic.masked_common.compare_mask);
        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else if constexpr (TYPE_BYTES == BYTES_64 && !IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_seg_t* atomic_seg = (__gm__ aclshmemi_xsc_wqe_atomic_seg_t*)cur_wqe_addr;
        atomic_seg->swap_add = aclshmemi_htobe64(wr.atomic.masked_common.swap_add_data);
        atomic_seg->compare = aclshmemi_htobe64(wr.atomic.masked_common.compare_data);
        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else if constexpr (TYPE_BYTES <= BYTES_32 && IS_MASKED) {
        __gm__ aclshmemi_xsc_wqe_atomic_32_masked_cas_seg_t* cas_seg =
            (__gm__ aclshmemi_xsc_wqe_atomic_32_masked_cas_seg_t*)cur_wqe_addr;
        cas_seg->swap_data = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.swap_add_data & UINT32_MAX);
        cas_seg->compare_data = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.compare_data & UINT32_MAX);
        cas_seg->swap_mask = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.swap_add_mask & UINT32_MAX);
        cas_seg->compare_mask = aclshmemi_htobe32((uint32_t)wr.atomic.masked_common.compare_mask & UINT32_MAX);

        return ACLSHMEMI_XSCALE_SND_WQE_SIZE;
    } else {
        return 0;
    }
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_xscale_fill_wqe_atomic(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    if constexpr (ATOMIC_OP_CODE == aclshmemi_rdma_atomic_op_t::OP_ATOMIC_FA) {
        return aclshmemi_rdma_xscdv_fill_wqe_fa<T, IS_MASKED>(wr, sq_context, wqe_addr, cur_head);
    } else if constexpr (ATOMIC_OP_CODE == aclshmemi_rdma_atomic_op_t::OP_ATOMIC_CAS) {
        return aclshmemi_rdma_xscdv_fill_wqe_cas<T, IS_MASKED>(wr, sq_context, wqe_addr, cur_head);
    } else {
        return 0;
    }
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_rdma_xscdv_post_send_atomic(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    constexpr int XSCALE_POLL_CQ_THRESHOLD = 16;
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();

    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    auto mem_info_table = rdma_info->mem_ptr;
    auto sq_base_addr = sq_context->buf_addr;
    auto wqe_size = sq_context->wqe_size;
    auto sq_pi_addr = sq_context->head_addr;
    dcci_cachelines((__gm__ uint8_t*)sq_pi_addr, 8);
    uint32_t cur_head = *(__gm__ uint32_t*)(sq_pi_addr);
    auto sq_ci_addr = sq_context->tail_addr;
    auto depth = sq_context->depth;
    uint32_t ret = 0;

    dcci_cachelines((__gm__ uint8_t*)sq_ci_addr, 8);
    uint32_t cur_tail = (*(__gm__ uint32_t*)(sq_ci_addr));
    // If the send queue (SQ) is about to be full, need to wait for CQE to arrive before continuing to send
    // Call the poll_cq function to wait for CQE arrival until cur_tail catches up with cur_head
    // Note: The poll_cq function will block the current thread until the condition is met or timeout occurs
    // Note: cq_context->depth needs to be greater than or equal to sq_context->depth
    if ((cur_head + XSCALE_POLL_CQ_THRESHOLD) % depth == (depth + cur_tail) % depth) {
        ret = aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::XSCALE>(
            pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (ret) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_printf,
                "Poll CQE error in aclshmemi_xscale_post_send_read_write: pe=%u, qp_idx=%u, cur_tail=%u, "
                "cur_head=%u\n",
                pe, qp_idx, cur_tail, cur_head);
            return;
        }
    }

    __gm__ uint8_t* wqe_addr = (__gm__ uint8_t*)(sq_base_addr + (uint64_t)wqe_size * (cur_head % depth));

    uint32_t wqe_total_size =
        aclshmemi_roce_xscale_fill_wqe_atomic<T, IS_MASKED, ATOMIC_OP_CODE>(wr, sq_context, wqe_addr, cur_head);

    dcci_cachelines(wqe_addr, wqe_total_size);
    cur_head++;

    aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::XSCALE>(
        sq_context, cur_head, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::XSCALE, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_xscale_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::XSCALE, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_xscale_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/*
 *  =====================================================================================================
 *  XSCALE Backend Atomic Operations Traits Specialization
 *  =====================================================================================================
 *  This specialization provides the atomic operations implementation for XSCALE backend.
 *  By specializing aclshmemi_backend_traits once, all combinations of T and IS_MASKED are
 *  automatically supported through the template functions below.
 *  =====================================================================================================
 */

template <>
struct aclshmemi_backend_traits<aclshmemi_rdma_backend_t::XSCALE> {
    template <typename T, bool IS_MASKED>
    struct atomic_op_traits {
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE uint32_t fill_wqe(
            aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
            uint32_t cur_head)
        {
            return aclshmemi_roce_xscale_fill_wqe_atomic<T, IS_MASKED, ATOMIC_OP_CODE>(
                wr, sq_context, wqe_addr, cur_head);
        }

        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE void post_send(
            aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
            AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
        {
            aclshmemi_rdma_xscdv_post_send_atomic<T, IS_MASKED, ATOMIC_OP_CODE>(
                wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
        }
    };
};

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_XSCALE_HPP
