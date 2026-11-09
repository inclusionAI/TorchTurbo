/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_HNS_1825_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_HNS_1825_HPP

#include "rdma_device_backend_base.h"

// WQE control segment (16B). Multi-byte fields are converted before the WQE is posted.
struct aclshmemi_hns_1825_wqe_ctrl_seg_t {
    uint8_t owner_sl;  // dw0[31:24]: owner(1) + ctrl_section_length(2) + csl(2) + difsl(3)
    uint8_t df_tsl;    // dw0[23:16]: cr(1) + df(1) + va(1) + tsl(5); tsl = task section length / 8B
    uint16_t wf_bdsl;  // dw0[15:0]: cf(1) + wf(1) + wqe_msn(2) + fde(1) + fast(1) + drv_sl(2) + bdsl(8)
    uint32_t cl_pi;    // dw1: cl(4) + reserved(8) + mask_pi(20); mask_pi only used by direct WQE
    uint64_t db;       // dw2-dw3: doorbell segment, not used by this path
    /**** 16 bytes ****/
};

// Task segment common first dword (dw0). opcode selects the RDMA operation.
union aclshmemi_hns_1825_wqe_tsk_com_seg_t {
    struct {
        uint32_t xrc_srqn : 18; // [17:0]  XRC SRQN, RC mode keeps 0
        uint32_t ext : 1;       // [18]    CMD64 extend, unused
        uint32_t dif : 1;       // [19]    reserved for RoCE
        uint32_t rsvd : 3;      // [22:20]
        uint32_t so : 1;        // [23]    strong ordering
        uint32_t opcode : 5;    // [28:24] 0x04 = RDMA WRITE, 0x08 = RDMA READ
        uint32_t signal : 1;    // [29]    request a CQE on completion
        uint32_t fence : 1;     // [30]    fence / ordering
        uint32_t se : 1;        // [31]    solicited event
    } bs;
    uint32_t value;
};

// WQE task segment for RDMA WRITE/READ (32B).
struct aclshmemi_hns_1825_wqe_rdma_task_seg_t {
    aclshmemi_hns_1825_wqe_tsk_com_seg_t com_tsk; // dw0: opcode / signal / fence (see union above)
    uint32_t data_len;  // total message length in bytes
    uint32_t imm_data;  // immediate data (0 for plain WRITE/READ)
    union {
        struct {
            uint32_t last_ext_len : 8; // [7:0] READ extension length; WRITE keeps 0
            uint32_t cmd_len : 8;      // [15:8] command length, unused for this WQE
            uint32_t pi : 16;          // [31:16] producer index, unused for this WQE
        } bs;
        uint32_t value; // dw3 is filled in host order, then byte-swapped before the NIC consumes it
    } dw3;
    uint64_t va;        // remote virtual address
    uint32_t rkey;      // remote memory key
    uint32_t ulp;       // upper-layer field; low 16 bits carry the local lkey
    /**** 32 bytes ****/
};

// WQE data segment / SGE (16B). WRITE/READ uses a single SGE.
struct aclshmemi_hns_1825_wqe_data_seg_t {
    uint64_t buf_addr; // local buffer address (source for WRITE, destination for READ)
    uint32_t r_len;    // r(1) + length(31): transfer length in bytes
    uint32_t le_key;   // L(1) + E(1) + key(30): local lkey; L is set for the single SGE
    /**** 16 bytes ****/
};

// Completion Queue Entry header (32B), RC service type. Only a subset of fields is consumed.
struct aclshmemi_hns_1825_cqe_t {
    uint32_t owner_id_qpn;     // dw1: owner(31) + cqe_size(30:29) + dif_en(28) + wq_id(27:24) + err_code(23:20) + qpn(19:0)
    uint32_t op_sr_wqebb;      // dw2: op_type(31:27) + s_r(26) + inline(25) + merge(24) + fake(23) + wqebb_cnt(19:0)
    uint32_t byte_cnt;         // dw3: transferred byte count
    uint32_t imm_data;         // dw4: immediate data / invalidate key (receive side)
    uint32_t rsvd_dw5;         // dw5: reserved for RC
    uint32_t wqe_num;          // dw6: merged wr count (RQ merge only)
    uint32_t vlan_queue_index; // dw7: srqn_rqpn (RC = SRQN, not read by SHMEM)
    uint8_t syndrome;          // dw8[7:0]: error syndrome, valid only when op_type = error(0x1e)
    uint8_t rsvd;              // dw8[15:8]
    uint16_t wqe_counter;      // dw8[31:16]: SQ WQE sequence number
    /**** 32 bytes ****/
};

// SQ hardware doorbell payload (64-bit), written to the UAR doorbell register via st_dev.
union aclshmemi_hns_1825_sq_db_t {
    struct {
        uint64_t qpn : 20;        // [19:0]  SQ QP number
        uint64_t cntx_size : 2;   // [21:20] QPC context size (0:256B / 1:512B / 2:1024B)
        uint64_t rsvd0 : 1;       // [22]    reserved
        uint64_t c : 1;           // [23]    C field, kept 0 for normal SQ doorbells
        uint64_t cos : 3;         // [26:24] hardware priority (class of service)
        uint64_t type : 5;        // [31:27] doorbell type (21 = SQ doorbell)
        uint64_t pi : 8;          // [39:32] SQ producer index, high 8 bits
        uint64_t rsvd1 : 8;       // [47:40] reserved
        uint64_t xrc_vld : 1;     // [48]    XRC valid
        uint64_t rsvd2 : 1;       // [49]    reserved
        uint64_t mtu_shift : 3;   // [52:50] path MTU shift
        uint64_t sgid_index : 7;  // [59:53] source GID index
        uint64_t sub_type : 4;    // [63:60] doorbell sub-type
    } bs;
    uint64_t value;
};

// HNS_1825 NIC consumes big-endian WQE dwords and software doorbell records while AscendC runs little-endian.
// SHMEM-owned SQ/CQ head-tail mirrors stay host-order; only NIC-visible payloads are byte-swapped.
ACLSHMEM_DEVICE uint16_t aclshmemi_hns_1825_htobe16(uint16_t host_val)
{
    return (uint16_t)(((host_val & 0x00ffU) << 8) | ((host_val & 0xff00U) >> 8));
}

ACLSHMEM_DEVICE uint32_t aclshmemi_hns_1825_htobe32(uint32_t host_val)
{
    return ((host_val & 0x000000ffU) << 24) | ((host_val & 0x0000ff00U) << 8) |
           ((host_val & 0x00ff0000U) >> 8) | ((host_val & 0xff000000U) >> 24);
}

ACLSHMEM_DEVICE uint64_t aclshmemi_hns_1825_htobe64(uint64_t host_val)
{
    return ((host_val & 0x00000000000000ffULL) << 56) | ((host_val & 0x000000000000ff00ULL) << 40) |
           ((host_val & 0x0000000000ff0000ULL) << 24) | ((host_val & 0x00000000ff000000ULL) << 8) |
           ((host_val & 0x000000ff00000000ULL) >> 8) | ((host_val & 0x0000ff0000000000ULL) >> 24) |
           ((host_val & 0x00ff000000000000ULL) >> 40) | ((host_val & 0xff00000000000000ULL) >> 56);
}

template <AscendC::HardEvent EVENT>
ACLSHMEM_DEVICE void aclshmemi_hns_1825_sync_func(uint32_t sync_id)
{
    AscendC::SetFlag<EVENT>((AscendC::TEventID)sync_id);
    AscendC::WaitFlag<EVENT>((AscendC::TEventID)sync_id);
}

/**
 * @brief Write UB data to GM with S/MTE3 synchronization.
 *
 * This function copies data from a UB LocalTensor to GM using the following order:
 * 1. S_MTE3 synchronization before DataCopyPad
 * 2. DataCopyPad from UB to GM
 * 3. PIPE_MTE3 barrier
 * 4. MTE3_S synchronization after the copy
 *
 * @note Used for NIC-visible GM records such as WQEs and software doorbells. Callers issue
 *       dcci_cachelines afterwards where a cache flush is required.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_hns_1825_write_ub_to_gm_with_sync(
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

ACLSHMEM_DEVICE void aclshmemi_hns_1825_clear_wqe_ub(AscendC::LocalTensor<uint32_t>& wqe_ub_local)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE = 64; // READ/WRITE WQE uses one 64B WQEBB
    for (uint32_t i = 0; i < ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE / sizeof(uint32_t); ++i) {
        wqe_ub_local.SetValue(i, 0);
    }
}

ACLSHMEM_DEVICE AscendC::LocalTensor<uint32_t> aclshmemi_hns_1825_make_wqe_ub(
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t wqe_size)
{
    AscendC::LocalTensor<uint32_t> wqe_ub_local;
    wqe_ub_local.address_.logicPos = ub_local32.address_.logicPos;
    wqe_ub_local.address_.bufferAddr = ub_local32.address_.bufferAddr;
    wqe_ub_local.address_.dataLen = wqe_size;
    return wqe_ub_local;
}

ACLSHMEM_DEVICE uint32_t aclshmemi_hns_1825_read_u32_gm(uint64_t addr)
{
    dcci_cachelines((__gm__ uint8_t*)addr, sizeof(uint32_t));
    return *(__gm__ volatile uint32_t*)addr;
}

ACLSHMEM_DEVICE void aclshmemi_hns_1825_write_u32_gm(uint64_t addr, uint32_t value)
{
    *(__gm__ volatile uint32_t*)addr = value;
    dcci_cachelines((__gm__ uint8_t*)addr, sizeof(uint32_t));
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_roce_hns_1825_get_send_wqe(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t idx)
{
    // SQ ring slot stride is one 64B WQEBB; it is independent from the control-plane wqe_size field.
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQEBB_SIZE = 64;
    return (__gm__ uint8_t*)sq_context->buf_addr + (uint64_t)idx * ACLSHMEMI_HNS_1825_WQEBB_SIZE;
}

// A CQE is ready when its owner bit matches the lap parity derived from the current consumer index.
// cq_ring is the physical CQ ring size; the lap bit is selected by (cur_tail & cq_ring).
ACLSHMEM_DEVICE bool aclshmemi_roce_hns_1825_check_cqe_owner(
    __ubuf__ aclshmemi_hns_1825_cqe_t* cqe, uint32_t cur_tail, uint32_t cq_ring)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_OWNER_SHIFT = 31;      // owner bit at dw1[31]
    uint32_t cur_owner = ((cqe->owner_id_qpn & (1U << ACLSHMEMI_HNS_1825_CQE_OWNER_SHIFT)) != 0);
    uint32_t expect_owner = (uint32_t)(((cur_tail & cq_ring) == 0));
    return (expect_owner ^ cur_owner) != 0;
}

// HNS_1825 CQ doorbell: keep host-order CI mirrors in CQ/SQ tail records and publish the big-endian CI to
// the NIC-visible CQ software doorbell (roceCq.dbSwVa).
template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(
    uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQ_UPDATE_CI_MASK = 0xffffff; // CQ consumer index is 24-bit

    (void)ub_local64;
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));

    // Host-order CI mirrors: cq tail_addr lets poll_cq resume, sq tail_addr feeds post_send SQ-full checks.
    aclshmemi_hns_1825_write_u32_gm(cq_context->tail_addr, cur_tail);
    aclshmemi_hns_1825_write_u32_gm(sq_context->tail_addr, cur_tail);

    ub_local32.SetValue(0, aclshmemi_hns_1825_htobe32(cur_tail & ACLSHMEMI_HNS_1825_CQ_UPDATE_CI_MASK));
    aclshmemi_hns_1825_write_ub_to_gm_with_sync(cq_context->db_sw_addr, ub_local32, sizeof(uint32_t), sync_id);
    dcci_cachelines((__gm__ uint8_t*)cq_context->db_sw_addr, sizeof(uint32_t));
}

// Poll until the CQ consumer index reaches target_idx, or return an error status on CQE error/timeout.
template <>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::HNS_1825>(
    uint32_t pe, uint32_t qp_idx, uint32_t target_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    constexpr uint32_t ACLSHMEMI_HNS_1825_CYCLE_TO_TIME_BASE = 50;
#else
    constexpr uint32_t ACLSHMEMI_HNS_1825_CYCLE_TO_TIME_BASE = 1000;
#endif
    constexpr uint32_t ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_DURATION = 5 * 60 * 1000000; // 5 minutes in microseconds
    constexpr uint64_t ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_CYCLES =
        (uint64_t)ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_DURATION * ACLSHMEMI_HNS_1825_CYCLE_TO_TIME_BASE;
    // Use a value outside the CQE syndrome range to report timeout.
    constexpr uint32_t ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_ERROR = 0x10000;
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_OPCODE_SHIFT = 27;     // op_type at dw2[31:27]
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_OPCODE_MASK = 0x1f;
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_OPTYPE_ERROR = 0x1e;   // op_type = error coding
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_OPTYPE_INVALID = 0x1f; // op_type = unused/placeholder slot
    // CQE slot stride in the CQ ring. The hardware slot is 64B even though SHMEM only parses the first
    // 32B (aclshmemi_hns_1825_cqe_t); used only when the control plane leaves cqe_size unset.
    constexpr uint32_t ACLSHMEMI_HNS_1825_DEFAULT_CQE_SIZE = 64;
    // Extra CQ slots included in the physical ring: physical ring = depth + this value.
    constexpr uint32_t ACLSHMEMI_HNS_1825_CQE_MAX_GEN_NUM = 1024;

    if (target_idx == 0) {
        return 0;
    }
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));

    uint32_t cqe_size = cq_context->cqe_size == 0 ? ACLSHMEMI_HNS_1825_DEFAULT_CQE_SIZE : cq_context->cqe_size;
    // depth is the usable CQE count; slot index and owner parity use the physical ring, not depth.
    uint32_t depth = cq_context->depth;
    uint32_t cq_ring = depth + ACLSHMEMI_HNS_1825_CQE_MAX_GEN_NUM;
    uint32_t cur_tail = aclshmemi_hns_1825_read_u32_gm(cq_context->tail_addr);
    uint32_t original_cur_tail = cur_tail;
    uint64_t run_cycles = 0;
    uint32_t status = 0;

    while (cur_tail != target_idx) {
        run_cycles = 0;
        __gm__ uint8_t* cqe_addr =
            (__gm__ uint8_t*)(cq_context->buf_addr + (uint64_t)(cur_tail & (cq_ring - 1)) * cqe_size);
        AscendC::GlobalTensor<uint32_t> cqe_gm;
        cqe_gm.SetGlobalBuffer((__gm__ uint32_t*)cqe_addr);
        AscendC::DataCopyExtParams copy_params{1, cqe_size, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pad_params{false, 0, 0, 0};
        __ubuf__ aclshmemi_hns_1825_cqe_t* cqe =
            (__ubuf__ aclshmemi_hns_1825_cqe_t*)(__ubuf__ void*)ub_local32.GetPhyAddr();
        uint32_t cqe_type = ACLSHMEMI_HNS_1825_CQE_OPTYPE_INVALID;

        // Copy the CQE into UB and wait until it carries the expected owner bit.
        while (run_cycles < ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_CYCLES) {
            AscendC::DataCopyPad(ub_local32, cqe_gm, copy_params, pad_params);
            AscendC::PipeBarrier<PIPE_ALL>();
            cqe_type = (cqe->op_sr_wqebb >> ACLSHMEMI_HNS_1825_CQE_OPCODE_SHIFT) & ACLSHMEMI_HNS_1825_CQE_OPCODE_MASK;
            if (cqe_type != ACLSHMEMI_HNS_1825_CQE_OPTYPE_INVALID &&
                aclshmemi_roce_hns_1825_check_cqe_owner(cqe, cur_tail, cq_ring)) {
                break;
            }
            run_cycles++;
            dcci_cachelines(cqe_addr, cqe_size);
        }
        if (run_cycles >= ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_CYCLES) {
            // No CQE with the expected owner bit was observed before timeout.
            status = ACLSHMEMI_HNS_1825_POLL_CQ_TIMEOUT_ERROR;
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf,
                "Poll CQE timeout: pe=%u, qp_idx=%u, cur_tail=%u, target_idx=%u, original_tail=%u, backend=%u\n",
                pe, qp_idx, cur_tail, target_idx, original_cur_tail, (uint32_t)aclshmemi_rdma_backend_t::HNS_1825);
            break;
        }

        // Check CQE status.
        if (cqe_type == ACLSHMEMI_HNS_1825_CQE_OPTYPE_ERROR) {
            // Return the syndrome from an error CQE.
            status = cqe->syndrome;
            uint32_t wqn = cqe->owner_id_qpn & 0xfffffU;
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf,
                "Receive CQE with error: syndrome=0x%x in pe %u, cur_tail: %u, wqn: %u, qp_idx: %u, "
                "backend %u\n",
                status, pe, cur_tail, wqn, qp_idx, (uint32_t)aclshmemi_rdma_backend_t::HNS_1825);
            cur_tail++;
            break;
        }
        cur_tail++;
    }

    aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(
        pe, qp_idx, cur_tail, ub_local64, ub_local32, sync_id);
    return status;
}

// Fill the 16B WQE control segment in the UB staging buffer. owner_sl carries the owner bit that toggles
// on every SQ wrap; the remaining fields select one task section plus one data segment.
ACLSHMEM_DEVICE __ubuf__ uint8_t* aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(
    AscendC::LocalTensor<uint32_t>& wqe_ub_local, uint32_t cur_head, uint32_t depth)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_CTRL_VALUE = 0x40;       // owner_sl fixed part
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_VA_VALUE = 0x20;         // df_tsl VA bit
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_CQE_SIGNAL_SHIFT = 7;    // df_tsl CR bit
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_OWNER_SHIFT = 7;         // owner_sl owner bit
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_CMP_TASK_LEN_SHIFT = 28; // cl_pi CL field
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_MSN_SHIFT = 12;          // wf_bdsl wqe_msn field
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_MSN_MASK = 0x3;          // low 2 bits of SQ WQE sequence number
    constexpr uint32_t ACLSHMEMI_HNS_1825_SEG_LEN_UNIT = sizeof(uint64_t); // hardware section length unit = 8B
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_DATA_SEG_BDSL =
        sizeof(aclshmemi_hns_1825_wqe_data_seg_t) / ACLSHMEMI_HNS_1825_SEG_LEN_UNIT;
    uint16_t wf_bdsl = (uint16_t)(ACLSHMEMI_HNS_1825_WQE_DATA_SEG_BDSL |
                                  ((cur_head & ACLSHMEMI_HNS_1825_WQE_MSN_MASK)
                                   << ACLSHMEMI_HNS_1825_WQE_MSN_SHIFT));

    aclshmemi_hns_1825_clear_wqe_ub(wqe_ub_local);
    __ubuf__ uint8_t* wqe_ub = (__ubuf__ uint8_t*)(__ubuf__ void*)wqe_ub_local.GetPhyAddr();
    __ubuf__ aclshmemi_hns_1825_wqe_ctrl_seg_t* ctrl =
        (__ubuf__ aclshmemi_hns_1825_wqe_ctrl_seg_t*)(__ubuf__ void*)wqe_ub;

    ctrl->owner_sl = (((cur_head & depth) == 0) ? 0 : (1U << ACLSHMEMI_HNS_1825_WQE_OWNER_SHIFT)) |
                     ACLSHMEMI_HNS_1825_WQE_CTRL_VALUE;
    ctrl->df_tsl = (uint8_t)((1U << ACLSHMEMI_HNS_1825_WQE_CQE_SIGNAL_SHIFT) |
                             ACLSHMEMI_HNS_1825_WQE_VA_VALUE |
                             (sizeof(aclshmemi_hns_1825_wqe_rdma_task_seg_t) /
                              ACLSHMEMI_HNS_1825_SEG_LEN_UNIT));
    ctrl->wf_bdsl = aclshmemi_hns_1825_htobe16(wf_bdsl);
    ctrl->cl_pi = aclshmemi_hns_1825_htobe32(1U << ACLSHMEMI_HNS_1825_WQE_CMP_TASK_LEN_SHIFT);
    return wqe_ub + sizeof(aclshmemi_hns_1825_wqe_ctrl_seg_t);
}

enum class aclshmemi_hns_1825_msg_type_t : uint32_t {
    ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE = 0x04,
    ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_READ = 0x08,
};

// Fill the 32B RDMA task segment in UB with opcode, message length, remote VA, rkey and ulp. Multi-byte
// fields are stored in the byte order consumed by the NIC.
ACLSHMEM_DEVICE __ubuf__ uint8_t* aclshmemi_roce_hns_1825_fill_wqe_task_seg(
    __ubuf__ uint8_t* wqe_addr, aclshmemi_rdma_send_wr& wr, aclshmemi_hns_1825_msg_type_t opcode)
{
    __ubuf__ aclshmemi_hns_1825_wqe_rdma_task_seg_t* task =
        (__ubuf__ aclshmemi_hns_1825_wqe_rdma_task_seg_t*)(__ubuf__ void*)wqe_addr;

    task->com_tsk.value = 0;
    task->com_tsk.bs.signal = 1;
    task->com_tsk.bs.opcode = (uint32_t)opcode;
    task->com_tsk.value = aclshmemi_hns_1825_htobe32(task->com_tsk.value);
    task->data_len = aclshmemi_hns_1825_htobe32((uint32_t)wr.message_len);
    task->imm_data = 0;
    task->dw3.value = 0;
    if (opcode == aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_READ) {
        constexpr uint32_t ACLSHMEMI_HNS_1825_RDMA_READ_LAST_EXT_LEN = 4;

        task->dw3.bs.last_ext_len = ACLSHMEMI_HNS_1825_RDMA_READ_LAST_EXT_LEN;
        task->dw3.value = aclshmemi_hns_1825_htobe32(task->dw3.value);
    }
    task->va = aclshmemi_hns_1825_htobe64((uint64_t)wr.remote_addr);
    task->rkey = aclshmemi_hns_1825_htobe32(wr.rkey);
    task->ulp = aclshmemi_hns_1825_htobe32(wr.lkey & 0xffffU);
    return wqe_addr + sizeof(aclshmemi_hns_1825_wqe_rdma_task_seg_t);
}

// Fill the 16B data segment (the single SGE) in UB, right after ctrl+task, so the whole 64B WQEBB is
// assembled contiguously and copied to GM in one transfer. le_key sets the L bit for the single SGE.
ACLSHMEM_DEVICE __ubuf__ uint8_t* aclshmemi_roce_hns_1825_fill_wqe_data_seg(
    __ubuf__ uint8_t* wqe_addr, aclshmemi_rdma_send_wr& wr)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_NEXT_SGE_INVALID = 1U << 31; // data seg le_key L bit
    constexpr uint32_t ACLSHMEMI_HNS_1825_WQE_LKEY_MASK = 0x3fffffffU;     // data seg le_key key[29:0]

    __ubuf__ aclshmemi_hns_1825_wqe_data_seg_t* data_ub =
        (__ubuf__ aclshmemi_hns_1825_wqe_data_seg_t*)(__ubuf__ void*)wqe_addr;

    data_ub->buf_addr = aclshmemi_hns_1825_htobe64((uint64_t)wr.local_addr);
    data_ub->r_len = aclshmemi_hns_1825_htobe32((uint32_t)wr.message_len);
    data_ub->le_key =
        aclshmemi_hns_1825_htobe32((wr.lkey & ACLSHMEMI_HNS_1825_WQE_LKEY_MASK) |
                                   ACLSHMEMI_HNS_1825_WQE_NEXT_SGE_INVALID);
    return wqe_addr + sizeof(aclshmemi_hns_1825_wqe_data_seg_t);
}

// Pre-mark the next WQEBB owner byte as invalid so the hardware stops there until the following WQE is
// posted.
ACLSHMEM_DEVICE void aclshmemi_roce_hns_1825_write_invalid_wqebb(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t idx)
{
    __gm__ aclshmemi_hns_1825_wqe_ctrl_seg_t* ctrl =
        (__gm__ aclshmemi_hns_1825_wqe_ctrl_seg_t*)aclshmemi_roce_hns_1825_get_send_wqe(
            sq_context, idx & (sq_context->depth - 1));
    ctrl->owner_sl = ((idx & sq_context->depth) == 0) ? 0xff : 0x7f;
    dcci_cachelines((__gm__ uint8_t*)ctrl, sizeof(uint8_t));
}

// Assemble the WRITE/READ WQE contiguously in the UB staging buffer, mark the next WQEBB invalid, then
// copy the whole WQE to the SQ in one transfer. Returns the WQE size for the caller's cache flush.
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_hns_1825_fill_wqe_write_read(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
    uint32_t cur_head, aclshmemi_hns_1825_msg_type_t opcode, AscendC::LocalTensor<uint32_t>& wqe_ub_local,
    uint32_t sync_id)
{
    // This single-SGE WRITE/READ WQE is one 64B WQEBB (ctrl 16B + task 32B + data 16B); other opcodes/tasks
    // may use a different WQE size, so this constant is scoped to WRITE/READ rather than a generic WQE size.
    constexpr uint32_t ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE = 64;

    __ubuf__ uint8_t* data_ub = aclshmemi_roce_hns_1825_fill_wqe_task_seg(
        aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(wqe_ub_local, cur_head, sq_context->depth), wr, opcode);
    (void)aclshmemi_roce_hns_1825_fill_wqe_data_seg(data_ub, wr);
    aclshmemi_roce_hns_1825_write_invalid_wqebb(sq_context, cur_head + 1);
    aclshmemi_hns_1825_write_ub_to_gm_with_sync(
        (uint64_t)wqe_addr, wqe_ub_local, ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE, sync_id);
    return ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE;
}

// The WQE body is assembled in aclshmemi_roce_hns_1825_fill_wqe_write_read (called from
// aclshmemi_hns_1825_post_send_read_write), where the UB staging buffer and sync_id are available. These
// two entries only report the WQE size so the caller flushes the correct number of cachelines.
template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::HNS_1825, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE = 64; // a WRITE/READ WQE occupies one 64B WQEBB

    (void)wr;
    (void)sq_context;
    (void)wqe_addr;
    (void)cur_head;
    return ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE;
}

template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::HNS_1825, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE = 64; // a WRITE/READ WQE occupies one 64B WQEBB

    (void)wr;
    (void)sq_context;
    (void)wqe_addr;
    (void)cur_head;
    return ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE;
}

/**
 * @brief Ring the HNS_1825 Send Queue (SQ) doorbell.
 *
 * This function updates the SHMEM-owned SQ producer-index mirror, publishes the NIC-visible software
 * doorbell, and writes the hardware doorbell register with st_dev.
 *
 * @param sq_context SQ context containing queue addresses, QPN and doorbell registers
 * @param cur_head The next SQ producer index, equal to the total number of posted WQEs
 * @param ub_local64 64-bit UB workspace, unused by this backend
 * @param ub_local32 32-bit UB workspace used for the software doorbell record
 * @param sync_id Hardware synchronization event ID
 */
template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_PI_HIGH_SHIFT = 8;  // high 8 bits of the SQ producer index
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_PI_FIELD_SHIFT = 32; // pi field offset in the 64-bit doorbell
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_TYPE_21 = 21;        // doorbell type = SQ
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_SGID_IDX = 1;
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_CNTX_SIZE = 1;
    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_COS = 0x7;

    (void)ub_local64;
    // head_addr is SHMEM's host-order PI mirror, not the NIC-visible software doorbell.
    AscendC::DataCopyExtParams copy_params{1, sizeof(uint32_t), 0, 0, 0};
    aclshmemi_hns_1825_write_u32_gm(sq_context->head_addr, cur_head);

    AscendC::GlobalTensor<uint32_t> sq_sw_db;
    sq_sw_db.SetGlobalBuffer((__gm__ uint32_t*)sq_context->db_sw_addr, 1);
    ub_local32.SetValue(0, aclshmemi_hns_1825_htobe32(cur_head));
    aclshmemi_hns_1825_sync_func<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(sq_sw_db, ub_local32, copy_params);
    AscendC::PipeBarrier<PIPE_MTE3>();
    dcci_cachelines((__gm__ uint8_t*)sq_context->db_sw_addr, sizeof(uint32_t));

    // Program the hardware doorbell register with a single st_dev write.
    aclshmemi_hns_1825_sq_db_t sq_db;
    sq_db.value = 0;
    sq_db.bs.c = 0;
    sq_db.bs.rsvd0 = 0;
    sq_db.bs.cntx_size = ACLSHMEMI_HNS_1825_SQ_DB_CNTX_SIZE;
    sq_db.bs.qpn = sq_context->wqn;
    sq_db.bs.sub_type = 0;
    sq_db.bs.rsvd1 = 0;
    sq_db.bs.pi = 0;
    sq_db.bs.sgid_index = ACLSHMEMI_HNS_1825_SQ_DB_SGID_IDX;
    sq_db.bs.type = ACLSHMEMI_HNS_1825_SQ_DB_TYPE_21;
    sq_db.bs.mtu_shift = sq_context->mtu_shift;
    sq_db.bs.cos = ACLSHMEMI_HNS_1825_SQ_DB_COS;
    uint64_t db_value = sq_db.value |
        ((((uint64_t)cur_head >> ACLSHMEMI_HNS_1825_SQ_DB_PI_HIGH_SHIFT) & 0xffULL)
         << ACLSHMEMI_HNS_1825_SQ_DB_PI_FIELD_SHIFT);

    AscendC::GlobalTensor<uint32_t> db_gm;
    db_gm.SetGlobalBuffer((__gm__ uint32_t*)sq_context->db_addr, sizeof(uint64_t) / sizeof(uint32_t));
    aclshmemi_hns_1825_sync_func<AscendC::HardEvent::MTE3_S>(sync_id);
    st_dev(db_value, (__gm__ uint64_t*)db_gm.GetPhyAddr(), 0);
    aclshmemi_hns_1825_sync_func<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::PipeBarrier<PIPE_ALL>();
}

/**
 * @brief Post an HNS_1825 RDMA WRITE/READ WQE and update SQ state.
 *
 * @tparam OP_CODE RDMA opcode, OP_RDMA_READ or OP_RDMA_WRITE
 * @param wr [Input/Output] RDMA work request. The caller fills remote_addr, local_addr and message_len;
 *           this function fills rkey and lkey from the local/remote MR info tables.
 * @param pe Target PE number
 * @param qp_idx QP index
 * @param ub_local64 64-bit UB workspace, unused by this backend
 * @param ub_local32 32-bit UB workspace; a 64B WQE view is derived from its base address before WQE staging
 * @param sync_id Hardware synchronization event ID
 */
template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_hns_1825_post_send_read_write(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    constexpr aclshmemi_hns_1825_msg_type_t OPCODE =
        OP_CODE == aclshmemi_rdma_opcode_t::OP_RDMA_READ ?
            aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_READ :
            aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE;
    constexpr uint32_t ACLSHMEMI_HNS_1825_POLL_CQ_THRESHOLD = 10;

    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t my_pe = aclshmemi_get_my_pe();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    uint64_t mem_info_table = rdma_info->mem_ptr;
    // SQ/CQ depth is assumed to be a power of two: owner-bit toggling and slot masking rely on it.
    uint32_t depth = sq_context->depth;
    // SQ head/tail are SHMEM's host-order mirrors. The NIC-visible software doorbell is db_sw_addr.
    uint32_t cur_head = aclshmemi_hns_1825_read_u32_gm(sq_context->head_addr);
    uint32_t cur_tail = aclshmemi_hns_1825_read_u32_gm(sq_context->tail_addr);
    uint32_t ret = 0;

    // If the SQ is about to be full, wait for CQEs to drain before posting a new WQE.
    if ((cur_head + ACLSHMEMI_HNS_1825_POLL_CQ_THRESHOLD) % depth == (depth + cur_tail) % depth) {
        ret = aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::HNS_1825>(
            pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (ret) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf,
                "Poll CQE error in aclshmemi_hns_1825_post_send_read_write: pe=%u, qp_idx=%u, cur_tail=%u, "
                "cur_head=%u\n",
                pe, qp_idx, cur_tail, cur_head);
            return;
        }
    }

    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);
    __gm__ aclshmemi_rdma_mem_info* local_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * my_pe);
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = local_mem_info->lkey;

    // Create a UB view sized to the READ/WRITE WQE before staging it for the SQ copy.
    constexpr uint32_t ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE = 64;
    AscendC::LocalTensor<uint32_t> wqe_ub_local =
        aclshmemi_hns_1825_make_wqe_ub(ub_local32, ACLSHMEMI_HNS_1825_WRITE_READ_WQE_SIZE);
    __gm__ uint8_t* wqe_addr =
        aclshmemi_roce_hns_1825_get_send_wqe(sq_context, cur_head & (sq_context->depth - 1));
    uint32_t wqe_total_size = aclshmemi_roce_hns_1825_fill_wqe_write_read(
        wr, sq_context, wqe_addr, cur_head, OPCODE, wqe_ub_local, sync_id);
    dcci_cachelines(wqe_addr, wqe_total_size);
    cur_head++;

    aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(
        sq_context, cur_head, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::HNS_1825,
    aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_hns_1825_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::HNS_1825,
    aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_hns_1825_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

// HNS_1825 does not implement RDMA atomics; keep atomic paths as compile-time errors.
template <>
struct aclshmemi_backend_traits<aclshmemi_rdma_backend_t::HNS_1825> {
    template <typename T, bool IS_MASKED>
    struct atomic_op_traits {
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE uint32_t fill_wqe(
            aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
            uint32_t cur_head)
        {
            (void)wr;
            (void)sq_context;
            (void)wqe_addr;
            (void)cur_head;
            static_assert(aclshmemi_atomic_op_dependent_false<ATOMIC_OP_CODE>::value,
                "HNS_1825 backend does not support atomic operations yet.");
            return 0;
        }

        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE void post_send(
            aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
            AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
        {
            (void)wr;
            (void)pe;
            (void)qp_idx;
            (void)ub_local64;
            (void)ub_local32;
            (void)sync_id;
            static_assert(aclshmemi_atomic_op_dependent_false<ATOMIC_OP_CODE>::value,
                "HNS_1825 backend does not support atomic operations yet.");
        }
    };
};

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_HNS_1825_HPP
