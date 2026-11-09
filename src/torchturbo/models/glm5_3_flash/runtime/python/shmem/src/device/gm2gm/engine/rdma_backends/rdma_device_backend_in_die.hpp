/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_IN_DIE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_IN_DIE_HPP

#include "rdma_device_backend_base.h"

// vendor-specfic header
struct aclshmemi_wqe_ctx {
    uint32_t byte4;
    uint32_t msg_len;
    uint32_t immt_data;
    uint32_t byte16;
    uint32_t byte20;
    uint32_t rkey;
    uint64_t va;
};

struct aclshmemi_sge_ctx {
    uint32_t len;
    uint32_t lkey;
    uint64_t addr;
};

struct aclshmemi_cqe_ctx {
    uint32_t byte4;
    uint32_t immt_data;
    uint32_t byte12;
    uint32_t byte16;
    uint32_t byte_cnt;
    uint32_t smac;
    uint32_t byte28;
    uint32_t byte32;
};

enum class aclshmemi_rdma_in_die_opcode_t : uint32_t { OP_RDMA_WRITE = 3, OP_RDMA_WRITE_WITH_IMM, OP_RDMA_READ };

// Shard Helper: WQE file for RDMA WRITE/READ, Returns the final size of WQE
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_write_read(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head,
    aclshmemi_rdma_in_die_opcode_t opcode)
{
    constexpr uint32_t shift = 13;
    // Write WQE to HBM
    uint64_t owner_bit = (cur_head >> shift) & 0x1;
    uint32_t byte4 = (uint32_t)opcode & 0x1F;           // [0:4] opcode
    byte4 |= ((~owner_bit) << 7) & (1 << 7);            // [7] owner_bit
    byte4 |= 1 << 8;                                    // [8] IBV_SEND_SINGNALED
    *(__gm__ uint32_t*)(wqe_addr) = byte4;              // control set by local parameter, see above lines
    *(__gm__ uint32_t*)(wqe_addr + 4) = wr.message_len; // message size in bytes
    *(__gm__ uint32_t*)(wqe_addr + 8) = 0;              // immt_data is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqe_addr + 12) = 1 << 24;       // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqe_addr + 16) = 0;             // [128:151] start_sge_index = 0
    *(__gm__ uint32_t*)(wqe_addr + 20) = wr.rkey;       // rkey
    *(__gm__ uint64_t*)(wqe_addr + 24) = (uint64_t)wr.remote_addr; // remote VA

    // Write SGE to HBM
    __gm__ uint8_t* sge_addr = wqe_addr + sizeof(aclshmemi_wqe_ctx);
    *(__gm__ uint32_t*)(sge_addr) = wr.message_len;              // message size in bytes
    *(__gm__ uint32_t*)(sge_addr + 4) = wr.lkey;                 // lkey
    *(__gm__ uint64_t*)(sge_addr + 8) = (uint64_t)wr.local_addr; // local VA

    return sizeof(aclshmemi_wqe_ctx) + sizeof(aclshmemi_sge_ctx);
}

// Note: Here Map public aclshmemi_rdma_opcode_t to vendor-specific Opcode here
template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::IN_DIE, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    return aclshmemi_roce_fill_wqe_write_read(
        wr, sq_context, wqe_addr, cur_head, aclshmemi_rdma_in_die_opcode_t::OP_RDMA_WRITE);
}

// Note: Here Map public aclshmemi_rdma_opcode_t to vendor-specific Opcode here
template <>
ACLSHMEM_DEVICE uint32_t
aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::IN_DIE, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head)
{
    return aclshmemi_roce_fill_wqe_write_read(
        wr, sq_context, wqe_addr, cur_head, aclshmemi_rdma_in_die_opcode_t::OP_RDMA_READ);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::IN_DIE>(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    uint64_t doorbell_info = 0;
    doorbell_info |= sq_context->wqn;                    // [0:23] DB_TAG = qp_num
    doorbell_info |= 0 << 24;                            // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB(0)
    doorbell_info |= ((uint64_t)cur_head % 65536) << 32; // [32:47] DB_PI = sq.head
    doorbell_info |= (uint64_t)(sq_context->sl) << 48;   // [48:50] DB_SL = qp.sl

    auto sq_pi_addr = sq_context->head_addr;

    __gm__ uint64_t* doorbell_addr = (__gm__ uint64_t*)(sq_context->db_addr);

    ub_local64.SetValue(0, doorbell_info);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint64_t> db_gm;
    db_gm.SetGlobalBuffer(doorbell_addr);
    AscendC::DataCopyExtParams copy_params{1, 1 * sizeof(uint64_t), 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(db_gm, ub_local64, copy_params);

    ub_local32.SetValue(0, (uint32_t)cur_head);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint32_t> sq_head_gm;
    sq_head_gm.SetGlobalBuffer((__gm__ uint32_t*)sq_pi_addr);
    AscendC::DataCopyExtParams copy_params_head{1, 1 * sizeof(uint32_t), 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(sq_head_gm, ub_local32, copy_params_head);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

// Shard Helper: Post-Send orchestration for RDMA WRITE/READ
template <aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send_read_write(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;

    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    auto mem_info_table = rdma_info->mem_ptr;
    auto sq_base_addr = sq_context->buf_addr;
    auto wqe_size = sq_context->wqe_size;
    auto sq_pi_addr = sq_context->head_addr;
    dcci_cachelines((__gm__ uint8_t*)sq_pi_addr, 8);
    uint32_t cur_head = *(__gm__ uint32_t*)(sq_pi_addr);
    auto sq_ci_addr = sq_context->tail_addr;
    auto depth = sq_context->depth;

    // Poll CQ if send queue is full
    dcci_cachelines((__gm__ uint8_t*)sq_ci_addr, 8);
    if ((cur_head + 10) % depth == (*(__gm__ uint32_t*)(sq_ci_addr)) % depth) {
        aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::IN_DIE>(
            pe, qp_idx, *(__gm__ uint32_t*)(sq_ci_addr) + ACLSHMEM_NUM_CQE_PER_POLL_CQ, ub_local64, ub_local32,
            sync_id);
    }

    __gm__ uint8_t* wqe_addr = (__gm__ uint8_t*)(sq_base_addr + wqe_size * (cur_head % depth));

    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);
    __gm__ aclshmemi_rdma_mem_info* local_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * aclshmemi_get_my_pe());
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = local_mem_info->lkey;

    // Write WQE to HBM
    uint32_t wqe_total_size =
        aclshmemi_roce_fill_wqe<aclshmemi_rdma_backend_t::IN_DIE, OP_CODE>(wr, sq_context, wqe_addr, cur_head);
    ACLSHMEM_DEBUG_FUNC(
        aclshmemi_kernel_printf, "Opcode: %u, Get Wqe Size: %d for backend %u\n", (uint32_t)OP_CODE, wqe_total_size,
        (uint32_t)aclshmemi_rdma_backend_t::IN_DIE);

    // WQE & SGE cache flush
    dcci_cachelines(wqe_addr, wqe_total_size);
    cur_head++;

    aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::IN_DIE>(
        sq_context, cur_head, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::IN_DIE, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_roce_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<aclshmemi_rdma_backend_t::IN_DIE, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_roce_post_send_read_write<aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

template <>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::IN_DIE>(
    uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();

    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));
    AscendC::DataCopyExtParams copy_params_tail{1, 1 * sizeof(uint32_t), 0, 0, 0};

    // Ring CQ Doorbell
    auto cq_db_addr = cq_context->db_addr;
    if (cq_context->db_mode == aclshmemi_rdma_db_mode_t::SW_DB) {
        ub_local32.SetValue(0, (uint32_t)(cur_tail & 0xFFFFFF));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::GlobalTensor<uint32_t> cq_db_gm;
        cq_db_gm.SetGlobalBuffer((__gm__ uint32_t*)cq_db_addr);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::DataCopyPad(cq_db_gm, ub_local32, copy_params_tail);
    } else if (cq_context->db_mode == aclshmemi_rdma_db_mode_t::HW_DB) {
        uint64_t doorbell_info = 0;
        doorbell_info |= cq_context->cqn;                       // [0:23] DB_TAG = qp_num
        doorbell_info |= 3 << 24;                               // [24:27] DB_CMD = HNS_ROCE_V2_CQ_DB_PTR(3)
        doorbell_info |= (uint64_t)(cur_tail & 0xFFFFFF) << 32; // [32:55] DB_CQ_CI = cq.tail
        doorbell_info |= (uint64_t)1 << 56;                     // [56:56] DB_CQ_CMD_SN = 1
        ub_local64.SetValue(0, doorbell_info);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::GlobalTensor<uint64_t> db_gm;
        db_gm.SetGlobalBuffer((__gm__ uint64_t*)cq_db_addr);
        AscendC::DataCopyExtParams copy_params{1, 1 * sizeof(uint64_t), 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::DataCopyPad(db_gm, ub_local64, copy_params);
    }

    // Update WQ tail
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    auto sq_ci_addr = sq_context->tail_addr;
    ub_local32.SetValue(0, cur_tail);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint32_t> sq_tail_gm;
    sq_tail_gm.SetGlobalBuffer((__gm__ uint32_t*)sq_ci_addr);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(sq_tail_gm, ub_local32, copy_params_tail);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

template <>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq<aclshmemi_rdma_backend_t::IN_DIE>(
    uint32_t pe, uint32_t qp_idx, uint32_t target_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    if (target_idx == 0) {
        return 0;
    }
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();

    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_cq_ctx* cq_context =
        (__gm__ aclshmemi_rdma_cq_ctx*)(rdma_info->scq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_cq_ctx));
    auto cq_base_addr = cq_context->buf_addr;
    auto cqe_size = cq_context->cqe_size;
    auto depth = cq_context->depth;
    auto cq_ci_addr = cq_context->tail_addr;
    dcci_cachelines((__gm__ uint8_t*)cq_ci_addr, 8);
    uint32_t cur_tail = *(__gm__ uint32_t*)(cq_ci_addr);

    const uint32_t shift_width = 7;
    AscendC::DataCopyExtParams copy_params_tail{1, 1 * sizeof(uint32_t), 0, 0, 0};
    while (cur_tail != target_idx) {
        __gm__ aclshmemi_cqe_ctx* cqe_addr =
            (__gm__ aclshmemi_cqe_ctx*)(cq_base_addr + cqe_size * (cur_tail & (depth - 1)));
        uint32_t cqe_byte4 = *(__gm__ uint32_t*)cqe_addr;
        while (((cqe_byte4 & (1 << shift_width)) != 0) == ((cur_tail & depth) != 0)) {
            int64_t tmp = AscendC::GetSystemCycle(); // reserved for timeout check
            dcci_cachelines((__gm__ uint8_t*)cqe_addr, 32);
            cqe_byte4 = *(__gm__ uint32_t*)cqe_addr;
        }
        cur_tail++;
        uint32_t wqn = cqe_addr->byte16 & 0xFFFFFF; // reserved for multi WQ share the same CQ

        // Check CQE status
        uint32_t status = (cqe_addr->byte4 >> 8) & 0xFF;
        if (status) {
            return status;
        }
    }

    // Update CQ tail
    ub_local32.SetValue(0, (uint32_t)cur_tail);
    AscendC::GlobalTensor<uint32_t> cq_tail_gm;
    cq_tail_gm.SetGlobalBuffer((__gm__ uint32_t*)cq_ci_addr);
    AscendC::DataCopyPad(cq_tail_gm, ub_local32, copy_params_tail);

    aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::IN_DIE>(
        pe, qp_idx, cur_tail, ub_local64, ub_local32, sync_id);
    return 0;
}

/*
 *  =====================================================================================================
 *  IN_DIE Backend Atomic Operations Traits Specialization
 *  =====================================================================================================
 *  This specialization provides the atomic operations implementation for IN_DIE backend.
 *  Currently, IN_DIE backend does not support atomic operations. This placeholder will trigger
 *  a compilation error if atomic operations are attempted with IN_DIE backend.
 *  =====================================================================================================
 */

template <>
struct aclshmemi_backend_traits<aclshmemi_rdma_backend_t::IN_DIE> {
    template <typename T, bool IS_MASKED>
    struct atomic_op_traits {
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE uint32_t fill_wqe(
            aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
            uint32_t cur_head)
        {
            static_assert(
                aclshmemi_atomic_op_dependent_false<ATOMIC_OP_CODE>::value,
                "IN_DIE backend does not support atomic operations. "
                "Atomic operations are not available for this backend.");
            return 0;
        }

        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE void post_send(
            aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
            AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
        {
            static_assert(
                aclshmemi_atomic_op_dependent_false<ATOMIC_OP_CODE>::value,
                "IN_DIE backend does not support atomic operations. "
                "Atomic operations are not available for this backend.");
        }
    };
};

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_IN_DIE_HPP
