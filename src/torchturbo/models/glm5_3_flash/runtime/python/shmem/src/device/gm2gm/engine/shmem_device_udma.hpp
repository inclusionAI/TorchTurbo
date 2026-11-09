/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_UDMA_HPP
#define SHMEM_DEVICE_UDMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_common.h"
#include "shmemi_device_udma.h"
#include "utils/shmemi_kernel_debug.h"
#include "../host_device/shmemi_host_device_constant.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEM_UDMA_SUPPORTED 1
#else
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif

#if defined(ACLSHMEM_RELAY_SUPPORT)
#define ACLSHMEM_RELAY_SUPPORTED 1
#else
#define ACLSHMEM_RELAY_SUPPORTED 0
#endif

constexpr uint32_t MAX_RETRY_TIMES = 1000000;

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqe_addr, uint32_t sge_num);
ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqe_addr, uint32_t atomic_len);

ACLSHMEM_DEVICE __gm__ aclshmemi_aiv_udma_info_t* aclshmemi_udma_qp_info_fetch()
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = (__gm__ aclshmemi_aiv_udma_info_t*)(aclshmemi_get_qp_info_address(0));
    return udma_info;
}

// Active queue/memory table for this build. Compile-time selected, inlined, and (since the union
// members share storage) resolves to the same address as the other member -- zero runtime cost.
ACLSHMEM_DEVICE __gm__ aclshmemi_udma_qp_table_t* aclshmemi_udma_active_table(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        return &udma_info->relay;
    } else {
        return &udma_info->direct;
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_cqe(__gm__ aclshmemi_jfc_cqe_ctx_t* cqe_addr)
{
    if (cqe_addr == nullptr) {
        AscendC::printf("CQE: nullptr pointer\n");
        return;
    }
    uint32_t s_r = cqe_addr->s_r;
    uint32_t is_jetty = cqe_addr->is_jetty;
    uint32_t owner = cqe_addr->owner;
    uint32_t inline_en = cqe_addr->inline_en;
    uint32_t opcode = cqe_addr->opcode;
    uint32_t fd = cqe_addr->fd;
    uint32_t substatus = cqe_addr->substatus;
    uint32_t status = cqe_addr->status;
    uint32_t entry_idx = cqe_addr->entry_idx;
    uint32_t local_num_l = cqe_addr->local_num_l;
    uint32_t local_num_h = cqe_addr->local_num_h;
    uint32_t rmt_idx = cqe_addr->rmt_idx;
    uint32_t tpn = cqe_addr->tpn;
    AscendC::printf(
        "CQE: DW0 - s_r: %d, is_jetty: %d, owner: %d, inline_en: %d, opcode: %d, fd: %d, substatus: %d, status: %d\n",
        s_r, is_jetty, owner, inline_en, opcode, fd, substatus, status);
    AscendC::printf("CQE: DW1 - entry_idx: %d, local_num_l: %d\n", entry_idx, local_num_l);
    AscendC::printf("CQE: DW2 - local_num_h: %d, rmt_idx: %d\n", local_num_h, rmt_idx);
    AscendC::printf("CQE: DW3 - tpn: %d\n", tpn);
    AscendC::printf("CQE: DW4 - byte_cnt: %d\n", cqe_addr->byte_cnt);
    AscendC::printf("CQE: DW5-DW6 - userData: 0x%x%x\n", cqe_addr->user_data_h, cqe_addr->user_data_l);
    AscendC::printf(
        "CQE: DW7-DW10 - rmt_eid: [0x%x, 0x%x, 0x%x, 0x%x]\n", cqe_addr->rmt_eid[0], cqe_addr->rmt_eid[1],
        cqe_addr->rmt_eid[2], cqe_addr->rmt_eid[3]);
    AscendC::printf("CQE: DW11-DW12 - data: 0x%x%x\n", cqe_addr->data_h, cqe_addr->data_l);
    AscendC::printf(
        "CQE: DW13-DW15 - inline_data: [0x%x, 0x%x, 0x%x]\n", cqe_addr->inline_data[0], cqe_addr->inline_data[1],
        cqe_addr->inline_data[2]);
}

ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t slot, uint32_t qp_idx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    uint32_t qp_num = udma_info->qp_num;
    __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx_entry =
        (__gm__ aclshmemi_udma_cq_ctx_t*)(tbl->scq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_cq_ctx_t));
    auto cq_base_addr = cq_ctx_entry->buf_addr;
    auto cqe_size = cq_ctx_entry->cqe_size;
    uint32_t cur_tail = cq_ctx_entry->tail;
    while (cur_tail != idx) {
        __gm__ aclshmemi_jfc_cqe_ctx_t* cqe_addr =
            (__gm__ aclshmemi_jfc_cqe_ctx_t*)(cq_base_addr + cqe_size * (cur_tail & (shm::UDMA_CQ_DEPTH_DEFAULT - 1)));
        bool valid_owner = (cur_tail / shm::UDMA_CQ_DEPTH_DEFAULT) & 1;
        uint32_t times = 0;
        while ((valid_owner ^ cqe_addr->owner) == 0 && times < MAX_RETRY_TIMES) { // util cqe_addr->owner changed
            dcci_cachelines((__gm__ uint8_t*)cqe_addr, sizeof(aclshmemi_jfc_cqe_ctx_t));
            times++;
        }
        if (times >= MAX_RETRY_TIMES) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Poll cq timeout! cur_tail=%d idx=%d\n", cur_tail, idx);
            return 0xFF;
        }
        // Check CQE status
        uint8_t status = cqe_addr->status & 0xFF;
        uint8_t sub_status = cqe_addr->substatus & 0xFF;
        constexpr uint8_t STATUS_SHIFT = 8;
        if (status != 0 || sub_status != 0) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_cqe, cqe_addr);
            return (status << STATUS_SHIFT) | sub_status;
        }
        cur_tail++;
    }

    // Update CQ tail
    cq_ctx_entry->tail = cur_tail;
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx_entry =
        (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
    aclshmemi_udma_poll_cq_update_info(cur_tail, qp_idx, cq_ctx_entry, wq_ctx_entry);
    return 0;
}

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(
    uint32_t cur_tail, uint32_t qp_idx, __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx_entry,
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx_entry)
{
    // Ring CQ Doorbell (reference URMA implementation)
    auto cq_db_addr = cq_ctx_entry->db_addr;
    // For JFC, we write the consumer index (cur_tail) directly
    __gm__ uint32_t* db_addr = (__gm__ uint32_t*)cq_ctx_entry->db_addr;
    st_dev((uint32_t)(cur_tail & 0xFFFFFF), db_addr, 0);
    // Update WQ tail
    wq_ctx_entry->tail = cur_tail;
}

ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqe_addr, uint32_t atomic_len)
{
    if (wqe_addr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = (__gm__ aclshmemi_sqe_ctx_t*)wqe_addr;
    auto sqe_bb_idx = sqe_ctx->sqe_bb_idx;
    auto flag = sqe_ctx->flag;
    auto rsv0 = sqe_ctx->rsv0;
    auto nf = sqe_ctx->nf;
    auto token_en = sqe_ctx->token_en;
    auto rmt_jetty_type = sqe_ctx->rmt_jetty_type;
    AscendC::printf(
        "WQE: sqe_bb_idx: %x flag: %x rsv0: %x nf: %x token_en: %x rmt_jetty_type: %x\n", sqe_bb_idx, flag, rsv0, nf,
        token_en, rmt_jetty_type);
    auto owner = sqe_ctx->owner;
    auto target_hint = sqe_ctx->target_hint;
    auto opcode = sqe_ctx->opcode;
    auto rsv1 = sqe_ctx->rsv1;
    auto inline_msg_len = sqe_ctx->inline_msg_len;
    auto tp_id = sqe_ctx->tp_id;
    AscendC::printf(
        "WQE: owner: %x target_hint: %x opcode: %x rsv1: %x inline_msg_len: %x tp_id: %x\n", owner, target_hint, opcode,
        rsv1, inline_msg_len, tp_id);
    auto sge_num = sqe_ctx->sge_num;
    auto rmt_jetty_or_seg_id = sqe_ctx->rmt_jetty_or_seg_id;
    auto rsv2 = sqe_ctx->rsv2;
    AscendC::printf("WQE: sge_num: %x rmt_jetty_or_seg_id: %x rsv2: %x\n", sge_num, rmt_jetty_or_seg_id, rsv2);
    auto rmt_eid_l = sqe_ctx->rmt_eid_l;
    auto rmt_eid_h = sqe_ctx->rmt_eid_h;
    AscendC::printf("WQE: rmt_eid: %x, %x\n", rmt_eid_l, rmt_eid_h);
    auto rmt_token_value = sqe_ctx->rmt_token_value;
    auto udf_type = sqe_ctx->udf_type;
    auto reduce_data_type = sqe_ctx->reduce_data_type;
    auto reduce_opcode = sqe_ctx->reduce_opcode;
    auto rmt_addr_l_or_token_id = sqe_ctx->rmt_addr_l_or_token_id;
    auto rmt_addr_h_or_token_value = sqe_ctx->rmt_addr_h_or_token_value;
    AscendC::printf(
        "WQE: rmt_token_value: %x udf_type: %x reduce_data_type: %x reduce_opcode: %x\n", rmt_token_value, udf_type,
        reduce_data_type, reduce_opcode);
    AscendC::printf(
        "WQE: rmt_addr_l_or_token_id: %x rmt_addr_h_or_token_value: %x\n", rmt_addr_l_or_token_id,
        rmt_addr_h_or_token_value);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY)) {
        __gm__ aclshmemi_notify_ctx_t* notify_ctx =
            (__gm__ aclshmemi_notify_ctx_t*)((__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t));
        auto notify_token_id = notify_ctx->notify_token_id;
        auto notify_token_value = notify_ctx->notify_token_value;
        auto notify_addr_l = notify_ctx->notify_addr_l;
        auto notify_addr_h = notify_ctx->notify_addr_h;
        auto notify_data_l = notify_ctx->notify_data_l;
        auto notify_data_h = notify_ctx->notify_data_h;
        AscendC::printf(
            "WQE: notify_token_id: %x notify_token_value: %x notify_addr_l: %x notify_addr_h: %x notify_data_l: %x "
            "notify_data_h: %x \n",
            notify_token_id, notify_token_value, notify_addr_l, notify_addr_h, notify_data_l, notify_data_h);
    }
    aclshmemi_dump_sge(wqe_addr, sge_num);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA)) {
        __gm__ uint8_t* amo_data_addr =
            (__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t);
        uint64_t add_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amo_data_addr) :
                                                             *(__gm__ uint64_t*)amo_data_addr;
        AscendC::printf("SGE: add_data: 0x%llx \n", (unsigned long long)add_lo);
    } else if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_CAS)) {
        __gm__ uint8_t* amo_data_addr =
            (__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t);
        uint64_t swap_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amo_data_addr) :
                                                              *(__gm__ uint64_t*)amo_data_addr;
        uint64_t cond_lo = (atomic_len == sizeof(uint32_t)) ?
                               static_cast<uint64_t>(*(__gm__ uint32_t*)(amo_data_addr + atomic_len)) :
                               *(__gm__ uint64_t*)(amo_data_addr + atomic_len);
        AscendC::printf(
            "SGE: cond_data: 0x%llx, swap_data: 0x%llx\n", (unsigned long long)cond_lo, (unsigned long long)swap_lo);
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqe_addr, uint32_t sge_num)
{
    if (wqe_addr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ aclshmemi_sge_ctx_t* sge_ctx = (__gm__ aclshmemi_sge_ctx_t*)(wqe_addr + sizeof(aclshmemi_sqe_ctx_t));
    for (size_t i = 0; i < sge_num; i++) {
        auto sge_len = sge_ctx->len;
        auto sge_rmt_addr = sge_ctx->va;
        AscendC::printf("SGE: sge idx: %d, va: %p sge_len: %d\n", i, sge_rmt_addr, sge_len);
        sge_ctx++;
    }
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr uint32_t get_wqe_bb_cnt()
{
    // 暂时不考虑inline功能，inline场景需要考虑inline data的大小
    if constexpr (
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA || OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS ||
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        return 2;
    } else {
        return 1;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_reduce(SQE_PTR sqe_ctx)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        static_assert(
            AscendC::IsSameType<SQE_PTR, __gm__ aclshmemi_sqe_ctx_t*>::value,
            "WRITE_WITH_REDUCE requires SQE in HBM (PIPE_S path); UB-staged SQE not supported");

        sqe_ctx->udf_type = 0;        // inline reduce
        sqe_ctx->reduce_opcode = 0xa; // reduce add
        if constexpr (AscendC::IsSameType<T, float>::value) {
            sqe_ctx->reduce_data_type = 0x7; // fp32
        }
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_notify_data(
    SQE_PTR sqe_ctx, uint32_t tid, uint32_t token_value, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        // The notify ctx lives immediately after the SQE in the same address space.
        using NotifyPtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ aclshmemi_notify_ctx_t*,
            __gm__ aclshmemi_notify_ctx_t*>::type;
        using BytePtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ uint8_t*,
            __gm__ uint8_t*>::type;
        NotifyPtr notify_ctx = (NotifyPtr)((BytePtr)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t));
        notify_ctx->notify_token_id = tid & 0xFFFFF; // 20 bits
        notify_ctx->notify_token_value = token_value;
        notify_ctx->notify_addr_l = reinterpret_cast<uint64_t>(params.sig_addr) & 0xFFFFFFFF;
        notify_ctx->notify_addr_h = (reinterpret_cast<uint64_t>(params.sig_addr) >> 32) & 0xFFFFFFFF;
        notify_ctx->notify_data_l = params.signal & 0xFFFFFFFF;
        notify_ctx->notify_data_h = (params.signal >> 32) & 0xFFFFFFFF;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_fill_sge_ctx(
    __gm__ aclshmemi_sge_ctx_t* sge_ctx, uint64_t message_len, __gm__ uint8_t* local_addr,
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    // default
    sge_ctx->len = message_len;
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) { // fetch and add
        auto amo_addr = qp_ctx_entry->amo_addr;
        sge_ctx->va = amo_addr;
        __gm__ T* addDataAddr = (__gm__ T*)((__gm__ uint8_t*)sge_ctx + sizeof(aclshmemi_sge_ctx_t));
        *addDataAddr = params.value;                                        // fill in add_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) { // compare and swap
        auto amo_addr = qp_ctx_entry->amo_addr;
        sge_ctx->va = amo_addr;
        __gm__ T* swap_data_addr = (__gm__ T*)((__gm__ uint8_t*)sge_ctx + sizeof(aclshmemi_sge_ctx_t));
        *swap_data_addr = params.value; // fill in swap_data
        __gm__ T* cmp_data_addr = (__gm__ T*)((__gm__ uint8_t*)swap_data_addr + sizeof(T));
        *cmp_data_addr = params.cond; // fill in cmp_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        auto amo_addr = qp_ctx_entry->amo_addr;
        *(__gm__ T*)amo_addr = params.value;
        dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
        sge_ctx->va = amo_addr;
    } else {
        sge_ctx->va = reinterpret_cast<uint64_t>(local_addr);
    }
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_udma_get_sge_ctx(__gm__ uint8_t* wqe_addr)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        constexpr size_t offset = sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t);
        return (wqe_addr + offset);
    } else {
        return (wqe_addr + sizeof(aclshmemi_sqe_ctx_t));
    }
}

ACLSHMEM_DEVICE void poll_cq_when_sq_overflow(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t wqe_cnt, uint32_t slot, uint32_t qp_idx)
{
    // Poll CQ if send queue is full
    constexpr uint32_t POLL_CQ_THRESHOLD = 10;
    uint32_t cur_tail = qp_ctx_entry->tail;
    if ((wqe_cnt + POLL_CQ_THRESHOLD) % shm::UDMA_SQ_BASKBLK_CNT == (cur_tail) % shm::UDMA_SQ_BASKBLK_CNT) {
        uint32_t idx =
            (cur_tail + ACLSHMEM_NUM_CQE_PER_POLL_CQ) > wqe_cnt ? wqe_cnt : cur_tail + ACLSHMEM_NUM_CQE_PER_POLL_CQ;
        (void)aclshmemi_udma_poll_cq(slot, qp_idx, idx);
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR = __gm__ aclshmemi_sqe_ctx_t*>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_ctx(
    SQE_PTR sqe_ctx, __gm__ uint8_t* remote_addr, __gm__ aclshmemi_ubmem_info_t* remote_mem_info, uint32_t cur_head,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    // Fill SQE control information (reference udma_fill_write_sqe logic).
    // Templated on SQE_PTR so the same fill logic targets either HBM (PIPE_S path)
    // or UB scratch (PIPE_MTE3 path).
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        sqe_ctx->opcode = static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE);
        // udf_flag = 1 for write with reduce; last 2 bits = 00 (NO, no ordering) for higher throughput.
        sqe_ctx->flag = 0b10100000;
    } else {
        sqe_ctx->opcode = static_cast<uint32_t>(OP_CODE);
        // Last 2 bits = 00 (NO, no ordering) for higher throughput.
        // Note: aclshmemx_udma_quiet only guarantees completion of submitted WQEs,
        // it does NOT restore inter-submission ordering — the framework currently
        // does not provide a strong-ordering primitive for this path.
        sqe_ctx->flag = 0b00100000;
    }
    sqe_ctx->nf = 0; // Need fence
    sqe_ctx->token_en = remote_mem_info->token_value_valid;
    sqe_ctx->rmt_jetty_type = remote_mem_info->rmt_jetty_type;
    sqe_ctx->owner = (cur_head & shm::UDMA_SQ_BASKBLK_CNT) == 0 ? 1 : 0; // depth: baseblk_cnt
    sqe_ctx->target_hint = remote_mem_info->target_hint;
    sqe_ctx->inline_msg_len = 0; // No inline data
    sqe_ctx->tp_id = remote_mem_info->tpn;
    sqe_ctx->sge_num = 1; // Single SGE
    sqe_ctx->rmt_jetty_or_seg_id = remote_mem_info->tid;
    sqe_ctx->rmt_token_value = remote_mem_info->rmt_token_value;
    aclshmemi_fill_reduce<T, OP_CODE>(sqe_ctx);
    // Set remote address (reference udma_fill_write_sqe logic)
    uint64_t remote_addr_value = reinterpret_cast<uint64_t>(remote_addr);
    sqe_ctx->rmt_addr_l_or_token_id = remote_addr_value & 0xFFFFFFFF;
    sqe_ctx->rmt_addr_h_or_token_value = (remote_addr_value >> 32) & 0xFFFFFFFF;
    auto rmt_eid = (__gm__ uint64_t*)(remote_mem_info->eid_addr);
    sqe_ctx->rmt_eid_l = rmt_eid[0];
    sqe_ctx->rmt_eid_h = rmt_eid[1];
    aclshmemi_fill_notify_data<T, OP_CODE>(sqe_ctx, remote_mem_info->tid, remote_mem_info->rmt_token_value, params);
}

ACLSHMEM_DEVICE __gm__ aclshmemi_udma_wq_ctx_t* aclshmemi_udma_get_qp_ctx(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
        (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
    return qp_ctx_entry;
}

ACLSHMEM_DEVICE __gm__ aclshmemi_sqe_ctx_t* aclshmemi_udma_get_sqe_ctx(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    auto sq_base_addr = qp_ctx_entry->buf_addr;
    __gm__ uint8_t* wqe_addr = (__gm__ uint8_t*)(sq_base_addr + wqe_size * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = (__gm__ aclshmemi_sqe_ctx_t*)wqe_addr;
    return sqe_ctx;
}

ACLSHMEM_DEVICE void assert_qp_params_valid(__gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry)
{
    auto depth = qp_ctx_entry->depth;
    bool valid = (shm::UDMA_SQ_BASKBLK_CNT == depth);
    if (!valid) {
        AscendC::printf("sq depth [%d] is not equal with baskblkcnt [%d].\n", depth, shm::UDMA_SQ_BASKBLK_CNT);
        trap();
    }
}

ACLSHMEM_DEVICE void assert_not_self_send(uint32_t pe)
{
    if (pe == static_cast<uint32_t>(aclshmem_my_pe())) {
        AscendC::printf("udma_post_send: pe(%d) == my_pe, self-send not allowed\n", pe);
        trap();
    }
}

// Unified UDMA slot index, same signature for both builds; only the body is compile-time gated:
//   * OFF (direct): slot == pe. relay_pe ignored, myPe/rankCount not read -- zero hot-path cost.
//   * ON (relay): slot == pe * rankCount + actualRelayPe (actualRelayPe == myPe when relay_pe == -1).
// Overflow (RED-03 / TOP-03): slot indexes the [rankCount*rankCount] tables and is bounded by
// rankCount^2 << 2^32, so the uint32_t product cannot wrap; derived byte offsets widen to 64-bit at
// the pointer arithmetic (sizeof promotes to size_t) and host table sizing uses uint64_t (SlotCount()).
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_compute_slot(uint32_t pe, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        uint32_t my_pe = static_cast<uint32_t>(aclshmemi_get_my_pe());
        uint32_t rank_count = static_cast<uint32_t>(aclshmemi_get_total_pe());
        uint32_t actual_relay_pe = (relay_pe == static_cast<uint32_t>(-1)) ? my_pe : relay_pe;
        return pe * rank_count + actual_relay_pe;
    } else {
        (void)relay_pe; // direct path: ignore relay_pe, no extra computation
        return pe;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params, uint32_t relay_pe)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    ACLSHMEM_DEBUG_FUNC(assert_not_self_send, pe);
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry = aclshmemi_udma_get_qp_ctx(udma_info, slot, qp_idx);
    auto wqe_size = qp_ctx_entry->wqe_size;
    uint32_t cur_head = qp_ctx_entry->head;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qp_ctx_entry);
    uint32_t wqe_cnt = qp_ctx_entry->wqe_cnt;
    // Poll CQ if send queue is full
    poll_cq_when_sq_overflow(qp_ctx_entry, wqe_cnt, slot, qp_idx);
    // Get memory info for this (actualPe, relayPe) slot
    __gm__ aclshmemi_ubmem_info_t* remote_mem_info =
        (__gm__ aclshmemi_ubmem_info_t*)(aclshmemi_udma_active_table(udma_info)->mem_ptr +
                                         sizeof(aclshmemi_ubmem_info_t) * slot);
    // Write SQE to HBM
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = aclshmemi_udma_get_sqe_ctx(qp_ctx_entry, cur_head, wqe_size);
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE>(sqe_ctx, remote_addr, remote_mem_info, cur_head, params);
    // Write SGE to HBM
    __gm__ aclshmemi_sge_ctx_t* sge_ctx =
        (__gm__ aclshmemi_sge_ctx_t*)(aclshmemi_udma_get_sge_ctx<OP_CODE>((__gm__ uint8_t*)sqe_ctx));
    aclshmemi_fill_sge_ctx<T, OP_CODE>(sge_ctx, message_len, local_addr, qp_ctx_entry, params);
    // WQE & SGE cache flush
    constexpr uint32_t WQE_BB_CNT = get_wqe_bb_cnt<OP_CODE>();
    dcci_cachelines((__gm__ uint8_t*)sqe_ctx, wqe_size * WQE_BB_CNT);
    cur_head += WQE_BB_CNT;
    aclshmemi_udma_post_send_update_info(cur_head, qp_ctx_entry);
    wqe_cnt++;
    qp_ctx_entry->wqe_cnt = wqe_cnt;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        dcci_cachelines((__gm__ uint8_t*)qp_ctx_entry, sizeof(aclshmemi_udma_wq_ctx_t));
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_wqe, (__gm__ uint8_t*)sqe_ctx, (uint32_t)sizeof(T));
}

ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(
    uint32_t cur_head, __gm__ aclshmemi_udma_wq_ctx_t*& qp_ctx_entry)
{
    // Ring SQ Doorbell (reference udma_update_sq_db in UDMA)
    // Note: db address is 64-bit, but we only update 32-bit value
    __gm__ uint32_t* door_bell_addr = (__gm__ uint32_t*)qp_ctx_entry->db_addr;
    st_dev(cur_head, door_bell_addr, 0);
    qp_ctx_entry->head = cur_head;
    return;
}

ACLSHMEM_DEVICE void aclshmemi_udma_copy_wqe_from_ub(
    __gm__ uint8_t* dst_gm, AscendC::LocalTensor<uint8_t>& ub_local, uint32_t copy_len, uint32_t sync_id)
{
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    gm_tensor.SetGlobalBuffer(dst_gm);
    AscendC::DataCopyExtParams copyParams{1, copy_len, 0, 0, 0};

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(gm_tensor, ub_local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

// ---- MTE3-staged WQE construction (PIPE_MTE3 path) -------------------------------
// The default PIPE_S path above writes the SQE/SGE directly to HBM via scalar stores
// + dcci_cachelines. The MTE3 path stages the full WQE block in caller-provided UB
// scratch and lands it on the SQ ring with a single DataCopyPad. Useful for hot
// loops where the per-iteration scalar->HBM bursts dominate.
//
// MTE3 path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ
// (the data-mover opcodes). FAA / CAS / WRITE_WITH_REDUCE remain on PIPE_S because
// their SGE side fills GM-resident AMO data which is incompatible with UB staging.
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send_mte3(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id, const aclshmemi_udma_params_t<T, OP_CODE>& params = {},
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    static_assert(
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_READ,
        "PIPE_MTE3 WQE path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ");

    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry = aclshmemi_udma_get_qp_ctx(udma_info, slot, qp_idx);
    auto wqe_size = qp_ctx_entry->wqe_size;
    uint32_t cur_head = qp_ctx_entry->head;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qp_ctx_entry);
    uint32_t wqe_cnt = qp_ctx_entry->wqe_cnt;
    poll_cq_when_sq_overflow(qp_ctx_entry, wqe_cnt, slot, qp_idx);

    __gm__ aclshmemi_ubmem_info_t* remote_mem_info =
        (__gm__ aclshmemi_ubmem_info_t*)(aclshmemi_udma_active_table(udma_info)->mem_ptr +
                                         sizeof(aclshmemi_ubmem_info_t) * slot);

    // Stage WQE (SQE + optional notify + SGE) in caller's UB scratch. Reuse the
    // address-space-templated fill helper so SQE field assignments are not duplicated.
    __ubuf__ aclshmemi_sqe_ctx_t* sqeUb = (__ubuf__ aclshmemi_sqe_ctx_t*)ub_scratch;
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE, __ubuf__ aclshmemi_sqe_ctx_t*>(
        sqeUb, remote_addr, remote_mem_info, cur_head, params);

    constexpr size_t SGE_OFF = (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) ?
                                   sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t) :
                                   sizeof(aclshmemi_sqe_ctx_t);
    __ubuf__ aclshmemi_sge_ctx_t* sge_ub = (__ubuf__ aclshmemi_sge_ctx_t*)((__ubuf__ uint8_t*)ub_scratch + SGE_OFF);
    // OP_CODE is restricted to non-AMO/non-REDUCE here, so the SGE only needs
    // len + va. The full FAA/CAS/REDUCE fan-out is intentionally unused in this path.
    sge_ub->len = message_len;
    sge_ub->va = reinterpret_cast<uint64_t>(local_addr);

    // Single-shot DMA UB -> SQ ring entry. The helper uses the same S->MTE3 and
    // MTE3->S ordering as the RDMA backend before ringing the SQ doorbell.
    __gm__ aclshmemi_sqe_ctx_t* sqe_gm = aclshmemi_udma_get_sqe_ctx(qp_ctx_entry, cur_head, wqe_size);
    constexpr uint32_t WQE_BB_CNT = get_wqe_bb_cnt<OP_CODE>();
    AscendC::LocalTensor<uint8_t> ub_local;
    ub_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_local.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_scratch);
    aclshmemi_udma_copy_wqe_from_ub(
        (__gm__ uint8_t*)sqe_gm, ub_local, static_cast<uint32_t>(wqe_size * WQE_BB_CNT), sync_id);

    cur_head += WQE_BB_CNT;
    aclshmemi_udma_post_send_update_info(cur_head, qp_ctx_entry);
    wqe_cnt++;
    qp_ctx_entry->wqe_cnt = wqe_cnt;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        dcci_cachelines((__gm__ uint8_t*)qp_ctx_entry, sizeof(aclshmemi_udma_wq_ctx_t));
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_wqe, (__gm__ uint8_t*)sqe_gm, (uint32_t)sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write_mte3(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, ub_scratch, sync_id, {}, relay_pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify_mte3(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>& params,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id)
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, ub_scratch, sync_id, params);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, {}, relay_pe);
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, params);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_read(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t src_pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
        reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), src_pe,
        qp_idx, message_len, {}, relay_pe);
}

ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    uint32_t qp_num = udma_info->qp_num;
    uint32_t actual_pe = static_cast<uint32_t>(pe);
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        uint32_t rank_count = static_cast<uint32_t>(aclshmemi_get_total_pe());
        // Drain every (pe, relay_pe) slot. Slots that never received a post_send have wqe_cnt==0
        // and poll_cq returns immediately. UDMA does not support self ops (pe == myPe), so the
        // (pe, pe) diagonal is never posted to and host leaves it unfilled -- skip it.
        for (uint32_t relay_pe = 0; relay_pe < rank_count; ++relay_pe) {
            if (relay_pe == actual_pe) {
                continue;
            }
            uint32_t slot = actual_pe * rank_count + relay_pe;
            __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
                (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + 0) * sizeof(aclshmemi_udma_wq_ctx_t));
            uint32_t wqe_cnt = qp_ctx_entry->wqe_cnt;
            if (wqe_cnt == 0) {
                continue;
            }
            aclshmemi_udma_poll_cq(slot, 0, wqe_cnt);
        }
    } else {
        // Original direct path: poll only the single slot == pe. Does not read rankCount.
        __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
            (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (actual_pe * qp_num + 0) * sizeof(aclshmemi_udma_wq_ctx_t));
        uint32_t wqe_cnt = qp_ctx_entry->wqe_cnt;
        aclshmemi_udma_poll_cq(actual_pe, 0, wqe_cnt);
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(src, pe);
            // For UDMA_OP_READ, the SQE's "remote_addr" slot carries the src (remote)
            // and "local_addr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_get_nbi(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    aclshmemx_udma_get_nbi<T, WQE_PIPE>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst, pe);
            aclshmemi_udma_write_mte3<T>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto dst_phy_addr = (__gm__ T*)dst.GetPhyAddr();
        auto src_phy_addr = (__gm__ T*)src.GetPhyAddr();
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst_phy_addr, pe);
            aclshmemi_udma_write_mte3<T>(
                (__gm__ T*)ptr, src_phy_addr, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf.GetPhyAddr()), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi(dst_phy_addr, src_phy_addr, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay put: pe=%d relay_pe=%d\n", pe, relay_pe);
        int rank_count = aclshmemi_get_total_pe();
        int my_pe = aclshmemi_get_my_pe();
        if (pe < 0 || pe >= rank_count || relay_pe < 0 || relay_pe >= rank_count || pe == relay_pe || pe == my_pe ||
            relay_pe == my_pe) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "udma relay put: invalid pe=%d relay_pe=%d (myPe=%d rankCount=%d); "
                "require 0<=actual,relay<rankCount, actual!=relay, neither equals myPe\n",
                pe, relay_pe, my_pe, rank_count);
            return;
        }
        auto ptr = aclshmem_ptr(dst, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            aclshmemi_udma_write_mte3<uint8_t>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_write<uint8_t>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemx_udma_relay_put_nbi<T, WQE_PIPE>(
            (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
            elem_size, pe, relay_pe, sync_id);
    } else {
        (void)dst;
        (void)src;
        (void)buf;
        (void)elem_size;
        (void)pe;
        (void)relay_pe;
        (void)sync_id;
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay get: pe=%d relay_pe=%d\n", pe, relay_pe);
        int rank_count = aclshmemi_get_total_pe();
        int my_pe = aclshmemi_get_my_pe();
        if (pe < 0 || pe >= rank_count || relay_pe < 0 || relay_pe >= rank_count || pe == relay_pe || pe == my_pe ||
            relay_pe == my_pe) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "udma relay get: invalid pe=%d relay_pe=%d (myPe=%d rankCount=%d); "
                "require 0<=actual,relay<rankCount, actual!=relay, neither equals myPe\n",
                pe, relay_pe, my_pe, rank_count);
            return;
        }
        auto ptr = aclshmem_ptr(src, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            // For UDMA_OP_READ the SQE's "remote_addr" slot carries src (remote) and
            // "local_addr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, {}, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_read<uint8_t>(
                (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemx_udma_relay_get_nbi<T, WQE_PIPE>(
            (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
            elem_size, pe, relay_pe, sync_id);
    } else {
        (void)dst;
        (void)src;
        (void)buf;
        (void)elem_size;
        (void)pe;
        (void)relay_pe;
        (void)sync_id;
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
            .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
        aclshmemi_udma_write_notify<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
            (__gm__ T*)ptr, src, pe, 0, elem_size * sizeof(T), signal_params);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

// Buf-taking overload. PIPE_MTE3 (default) stages the WRITE_WITH_NOTIFY WQE in the
// caller-provided UB scratch (size one full WQE block; wqe_size * 2 bytes, 128 B is
// safe for the current SQ basebk_shift). PIPE_S falls through to the no-buf path and
// ignores buf/sync_id, mirroring the put_nbi(buf) overload's S/MTE3 split.
template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    __ubuf__ uint8_t* buf, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst, pe);
            auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
            aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
                .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
            aclshmemi_udma_write_notify_mte3<T>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T), signal_params, buf, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemx_udma_put_signal_nbi<T>(dst, src, elem_size, sig_addr, signal, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr bool aclshmemi_udma_check_atomic_len()
{
    size_t atomic_len = sizeof(T);
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) {
        if (atomic_len != 4 && atomic_len != 8) {
            return false;
        }
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) {
        if (atomic_len != 4 && atomic_len != 8 && atomic_len != 16) {
            return false;
        }
    }
    return true;
}

ACLSHMEM_DEVICE uint64_t aclshmemi_udma_get_amo_addr(uint32_t pe, uint32_t qp_idx)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    uint32_t qp_num = udma_info->qp_num;
    // Atomic ops only run on the direct path, so reuse the same slot as aclshmemi_udma_post_send's
    // default (relay = self). Gate the layout exactly like compute_slot: OFF is the original
    // single-dimension table where slot = pe; ON is the N*N table where the direct bucket
    // is (actual=pe, relay=self) = pe*N + myPe. Using the N*N formula on OFF would read amo_addr
    // from the wrong (out-of-bounds) slot and corrupt atomic fetch data.
    uint32_t slot = aclshmemi_udma_compute_slot(pe);
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
        (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
    auto amo_addr = qp_ctx_entry->amo_addr;
    return amo_addr;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_amo_addr_value(uint64_t amo_addr)
{
    dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
    __gm__ T* fetch_addr = reinterpret_cast<__gm__ T*>(amo_addr);
    T fetch_data = *fetch_addr;
    return fetch_data;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_atomic_fetch_data(uint32_t pe, uint32_t qp_idx)
{
    auto amo_addr = aclshmemi_udma_get_amo_addr(pe, qp_idx);
    return aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic add\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        if constexpr (AscendC::IsSameType<T, float>::value) { // float使用write with reduce逻辑处理
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        } else {
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic fetch add\n", sizeof(T));
        }

        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);

        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic compare swap\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> cas_params{
            .value = value, .cond = cond};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), cas_params);
        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T* dst, int32_t pe)
{
    return aclshmemx_udma_atomic_fetch_add<T>(dst, 0, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_set timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_swap timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T* dst, int32_t pe)
{
    return aclshmemx_udma_atomic_fetch_add<T>(dst, 1, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T* dst, int32_t pe)
{
    aclshmemx_udma_atomic_add<T>(dst, 1, pe);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_and timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_and timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_or timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_or timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_xor timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_xor timeout!\n");
}

#endif
