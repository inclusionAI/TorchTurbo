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
#ifndef SHMEMI_DEVICE_UDMA_H
#define SHMEMI_DEVICE_UDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

enum class aclshmemi_udma_opcode_t : uint32_t {
    UDMA_OP_SEND = 0,
    UDMA_OP_SEND_WITH_IMM,
    UDMA_OP_SEND_WITH_INV,
    UDMA_OP_WRITE,
    UDMA_OP_WRITE_WITH_IMM,
    UDMA_OP_WRITE_WITH_NOTIFY,
    UDMA_OP_READ,
    UDMA_OP_CAS,
    UDMA_OP_ATOMIC_SWAP,
    UDMA_OP_ATOMIC_STORE,
    UDMA_OP_ATOMIC_LOAD,
    UDMA_OPCODE_FAA = 0xb,
    UDMA_OP_WRITE_WITH_REDUCE = 0x10, // self-defined opcode, will be converted to UDMA_OP_WRITE in actual usage
    UDMA_OPCODE_NOP = 0x11
};

// Queue / memory-region base-address table. All arrays are indexed by "slot"
// (see aclshmemi_udma_compute_slot); the slot count differs per path (see below).
struct aclshmemi_udma_qp_table_t {
    uint64_t sq_ptr;  // send queue address array,             [slot_count][qp_num]
    uint64_t rq_ptr;  // receive queue address array,          [slot_count][qp_num]
    uint64_t scq_ptr; // send completion queue address array,  [slot_count][qp_num]
    uint64_t rcq_ptr; // receive completion queue address array,[slot_count][qp_num]
    uint64_t mem_ptr; // memory region array,                  [slot_count]
};

struct aclshmemi_aiv_udma_info_t {
    uint32_t qp_num; // number of QP per connection
    // The two paths are mutually exclusive per build (ACLSHMEM_RELAY_SUPPORT), so they share
    // storage. Pick the active table via aclshmemi_udma_active_table().
    //   * direct table: slot_count == PE_NUM,          slot == pe
    //   * relay table:  slot_count == PE_NUM*rank_count, slot == pe * rank_count + relay_pe
    union {
        aclshmemi_udma_qp_table_t direct;
        aclshmemi_udma_qp_table_t relay;
    };
};

struct aclshmemi_ubmem_info_t {
    bool token_value_valid;      // token_en 表示是否使能token
    uint32_t rmt_jetty_type : 2; // 表示远端jetty的类型
    uint8_t target_hint;         // jettygrp场景使用
    uint32_t tpn;                // 对应着tp_id 区分传输层是简易传输层还是完整传输层
    uint32_t tid;                // 对应着SQE的rmt_jetty_or_seg_id，来源是udma_seg->tid;
    uint32_t rmt_token_value;    // 对应着SQE的rmt_token_value，来源是udma_seg->token_value.token;
    uint32_t len;
    uint64_t addr; // 来源urma_sge的addr，对应SQE的rmt_addr_l_or_token_id，rmt_addr_h_or_token_value
    uint64_t eid_addr;
};

enum class aclshmemi_udma_db_mode_t : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct aclshmemi_udma_wq_ctx_t {
    uint32_t wqn;      // work queue number
    uint64_t buf_addr; // start address of ring buffer
    uint32_t wqe_size; // size in bytes of each WQE
    uint32_t depth;    // depth of ring buffer
    uint32_t head;     // work queue head (Producer Index)
    uint32_t tail;     // work queue tail (Consumer Index)
    aclshmemi_udma_db_mode_t db_mode;
    uint64_t db_addr;  // doorbell address
    uint32_t sl;       // service level
    uint32_t wqe_cnt;  // wqe count
    uint64_t amo_addr; // amo address to store fetch data
};

struct aclshmemi_udma_cq_ctx_t {
    uint32_t cqn;      // completion queue number
    uint64_t buf_addr; // start address of ring buffer
    uint32_t cqe_size; // size in bytes of each CQE
    uint32_t depth;    // depth of ring buffer
    uint32_t head;     // completion queue head (Producer Index)
    uint32_t tail;     // completion queue tail (Consumer Index)
    aclshmemi_udma_db_mode_t db_mode;
    uint64_t db_addr; // doorbell address
};

struct aclshmemi_sqe_ctx_t { // 对应着 ACLSHMEMwqeCtx
    /* byte 4 */
    uint32_t sqe_bb_idx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t token_en : 1;
    uint32_t rmt_jetty_type : 2;
    uint32_t owner : 1;
    /* byte 8 */
    uint32_t target_hint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inline_msg_len : 10;
    /* byte 12 */
    uint32_t tp_id : 24;
    uint32_t sge_num : 8;
    /* byte 16 */
    uint32_t rmt_jetty_or_seg_id : 20;
    uint32_t rsv2 : 12;
    /* byte 20 - 32 */
    // For better perf, use 2 uint64_t to represent int8[16]
    uint64_t rmt_eid_l;
    uint64_t rmt_eid_h;
    /* byte 36 */
    uint32_t rmt_token_value;
    /* byte 40 */
    uint32_t udf_type : 8;
    uint32_t reduce_data_type : 4;
    uint32_t reduce_opcode : 4;
    uint32_t rsv3 : 16;
    /* byte 44 - 48*/
    uint32_t rmt_addr_l_or_token_id;
    uint32_t rmt_addr_h_or_token_value;
};

struct aclshmemi_sge_ctx_t { // 对应着ACLSHMEMsegCtx
    uint32_t len;
    uint32_t token_id;
    uint64_t va;
};

struct aclshmemi_notify_ctx_t {
    /* byte 48 - 52 */
    uint32_t notify_token_id : 20;
    uint32_t rsv : 12;
    /* byte 52 - 56 */
    uint32_t notify_token_value;
    /* byte 56 - 60 */
    uint32_t notify_addr_l;
    /* byte 60 - 64 */
    uint32_t notify_addr_h;
    /* byte 64 - 68 */
    uint32_t notify_data_l;
    /* byte 68 - 72 */
    uint32_t notify_data_h;
    /* byte 72 - 80 */
    uint32_t rsv2[2];
};

struct aclshmemi_jfc_cqe_ctx_t { // 对应ACLSHMEMcqeCtx
    /* DW0 */
    uint32_t s_r : 1;
    uint32_t is_jetty : 1;
    uint32_t owner : 1;
    uint32_t inline_en : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    /* DW1 */
    uint32_t entry_idx : 16;
    uint32_t local_num_l : 16;
    /* DW2 */
    uint32_t local_num_h : 4;
    uint32_t rmt_idx : 20;
    uint32_t rsv1 : 8;
    /* DW3 */
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    /* DW4 */
    uint32_t byte_cnt;
    /* DW5 ~ DW6 */
    uint32_t user_data_l;
    uint32_t user_data_h;
    /* DW7 ~ DW10 */
    uint32_t rmt_eid[4];
    /* DW11 ~ DW12 */
    uint32_t data_l;
    uint32_t data_h;
    /* DW13 ~ DW15 */
    uint32_t inline_data[3];
};

struct aclshmemi_udma_device_meta_t {
    uint32_t entity_id;
    uint32_t rank_id;
    uint32_t rank_size;
    uint32_t extra_context_size;
    uint64_t symmetric_size;
    uint64_t qp_info_address; // 对应着aclshmem_aiv_udma_info_t
    uint64_t reserved[12];    // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

struct default_op_tag_t {}; // 默认类别的标记
struct atomic_op_tag_t {};  // atomic语义类别的标记
struct signal_op_tag_t {};  // 信号操作语义类别的标记
template <aclshmemi_udma_opcode_t OP_CODE>
struct aclshmemi_op_category_t {
    using type = default_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_CAS> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> {
    using type = signal_op_tag_t;
};

template <typename T, typename OP_TAG>
struct aclshmemi_udma_params_impl_t {};

template <typename T>
struct aclshmemi_udma_params_impl_t<T, atomic_op_tag_t> {
    T value;
    T cond;
};

template <typename T>
struct aclshmemi_udma_params_impl_t<T, signal_op_tag_t> {
    __gm__ uint64_t* sig_addr;
    uint64_t signal;
};

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
using aclshmemi_udma_params_t = aclshmemi_udma_params_impl_t<T, typename aclshmemi_op_category_t<OP_CODE>::type>;

ACLSHMEM_DEVICE __gm__ aclshmemi_aiv_udma_info_t* aclshmemi_udma_qp_info_fetch();

/**
 * @brief UDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @param slot                   [in] queue slot index (see aclshmemi_udma_compute_slot): direct path
 *                                    slot == pe; relay path slot == pe * rank_count + relay_pe
 * @param qp_idx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param idx                    [in] expect completion queue consumer index after polling
 */
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t slot, uint32_t qp_idx, uint32_t idx);

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(
    uint32_t cur_tail, uint32_t qp_idx, __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx_entry,
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx_entry);

/**
 * @brief AIV direct UDMA helper function for post send, prepare WQE and ring doorbell.
 *
 * @param remote_addr             [in] address in remote HBM
 * @param local_addr              [in] address in lcoal HBM
 * @param pe                     [in] destination PE ID (actual destination)
 * @param qp_idx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len             [in] message length in Bytes
 * @param params                 [in] extra parameters
 * @param relay_pe               [in] PE whose port EID is used as the WQE rmt_eid. Defaults to `pe`
 *                                    (direct path). When != pe, the SQE rmt_eid is the actual_pe's
 *                                    local EID toward relay_pe, so the fabric routes via relay_pe.
 */
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params = {}, uint32_t relay_pe = static_cast<uint32_t>(-1));

ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(
    uint32_t cur_head, __gm__ aclshmemi_udma_wq_ctx_t*& qp_ctx_entry);

/**
 * @brief Asynchronous UDMA Write function.
 *
 * @param dest_dma_addr            [in] destination address in remote HBM
 * @param src_dma_addr             [in] source address in local HBM
 * @param pe                     [in] destination PE ID
 * @param qp_idx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len             [in] message length in Bytes
 * @param relay_pe               [in] PE whose port EID is used to forward the packet. Defaults to
 *                                    `pe` (direct path).
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe = static_cast<uint32_t>(-1));

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params = {});

/**
 * @brief Asynchronous UDMA READ function.
 *
 * @param dest_dma_addr            [in] destination address in local HBM
 * @param src_dma_addr             [in] source address in remote HBM
 * @param src_pe                  [in] source PE ID
 * @param qp_idx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len             [in] message length in Bytes
 * @param relay_pe               [in] PE whose port EID is used to forward the packet. Defaults to
 *                                    `src_pe` (direct path).
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_read(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t src_pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe = static_cast<uint32_t>(-1));

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);

#endif
