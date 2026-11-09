/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SDMA_H
#define SHMEMI_DEVICE_SDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

struct sdma_config_t {
    uint64_t block_bytes;
    uint64_t per_core_bytes;
    uint32_t queue_num;
    uint32_t iter_num;
};

struct workspace_layout_t {
    __gm__ uint8_t* send_workspace;
    __gm__ uint8_t* recv_workspace;
    __gm__ uint8_t* remote_recv_workspace;
};

struct stars_channel_flag_info_t {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56]; // 对齐到64字节
};

struct stars_channel_info_t {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint64_t sq_base;        // SQ缓冲区基地址
    uint64_t sq_reg_base;    // SQ寄存器基地址
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4]; // 对齐到64字节
};

struct stars_sqe_header_t {
    uint8_t type : 6;
    uint16_t res1 : 10;
    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;
};

struct stars_sdma_sqe_t {
    stars_sqe_header_t header;
    /**** 8 bytpes ****/

    uint32_t res3;
    /* *******12 bytes********* */

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;
    /**** 16 bytpes ****/

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;
    /**** 20 bytpes ****/

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    /**** 24 bytpes ****/

    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;
    /**** 32 bytpes ****/

    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;
    /**** 48 bytpes ****/

    uint8_t link_type;
    uint8_t resvered[3];
    uint32_t reslast[3];
    /**** 64 bytpes ****/
};

struct stars_notify_sqe_t {
    stars_sqe_header_t header;
    /**** 8 bytpes ****/

    uint32_t notify_id : 13;
    uint32_t res2 : 19;

    uint16_t res3;
    uint8_t kernel_credit;
    uint8_t res4;
    /**** 16 bytpes ****/

    uint32_t timeout;
    uint32_t res5[11];
    /**** 64 bytpes ****/
};

enum class ACLSHMEMCMOTYPE : uint32_t {
    CMO_TYPE_PREFETCH = 6,  // Load data from memory into l2 cache
    CMO_TYPE_WRITEBACK,     // Flush modified data from l2 cache to main memory while retaining a copy in cache
    CMO_TYPE_INVALID,       // Discard the data block in l2 cache, making it invalid
    CMO_TYPE_FLUSH,         // Forcefully write data from l2 cache to main memory and remove it from cache
    CMO_TYPE_MAX,
};

// STARS SQE layout is selected at compile time by __NPU_ARCH__.
// Ascend950 (dav-3510) uses STARS v2 layout and SQ tail doorbell offset 0.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr bool ACLSHMEM_STARS_V2_LAYOUT = true;
constexpr uint32_t ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x0;
#else
constexpr bool ACLSHMEM_STARS_V2_LAYOUT = false;
constexpr uint32_t ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x8;
#endif

struct stars_v2_sqe_header_t {
    uint8_t type : 6;
    uint8_t l1_lock : 1;
    uint8_t l1_unlock : 1;
    uint8_t ie : 1;
    uint8_t pre_p : 1;
    uint8_t post_p : 1;
    uint8_t wr_cqe : 1;
    uint8_t ptr_mode : 1;
    uint8_t rtt_mode : 1;
    uint8_t head_update : 1;
    uint8_t reserved : 1;
    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;
};

struct stars_v2_sdma_cmo_sqe_t {
    stars_v2_sqe_header_t header;
    /********8 ~ 15 bytes**********/
    uint32_t res3;
    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;
    /********16 ~ 19 bytes**********/
    uint32_t opcode : 8;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t stride : 2;
    uint32_t ie2 : 1;
    uint32_t comp_en : 1;
    uint32_t res6 : 14;
    /********20 ~ 23 bytes**********/
    uint16_t sqe_id;
    uint8_t mpam_partid;
    uint8_t mpamns : 1;
    uint8_t pmg : 2;
    uint8_t qos : 4;
    uint8_t res7 : 1;
    /********24 ~ 31 bytes**********/
    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;
    /********32 ~ 47 bytes**********/
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;
    /********48 ~ 63 bytes**********/
    uint32_t length;
    uint32_t src_stride_len;
    uint32_t dst_stride_len;
    uint32_t stride_num;
};

struct stars_sdma_cmo_sqe_t {
    uint8_t type : 6;
    uint8_t l1_lock : 1;
    uint8_t l1_unlock : 1;

    uint8_t ie : 2;
    uint8_t pre_p : 2;
    uint8_t post_p : 2;
    uint8_t wr_cqe : 1;
    uint8_t reserved : 1;

    uint16_t block_dim;

    uint16_t rt_streamid;
    uint16_t task_id;

    uint32_t res3;
    /********12 bytes**********/

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;
    /********16 bytes**********/

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t d2d_offset_flag : 1;
    uint32_t res6 : 3;
    /********20 bytes**********/

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;
    /********28 bytes**********/

    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    uint32_t src_offset_low;
    uint32_t dst_offset_low;
    uint16_t src_offset_high;
    uint16_t dst_offset_high;
    uint32_t res_last[1];
    /********64 bytes**********/
};

ACLSHMEM_DEVICE void aclshmemi_stars_submit_notify_record(AscendC::LocalTensor<uint32_t> &tmp_local,
                                                        AscendC::TEventID sync_id);

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_data_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     __gm__ uint8_t *send_buffer, __gm__ uint8_t *recv_buffer,
                                                     const sdma_config_t &config, uint32_t *sq_tail);

ACLSHMEM_DEVICE void aclshmemi_cmo_submit_data_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                    __gm__ uint8_t *src, uint32_t size, ACLSHMEMCMOTYPE cmo_type, uint32_t *sq_tail);

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_flag_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     const workspace_layout_t &layout, 
                                                     AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id);

ACLSHMEM_DEVICE void aclshmemi_sdma_poll_for_completion(const workspace_layout_t &layout, 
                                                        AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id);

/**
 * @brief AIV direct STARS helper function for post send, prepare SQE and ring doorbell.
 *
 * @param recv_buffer               [in] Destination buffer for received data
 * @param send_buffer               [in] Source buffer for data to be sent
 * @param message_len               [in] message length in Bytes
 * @param tmp_local                 [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
ACLSHMEM_DEVICE void aclshmemi_sdma_post_send(__gm__ uint8_t *recv_buffer, __gm__ uint8_t *send_buffer,
                                              uint64_t message_len, AscendC::LocalTensor<uint32_t> &tmp_local,
                                              uint32_t sync_id);

/**
 * @brief AIV direct STARS helper function for cache manager operation, prepare SQE and ring doorbell.
 *
 * @param src                       [in] Address of device memory to operate on.
 * @param size                      [in] Size of memory in bytes.
 * @param cmo_type                  [in] Cache operation type. Currently only supports prefetch.
 * @param tmp_local                 [in] Temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
ACLSHMEM_DEVICE void aclshmemi_cmo_async(__gm__ uint8_t* src,
                                uint32_t size,
                                ACLSHMEMCMOTYPE cmo_type, 
                                AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id);

/**
 * @brief Calculate the base address of batch write channel info from device state.
 *        This function retrieves the device state and calculates the base address
 *        by adding the offset of stars_channel_flag_info_t to the SDMA workspace address.
 *
 * @return                          Pointer to the base of batch write channel info.
 */
ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_sdma_get_channel_base();

#endif
