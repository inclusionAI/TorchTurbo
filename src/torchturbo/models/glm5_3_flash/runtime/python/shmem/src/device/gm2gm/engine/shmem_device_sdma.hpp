/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SDMA_HPP
#define SHMEM_DEVICE_SDMA_HPP

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"
#include "device/shmem_def.h"
#include "shmemi_device_sdma.h"

constexpr uint8_t ACLSHMEM_SQE_TYPE_SDMA = 11;
constexpr uint8_t ACLSHMEM_SQE_TYPE_NOTIFY_RECORD = 6;
constexpr uint8_t ACLSHMEM_STARS_DEFAULT_KERNEL_CREDIT = 240U;
constexpr uint8_t ACLSHMEM_DEFAULT_KERNEL_CREDIT = 254U;
constexpr uint8_t ACLSHMEM_STARS_NEVER_TIMEOUT_KERNEL_CREDIT = 255U;

template <typename T>
ACLSHMEM_DEVICE void copy_gm_to_gm(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t size,
                                   AscendC::LocalTensor<T> &tmp_local, uint32_t sync_id)
{
    AscendC::GlobalTensor<T> gm_src;
    AscendC::GlobalTensor<T> gm_dst;
    gm_src.SetGlobalBuffer((__gm__ T *)src, size);
    gm_dst.SetGlobalBuffer((__gm__ T *)dst, size);

    uint32_t cp_len = size * sizeof(T);
    AscendC::DataCopyExtParams cp_params{1, cp_len, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    // gm2ub
    AscendC::DataCopyPad(tmp_local, gm_src, cp_params, pad_params);
    AscendC::PipeBarrier<PIPE_ALL>();
    // ub2gm
    AscendC::DataCopyPad(gm_dst, tmp_local, cp_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_set_value(__gm__ uint8_t *addr, T x, AscendC::LocalTensor<T> &tmp_local,
                                         uint32_t sync_id)
{
    AscendC::GlobalTensor<T> gm_dst;
    gm_dst.SetGlobalBuffer((__gm__ T *)addr);
    tmp_local.SetValue(0, x);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp_out_params{1, sizeof(T), 0, 0, 0};
    // ub2gm
    AscendC::DataCopyPad(gm_dst, tmp_local, cp_out_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
}

ACLSHMEM_DEVICE void aclshmemi_fill_sdma_sqe(__gm__ stars_channel_info_t* channel_info, __gm__ uint8_t* src,
                                             __gm__ uint8_t* dst, uint32_t length, uint32_t sq_tail, uint32_t task_id)
{
    __gm__ stars_sdma_sqe_t *sqe = (__gm__ stars_sdma_sqe_t *)((channel_info->sq_base));
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->header.type = ACLSHMEM_SQE_TYPE_SDMA;
    sqe->header.block_dim = 0;
    sqe->header.rt_streamid = channel_info->stream_id;
    sqe->header.task_id = task_id;

    sqe->kernel_credit = ACLSHMEM_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->ptr_mode = 0;

    sqe->opcode = 0;
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6; // 6 is HCCL QoS
    sqe->partid = 0U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);

    sqe->src_addr_low = static_cast<uint32_t>(src_addr & 0xFFFFFFFF);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFF);
    sqe->dst_addr_low = static_cast<uint32_t>(dst_addr & 0xFFFFFFFF);
    sqe->dst_addr_high = static_cast<uint32_t>((dst_addr >> 32) & 0xFFFFFFFF);
    sqe->link_type = static_cast<uint8_t>(255U);
}

ACLSHMEM_DEVICE void aclshmemi_fill_notify_record_sqe(__gm__ stars_channel_info_t *channel_info, uint32_t sq_tail,
                                                    uint32_t task_id, uint32_t notify_id)
{
    __gm__ stars_notify_sqe_t *sqe = (__gm__ stars_notify_sqe_t *)((channel_info->sq_base));
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->header.type = ACLSHMEM_SQE_TYPE_NOTIFY_RECORD;
    sqe->header.block_dim = 0;
    sqe->header.rt_streamid = channel_info->stream_id;
    sqe->header.task_id = task_id;
    sqe->notify_id = notify_id;
    sqe->kernel_credit = ACLSHMEM_DEFAULT_KERNEL_CREDIT;
}

ACLSHMEM_DEVICE void aclshmemi_fill_stars_v2_cmo_sqe(__gm__ stars_channel_info_t* channel_info, __gm__ uint8_t* src,
                                                     uint32_t length, uint32_t opcode, uint32_t sq_tail,
                                                     uint32_t task_id)
{
    __gm__ stars_v2_sdma_cmo_sqe_t *sqe = (__gm__ stars_v2_sdma_cmo_sqe_t *)((channel_info->sq_base));
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->header.type = ACLSHMEM_SQE_TYPE_SDMA;
    sqe->header.l1_lock = 0U;
    sqe->header.l1_unlock = 0U;
    sqe->header.ie = 0U;
    sqe->header.pre_p = 0U;
    sqe->header.post_p = 0U;
    sqe->header.wr_cqe = 1U;
    sqe->header.ptr_mode = 0U;
    sqe->header.rtt_mode = 0U;
    sqe->header.head_update = 0U;
    sqe->header.reserved = 0U;
    sqe->header.block_dim = 0U;
    sqe->header.rt_streamid = static_cast<uint16_t>(channel_info->stream_id);
    // task_id is derived from SQ tail/head distance and is bounded by SQ depth.
    sqe->header.task_id = static_cast<uint16_t>(task_id);

    sqe->res3 = 0U;
    sqe->res4 = 0U;
    sqe->kernel_credit = ACLSHMEM_DEFAULT_KERNEL_CREDIT;
    sqe->ptr_mode = 0U;
    sqe->res5 = 0U;

    sqe->opcode = opcode;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->sro = 0U;
    sqe->dro = 0U;
    sqe->stride = 0U;
    sqe->ie2 = 0U;
    sqe->comp_en = 0U;
    sqe->res6 = 0U;

    sqe->sqe_id = 0U;
    sqe->mpam_partid = 0U;
    sqe->mpamns = 0U;
    sqe->pmg = 0U;
    sqe->qos = 6U; // HCCL QoS
    sqe->res7 = 0U;

    sqe->src_streamid = 0U;
    sqe->src_sub_streamid = 0U;
    sqe->dst_streamid = 0U;
    sqe->dst_sub_streamid = 0U;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    sqe->src_addr_low = static_cast<uint32_t>(src_addr & 0x00000000FFFFFFFFU);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr & 0xFFFFFFFF00000000U) >> 32);
    sqe->dst_addr_low = 0U;
    sqe->dst_addr_high = 0U;

    sqe->length = length;
    sqe->src_stride_len = 0U;
    sqe->dst_stride_len = 0U;
    sqe->stride_num = 0U;
}

ACLSHMEM_DEVICE void aclshmemi_fill_cmo_sqe(__gm__ stars_channel_info_t* channel_info, __gm__ uint8_t* src,
                                                    uint32_t length, uint32_t opcode, uint32_t sq_tail, uint32_t task_id)
{
    __gm__ stars_sdma_cmo_sqe_t *sqe = (__gm__ stars_sdma_cmo_sqe_t *)((channel_info->sq_base));
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->type = ACLSHMEM_SQE_TYPE_SDMA;
    sqe->ie = 0U;
    sqe->pre_p = 0U;
    sqe->post_p = 0U;
    sqe->wr_cqe = 0U;
    sqe->rt_streamid = static_cast<uint16_t>(channel_info->stream_id);
    // task_id is derived from SQ tail/head distance and is bounded by SQ depth.
    sqe->task_id = static_cast<uint16_t>(task_id);
    sqe->kernel_credit = ACLSHMEM_DEFAULT_KERNEL_CREDIT;

    // for prefetch
    sqe->opcode = opcode;   // 0x6U is cmoTask for prefetch Load
    sqe->ptr_mode = 0U;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);

    sqe->src_addr_low  = static_cast<uint32_t>(src_addr & 0x00000000FFFFFFFFU);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr & 0xFFFFFFFF00000000U) >> 32);
    // sdmaCmo task not preempting other resources
    sqe->qos  = 6U; // only CHIP_910_B_93 sdma  task qos: 6; partid: 63
    sqe->partid = 63U;

    sqe->src_streamid = 0U;    // get sid and ssid from sq, leave 0 here
    sqe->dst_streamid = 0U;
    sqe->src_sub_streamid = 0U;
    sqe->dst_sub_streamid = 0U;
    sqe->ie2  = 0U;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns  = 1U;
    sqe->dns  = 1U;
    sqe->sro  = 0U;
    sqe->dro  = 0U;
    sqe->mpam = 0U;
    sqe->res3 = 0U;
    sqe->res4 = 0U;
    sqe->res5 = 0U;
    sqe->res6 = 0U;
    sqe->d2d_offset_flag = 0U;
    sqe->src_offset_low = 0U;
    sqe->dst_offset_low = 0U;
    sqe->src_offset_high = 0U;
    sqe->dst_offset_high = 0U;
}

ACLSHMEM_DEVICE void aclshmemi_stars_submit_notify_record(AscendC::LocalTensor<uint32_t> &tmp_local,
                                                        AscendC::TEventID sync_id)
{
    const auto cur_block_idx = AscendC::GetBlockIdx();
    const uint32_t queue_num = 1;

    __gm__ uint8_t *channel_base = aclshmemi_sdma_get_channel_base();
    __gm__ stars_channel_info_t *batch_write_channel_info = (__gm__ stars_channel_info_t *)(channel_base)
                                                            + cur_block_idx * queue_num;
    __gm__ uint32_t *notify_addr = reinterpret_cast<__gm__ uint32_t *>(channel_base - sizeof(stars_channel_flag_info_t)
                                + ACLSHMEM_STARS_NOTIFY_ADDR_OFFSET);
    
    for (uint32_t queue_id = 0U; queue_id < queue_num; ++queue_id) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;
        dcci_cacheline(((__gm__ uint8_t *)channel_info) + 4);
        uint32_t sq_tail =  *((__gm__ uint32_t *)(((__gm__ uint8_t *)channel_info) + 4));

        aclshmemi_fill_notify_record_sqe(channel_info, sq_tail, sq_tail - channel_info->sq_head, notify_addr[cur_block_idx * queue_num + queue_id]);
        
        sq_tail = (sq_tail + 1) % (channel_info->sq_depth);

        // Flush data cache to ensure all SQEs are written back to HBM
        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), sizeof(stars_notify_sqe_t));
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(write_info);

        // Ring Doorbell
        aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET, sq_tail, tmp_local, sync_id);
        aclshmemi_set_value<uint32_t>(((__gm__ uint8_t *)channel_info) + 4, sq_tail, tmp_local, sync_id);
    }
}

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_data_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     __gm__ uint8_t *send_buffer, __gm__ uint8_t *recv_buffer,
                                                     const sdma_config_t &config, uint32_t *sq_tail)
{
    for (uint32_t idx = 0U; idx < config.iter_num; ++idx) {
        uint32_t queue_idx = idx % config.queue_num;
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_idx;

        // This transfer data length
        uint32_t transfer_bytes = config.block_bytes;
        if (idx == config.iter_num - 1) {
            transfer_bytes = config.per_core_bytes - idx * config.block_bytes;
        }

        __gm__ uint8_t *src_addr = send_buffer + idx * config.block_bytes;
        __gm__ uint8_t *dst_addr = recv_buffer + idx * config.block_bytes;

        aclshmemi_fill_sdma_sqe(channel_info, src_addr, dst_addr, transfer_bytes, sq_tail[queue_idx],
            sq_tail[queue_idx] - channel_info->sq_head);

        sq_tail[queue_idx] = (sq_tail[queue_idx] + 1) % (channel_info->sq_depth);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_flag_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     const workspace_layout_t &layout, 
                                                     AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id)
{
    const uint32_t queue_num = 1;
    for (uint32_t queue_id = 0U; queue_id < queue_num; ++queue_id) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;
        dcci_cachelines((__gm__ uint8_t *)channel_info + 4, 4);
        uint32_t sq_tail = *((__gm__ uint32_t *)((__gm__ uint8_t *)channel_info + 4));

        const uint32_t flag_size = 8;

        aclshmemi_fill_sdma_sqe(channel_info, layout.send_workspace, // send flag buffer of current core
            layout.remote_recv_workspace + queue_id * ACLSHMEM_SDMA_FLAG_LENGTH, // write flag location for this channel
            flag_size, sq_tail, sq_tail - channel_info->sq_head);

        sq_tail = (sq_tail + 1) % (channel_info->sq_depth);

        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), sizeof(stars_channel_info_t));
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(write_info);

        // Ring Doorbell
        aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET, sq_tail, tmp_local,
                                      sync_id);
        aclshmemi_set_value<uint32_t>(((__gm__ uint8_t *)channel_info) + 4, sq_tail, tmp_local, sync_id);
    }
}

ACLSHMEM_DEVICE void aclshmemi_sdma_poll_for_completion(const workspace_layout_t &layout, 
                                                        AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id)
{
    uint32_t queue_num = 1;
    const uint32_t max_times = 1000000;
    for (uint8_t queue_id = 0; queue_id < queue_num; queue_id++) {
        auto local_recv_workspace = layout.recv_workspace + queue_id * ACLSHMEM_SDMA_FLAG_LENGTH;
        auto remote_recv_workspace = layout.remote_recv_workspace + queue_id * ACLSHMEM_SDMA_FLAG_LENGTH;

        uint32_t send_value = 0;
        uint32_t times = 0;

        // Poll until flag is received or timeout occurs
        while (send_value == 0 && times < max_times) {
            copy_gm_to_gm<uint32_t>(local_recv_workspace, remote_recv_workspace, 1, tmp_local, sync_id);
            dcci_cacheline(local_recv_workspace);
            send_value = *((__gm__ uint32_t *)local_recv_workspace);
            times++;
        }

        // Clear
        aclshmemi_set_value<uint32_t>(remote_recv_workspace, 0, tmp_local, sync_id);
        aclshmemi_set_value<uint32_t>(local_recv_workspace, 0, tmp_local, sync_id);
    }
}

ACLSHMEM_DEVICE void aclshmemi_sdma_post_send(__gm__ uint8_t *recv_buffer, __gm__ uint8_t *send_buffer,
                                              uint64_t message_len, AscendC::LocalTensor<uint32_t> &tmp_local,
                                              uint32_t sync_id)
{
    __gm__ uint8_t *channel_base = aclshmemi_sdma_get_channel_base();

    const auto cur_block_idx = AscendC::GetBlockIdx();
    // 1. Initialize per-core parameters, including channel count, etc.
    sdma_config_t config;
    config.queue_num = 1;
    config.block_bytes = 1024 * 1024; // 1MB per SQE
    config.per_core_bytes = message_len;
    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes;

    if (config.iter_num == 0 || cur_block_idx >= (ACLSHMEM_MAX_AIV_PER_NPU / config.queue_num)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // 2. Calculate base channel index for the current core
    __gm__ stars_channel_info_t *batch_write_channel_base = 
        (__gm__ stars_channel_info_t *)(channel_base);
    __gm__ stars_channel_info_t *batch_write_channel_info = batch_write_channel_base + cur_block_idx * config.queue_num;

    // 3. Calculate base addresses of send and receive flags
    __gm__ uint8_t *workspace = channel_base + ACLSHMEM_MAX_AIV_PER_NPU * sizeof(stars_channel_info_t);
    workspace_layout_t workspace_layout;
    uint64_t per_core_workspace_size = config.queue_num * ACLSHMEM_SDMA_FLAG_LENGTH;
    __gm__ uint8_t *my_workspace = workspace + ACLSHMEM_SDMA_FLAG_LENGTH + (cur_block_idx * per_core_workspace_size);

    workspace_layout.send_workspace = workspace;
    workspace_layout.recv_workspace = my_workspace;
    workspace_layout.remote_recv_workspace = my_workspace + ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_SDMA_FLAG_LENGTH;

    aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)workspace_layout.send_workspace, config.queue_num, tmp_local,
        sync_id);

    // 4. Initialize the sq_tail array
    uint32_t sq_tail[ACLSHMEM_MAX_AIV_PER_NPU] = {0};
    for (uint32_t queue_id = 0U; queue_id < config.queue_num; ++queue_id) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;
        // Load sq_tail field at 4-byte offset
        dcci_cacheline(((__gm__ uint8_t *)channel_info) + 4);
        sq_tail[queue_id] =  *((__gm__ uint32_t *)(((__gm__ uint8_t *)channel_info) + 4));
    }

    // 5. Submit data transfer SQE
    aclshmemi_sdma_submit_data_sqes(batch_write_channel_info, send_buffer, recv_buffer, config, sq_tail);

    // 6. Ring doorbell
    auto item_size = config.iter_num * sizeof(stars_sdma_sqe_t);
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;

        // Flush data cache to ensure all SQEs are written back to HBM
        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), item_size);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(write_info);

        // Ring Doorbell
        aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET, sq_tail[queue_id], tmp_local,
                                      sync_id);
        aclshmemi_set_value<uint32_t>(((__gm__ uint8_t *)channel_info) + 4, sq_tail[queue_id], tmp_local, sync_id);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

ACLSHMEM_DEVICE void aclshmemi_cmo_submit_data_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                    __gm__ uint8_t *src, uint32_t size, ACLSHMEMCMOTYPE cmo_type, uint32_t& sq_tail)
{
    __gm__ stars_channel_info_t *channel_info = batch_write_channel_info;

    if constexpr (ACLSHMEM_STARS_V2_LAYOUT) {
        aclshmemi_fill_stars_v2_cmo_sqe(
            channel_info, src, size, (uint32_t)cmo_type, sq_tail, sq_tail - channel_info->sq_head);
    } else {
        aclshmemi_fill_cmo_sqe(channel_info, src, size, (uint32_t)cmo_type, sq_tail, sq_tail - channel_info->sq_head);
    }

    sq_tail = (sq_tail + 1) % (channel_info->sq_depth);
    AscendC::PipeBarrier<PIPE_ALL>();
}

ACLSHMEM_DEVICE void aclshmemi_cmo_async(__gm__ uint8_t *src, uint32_t size, ACLSHMEMCMOTYPE cmo_type,
                                         AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    __gm__ uint8_t *context_gm = reinterpret_cast<__gm__ uint8_t *>(device_state->sdma_workspace_addr);
    if (context_gm == nullptr)
        return;

    const auto cur_block_idx = AscendC::GetBlockIdx();

    // Calculate base channel index for the current core
    __gm__ stars_channel_info_t *batch_write_channel_base = 
        (__gm__ stars_channel_info_t *)(context_gm + sizeof(stars_channel_flag_info_t));
    __gm__ stars_channel_info_t *batch_write_channel_info = batch_write_channel_base + cur_block_idx;

    // Calculate base addresses of send and receive flags
    __gm__ uint8_t *workspace = context_gm + sizeof(stars_channel_flag_info_t) +
                                ACLSHMEM_MAX_AIV_PER_NPU * sizeof(stars_channel_info_t);
    workspace_layout_t workspace_layout;
    uint64_t per_core_workspace_size = ACLSHMEM_SDMA_FLAG_LENGTH;
    __gm__ uint8_t *sdma_recv_workspace = workspace + ACLSHMEM_SDMA_FLAG_LENGTH + (cur_block_idx * per_core_workspace_size);
    __gm__ uint8_t *cmo_recv_workspace = sdma_recv_workspace + 2 * ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_SDMA_FLAG_LENGTH;

    workspace_layout.send_workspace = workspace;
    workspace_layout.recv_workspace = sdma_recv_workspace;
    workspace_layout.remote_recv_workspace = sdma_recv_workspace + ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_SDMA_FLAG_LENGTH;

    aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)workspace_layout.send_workspace, 1, tmp_local, sync_id);

    // Initialize the sq_tail
    uint32_t sq_tail = 0;
    __gm__ stars_channel_info_t *channel_info = batch_write_channel_info;
    // Load sq_tail field at 4-byte offset
    dcci_cacheline(((__gm__ uint8_t *)channel_info) + 4);
    sq_tail = *((__gm__ uint32_t *)(((__gm__ uint8_t *)channel_info) + 4));

    // Submit cmo SQE
    aclshmemi_cmo_submit_data_sqes(batch_write_channel_info, src, size, cmo_type, sq_tail);

    auto item_size = 2 * sizeof(stars_sdma_sqe_t);  // cmo and flag
    // Flush data cache to ensure all SQEs are written back to HBM
    AscendC::GlobalTensor<uint8_t> write_info;
    write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), item_size);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(write_info);

    // Ring Doorbell
    aclshmemi_set_value<uint32_t>((__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET, sq_tail, tmp_local, sync_id);
    aclshmemi_set_value<uint32_t>(((__gm__ uint8_t *)channel_info) + 4, sq_tail, tmp_local, sync_id);

    AscendC::PipeBarrier<PIPE_ALL>();
}

// Set SDMA Interfaces necessary UB Buffer.
ACLSHMEM_DEVICE void aclshmemx_set_sdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    device_state->sdma_config.aclshmem_ub = offset;
    device_state->sdma_config.ub_size = ub_size;
    device_state->sdma_config.sync_id = sync_id;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);

    // Create LocalTensor from buf pointer and ub_size
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_sdma_post_send((__gm__ uint8_t *)ptr, (__gm__ uint8_t *)src, elem_size * sizeof(T), ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(AscendC::GlobalTensor<T> &dst, AscendC::GlobalTensor<T> &src,
                                            AscendC::LocalTensor<T> &buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor.address_.dataLen = UB_ALIGN_SIZE_64;

    aclshmemi_sdma_post_send((__gm__ uint8_t *)ptr, (__gm__ uint8_t *)(src.GetPhyAddr()), elem_size * sizeof(T),
        ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);

    // Create LocalTensor from buf pointer and ub_size
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_sdma_post_send((__gm__ uint8_t *)dst, (__gm__ uint8_t *)ptr, elem_size * sizeof(T), ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(AscendC::GlobalTensor<T> &dst, AscendC::GlobalTensor<T> &src,
                                            AscendC::LocalTensor<T> &buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor.address_.dataLen = UB_ALIGN_SIZE_64;

    aclshmemi_sdma_post_send((__gm__ uint8_t *)(dst.GetPhyAddr()), (__gm__ uint8_t *)ptr, elem_size * sizeof(T),
        ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(AscendC::LocalTensor<T> &buf, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor.address_.dataLen = UB_ALIGN_SIZE_64;

    const uint32_t queue_num = 1;
    const auto cur_block_idx = AscendC::GetBlockIdx();
    __gm__ uint8_t *channel_base = aclshmemi_sdma_get_channel_base();
    __gm__ stars_channel_info_t *batch_write_channel_info = (__gm__ stars_channel_info_t *)(channel_base) + cur_block_idx * queue_num;

    workspace_layout_t layout;
    layout.send_workspace = channel_base + ACLSHMEM_MAX_AIV_PER_NPU * sizeof(stars_channel_info_t);
    layout.recv_workspace = layout.send_workspace + ACLSHMEM_SDMA_FLAG_LENGTH * (cur_block_idx * queue_num + 1);
    layout.remote_recv_workspace = layout.recv_workspace + ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_SDMA_FLAG_LENGTH;
    
    aclshmemi_sdma_submit_flag_sqes(batch_write_channel_info, layout, ub_tensor, sync_id);
    aclshmemi_sdma_poll_for_completion(layout, ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(__ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    const uint32_t queue_num = 1;
    const auto cur_block_idx = AscendC::GetBlockIdx();
    __gm__ uint8_t *channel_base = aclshmemi_sdma_get_channel_base();
    __gm__ stars_channel_info_t *batch_write_channel_info = (__gm__ stars_channel_info_t *)(channel_base) + cur_block_idx * queue_num;

    workspace_layout_t layout;
    layout.send_workspace = channel_base + ACLSHMEM_MAX_AIV_PER_NPU * sizeof(stars_channel_info_t);
    layout.recv_workspace = layout.send_workspace + ACLSHMEM_SDMA_FLAG_LENGTH * (cur_block_idx * queue_num + 1);
    layout.remote_recv_workspace = layout.recv_workspace + ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_SDMA_FLAG_LENGTH;
    
    aclshmemi_sdma_submit_flag_sqes(batch_write_channel_info, layout, ub_tensor, sync_id);
    aclshmemi_sdma_poll_for_completion(layout, ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(AscendC::LocalTensor<T> &buf, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor.address_.dataLen = UB_ALIGN_SIZE_64;

    aclshmemi_stars_submit_notify_record(ub_tensor, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(__ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_stars_submit_notify_record(ub_tensor, sync_id);
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_sdma_get_channel_base()
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    ASCENDC_ASSERT((device_state != nullptr), "device_state is null");
    __gm__ uint8_t *context_gm = reinterpret_cast<__gm__ uint8_t *>(device_state->sdma_workspace_addr);
    return context_gm + sizeof(stars_channel_flag_info_t);
}

// Set CMO Interfaces necessary UB Buffer.
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                                        __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id)
{
    if (cmo_type != ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH) {
        return;
    }
    // Create LocalTensor from buf pointer and ub_size
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_cmo_async((__gm__ uint8_t *)src, elem_size * sizeof(T), cmo_type, 
                            ub_tensor, (AscendC::TEventID)sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(AscendC::GlobalTensor<T> &src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                                            AscendC::LocalTensor<T> &buf, uint32_t sync_id)
{
    if (cmo_type != ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH) {
        return;
    }
    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor.address_.dataLen = UB_ALIGN_SIZE_64;

    aclshmemi_cmo_async((__gm__ uint8_t *)(src.GetPhyAddr()), elem_size * sizeof(T), cmo_type, 
                            ub_tensor, (AscendC::TEventID)sync_id);
}

#endif
