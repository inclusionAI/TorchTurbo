/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP

#include "kernel_operator.h"
#include "rdma_device_backend_in_die.hpp"
#include "rdma_device_backend_xscale.hpp"
#include "rdma_device_backend_hns_1825.hpp"
#include "rdma_device_backend_base.h"

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_fetch_and_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    auto mem_info_table = rdma_info->mem_ptr;
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);

    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)dst;
    wr.local_addr = (__gm__ uint8_t*)sq_context->amo_addr;
    wr.message_len = 0;
    wr.atomic.masked_common.swap_add_data = add_val;
    wr.atomic.masked_common.swap_add_mask = boundary;
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = sq_context->amo_lkey;

    aclshmemi_backend_traits<B>::template atomic_op_traits<T, IS_MASKED>::template post_send<
        aclshmemi_rdma_atomic_op_t::OP_ATOMIC_FA>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);

    return T(0);
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_compare_and_swap(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    auto mem_info_table = rdma_info->mem_ptr;
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);

    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)dst;
    wr.local_addr = (__gm__ uint8_t*)sq_context->amo_addr;
    wr.message_len = 0;
    wr.atomic.masked_common.swap_add_data = swap_val;
    wr.atomic.masked_common.compare_data = comp_val;
    wr.atomic.masked_common.swap_add_mask = swap_mask;
    wr.atomic.masked_common.compare_mask = comp_mask;
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = sq_context->amo_lkey;

    aclshmemi_backend_traits<B>::template atomic_op_traits<T, IS_MASKED>::template post_send<
        aclshmemi_rdma_atomic_op_t::OP_ATOMIC_CAS>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);

    return T(0);
}
#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
