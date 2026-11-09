/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_GM2GM_RMA_HPP
#define SHMEM_DEVICE_GM2GM_RMA_HPP

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"
#include "gm2gm/engine/shmemi_device_udma.h"
#include "host/shmem_host_def.h"
#include "shmemi_device_rma.h"

#ifndef ACLSHMEM_UDMA_SUPPORTED
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif

#define ACLSHMEM_UDMA_TRANSPORT_ENABLED(STATE, PE) \
    (ACLSHMEM_UDMA_SUPPORTED && (((STATE)->topo_list[(PE)] & ACLSHMEM_TRANSPORT_UDMA) != 0))

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(double, double);        \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, char);            \
    FUNC(bfloat16, bfloat16_t)

#define ACLSHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe)    \
    {                                                                                       \
        auto ptr = aclshmem_ptr(dst, pe);                                                   \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        *addr_gm = value;                                                                   \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);

#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_g(__gm__ TYPE *src, int32_t pe)                  \
    {                                                                                       \
        auto ptr = aclshmem_ptr(src, pe);                                                   \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
        return *addr_gm;                                                                    \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_default_nbi(
    __gm__ aclshmem_device_host_state_t *device_state, __gm__ T *dst, __gm__ T *src, uint32_t elem_size, int32_t pe)
{
    uint64_t copy_ub = device_state->udma_config.aclshmem_ub;
    uint32_t sync_id = device_state->udma_config.sync_id;
    aclshmemx_udma_get_nbi<T>(
        dst, src, reinterpret_cast<__ubuf__ T *>(copy_ub), elem_size, pe, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_default_nbi(
    __gm__ aclshmem_device_host_state_t *device_state, __gm__ T *dst, __gm__ T *src, uint32_t elem_size, int32_t pe)
{
    uint64_t copy_ub = device_state->udma_config.aclshmem_ub;
    uint32_t sync_id = device_state->udma_config.sync_id;
    aclshmemx_udma_put_nbi<T>(
        dst, src, reinterpret_cast<__ubuf__ T *>(copy_ub), elem_size, pe, sync_id);
}

ACLSHMEM_DEVICE void aclshmem_getmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {
        /* SDMA */
        uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->sdma_config.ub_size;
        uint32_t sync_id = device_state->sdma_config.sync_id;
        aclshmemx_sdma_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
        aclshmemx_sdma_quiet(reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, sync_id);
    } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {
        /* UDMA */
        aclshmemi_udma_get_default_nbi<char>(device_state, reinterpret_cast<__gm__ char *>(dst),
            reinterpret_cast<__gm__ char *>(src), elem_size, pe);
        aclshmemx_udma_quiet(pe);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {
        /* MTE */
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;
        aclshmemx_mte_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                            reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
        aclshmemx_mte_quiet();
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {
        /* ROCE */
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
        uint32_t sync_id = device_state->rdma_config.sync_id;
        aclshmemx_roce_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                            reinterpret_cast<__ubuf__ char *>(copy_ub), elem_size, pe, sync_id);
        aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char *>(copy_ub), sync_id);
    }
}

#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)   \
    {                                                                                                                \
        /* Global State Get */                                                                                       \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                   \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                 \
            /* SDMA */                                                                                               \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                \
            uint32_t copy_ub_size = device_state->sdma_config.ub_size;                                               \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                    \
            aclshmemx_sdma_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,\
                                   sync_id);                                                                         \
            aclshmemx_sdma_quiet(reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, sync_id);                 \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                             \
            /* UDMA */                                                                                               \
            aclshmemi_udma_get_default_nbi<TYPE>(device_state, dst, src, elem_size, pe);                             \
            aclshmemx_udma_quiet(pe);                                                                                \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE)  {                                          \
            /* MTE */                                                                                                \
            /* CopyUB Config Set */                                                                                  \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                 \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                \
            AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                         \
            aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe, \
                                  sync_id);                                                                          \
            aclshmemx_mte_quiet();                                                                                   \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                          \
            /* ROCE */                                                                                               \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                    \
            aclshmemx_roce_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), elem_size, pe, sync_id);    \
            aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), sync_id);                           \
        }                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);

#define ACLSHMEM_IGET_TYPENAME_MEM(NAME, TYPE)                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_iget(__gm__ TYPE *dest, __gm__ TYPE *source, ptrdiff_t dst, ptrdiff_t sst,     \
                                                size_t nelems, int pe)                                                    \
    {                                                                                                                     \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                        \
        uint64_t buf = device_state->mte_config.aclshmem_ub;                                                              \
        uint32_t ub_size = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                                 \
                                                                                                                          \
        auto ptr = aclshmem_ptr(source, pe);                                                                              \
        __gm__ TYPE *remote_ptr = reinterpret_cast<__gm__ TYPE *>(ptr);                                                   \
                                                                                                                          \
        non_contiguous_copy_param copy_params {static_cast<uint32_t>(nelems), 1,                                          \
                                               static_cast<uint32_t>(sst), static_cast<uint32_t>(dst)};                   \
                                                                                                                          \
        uint32_t ascendc_block_count_limit = 4095;                                                                        \
        uint32_t block_size = ub_size / sizeof(TYPE) / copy_params.length;                                                \
        block_size = block_size > ascendc_block_count_limit ? ascendc_block_count_limit : block_size;                     \
                                                                                                                          \
        uint32_t repeat_times = copy_params.repeat / block_size;                                                          \
        uint32_t remain = copy_params.repeat % block_size;                                                                \
                                                                                                                          \
        uint32_t src_offset_unit = block_size * copy_params.src_ld;                                                       \
        uint32_t dst_offset_unit = block_size * copy_params.dst_ld;                                                       \
        uint32_t ascendc_block_len_bytes = copy_params.length * sizeof(TYPE);                                             \
        uint32_t ascendc_src_stride_bytes = (copy_params.src_ld - copy_params.length) * sizeof(TYPE);                     \
        uint32_t ascendc_dst_stride_bytes = (copy_params.dst_ld - copy_params.length) * sizeof(TYPE);                     \
                                                                                                                          \
        AscendC::GlobalTensor<TYPE> src_tensor;                                                                           \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                             \
        AscendC::GlobalTensor<TYPE> dst_tensor;                                                                           \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                    \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);                                                  \
        ub_tensor.address_.dataLen = ALIGN_UP(block_size * ascendc_block_len_bytes, UB_ALIGN_SIZE);                       \
                                                                                                                          \
        for (uint64_t i = 0; i < repeat_times; i++) {                                                                     \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(remote_ptr + i * src_offset_unit));                \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(dest + i * dst_offset_unit));                      \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(block_size, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0); \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(block_size, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes , 0);\
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
                                                                                                                          \
        if (remain > 0) {                                                                                                 \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(remote_ptr + repeat_times * src_offset_unit));     \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(dest + repeat_times * dst_offset_unit));           \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(remain, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0);     \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(remain, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes, 0);     \
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_IGET_TYPENAME_MEM);
#undef ACLSHMEM_IGET_TYPENAME_MEM

#define ACLSHMEM_GET_SIZE_MEM(BITS)                                                                                     \
    ACLSHMEM_DEVICE void aclshmem_get##BITS(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)         \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        aclshmem_getmem(dst, src, elem_size * ((BITS) / 8), pe);                                                        \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM);
#undef ACLSHMEM_GET_SIZE_MEM

#define ACLSHMEM_IGET_SIZE_MEM(BITS)                                                                                      \
    ACLSHMEM_DEVICE void aclshmem_iget##BITS(__gm__ void *dest, __gm__ void *source, ptrdiff_t dst, ptrdiff_t sst,        \
                                             size_t nelems, int pe)                                                       \
    {                                                                                                                     \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                        \
        uint64_t buf = device_state->mte_config.aclshmem_ub;                                                              \
        uint32_t ub_size = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                                 \
                                                                                                                          \
        auto ptr = aclshmem_ptr(source, pe);                                                                              \
        __gm__ uint8_t *remote_ptr = reinterpret_cast<__gm__ uint8_t *>(ptr);                                             \
        __gm__ uint8_t *dest_ptr = reinterpret_cast<__gm__ uint8_t *>(dest);                                              \
                                                                                                                          \
        uint32_t bytes = (BITS) / 8;                                                                                      \
        non_contiguous_copy_param copy_params {static_cast<uint32_t>(nelems), bytes,                                      \
                                               static_cast<uint32_t>(sst) * bytes, static_cast<uint32_t>(dst) * bytes};   \
                                                                                                                          \
        uint32_t ascendc_block_count_limit = 4095;                                                                        \
        uint32_t block_size = ub_size / sizeof(uint8_t) / copy_params.length;                                             \
        block_size = block_size > ascendc_block_count_limit ? ascendc_block_count_limit : block_size;                     \
                                                                                                                          \
        uint32_t repeat_times = copy_params.repeat / block_size;                                                          \
        uint32_t remain = copy_params.repeat % block_size;                                                                \
                                                                                                                          \
        uint32_t src_offset_unit = block_size * copy_params.src_ld;                                                       \
        uint32_t dst_offset_unit = block_size * copy_params.dst_ld;                                                       \
        uint32_t ascendc_block_len_bytes = copy_params.length * sizeof(uint8_t);                                          \
        uint32_t ascendc_src_stride_bytes = (copy_params.src_ld - copy_params.length) * sizeof(uint8_t);                  \
        uint32_t ascendc_dst_stride_bytes = (copy_params.dst_ld - copy_params.length) * sizeof(uint8_t);                  \
                                                                                                                          \
        AscendC::GlobalTensor<uint8_t> src_tensor;                                                                        \
        AscendC::LocalTensor<uint8_t> ub_tensor;                                                                          \
        AscendC::GlobalTensor<uint8_t> dst_tensor;                                                                        \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                    \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);                                                  \
        ub_tensor.address_.dataLen = ALIGN_UP(block_size * ascendc_block_len_bytes, UB_ALIGN_SIZE);                       \
                                                                                                                          \
        for (uint64_t i = 0; i < repeat_times; i++) {                                                                     \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(remote_ptr + i * src_offset_unit));             \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dest_ptr + i * dst_offset_unit));               \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(block_size, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0); \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(block_size, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes , 0);\
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
                                                                                                                          \
        if (remain > 0) {                                                                                                 \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(remote_ptr + repeat_times * src_offset_unit));  \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dest_ptr + repeat_times * dst_offset_unit));    \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(remain, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0);     \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(remain, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes, 0);     \
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_IGET_SIZE_MEM);
#undef ACLSHMEM_IGET_SIZE_MEM

ACLSHMEM_DEVICE void aclshmem_putmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA){
        /* SDMA */
        uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->sdma_config.ub_size;
        uint32_t sync_id = device_state->sdma_config.sync_id;
        aclshmemx_sdma_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
        aclshmemx_sdma_quiet(reinterpret_cast<__ubuf__ char *>(copy_ub),copy_ub_size, sync_id);
    } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {
        /* UDMA */
        aclshmemi_udma_put_default_nbi<char>(device_state, reinterpret_cast<__gm__ char *>(dst),
            reinterpret_cast<__gm__ char *>(src), elem_size, pe);
        aclshmemx_udma_quiet(pe);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {
        /* MTE */
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;
        aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                              reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
        aclshmemx_mte_quiet();
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {
        /* ROCE */
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
        uint32_t sync_id = device_state->rdma_config.sync_id;
        aclshmemx_roce_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), elem_size, pe, sync_id);
        aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char *>(copy_ub), sync_id);
    }
}

#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                   \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                    \
            /* SDMA */                                                                                                  \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                   \
            uint32_t copy_ub_size = device_state->sdma_config.ub_size;                                                  \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                       \
            aclshmemx_sdma_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,   \
                                   sync_id);                                                                            \
            aclshmemx_sdma_quiet(reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, sync_id);                    \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                                \
            /* UDMA */                                                                                                  \
            aclshmemi_udma_put_default_nbi<TYPE>(device_state, dst, src, elem_size, pe);                                \
            aclshmemx_udma_quiet(pe);                                                                                   \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                              \
            /* MTE */                                                                                                   \
            /* CopyUB Config Set */                                                                                     \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                    \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                   \
            AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                            \
            aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,    \
                                sync_id);                                                                               \
            aclshmemx_mte_quiet();                                                                                           \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                             \
            /* ROCE */                                                                                                  \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                   \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                       \
            aclshmemx_roce_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), elem_size, pe, sync_id);       \
            aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), sync_id);                              \
        }                                                                                                               \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);

#define ACLSHMEM_IPUT_TYPENAME_MEM(NAME, TYPE)                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_iput(__gm__ TYPE *dest, __gm__ TYPE *source, ptrdiff_t dst, ptrdiff_t sst,     \
                                                size_t nelems, int pe)                                                    \
    {                                                                                                                     \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                        \
        uint64_t buf = device_state->mte_config.aclshmem_ub;                                                              \
        uint32_t ub_size = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                                 \
                                                                                                                          \
        auto ptr = aclshmem_ptr(dest, pe);                                                                                \
        __gm__ TYPE *remote_ptr = reinterpret_cast<__gm__ TYPE *>(ptr);                                                   \
                                                                                                                          \
        non_contiguous_copy_param copy_params {static_cast<uint32_t>(nelems), 1,                                          \
                                               static_cast<uint32_t>(sst), static_cast<uint32_t>(dst)};                   \
                                                                                                                          \
        uint32_t ascendc_block_count_limit = 4095;                                                                        \
        uint32_t block_size = ub_size / sizeof(TYPE) / copy_params.length;                                                \
        block_size = block_size > ascendc_block_count_limit ? ascendc_block_count_limit : block_size;                     \
                                                                                                                          \
        uint32_t repeat_times = copy_params.repeat / block_size;                                                          \
        uint32_t remain = copy_params.repeat % block_size;                                                                \
                                                                                                                          \
        uint32_t src_offset_unit = block_size * copy_params.src_ld;                                                       \
        uint32_t dst_offset_unit = block_size * copy_params.dst_ld;                                                       \
        uint32_t ascendc_block_len_bytes = copy_params.length * sizeof(TYPE);                                             \
        uint32_t ascendc_src_stride_bytes = (copy_params.src_ld - copy_params.length) * sizeof(TYPE);                     \
        uint32_t ascendc_dst_stride_bytes = (copy_params.dst_ld - copy_params.length) * sizeof(TYPE);                     \
                                                                                                                          \
        AscendC::GlobalTensor<TYPE> src_tensor;                                                                           \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                             \
        AscendC::GlobalTensor<TYPE> dst_tensor;                                                                           \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                    \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);                                                  \
        ub_tensor.address_.dataLen = ALIGN_UP(block_size * ascendc_block_len_bytes, UB_ALIGN_SIZE);                       \
                                                                                                                          \
        for (uint64_t i = 0; i < repeat_times; i++) {                                                                     \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(source + i * src_offset_unit));                    \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(remote_ptr + i * dst_offset_unit));                \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(block_size, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0); \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(block_size, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes , 0);\
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
                                                                                                                          \
        if (remain > 0) {                                                                                                 \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(source + repeat_times * src_offset_unit));         \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(remote_ptr + repeat_times * dst_offset_unit));     \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(remain, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0);     \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(remain, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes, 0);     \
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_IPUT_TYPENAME_MEM);
#undef ACLSHMEM_IPUT_TYPENAME_MEM

#define ACLSHMEM_PUT_SIZE_MEM(BITS)                                                                                     \
    ACLSHMEM_DEVICE void aclshmem_put##BITS(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)         \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        aclshmem_putmem(dst, src, elem_size * ((BITS) / 8), pe);                                                        \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM);
#undef ACLSHMEM_PUT_SIZE_MEM

#define ACLSHMEM_IPUT_SIZE_MEM(BITS)                                                                                      \
    ACLSHMEM_DEVICE void aclshmem_iput##BITS(__gm__ void *dest, __gm__ void *source, ptrdiff_t dst, ptrdiff_t sst,        \
                                             size_t nelems, int pe)                                                       \
    {                                                                                                                     \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                        \
        uint64_t buf = device_state->mte_config.aclshmem_ub;                                                              \
        uint32_t ub_size = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                                 \
                                                                                                                          \
        auto ptr = aclshmem_ptr(dest, pe);                                                                                \
        __gm__ uint8_t *remote_ptr = reinterpret_cast<__gm__ uint8_t *>(ptr);                                             \
        __gm__ uint8_t *source_ptr = reinterpret_cast<__gm__ uint8_t *>(source);                                          \
                                                                                                                          \
        uint32_t bytes = (BITS) / 8;                                                                                      \
        non_contiguous_copy_param copy_params {static_cast<uint32_t>(nelems), bytes,                                      \
                                               static_cast<uint32_t>(sst) * bytes, static_cast<uint32_t>(dst) * bytes};   \
                                                                                                                          \
        uint32_t ascendc_block_count_limit = 4095;                                                                        \
        uint32_t block_size = ub_size / sizeof(uint8_t) / copy_params.length;                                             \
        block_size = block_size > ascendc_block_count_limit ? ascendc_block_count_limit : block_size;                     \
                                                                                                                          \
        uint32_t repeat_times = copy_params.repeat / block_size;                                                          \
        uint32_t remain = copy_params.repeat % block_size;                                                                \
                                                                                                                          \
        uint32_t src_offset_unit = block_size * copy_params.src_ld;                                                       \
        uint32_t dst_offset_unit = block_size * copy_params.dst_ld;                                                       \
        uint32_t ascendc_block_len_bytes = copy_params.length * sizeof(uint8_t);                                          \
        uint32_t ascendc_src_stride_bytes = (copy_params.src_ld - copy_params.length) * sizeof(uint8_t);                  \
        uint32_t ascendc_dst_stride_bytes = (copy_params.dst_ld - copy_params.length) * sizeof(uint8_t);                  \
                                                                                                                          \
        AscendC::GlobalTensor<uint8_t> src_tensor;                                                                        \
        AscendC::LocalTensor<uint8_t> ub_tensor;                                                                          \
        AscendC::GlobalTensor<uint8_t> dst_tensor;                                                                        \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                    \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);                                                  \
        ub_tensor.address_.dataLen = ALIGN_UP(block_size * ascendc_block_len_bytes, UB_ALIGN_SIZE);                       \
                                                                                                                          \
        for (uint64_t i = 0; i < repeat_times; i++) {                                                                     \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_ptr + i * src_offset_unit));             \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(remote_ptr + i * dst_offset_unit));             \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(block_size, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0); \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(block_size, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes , 0);\
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
                                                                                                                          \
        if (remain > 0) {                                                                                                 \
            src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_ptr + repeat_times * src_offset_unit));  \
            dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(remote_ptr + repeat_times * dst_offset_unit));  \
                                                                                                                          \
            AscendC::DataCopyExtParams gm2ub_params(remain, ascendc_block_len_bytes, ascendc_src_stride_bytes, 0, 0);     \
            aclshmemi_copy_gm2ub(ub_tensor, src_tensor, gm2ub_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);                                                   \
                                                                                                                          \
            AscendC::DataCopyExtParams ub2gm_params(remain, ascendc_block_len_bytes, 0, ascendc_dst_stride_bytes, 0);     \
            aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, ub2gm_params);                                                    \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);                                                   \
        }                                                                                                                 \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_IPUT_SIZE_MEM);
#undef ACLSHMEM_IPUT_SIZE_MEM

ACLSHMEM_DEVICE void aclshmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {
        /* SDMA */
        uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->sdma_config.ub_size;
        uint32_t sync_id = device_state->sdma_config.sync_id;
        aclshmemx_sdma_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
    } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {
        /* UDMA */
        aclshmemi_udma_get_default_nbi<char>(device_state, reinterpret_cast<__gm__ char *>(dst),
            reinterpret_cast<__gm__ char *>(src), elem_size, pe);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {
        /* MTE  */
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
        aclshmemx_mte_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                              reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {
        /* ROCE */
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
        uint32_t sync_id = device_state->rdma_config.sync_id;
        aclshmemx_roce_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), elem_size, pe, sync_id);
    }
}

#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                       \
        /* Global State Get */                                                                                              \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                        \
            /* SDMA */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                       \
            uint32_t copy_ub_size = device_state->sdma_config.ub_size;                                                      \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                     \
            aclshmemx_sdma_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,       \
                                   sync_id);                                                                          \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                                    \
            /* UDMA */                                                                                                      \
            aclshmemi_udma_get_default_nbi<TYPE>(device_state, dst, src, elem_size, pe);                                    \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                  \
            /* MTE  */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
            aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                                  copy_event_id);                                                                           \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                                 \
            /* RoCE */                                                                                                      \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                       \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                           \
            aclshmemx_roce_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), elem_size, pe, sync_id);            \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);

#define ACLSHMEM_GET_SIZE_MEM_NBI(BITS)                                                                                 \
    ACLSHMEM_DEVICE void aclshmem_get##BITS##_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        aclshmem_getmem_nbi(dst, src, elem_size * ((BITS) / 8), pe);                                                    \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_MEM_NBI);
#undef ACLSHMEM_GET_SIZE_MEM_NBI

#define ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                                  \
                                                   const non_contiguous_copy_param &copy_params, int32_t pe)            \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,      \
                              copy_event_id);                                                                           \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                   uint32_t elem_size, int pe)                                           \
    {                                                                                                                    \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                     \
            /* SDMA */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                    \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->sdma_config.ub_size;                                              \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                        \
            aclshmemx_sdma_get_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                         \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                                 \
            /* UDMA */                                                                                                   \
            aclshmemi_udma_get_default_nbi<TYPE>(device_state, (__gm__ TYPE*)dst.GetPhyAddr(),                         \
                (__gm__ TYPE*)src.GetPhyAddr(), elem_size, pe);                                                        \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                               \
            /* MTE  */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                               \
            AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                             \
            aclshmemx_mte_get_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                          \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                              \
            /* ROCE */                                                                                                   \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                    \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->rdma_config.ub_size;                                              \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                        \
            aclshmemx_roce_get_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                         \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                   const non_contiguous_copy_param &copy_params, int pe)                 \
    {                                                                                                                    \
        /* ROCE */                                                                                                       \
        /* RDMA */                                                                                                       \
        /* MTE  */                                                                                                       \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        /* CopyUB Config Set */                                                                                          \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                         \
        /* Create LocalTensor */                                                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                            \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                             \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                           \
        aclshmemx_mte_get_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                      \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                       \
        /* Global State Get */                                                                                              \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                        \
            /* SDMA */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                       \
            uint32_t copy_ub_size = device_state->sdma_config.ub_size;                                                      \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                           \
            aclshmemx_sdma_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,       \
                                   sync_id);                                                                                \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                                    \
            /* UDMA */                                                                                                      \
            aclshmemi_udma_put_default_nbi<TYPE>(device_state, dst, src, elem_size, pe);                                    \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                  \
            /* MTE  */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
            AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                                \
            aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                                  sync_id);                                                                                 \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                                 \
            /* RoCE */                                                                                                      \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                       \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                           \
            aclshmemx_roce_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), elem_size, pe, sync_id);           \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);

#define ACLSHMEM_PUT_SIZE_MEM_NBI(BITS)                                                                                 \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        aclshmem_putmem_nbi(dst, src, elem_size * ((BITS) / 8), pe);                                                    \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_NBI);
#undef ACLSHMEM_PUT_SIZE_MEM_NBI

#define ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                                \
                                                   const non_contiguous_copy_param &copy_params, int32_t pe)          \
    {                                                                                                                 \
        /* ROCE */                                                                                                    \
        /* MTE  */                                                                                                    \
        /* Global State Get */                                                                                        \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                    \
        /* CopyUB Config Set */                                                                                       \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                        \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,    \
                              copy_event_id);                                                                         \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                   uint32_t elem_size, int pe)                                           \
    {                                                                                                                    \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA) {                                                     \
            /* SDMA */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;                                                    \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->sdma_config.ub_size;                                              \
            uint32_t sync_id = device_state->sdma_config.sync_id;                                                        \
            aclshmemx_sdma_put_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                         \
        } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {                                                 \
            /* UDMA */                                                                                                   \
            aclshmemi_udma_put_default_nbi<TYPE>(device_state, (__gm__ TYPE*)dst.GetPhyAddr(),                         \
                (__gm__ TYPE*)src.GetPhyAddr(), elem_size, pe);                                                        \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                               \
            /* MTE  */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                               \
            AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                             \
            aclshmemx_mte_put_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                          \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                              \
            /* RoCE */                                                                                                   \
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                                    \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->rdma_config.ub_size;                                              \
            uint32_t sync_id = device_state->rdma_config.sync_id;                                                        \
            aclshmemx_roce_put_nbi(dst, src, ub_tensor, elem_size, pe, sync_id);                                         \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                   const non_contiguous_copy_param &copy_params, int pe)                 \
    {                                                                                                                    \
        /* ROCE */                                                                                                       \
        /* MTE  */                                                                                                       \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        /* CopyUB Config Set */                                                                                          \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                         \
        /* Create LocalTensor */                                                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                            \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                             \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                           \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                      \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);

ACLSHMEM_DEVICE void aclshmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* MTE  */
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_SDMA)
    {
        /* SDMA */
        uint64_t copy_ub = device_state->sdma_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->sdma_config.ub_size;
        uint32_t sync_id = device_state->sdma_config.sync_id;
        aclshmemx_sdma_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src), 
                               reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
    } else if (ACLSHMEM_UDMA_TRANSPORT_ENABLED(device_state, pe)) {
        /* UDMA */
        aclshmemi_udma_put_default_nbi<char>(device_state, reinterpret_cast<__gm__ char *>(dst),
            reinterpret_cast<__gm__ char *>(src), elem_size, pe);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {
        /* MTE */
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;
        aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                              reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, sync_id);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {
        /* ROCE */
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
        uint32_t sync_id = device_state->rdma_config.sync_id;
        aclshmemx_roce_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                               reinterpret_cast<__ubuf__ char *>(copy_ub), elem_size, pe, sync_id);
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
ACLSHMEM_DEVICE void aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    
    device_state->mte_config.aclshmem_ub = offset;
    device_state->mte_config.ub_size = ub_size;
    device_state->mte_config.sync_id = sync_id;
}

ACLSHMEM_DEVICE void aclshmemx_set_udma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    if (ub_size < ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort,
            "UDMA MTE staging UB size %u is smaller than minimum %u bytes\n",
            ub_size, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
        return;
    }

    device_state->udma_config.aclshmem_ub = offset;
    device_state->udma_config.ub_size = ub_size;
    device_state->udma_config.sync_id = sync_id;
}

#undef ACLSHMEM_UDMA_TRANSPORT_ENABLED

#endif
