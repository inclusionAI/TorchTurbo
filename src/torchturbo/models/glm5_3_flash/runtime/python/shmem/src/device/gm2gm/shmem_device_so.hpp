/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SO_HPP
#define SHMEM_DEVICE_SO_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "host/shmem_host_def.h"
#include "shmemi_device_rma.h"
#include "utils/mstx/shmemi_mstx_report.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr int ACLSHMEM_RDMA_V2_SUPPORTED = 1;
#else
constexpr int ACLSHMEM_RDMA_V2_SUPPORTED = 0;
#endif

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


ACLSHMEM_DEVICE void aclshmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_quiet();
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,          \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)    \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                        \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                      \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                    \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,     \
                              copy_event_id);                                                                        \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                               \
        aclshmem_quiet();                                                                                            \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                           \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                    size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal,             \
                                                    int sig_op, int pe)                                                     \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                               \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                             \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                            \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                               \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                      \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                     \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                           \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                      \
        aclshmem_quiet();                                                                                                   \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                                  \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED(NAME, TYPE)                                                         \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src,                             \
                                                    const non_contiguous_copy_param &copy_params,                     \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)     \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                       \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,    \
                              copy_event_id);                                                                         \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        aclshmem_quiet();                                                                                             \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                    const non_contiguous_copy_param &copy_params,                           \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)           \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                               \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                             \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                            \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                               \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                      \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                     \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                         \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                      \
        aclshmem_quiet();                                                                                                   \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                                  \
    }

#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAIL(BITS)                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_signal(__gm__ void *dest, __gm__ void *src, size_t nelems,                        \
                                                     __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)              \
    {                                                                                                                           \
        aclshmem_putmem_signal(dest, src, nelems * (BITS / 8), sig_addr, signal, sig_op, pe);                                   \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAIL);

ACLSHMEM_DEVICE void aclshmem_putmem_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size,
                                          __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_fence();
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,          \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)    \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                            \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                         \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                        \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,         \
                              copy_event_id);                                                                            \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                   \
        aclshmem_fence();                                                                                                \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                               \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI(NAME, TYPE)                                                       \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(AscendC::GlobalTensor<TYPE> dst,                            \
                                                        AscendC::GlobalTensor<TYPE> src, size_t elem_size,            \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                       \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                      \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                         \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                          \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                               \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                     \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        aclshmem_fence();                                                                                             \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI(NAME, TYPE)                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                         \
                                                        const non_contiguous_copy_param &copy_params,                 \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                         \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                       \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,    \
                              copy_event_id);                                                                         \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                                \
        aclshmem_fence();                                                                                             \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI(NAME, TYPE)                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal_nbi(                                                          \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,                                           \
        const non_contiguous_copy_param &copy_params, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                       \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                  \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                     \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                    \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                       \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                              \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                        \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                             \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                 \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                              \
        aclshmem_fence();                                                                                           \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                          \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAILED_NBI(BITS)                                                               \
    ACLSHMEM_DEVICE void aclshmem_put##BITS##_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t nelems,           \
                                                       __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)  \
    {                                                                                                                 \
        aclshmem_putmem_signal_nbi(dst, src, nelems * (BITS / 8), sig_addr, signal, sig_op, pe);                      \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_DETAILED_NBI);

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    MSTX_FUSE_SCOPE_START();
    aclshmemi_store(addr, val);

    // flush data cache to GM after signal to ensure it is visible to other ranks
    dcci_cacheline((__gm__ uint8_t *)addr);
    MSTX_FUSE_SCOPE_END();
    MSTX_SIGNAL_SET_REPORT(addr, val);
}

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {
        aclshmemi_signal_set((__gm__ int32_t *)aclshmem_ptr(addr, pe), val);
    } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {
        if constexpr (ACLSHMEM_RDMA_V2_SUPPORTED) {
            aclshmemx_roce_atomic_set(addr, val, pe);
            uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
            uint32_t sync_id = device_state->rdma_config.sync_id;
            aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), sync_id);
        } else {
            __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(device_state->signal_addr);
            aclshmemi_store(sig_addr_int32, val);
            // flush data cache to GM after signal to ensure it is visible to other ranks
            dcci_cachelines((__gm__ uint8_t *)sig_addr_int32, sizeof(int32_t));
            aclshmemi_highlevel_signal_set(addr, sig_addr_int32, pe);
        } 
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe, __ubuf__ T* buf, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)aclshmem_roce_ptr(dst, pe), (__gm__ uint8_t*)src, pe, 0, sizeof(int32_t),
        ub_tensor_64, ub_tensor_32, sync_id);
    aclshmemi_roce_quiet(pe, 0, ub_tensor_64, ub_tensor_32, sync_id);
}

ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)aclshmem_roce_ptr(dst, pe), (__gm__ uint8_t*)src, pe, 0, sizeof(int32_t),
        ub_tensor_64, ub_tensor_32, sync_id);
    aclshmemi_roce_quiet(pe, 0, ub_tensor_64, ub_tensor_32, sync_id);
}

ACLSHMEM_DEVICE void aclshmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    __gm__ int32_t* remote_ptr = reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(addr, pe));
    __ubuf__ int32_t* buf =(__ubuf__ int32_t*)(device_state->mte_config.aclshmem_ub);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LocalTensor<int32_t> ub_tensor_32(AscendC::TPosition::VECIN, device_state->mte_config.aclshmem_ub, 1);
    ub_tensor_32.SetValue(0, val);
    AscendC::TEventID sync_id = device_state->mte_config.sync_id;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::SetAtomicAdd<int32_t>();
    aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(int32_t));
    AscendC::SetAtomicNone();
    AscendC::PipeBarrier<PIPE_ALL>();
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    // never reach
    return -1;
}

ACLSHMEM_DEVICE void aclshmemi_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    switch (sig_op) {
        case ACLSHMEM_SIGNAL_SET:
            aclshmemi_signal_set(sig_addr, pe, signal);
            break;
        case ACLSHMEM_SIGNAL_ADD:
            aclshmemi_signal_add(sig_addr, pe, signal);
            break;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

#ifdef __cplusplus
}
#endif

#endif
