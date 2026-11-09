/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_UB2GM_MTE_HPP
#define SHMEM_DEVICE_UB2GM_MTE_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    uint32_t size, bool toL2Cache)
{
    ASCENDC_ASSERT((dstGva != nullptr), "input gva is null");

    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    ubTensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, uint32_t size)
{
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    AscendC::DataCopyPad(dstGva, srcUb, dataCopyParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    AscendC::DataCopyExtParams &copyParams)
{
    ASCENDC_ASSERT((dstGva != nullptr), "input gva is null");

    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    ubTensor.address_.dataLen = ALIGN_UP(copyParams.blockCount * copyParams.blockLen, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGva));
    AscendC::DataCopyPad(gmTensor, ubTensor, copyParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, AscendC::DataCopyExtParams &copyParams)
{
    AscendC::DataCopyPad(dstGva, srcUb, copyParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    uint32_t size, bool toL2Cache)
{
    ASCENDC_ASSERT((srcGva != nullptr), "input gva is null");
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    ubTensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, uint32_t size)
{
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(dstUb, srcGva, dataCopyParams, padParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    AscendC::DataCopyExtParams &copyParams)
{
    ASCENDC_ASSERT((srcGva != nullptr), "input gva is null");
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    ubTensor.address_.dataLen = ALIGN_UP(copyParams.blockCount * copyParams.blockLen, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGva));
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, copyParams, padParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, AscendC::DataCopyExtParams &copyParams)
{
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(dstUb, srcGva, copyParams, padParams);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    aclshmemi_copy_gm2ub(dst, remote_ptr, elem_size * sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    aclshmemi_copy_gm2ub(dst, remote_buff, elem_size * sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst);
    ub_tensor.address_.dataLen = ALIGN_UP(copy_params.repeat * copy_params.length * sizeof(T), UB_ALIGN_SIZE);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           const non_contiguous_copy_param &copy_params, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(dst, remote_buff, data_copy_params_gm2ub);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    aclshmemi_copy_ub2gm(remote_ptr, src, elem_size * sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t elem_size,
                                           int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    aclshmemi_copy_ub2gm(remote_buff, src, elem_size * sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src);
    ub_tensor.address_.dataLen = ALIGN_UP(copy_params.repeat * copy_params.length * sizeof(T), UB_ALIGN_SIZE);
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                           const non_contiguous_copy_param &copy_params, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(remote_buff, src, data_copy_params_ub2gm);
}

#endif