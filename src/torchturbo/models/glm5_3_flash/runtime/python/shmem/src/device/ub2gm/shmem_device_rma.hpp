/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_UB2GM_RMA_HPP
#define SHMEM_DEVICE_UB2GM_RMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"

#define ACLSHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)     \
    {                                                                                                                    \
        /* MTE  */                                                                                                       \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                           \
        aclshmemx_mte_get_nbi(dst, src, elem_size, pe, copy_event_id);                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)                                            \
    {                                                                                                                   \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        aclshmemx_mte_get_nbi(dst, src, elem_size, pe, copy_event_id);                                                  \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src,                   \
                                                 const non_contiguous_copy_param &copy_params, int pe)     \
    {                                                                                                      \
        /* MTE  */                                                                                         \
        /* Global State Get */                                                                             \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                         \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;             \
        aclshmemx_mte_get_nbi(dst, src, copy_params, pe, copy_event_id);                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                    \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)                  \
    {                                                                                                                   \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        aclshmemx_mte_get_nbi(dst, src, copy_params, pe, copy_event_id);                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                         \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)     \
    {                                                                                                                        \
        /* MTE  */                                                                                                           \
        /* Global State Get */                                                                                               \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                           \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                               \
        aclshmemx_mte_put_nbi(dst, src, elem_size, pe, copy_event_id);                                                       \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int32_t pe)                                        \
    {                                                                                                                   \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        aclshmemx_mte_put_nbi(dst, src, elem_size, pe, copy_event_id);                                                  \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src,                     \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)   \
    {                                                                                                        \
        /* MTE  */                                                                                           \
        /* Global State Get */                                                                               \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                           \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;               \
        aclshmemx_mte_put_nbi(dst, src, copy_params, pe, copy_event_id);                                     \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                    \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)              \
    {                                                                                                                   \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                          \
        aclshmemx_mte_put_nbi(dst, src, copy_params, pe, copy_event_id);                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

#endif