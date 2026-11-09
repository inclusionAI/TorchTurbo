/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_AMO_HPP
#define SHMEM_DEVICE_AMO_HPP

#include "kernel_operator.h"
#include "host/shmem_host_def.h"
#include "shmemi_device_cc.h"
#include "device/gm2gm/engine/shmem_device_udma.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_mte.h"

/**
 * @brief Direct support types for atomic_add.
 *        Supported types: int8, int16, bfloat16, half.
 *        Supported hardware platform: A2/A3/Ascend_950 (MTE path).
 */
#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE)                                                                      \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)                       \
    {                                                                                                                 \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                    \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                   \
            /* MTE path - supports A2/A3/Ascend_950 */                                                                \
            auto ptr = aclshmem_ptr(dst, pe);                                                                         \
            __gm__ TYPE* remote_ptr = reinterpret_cast<__gm__ TYPE*>(ptr);                                            \
            __ubuf__ TYPE* buf = reinterpret_cast<__ubuf__ TYPE*>(device_state->mte_config.aclshmem_ub);              \
            AscendC::LocalTensor<TYPE> ub_tensor(AscendC::TPosition::VECIN, device_state->mte_config.aclshmem_ub, 1); \
            ub_tensor.SetValue(0, value);                                                                             \
            AscendC::TEventID sync_id = device_state->mte_config.sync_id;                                             \
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);                                                   \
            AscendC::SetAtomicAdd<TYPE>();                                                                            \
            aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(TYPE));                                                      \
            AscendC::SetAtomicNone();                                                                                 \
        } else {                                                                                                      \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_add for int8/int16/bfloat16/half is MTE-only.\n");    \
        }                                                                                                             \
    }

/**
 * @brief Direct support types for atomic_add.
 *        Supported types: int32_t, float.
 *        Supported hardware platform: A2/A3/Ascend_950 (MTE path), Ascend_950 (ROCE path).
 */
#define ACLSHMEM_ATOMIC_ADD_910_TYPENAME(NAME, TYPE)                                                                  \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)                       \
    {                                                                                                                 \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                    \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                   \
            /* MTE path - supports A2/A3/Ascend_950 */                                                                \
            auto ptr = aclshmem_ptr(dst, pe);                                                                         \
            __gm__ TYPE* remote_ptr = reinterpret_cast<__gm__ TYPE*>(ptr);                                            \
            __ubuf__ TYPE* buf = reinterpret_cast<__ubuf__ TYPE*>(device_state->mte_config.aclshmem_ub);              \
            AscendC::LocalTensor<TYPE> ub_tensor(AscendC::TPosition::VECIN, device_state->mte_config.aclshmem_ub, 1); \
            ub_tensor.SetValue(0, value);                                                                             \
            AscendC::TEventID sync_id = device_state->mte_config.sync_id;                                             \
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);                                                    \
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);                                                   \
            AscendC::SetAtomicAdd<TYPE>();                                                                            \
            aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(TYPE));                                                      \
            AscendC::SetAtomicNone();                                                                                 \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                           \
            /* ROCE path - supports Ascend_950 only */                                                                \
            if constexpr (std::is_same_v<TYPE, float>) {                                                              \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE does not support float for atomic_add\n");          \
            } else {                                                                                                  \
                aclshmemx_roce_atomic_add(dst, value, pe);                                                            \
            }                                                                                                         \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                           \
            /* UDMA path - supports Ascend_950 only */                                                                \
            aclshmemx_udma_atomic_add(dst, value, pe);                                                                \
        } else {                                                                                                      \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_add is only supported on MTE, ROCE and UDMA path.");  \
        }                                                                                                             \
    }

/**
 * @brief Support types for atomic_add.
 *        Supported types: uint32, uint64, int64 (MTE/ROCE/UDMA).
 *        Supported hardware platform: Ascend_950(MTE, ROCE and UDMA path).
 */
#define ACLSHMEM_ATOMIC_ADD_EXT_TYPENAME(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE* dst, TYPE value, int32_t pe)                      \
    {                                                                                                                \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                   \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                  \
            /* MTE path - supports Ascend_950 only, does not support int64 */                                        \
            aclshmemx_mte_atomic_add(dst, value, pe);                                                                \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                          \
            /* ROCE path - supports Ascend_950 only */                                                               \
            aclshmemx_roce_atomic_add(dst, value, pe);                                                               \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                          \
            /* UDMA path - supports Ascend_950 only */                                                               \
            aclshmemx_udma_atomic_add(dst, value, pe);                                                               \
        } else {                                                                                                     \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_add is only supported on MTE, ROCE and UDMA path."); \
        }                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_910(ACLSHMEM_ATOMIC_ADD_910_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_EXT(ACLSHMEM_ATOMIC_ADD_EXT_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch_add.
 *        Supported types: uint32, uint64, int32, int64, float.
 *        Supported hardware platform: Ascend_950(MTE, ROCE and UDMA path).
 */
#define ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_add(__gm__ TYPE* dest, TYPE value, int32_t pe)             \
    {                                                                                                              \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                 \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                \
            /* MTE path - supports Ascend_950 only */                                                              \
            TYPE res = aclshmemx_mte_atomic_fetch_add(dest, value, pe);                                            \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                        \
            return res;                                                                                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                        \
            /* ROCE path - supports Ascend_950 only */                                                             \
            if constexpr (std::is_same_v<TYPE, float>) {                                                           \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE does not support float for atomic_fetch_add\n"); \
                return 0;                                                                                          \
            } else {                                                                                               \
                return aclshmemx_roce_atomic_fetch_add(dest, value, pe);                                           \
            }                                                                                                      \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                        \
            /* UDMA path - supports Ascend_950 only */                                                             \
            return aclshmemx_udma_atomic_fetch_add(dest, value, pe);                                               \
        } else {                                                                                                   \
            ACLSHMEM_DEBUG_FUNC(                                                                                   \
                aclshmemi_kernel_abort, "atomic_fetch_add is only supported on MTE, ROCE and UDMA path. \n");      \
            return 0;                                                                                              \
        }                                                                                                          \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME);

/**
 * @brief Direct support types for atomic_inc.
 *        Supported types: uint32, uint64, int32, int64, float.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_INC_TYPENAME(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_inc(__gm__ TYPE* dst, int32_t pe)                          \
    {                                                                                                        \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                           \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                          \
            aclshmemx_mte_atomic_inc(dst, pe);                                                               \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                  \
            if constexpr (std::is_same_v<TYPE, float>) {                                                     \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE does not support float for atomic_inc\n"); \
            } else {                                                                                         \
                aclshmemx_roce_atomic_inc(dst, pe);                                                          \
            }                                                                                                \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                  \
            aclshmemx_udma_atomic_inc(dst, pe);                                                              \
        } else {                                                                                             \
            ACLSHMEM_DEBUG_FUNC(                                                                             \
                aclshmemi_kernel_abort, "atomic_inc is only supported on MTE, ROCE and UDMA path. \n");      \
        }                                                                                                    \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_INC_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch_inc.
 *        Supported types: uint32, uint64, int32, int64, float.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_inc(__gm__ TYPE* dest, int32_t pe)                         \
    {                                                                                                              \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                 \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                \
            /* MTE path - supports Ascend_950 only */                                                              \
            TYPE res = aclshmemx_mte_atomic_fetch_inc(dest, pe);                                                   \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                        \
            return res;                                                                                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                        \
            if constexpr (std::is_same_v<TYPE, float>) {                                                           \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE does not support float for atomic_fetch_inc\n"); \
                return 0;                                                                                          \
            } else {                                                                                               \
                return aclshmemx_roce_atomic_fetch_inc(dest, pe);                                                  \
            }                                                                                                      \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                        \
            if constexpr (std::is_same_v<TYPE, float>) {                                                           \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA does not support float for atomic_fetch_inc\n"); \
                return 0;                                                                                          \
            } else {                                                                                               \
                return aclshmemx_udma_atomic_fetch_inc(dest, pe);                                                  \
            }                                                                                                      \
        } else {                                                                                                   \
            ACLSHMEM_DEBUG_FUNC(                                                                                   \
                aclshmemi_kernel_abort, "atomic_fetch_inc is only supported on MTE, ROCE and UDMA path. \n");      \
            return 0;                                                                                              \
        }                                                                                                          \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME);

/**
 * @brief Direct support types for atomic_and.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_AND_TYPENAME(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_and(__gm__ TYPE* dest, TYPE value, int32_t pe)                   \
    {                                                                                                              \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                 \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                               \
            /* ROCE path - supports Ascend_950 only */                                                             \
            aclshmemx_roce_atomic_and(dest, value, pe);                                                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                        \
            aclshmemx_udma_atomic_and(dest, value, pe);                                                            \
        } else {                                                                                                   \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_and is only supported on ROCE and UDMA path. \n"); \
        }                                                                                                          \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_AND_TYPENAME);

/**
 * @brief Direct support types for atomic_or.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_OR_TYPENAME(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_or(__gm__ TYPE* dest, TYPE value, int32_t pe)                   \
    {                                                                                                             \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                              \
            /* ROCE path - supports Ascend_950 only */                                                            \
            aclshmemx_roce_atomic_or(dest, value, pe);                                                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                       \
            aclshmemx_udma_atomic_or(dest, value, pe);                                                            \
        } else {                                                                                                  \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_or is only supported on ROCE and UDMA path. \n"); \
        }                                                                                                         \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_OR_TYPENAME);

/**
 * @brief Direct support types for atomic_xor.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_XOR_TYPENAME(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_xor(__gm__ TYPE* dest, TYPE value, int32_t pe)                   \
    {                                                                                                              \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                 \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                               \
            /* ROCE path - supports Ascend_950 only */                                                             \
            aclshmemx_roce_atomic_xor(dest, value, pe);                                                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                        \
            aclshmemx_udma_atomic_xor(dest, value, pe);                                                            \
        } else {                                                                                                   \
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "atomic_xor is only supported on ROCE and UDMA path. \n"); \
        }                                                                                                          \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_XOR_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch_and.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME(NAME, TYPE)                                                   \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_and(__gm__ TYPE* dest, TYPE value, int32_t pe)   \
    {                                                                                                    \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                     \
            /* ROCE path - supports Ascend_950 only */                                                   \
            return aclshmemx_roce_atomic_fetch_and(dest, value, pe);                                     \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                              \
            return aclshmemx_udma_atomic_fetch_and(dest, value, pe);                                     \
        } else {                                                                                         \
            ACLSHMEM_DEBUG_FUNC(                                                                         \
                aclshmemi_kernel_abort, "atomic_fetch_and is only supported on ROCE and UDMA path. \n"); \
            return 0;                                                                                    \
        }                                                                                                \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch_or.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME(NAME, TYPE)                                                   \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_or(__gm__ TYPE* dest, TYPE value, int32_t pe)   \
    {                                                                                                   \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                      \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                    \
            /* ROCE path - supports Ascend_950 only */                                                  \
            return aclshmemx_roce_atomic_fetch_or(dest, value, pe);                                     \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                             \
            return aclshmemx_udma_atomic_fetch_or(dest, value, pe);                                     \
        } else {                                                                                        \
            ACLSHMEM_DEBUG_FUNC(                                                                        \
                aclshmemi_kernel_abort, "atomic_fetch_or is only supported on ROCE and UDMA path. \n"); \
            return 0;                                                                                   \
        }                                                                                               \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch_xor.
 *        Supported types: uint32, uint64, int32, int64.
 *        Supported hardware platform: Ascend_950(ROCE path).
 */
#define ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME(NAME, TYPE)                                                   \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_xor(__gm__ TYPE* dest, TYPE value, int32_t pe)   \
    {                                                                                                    \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                     \
            /* ROCE path - supports Ascend_950 only */                                                   \
            return aclshmemx_roce_atomic_fetch_xor(dest, value, pe);                                     \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                              \
            return aclshmemx_udma_atomic_fetch_xor(dest, value, pe);                                     \
        } else {                                                                                         \
            ACLSHMEM_DEBUG_FUNC(                                                                         \
                aclshmemi_kernel_abort, "atomic_fetch_xor is only supported on ROCE and UDMA path. \n"); \
            return 0;                                                                                    \
        }                                                                                                \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_LOGIC(ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME);

/**
 * @brief Direct support types for atomic_fetch.
 *        Supported types: uint32, uint64, int32, int64, float.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_FETCH_TYPENAME(NAME, TYPE)                                                               \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch(__gm__ const TYPE* source, int32_t pe)                   \
    {                                                                                                            \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                               \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                              \
            /* MTE path - supports Ascend_950 only */                                                            \
            TYPE res = aclshmemx_mte_atomic_fetch(const_cast<__gm__ TYPE*>(source), pe);                         \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                      \
            return res;                                                                                          \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                      \
            /* ROCE path - supports Ascend_950 only */                                                           \
            return aclshmemx_roce_atomic_fetch(const_cast<__gm__ TYPE*>(source), pe);                            \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                      \
            /* UDMA path - supports Ascend_950 only */                                                           \
            if constexpr (std::is_same_v<TYPE, float>) {                                                         \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA does not support float for atomic_fetch. \n"); \
                return 0;                                                                                        \
            } else {                                                                                             \
                return aclshmemx_udma_atomic_fetch(const_cast<__gm__ TYPE*>(source), pe);                        \
            }                                                                                                    \
        } else {                                                                                                 \
            ACLSHMEM_DEBUG_FUNC(                                                                                 \
                aclshmemi_kernel_abort, "atomic_fetch is only supported on MTE, ROCE and UDMA path. \n");        \
            return 0;                                                                                            \
        }                                                                                                        \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD_950(ACLSHMEM_ATOMIC_FETCH_TYPENAME);

/**
 * @brief Direct support types for atomic_set.
 *        Supported types: uint32, uint64.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_SET_TYPENAME(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_set(__gm__ TYPE* dest, TYPE value, int32_t pe)        \
    {                                                                                                   \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                      \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                     \
            /* MTE path - supports Ascend_950 only */                                                   \
            aclshmemx_mte_atomic_set(dest, value, pe);                                                  \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                             \
            /* ROCE path - supports Ascend_950 only */                                                  \
            aclshmemx_roce_atomic_set(dest, value, pe);                                                 \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                             \
            /* UDMA path - supports Ascend_950 only */                                                  \
            aclshmemx_udma_atomic_set(dest, value, pe);                                                 \
        } else {                                                                                        \
            ACLSHMEM_DEBUG_FUNC(                                                                        \
                aclshmemi_kernel_abort, "atomic_set is only supported on MTE, ROCE and UDMA path. \n"); \
        }                                                                                               \
    }

/**
 * @brief CAST support types for atomic_set.
 *        Supported types: int32, int64, float.
 *        Converts dst, value to underlying unsigned type.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_SET_TYPENAME_CAST(NAME, TYPE, SUBTYPE)                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_set(__gm__ TYPE* dest, TYPE value, int32_t pe)               \
    {                                                                                                          \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                             \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                            \
            /* MTE path - supports Ascend_950 only */                                                          \
            aclshmemx_mte_atomic_set(reinterpret_cast<__gm__ SUBTYPE*>(dest), *((SUBTYPE*)&value), pe);        \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                    \
            /* ROCE path - supports Ascend_950 only */                                                         \
            aclshmemx_roce_atomic_set(reinterpret_cast<__gm__ SUBTYPE*>(dest), *((SUBTYPE*)&value), pe);       \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                    \
            /* UDMA path - supports Ascend_950 only */                                                         \
            if constexpr (std::is_same_v<TYPE, float>) {                                                       \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA does not support float for atomic_set. \n"); \
            } else {                                                                                           \
                aclshmemx_udma_atomic_set(dest, value, pe);                                                    \
            }                                                                                                  \
        } else {                                                                                               \
            ACLSHMEM_DEBUG_FUNC(                                                                               \
                aclshmemi_kernel_abort, "atomic_set is only supported on MTE, ROCE and UDMA path. \n");        \
        }                                                                                                      \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_SET_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST(ACLSHMEM_ATOMIC_SET_TYPENAME_CAST);

/**
 * @brief Direct support types for atomic_swap.
 *        Supported types: uint32, uint64.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_SWAP_TYPENAME(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_swap(__gm__ TYPE* dest, TYPE value, int32_t pe)        \
    {                                                                                                    \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                      \
            /* MTE path - supports Ascend_950 only */                                                    \
            TYPE res = aclshmemx_mte_atomic_swap(dest, value, pe);                                       \
            AscendC::PipeBarrier<PIPE_ALL>();                                                              \
            return res;                                                                                  \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                              \
            /* ROCE path - supports Ascend_950 only */                                                   \
            return aclshmemx_roce_atomic_swap(dest, value, pe);                                          \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                              \
            /* UDMA path - supports Ascend_950 only */                                                   \
            return aclshmemx_udma_atomic_swap(dest, value, pe);                                          \
        } else {                                                                                         \
            ACLSHMEM_DEBUG_FUNC(                                                                         \
                aclshmemi_kernel_abort, "atomic_swap is only supported on MTE, ROCE and UDMA path. \n"); \
            return 0;                                                                                    \
        }                                                                                                \
    }

/**
 * @brief CAST support types for atomic_swap.
 *        Supported types: int32, int64, float.
 *        Converts dest, value to underlying unsigned type.
 *        Supported hardware platform: Ascend_950(MTE and ROCE path).
 */
#define ACLSHMEM_ATOMIC_SWAP_TYPENAME_CAST(NAME, TYPE, SUBTYPE)                                               \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_swap(__gm__ TYPE* dest, TYPE value, int32_t pe)             \
    {                                                                                                         \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                            \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                           \
            /* MTE path - supports Ascend_950 only */                                                         \
            SUBTYPE temp =                                                                                    \
                aclshmemx_mte_atomic_swap(reinterpret_cast<__gm__ SUBTYPE*>(dest), *((SUBTYPE*)&value), pe);  \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                   \
            return *((TYPE*)&temp);                                                                           \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                   \
            /* ROCE path - supports Ascend_950 only */                                                        \
            SUBTYPE temp =                                                                                    \
                aclshmemx_roce_atomic_swap(reinterpret_cast<__gm__ SUBTYPE*>(dest), *((SUBTYPE*)&value), pe); \
            return *((TYPE*)&temp);                                                                           \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                   \
            /* UDMA path - supports Ascend_950 only */                                                        \
            if constexpr (std::is_same_v<TYPE, float>) {                                                      \
                ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA does not support float for atomic_swap\n"); \
                return 0;                                                                                     \
            } else {                                                                                          \
                return aclshmemx_udma_atomic_swap(dest, value, pe);                                           \
            }                                                                                                 \
        } else {                                                                                              \
            ACLSHMEM_DEBUG_FUNC(                                                                              \
                aclshmemi_kernel_abort, "atomic_swap is only supported on MTE, ROCE and UDMA path. \n");      \
            return 0;                                                                                         \
        }                                                                                                     \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_SWAP_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST(ACLSHMEM_ATOMIC_SWAP_TYPENAME_CAST);

/**
 * @brief Direct support types for atomic_compare_swap.
 *        Supported types: uint32, uint64.
 *        Supported hardware platform: Ascend_950(MTE, ROCE and UDMA path).
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME(NAME, TYPE)                                                            \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe) \
    {                                                                                                                \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                   \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                  \
            /* MTE path - supports Ascend_950 only */                                                                \
            TYPE res = aclshmemx_mte_atomic_compare_swap(dest, cond, value, pe);                                     \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                          \
            return res;                                                                                              \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                          \
            /* ROCE path - supports Ascend_950 only */                                                               \
            return aclshmemx_roce_atomic_compare_swap(dest, cond, value, pe);                                        \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                          \
            /* UDMA path - supports Ascend_950 only */                                                               \
            return aclshmemx_udma_atomic_compare_swap(dest, cond, value, pe);                                        \
        } else {                                                                                                     \
            ACLSHMEM_DEBUG_FUNC(                                                                                     \
                aclshmemi_kernel_abort, "atomic_compare_swap is only supported on MTE, ROCE and UDMA path. \n");     \
            return 0;                                                                                                \
        }                                                                                                            \
    }

/**
 * @brief CAST support types for atomic_compare_swap.
 *        Supported types: int32, int64.
 *        Converts dest, cond, value to underlying unsigned type.
 *        Supported hardware platform: Ascend_950(MTE, ROCE and UDMA path).
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_CAST(NAME, TYPE, SUBTYPE)                                              \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe) \
    {                                                                                                                \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                                   \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                  \
            /* MTE path - supports Ascend_950 only */                                                                \
            SUBTYPE temp = aclshmemx_mte_atomic_compare_swap(                                                        \
                reinterpret_cast<__gm__ SUBTYPE*>(dest), *((SUBTYPE*)&cond), *((SUBTYPE*)&value), pe);               \
            AscendC::PipeBarrier<PIPE_ALL>();                                                                          \
            return *((TYPE*)&temp);                                                                                  \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                          \
            /* ROCE path - supports Ascend_950 only */                                                               \
            return aclshmemx_roce_atomic_compare_swap(dest, cond, value, pe);                                        \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                          \
            /* UDMA path - supports Ascend_950 only */                                                               \
            return aclshmemx_udma_atomic_compare_swap(dest, cond, value, pe);                                        \
        } else {                                                                                                     \
            ACLSHMEM_DEBUG_FUNC(                                                                                     \
                aclshmemi_kernel_abort, "atomic_compare_swap is only supported on MTE, ROCE and UDMA path. \n");     \
            return 0;                                                                                                \
        }                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME);
ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_CAST);

#endif // SHMEM_DEVICE_AMO_HPP