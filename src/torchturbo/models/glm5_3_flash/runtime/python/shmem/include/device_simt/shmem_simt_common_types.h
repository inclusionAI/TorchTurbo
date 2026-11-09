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
#ifndef SHMEM_SIMT_COMMON_TYPES_H
#define SHMEM_SIMT_COMMON_TYPES_H

#include "device/shmem_def.h"

#if defined(USE_SIMT)
#include "simt_api/asc_simt.h"

typedef struct {
    int version;             ///< Version number of the state structure
    int mype;                ///< Global number of the local PE
    int npes;                ///< Total number of PEs in the system
    uint64_t heap_base;      ///< Start address of the local memory heap
    uint64_t host_heap_base; ///< heap_base for host side memory type

    // Store All Devices' heap_base.
    uint64_t p2p_device_heap_base;  ///< Array of P2P memory heap base addresses for all devices
    uint64_t rdma_device_heap_base; ///< Array of RDMA memory heap base addresses for all devices
    uint64_t sdma_device_heap_base; ///< Array of SDMA memory heap base addresses for all devices

    // Store All Host' heap_base.
    uint64_t p2p_host_heap_base;  ///< Array of P2P memory heap base addresses for all hosts
    uint64_t rdma_host_heap_base; ///< Array of RDMA memory heap base addresses for all hosts
    uint64_t sdma_host_heap_base; ///< Array of SDMA memory heap base addresses for all hosts

    uint8_t
        topo_list[ACLSHMEM_MAX_PES]; ///< PE topology list, storing the topological relationship of PEs in the system
    size_t heap_size;                ///< Total size of the local memory heap in bytes

    uint64_t team_pools[ACLSHMEM_MAX_TEAMS]; ///< Team pool array, storing all created team instances

    // Using aclshmemx_sync_bit instead of basic types as sync flag,
    // avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_device_cc.h for more details.
    // These members are 'aclshmemx_sync_bit *' types actually, but are defined as 'uint64_t' due to compiler
    // restriction.
    uint64_t sync_pool; ///< NPU-level sync pool pointer (actual type is aclshmemx_sync_bit*, defined as uint64_t due to
                        ///< compiler restrictions)
    uint64_t sync_counter; ///< NPU-level sync counter pointer (actual type is aclshmemx_sync_bit*, defined as uint64_t
                           ///< due to compiler restrictions)
    uint64_t core_sync_pool; ///< Core-level sync pool pointer (actual type is aclshmemx_sync_bit*, defined as uint64_t
                             ///< due to compiler restrictions)
    uint64_t core_sync_counter; ///< Core-level sync counter pointer (actual type is aclshmemx_sync_bit*, defined as
                                ///< uint64_t due to compiler restrictions)

    bool is_aclshmem_initialized; ///< Flag indicating whether ACLSHMEM has completed initialization
    bool is_aclshmem_created;     ///< Flag indicating whether ACLSHMEM has been created

    aclshmem_mte_config_t mte_config;   ///< Configuration information of the MTE memory transfer engine
    aclshmem_sdma_config_t sdma_config; ///< Configuration information of the SDMA memory transfer engine
    aclshmem_rdma_config_t rdma_config; ///< Configuration information of RDMA
    aclshmem_udma_config_t udma_config; ///< Configuration information of UDMA MTE staging
    uint64_t qp_info;                   ///< Queue Pair (QP) information, used for communication mechanisms such as RDMA

    uint64_t sdma_workspace_addr; /// sdma aicpu和aiv的共享内存
    uint64_t profs;               ///< for profiling
    uint64_t signal_addr;         ///< Address of the signal counter for put and get
} aclshmem_device_host_state_simt_t;

static_assert(
    sizeof(aclshmem_device_host_state_simt_t) == sizeof(aclshmem_device_host_state_t),
    "Please make sure sizeof(aclshmem_device_host_state_simt_t) == sizeof(aclshmem_device_host_state_t)");

#endif

#endif // !SHMEM_SIMT_COMMON_TYPES_H
