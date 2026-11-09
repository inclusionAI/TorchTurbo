/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_API_H
#define SHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/gm2gm/shmem_device_amo.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "device/gm2gm/shmem_device_p2p_sync.h"
#include "device/gm2gm/shmem_device_rma.h"
#include "device/gm2gm/shmem_device_so.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/gm2gm/engine/shmem_device_udma.h"
#include "device/ub2gm/shmem_device_rma.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "device/shmem_def.h"
#include "device/team/shmem_device_team.h"

// simt
#if defined(USE_SIMT)
#include "device_simt/gm2gm/shmem_device_simt_rma.h"
#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/ub2gm/shmem_device_simt_rma.h"
#include "device_simt/ub2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/team/shmem_device_simt_team.h"
#endif
#endif

#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_so.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "host/team/shmem_host_team.h"
#include "host/utils/shmem_log.h"

#endif // SHMEM_API_H
