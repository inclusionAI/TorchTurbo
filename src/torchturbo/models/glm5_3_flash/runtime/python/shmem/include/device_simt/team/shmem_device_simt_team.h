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
#ifndef SHMEM_DEVICE_SIMT_TEAM_H
#define SHMEM_DEVICE_SIMT_TEAM_H

#include "device_simt/shmem_simt_common_types.h"
#include "team/shmem_device_simt_team.hpp"

namespace simt {

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
__simt_callee__ inline int aclshmem_my_pe(void);

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
__simt_callee__ inline int aclshmem_n_pes(void);

} // namespace simt

#endif
