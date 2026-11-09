/**
 * @cond IGNORE_COPYRIGHT
?* Copyright (c) 2026 Huawei Technologies Co., Ltd.
?* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
?* CANN Open Software License Agreement Version 2.0 (the "License").
?* Please refer to the License for details. You may not use this file except in compliance with the License.
?* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
?* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
?* See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
?*/

#ifndef SHMEM_MACROS_H
#define SHMEM_MACROS_H

#define ACLSHMEM_SIZE_FUNC(FUNC)  \
    FUNC(8);                      \
    FUNC(16);                     \
    FUNC(32);                     \
    FUNC(64);                     \
    FUNC(128)

#endif