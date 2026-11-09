# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    set(KERNEL_COMPUTE_UNIT ascend910b1)
elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
    set(KERNEL_COMPUTE_UNIT ascend910_9391)
else()
    message(
        FATAL_ERROR
        "Unsupported MegaMoe compute unit: ${ASCEND_COMPUTE_UNIT}"
    )
endif()

set(CANN_OP_COMMON_INCLUDE ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${MEGA_MOE_SOURCE_ROOT}
    TILING_LIBRARY cust_optiling
)

npu_op_kernel_options(ascendc_kernels ALL OPTIONS
    -DASCENDC_DUMP=0
    -I${CANN_OP_COMMON_INCLUDE}
    -I${SHMEM_SDK_ROOT}/include
    -I${SHMEM_SDK_ROOT}/include/device
    -I${SHMEM_SDK_ROOT}/src/device
    -Wno-macro-redefined
    -Wno-ignored-attributes
)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE MegaMoe
    KERNEL_DIR ./mega_moe/op_kernel
    KERNEL_FILE mega_moe.cpp
    COMPUTE_UNIT ${KERNEL_COMPUTE_UNIT}
)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE MegaMoeGrad
    KERNEL_DIR ./mega_moe_grad/op_kernel
    KERNEL_FILE mega_moe_grad.cpp
    COMPUTE_UNIT ${KERNEL_COMPUTE_UNIT}
)




npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
