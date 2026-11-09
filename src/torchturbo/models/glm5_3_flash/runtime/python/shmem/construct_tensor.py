#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import torch
import torch_npu

def calc_nbytes(shape, dtype: torch.dtype):
    if shape is None:
        return 0

    ele_num = 1
    for n in shape:
        ele_num *= n
    return ele_num * torch.tensor([], dtype=dtype).element_size()

def construct_tensor_from_ptr(data_ptr, shape, dtype, device, stride=None, storage_offset=None):
    if not isinstance(data_ptr, int):
        raise TypeError(f"data_ptr must be integer type, got {type(data_ptr)}")
    if data_ptr <= 0:
        raise ValueError(f"data_ptr must be positive, got {data_ptr}")
    if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be torch.dtype type, got {type(dtype)}")
    if stride is None:
        if len(shape) == 0:
            stride = ()
        else:
            stride_list = [1]
            for i in range(-1, -len(shape), -1):
                dim = shape[i]
                if dim < 0:
                    raise ValueError(f"shape dim must be positive, got {dim}")
                stride_list = [stride_list[0] * dim] + stride_list
            stride = tuple(stride_list)

    if storage_offset is None:
        storage_offset = 0

    nbytes = calc_nbytes(shape, dtype)
    metadata = {
        "data_ptr": data_ptr,
        "device": device,
        "nbytes": nbytes,
        "dtype": dtype,
        "size": shape,
        "stride": stride,
        "storage_offset": storage_offset
    }

    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )

    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
