#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
import logging

import shmem._pyshmem as _pyshmem
from shmem.core.utils import Buffer, AclshmemError

__all__ = ['buffer', 'free', 'get_peer_buffer']

logger = logging.getLogger("aclshmem")


def buffer(size, release=False, except_on_del=True) -> Buffer:
    """
    Allocates an ACLSHMEM-backed npu buffer.

    Args:
        size (int): The size in bytes of the buffer to allocate.
        release: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``False``.
        except_on_del: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``True``.

    Returns:
        Buffer: A raw memory buffer via its address and byte length.

    Raises:
        AclshmemError: If the buffer could not be allocated properly.
    """
    try:
        dev_ptr = _pyshmem.aclshmem_malloc(size)
    except:
        raise AclshmemError("Allocate buffer failed.")

    buf = Buffer(dev_ptr, size)
    return buf


def free(buf: Buffer) -> None:
    """
    Free an ACLSHMEM buffer previously allocated with ``alloc``.

    Args:
        buf (Buffer): The buffer to be freed.
    """

    _pyshmem.aclshmem_free(buf.addr)


def get_peer_buffer(buf: Buffer, pe: int) -> Buffer:
    """
    Get address that may be used to directly reference dest on the specified PE.

    Args:
        buf (Buffer): The symmetric address of the remotely accessible data.
        pe (int): PE number

    Returns:
        Buffer: A remote symmetric address on the specified PE that can be accessed using memory loads and stores.

    Raises:
        AclshmemError:  If the input address is illegal.
    """
    peer_addr = _pyshmem.aclshmem_ptr(buf.addr, pe)
    if peer_addr == 0:
        raise AclshmemError("Get the symmetric address on a specified PE failed.")

    peer_buffer = Buffer(peer_addr, buf.length)
    return peer_buffer
