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
from shmem.core.utils import Buffer, AclshmemInvalid
from shmem.core.direct import ComparisonType, SignalOp

__all__ = ['put_signal', 'signal_op', 'signal_wait', 'put', 'get', 'quiet']

logger = logging.getLogger("aclshmem")


def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp,
               remote_pe: int=-1, stream=None) -> None:
    """
    Synchronous (blocking) interface. Copy contiguous data from the local PE to a
    symmetric memory address on the specified PE, and update a remote signal variable
    on completion.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        dst (Buffer):
            [in] Symmetric address of the destination data on the remote PE.
        src (Buffer):
            [in] Local memory of the source data.
        signal_var (Buffer):
            [in] Symmetric address of the signal word to be updated on the remote PE.
        signal_val (int):
            [in] The value used to update the signal variable.
        signal_operation (SignalOp):
            [in] Operation used to update the signal variable with signal_val.
            Supported: ``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``.
        remote_pe (int):
            [in] PE number of the remote PE. Defaults to -1.
        stream:
            [in] Reserved parameter, ignored. The underlying call uses the default
            stream internally.

    Returns:
        None: This function has no return value.
    """
    _pyshmem.aclshmemx_putmem_signal(
        dst.addr, src.addr, src.length, signal_var.addr, signal_val, signal_operation, remote_pe
    )


def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1,
              stream: int=None) -> None:
    """
    Non-blocking interface. Performs an atomic operation on a remote signal variable
    at the specified PE, with the operation executed on the given stream. The caller
    must synchronize the stream to observe the result.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        signal_var (Buffer):
            [in] Local address of the signal variable that is accessible at the target PE.
        signal_val (int):
            [in] The value to be used in the atomic operation.
        signal_operation (SignalOp):
            [in] The operation to perform on the remote signal.
            Supported: ``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``.
        remote_pe (int):
            [in] The PE number on which the remote signal variable is to be updated.
            Defaults to -1.
        stream (int):
            [in] ACL stream object used for execution ordering. Must be a valid stream
            created via ACL runtime. Passing ``None`` raises ``AclshmemInvalid``.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If ``stream`` is ``None``.
    """
    if stream is None:
        logger.error("Signal operations without an explicit stream are not supported.")
        raise AclshmemInvalid("Signal operations without an explicit stream are not supported.")
    _pyshmem.aclshmemx_signal_op_on_stream(signal_var.addr, signal_val, signal_operation, remote_pe, stream)


def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None:
    """
    Waits until a symmetric signal variable satisfies a given condition. The wait
    is performed on the specified stream; the call returns immediately on the host.
    When the stream is synchronized, the condition
    ``signal_var`` ``cmp`` ``signal_val`` is guaranteed to be true.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        signal_var (Buffer):
            [in] Local address of the source signal variable.
        signal_val (int):
            [in] The value against which the object pointed to by signal_var will be
            compared.
        signal_operation (ComparisonType):
            [in] The comparison operator used to evaluate the condition.
            Supported: ``ComparisonType.CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` /
            ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``.
        stream (int):
            [in] ACL stream object used for execution ordering. Must be a valid
            stream created via ACL runtime.

    Returns:
        None: This function has no return value.
    """
    if stream is None:
        logger.error("Signal wait operations without an explicit stream are not supported.")
        raise AclshmemInvalid("Signal wait operations without an explicit stream are not supported.")
    _pyshmem.aclshmemx_signal_wait_until_on_stream(
        signal_var.addr, signal_operation, signal_val, stream
    )


def put(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None:
    """
    Non-blocking interface. Copy contiguous data from the local PE to a symmetric
    memory address on a remote PE, ordered on the given stream. The caller must
    synchronize the stream to ensure the transfer is complete.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        dst (Buffer):
            [in] Symmetric address of the destination data on the remote PE.
        src (Buffer):
            [in] Local memory of the source data.
        remote_pe (int):
            [in] PE number of the remote PE. Defaults to -1.
        stream (int):
            [in] ACL stream object used for execution ordering. Passing ``0`` or
            ``None`` uses the default stream.

    Returns:
        None: This function has no return value.
    """
    _pyshmem.aclshmemx_putmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream)

def get(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None:
    """
    Non-blocking interface. Copy contiguous data from symmetric memory on a remote PE
    to a local buffer, ordered on the given stream. The caller must synchronize the
    stream to ensure the transfer is complete.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        dst (Buffer):
            [in] Local memory of the destination data.
        src (Buffer):
            [in] Symmetric address of the source data on the remote PE.
        remote_pe (int):
            [in] PE number of the remote PE. Defaults to -1.
        stream (int):
            [in] ACL stream object used for execution ordering. Passing ``0`` or
            ``None`` uses the default stream.

    Returns:
        None: This function has no return value.
    """
    _pyshmem.aclshmemx_getmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream)


def quiet(stream: int) -> None:
    """
    Ensures completion of all previously issued operations on symmetric data
    objects on the given stream. The quiet is queued on the specified stream;
    the caller must synchronize the stream to observe completion from the host.

    .. note::
        Currently only MTE (Memory Transfer Engine) is supported.

    Args:
        stream (int):
            [in] ACL stream on which to queue the quiet operation. Must be a
            valid stream created via ACL runtime.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If ``stream`` is ``None``.
    """
    if stream is None:
        logger.error("quiet operations without an explicit stream are not supported.")
        raise AclshmemInvalid("quiet operations without an explicit stream are not supported.")
    _pyshmem.aclshmemx_quiet_on_stream(stream)

