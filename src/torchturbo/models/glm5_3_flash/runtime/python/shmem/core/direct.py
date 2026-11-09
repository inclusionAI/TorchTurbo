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
import shmem._pyshmem as _pyshmem

__all__ = ["ComparisonType", "SignalOp", "InitStatus", "my_pe", "team_my_pe", "team_n_pes", "n_pes", 'init_status']

"""Python IntEnum equivalent of the C enum `aclshmemx_init_status_t`."""
InitStatus = _pyshmem.InitStatus

"""Python IntEnum equivalent of the C enum `aclshmem_signal_op_type_t`."""
SignalOp = _pyshmem.SignalOp

"""Python IntEnum equivalent of the C enum `aclshmem_cmp_op_type_t`."""
ComparisonType = _pyshmem.CmpOp


def my_pe() -> int:
    """
    Get the Processing Element (PE) ID of the calling process.

    Returns:
        int: Returns the PE number of the local PE.
    """
    return _pyshmem.my_pe()


def n_pes() -> int:
    """
    Get the total number of Processing Elements (PEs) in the world team.

    Returns:
        int: Returns the number of PEs running in the program.
    """
    return _pyshmem.pe_count()


def team_my_pe(team) -> int:
    """
    Get the PE ID of this process within a specified team.

    Args:
        team (int): Target team id.

    Returns:
        int: The number of the calling PE within the specified team.
             If the team handle is ACLSHMEM_TEAM_INVALID, returns -1.
    """
    return _pyshmem.team_my_pe(team)


def team_n_pes(team) -> int:
    """
    Return the number of processing elements (PEs) in the given team.

    Args:
        team (int): Target team id.

    Returns:
        int: Number of PEs in the team. Returns -1 if the team ID is invalid.
    """
    return _pyshmem.team_n_pes(team)


def init_status() -> InitStatus:
    """
    Return the current initialization status of the ACLSHMEM library.

    Returns:
        InitStatus: Enumeration indicating whether the library is uninitialized,
                    initialized, or in an error state.
    """
    return _pyshmem.aclshmemx_init_status()
