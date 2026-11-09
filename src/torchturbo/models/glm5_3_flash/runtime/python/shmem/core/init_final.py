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
import os

import shmem._pyshmem as _pyshmem
from shmem.core.utils import setup_aclshmem_logger, AclshmemError, AclshmemInvalid

__all__ = ['get_unique_id', 'init', 'finalize', 'get_version', 'UniqueID']

logger = logging.getLogger("aclshmem")

UniqueID = _pyshmem.UniqueId


def get_version() -> str:
    """
    Get the ACLSHMEM library version as a formatted string.

    Returns:
        str: Version string in the format ``"libaclshmem_version=X.Y"``.
    """
    major, minor = _pyshmem.aclshmem_info_get_version()
    res = f"libaclshmem_version={major}.{minor}"
    return res


def get_unique_id(empty: bool=False) -> UniqueID:
    """
    Create a unique ID used for UID-based ACLSHMEM initialization.

    This function generates an unique ID to initialize the aclshmem process. The ID should be generated
    by a single process (e.g., rank 0) and distributed to other processes via broadcast.

    Args:
        empty: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``False``.

    Returns:
        UniqueID: A handle representing the generated unique ID (internally stored as bytes).
                  This object is opaque but can be pickled and sent across processes.

    Raises:
        AclshmemError: If generation of a unique ID fails.
    """
    try:
        u_id = _pyshmem.aclshmem_get_unique_id()
    except Exception as e:
        raise AclshmemError("Generate a unique ID fails.") from e

    return u_id


def init(device: int=None, uid: UniqueID=None, rank: int=None, nranks: int=None, mpi_comm=None,
         initializer_method: str="", mem_size: int=None) -> None:
    """
    Initialize the ACLSHMEM runtime with unique ID.

    Args:
        device: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``None``.
        uid (aclshmem_uid, required): A unique identifier used for initialization.
        rank (int, required): The rank (0-based index) of the current process within the ACLSHMEM job.
        nranks (int, required): The total number of processes (ranks) participating in the ACLSHMEM job.
        mpi_comm: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``None``.
        initializer_method (str): Specifies the initialization method. Must be "uid".
        mem_size (int, required): Memory size for each processing element in bytes.

    Raises:
        AclshmemInvalid: If the required arguments for the selected method are missing or incorrect.
        AclshmemError: If ACLSHMEM fails to initialize using the specified method.
    """
    log_level = os.environ.get("SHMEM_LOG_LEVEL")

    if not log_level or log_level not in ["DEBUG", "INFO", "WARN", "ERROR", None]:
        logger.warning("Set log level to 'ERROR'.")
        log_level = "ERROR"
    if log_level == "WARN":
        log_level = "WARNING"
    setup_aclshmem_logger(log_level=log_level)
    if initializer_method not in ["uid"]:
        raise AclshmemInvalid("Invalid init method requested")

    if any(arg is None for arg in (uid, rank, nranks, mem_size)):
        raise AclshmemInvalid("uid, rank and nranks must be specified.")

    ret = _pyshmem.aclshmem_init_using_unique_id(rank, nranks, mem_size, uid)
    if ret != 0:
        raise AclshmemError("ACLSHMEM initialization fails.")


def finalize() -> None:
    """
    Finalize the ACLSHMEM runtime.

    This function should be called once per process after all ACLSHMEM operations are complete,
    typically before application exit, to release resources.

    Raises:
        AclshmemError: If the ACLSHMEM finalization fails.
    """
    ret = _pyshmem.aclshmem_finalize()
    if ret != 0:
        raise AclshmemError("ACLSHMEM finalization fails.")
