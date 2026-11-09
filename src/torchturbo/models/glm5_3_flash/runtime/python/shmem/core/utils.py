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
import sys
import os
import socket
import threading

import shmem._pyshmem as _pyshmem

def setup_aclshmem_logger(log_level: str="ERROR", log_file_path: str=None) -> None:
    """
    Initialize and configure the logger for the ACLSHMEM Python interface.

    The log format includes host, process ID, thread ID, and PE rank (if available):
        <host>:<pid>:<tid> [<PE>] ACLSHMEM <LEVEL> : <message>

    Args:
        log_level (str): Desired logging level. One of 'DEBUG', 'INFO', 'WARNING',
                         'ERROR', or 'CRITICAL'. Defaults to 'ERROR'.
        log_file_path (str, optional): If specified, logs will also be written to this file.
    """
    try:
        pe_rank = _pyshmem.my_pe()
    except Exception as e:
        print("Unable to retrieve PE rank. Ensure ACLSHMEM is initialized.")
        pe_rank = -1

    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Build log prefix
    host = socket.getfqdn().split(".")[0]
    pid = os.getpid()
    tid = threading.get_native_id()
    log_format = f"{host}:{pid}:{tid} [{pe_rank}] ACLSHMEM %(levelname)s : %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Set up dedicated logger (not root)
    logger = logging.getLogger("aclshmem")
    logger.setLevel(numeric_level)
    logger.handlers.clear()  # Remove existing handlers to avoid duplication
    logger.addHandler(console_handler)

    # Optional file logging
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

class AclshmemError(Exception):
    def __init__(self,  msg):
        super().__init__(msg)
        self.msg = msg

    def __repr__(self):
        return f"<AclshmemError: {self.msg}>"


class AclshmemInvalid(Exception):
    def __init__(self,  msg):
        super().__init__(msg)
        self.msg = msg

    def __repr__(self):
        return f"<AclshmemInvalid: {self.msg}>"


class Buffer:
    """
    A lightweight wrapper representing a memory buffer by its address and size.

    Attributes:
        addr (int): Memory address (e.g., from CuPy, PyTorch, or CUDA driver).
        length (int): Size of the buffer in bytes.
    """
    __slots__ = ('addr', 'length')

    def __init__(self, addr: int, length: int):
        if addr < 0:
            raise ValueError("Address must be non-negative")
        if length <= 0:
            raise ValueError("Length must be positive")
        self.addr = addr
        self.length = length

    def __repr__(self):
        return f"Buffer(addr=0x{self.addr:x}, length={self.length})"

    def __eq__(self, other):
        if not isinstance(other, Buffer):
            return False
        return self.addr == other.addr and self.length == other.length