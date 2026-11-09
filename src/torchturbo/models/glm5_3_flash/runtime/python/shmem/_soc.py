#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared runtime utilities: SoC detection and backend selection.

Reuses the same aclrtGetSocName() path as the C++ examples
(see examples/combine/combine_classic/main.cpp) instead of the unreliable
acl.rt.get_device_info Python wrapper.
"""
import ctypes
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)


def _read_proc_maps():
    """Read /proc/self/maps lines; returns [] if inaccessible (container/sandbox/non-Linux)."""
    try:
        return Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def detect_soc():
    """Detect the SoC type via aclrtGetSocName (stable C API, no set_device needed).

    Uses RTLD_LOCAL to avoid polluting the global symbol table, since this
    function may be called from shmem-config (standalone, no torch_npu preload).
    """
    try:
        lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_LOCAL)
        func = lib.aclrtGetSocName
        func.restype = ctypes.c_char_p
        soc = func()
        if soc:
            soc_str = soc.decode("utf-8")
            soc_lower = soc_str.lower()
            if any(k in soc_lower for k in ("ascend950", "dav-3510", "3510")):
                return "950"
            if any(k in soc_lower for k in ("ascend910", "dav-220", "dav-2201")):
                return "910"
    except Exception as e:
        _logger.debug("aclrtGetSocName SoC detection failed: %s", e)

    return None


def select_backend():
    """Auto-detect backend via aclrtGetSocName, falling back to "910"."""
    soc = detect_soc()
    return soc if soc else "910"
