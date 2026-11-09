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

import os
import re
import sys
import ctypes
import logging
import subprocess
import torch
import torch_npu
from pathlib import Path

from ._soc import detect_soc, select_backend, _read_proc_maps
from .shmem_config import _read_version

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)

_PKG_ROOT = Path(current_dir)

def _is_debug_build():
    """Return True for debug builds (parses 'Build Type' key from version.info)."""
    version_file = _PKG_ROOT / "version.info"
    if version_file.exists():
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if re.match(r'build[_:\s]*type[_:\s]*:', line, re.IGNORECASE):
                return line.split(":", 1)[-1].strip().lower() == "debug"
    return False


_STRICT_MODE = not _is_debug_build()

_env_already_checked = False


def _hint():
    """Return a shmem-config --diagnose hint, only in debug builds."""
    return ("\nRun 'shmem-config --diagnose' for details." if not _STRICT_MODE else "")


def _get_backend_so_dir():
    backend = select_backend()
    return _PKG_ROOT / "backends" / backend


def _pre_load_guard():
    """Validate backend artifacts exist and SoC is detectable, before loading .so files.

    Runs before any CDLL() call, so /proc/self/maps scanning is deferred to
    _post_load_guard().

    In release builds issues raise RuntimeError immediately.
    In debug builds (built with -debug) issues are logged as warnings so that
    `shmem-config --diagnose` always runs.
    """
    warnings = []

    backend_so_dir = _get_backend_so_dir()

    if not backend_so_dir.exists():
        msg = (
            f"Backend directory not found: {backend_so_dir}. "
            f"Check that cann-shmem was built for the correct SOC_TYPE."
            f"{_hint()}"
        )
        if _STRICT_MODE:
            raise RuntimeError(msg)
        warnings.append(msg)

    expected_libshmem = backend_so_dir / "libshmem.so"
    if not expected_libshmem.exists():
        msg = (
            f"libshmem.so not found in backend dir: {expected_libshmem}. "
            f"Backend artifacts may be incomplete. Rebuild the wheel."
            f"{_hint()}"
        )
        if _STRICT_MODE:
            raise RuntimeError(msg)
        warnings.append(msg)

    soc = detect_soc()
    if soc is None:
        warnings.append(
            "Auto SoC detection failed, fallback to default 910 backend (degraded mode). "
            "Ensure Ascend driver is loaded. "
            "Run 'shmem-config --diagnose' for details."
        )

    for w in warnings:
        logging.warning(f"[SHMEM startup guard] {w}")


def _post_load_guard():
    """Validate libshmem.so runtime state after CDLL loads (maps scan, conflicts).

    In release builds issues raise RuntimeError.  In debug builds they are
    logged as warnings.
    """
    warnings = []

    backend_so_dir = _get_backend_so_dir()
    expected_libshmem = backend_so_dir / "libshmem.so"

    loaded_paths = []
    for line in _read_proc_maps():
        if "libshmem.so" in line:
            parts = line.split()
            if len(parts) >= 6:
                loaded_paths.append(parts[5])

    seen = set()
    unique_paths = []
    for p in loaded_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(Path(p).resolve())

    if len(unique_paths) > 1:
        paths_str = "\n  - ".join(str(p) for p in unique_paths)
        msg = (
            f"Multiple libshmem.so instances detected in the same process. "
            f"This may cause symbol conflicts, ABI mismatch, or initialization state corruption.\n"
            f"  Loaded paths:\n  - {paths_str}\n"
            f"  Action: Uninstall duplicate SHMEM installations, "
            f"or clean LD_LIBRARY_PATH to ensure only one libshmem.so is resolved."
            f"{_hint()}"
        )
        if _STRICT_MODE:
            raise RuntimeError(msg)
        warnings.append(msg)

    pkg_lib_path = expected_libshmem.resolve()
    if unique_paths and unique_paths[0] != pkg_lib_path:
        msg = (
            f"libshmem.so resolved to an unexpected path.\n"
            f"  Expected: {pkg_lib_path}\n"
            f"  Actual:   {unique_paths[0]}\n"
            f"  Action: Check LD_LIBRARY_PATH and ensure cann-shmem package is correctly installed."
            f"{_hint()}"
        )
        if _STRICT_MODE:
            raise RuntimeError(msg)
        warnings.append(msg)

    for w in warnings:
        logging.warning(f"[SHMEM startup guard] {w}")


def _run_pre_import_env_check():
    """首次 import shmem 时执行一次环境检测，后续跳过。

    调用打包的 preinstall_check.sh，与 shmem-config check 使用同一检测逻辑。
    检测结果写入 ~/.cache/shmem/.env_checked 标记文件，
    之后 import 不再重复检测。用户可随时手动运行 shmem-config check 命令。

    哨兵文件读写全部包裹 OSError 保护，避免 HOME 不可写或只读文件系统
    导致 import shmem 直接崩溃。
    """
    global _env_already_checked
    if _env_already_checked:
        return

    # 通过 shmem-config --check 调用时跳过自动检测，由 cmd_check() 统一处理，
    # 避免 import __init__.py 和 cmd_check() 各跑一次 preinstall_check.sh 导致两份报告。
    if "--check" in sys.argv:
        _env_already_checked = True
        return

    sentinel = Path.home() / ".cache" / "shmem" / ".env_checked"
    current_version = _read_version()

    try:
        if sentinel.exists():
            cached_version = sentinel.read_text(encoding="utf-8").strip()
            if cached_version == current_version:
                _env_already_checked = True
                return
    except OSError:
        pass

    check_script = _PKG_ROOT / "scripts" / "preinstall_check.sh"
    if not check_script.exists():
        _env_already_checked = True
        return

    try:
        subprocess.run(
            [str(check_script), "--package", str(_PKG_ROOT)],
            timeout=15
        )
    except subprocess.TimeoutExpired:
        print("\n[SHMEM env check] 环境检测超时（>15s），跳过。请手动运行 shmem-config check。",
              file=sys.stderr)
        _env_already_checked = True
        return
    except Exception as e:
        logging.warning("[SHMEM env check] 环境检测执行失败：%s", e)
        _env_already_checked = True
        return

    # 写入标记，后续不再检测
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(current_version or "unknown", encoding="utf-8")
    except OSError:
        pass
    _env_already_checked = True


# -- 装包后首次 import 自动检测一次，后续跳过 --
_run_pre_import_env_check()

_pre_load_guard()


_CANN_RUNTIME_ONLY_SYMBOLS = frozenset({
    "aclrtGetFuncBySymbol",
    "aclrtBinaryLoadFromData",
    "aclrtLaunchKernelWithHostArgs",
})


def _diagnose_cann_mismatch(oserror_msg):
    """Check whether an OSError from CDLL is caused by CANN version mismatch.

    When libshmem.so is built with a newer CANN toolkit (bisheng >= 2026-03-27,
    -xasc mode), the CCE compiler injects references to runtime symbols that
    don't exist in older CANN releases.  This function extracts the missing
    symbol from the error message and returns a targeted diagnostic, or None
    if the failure is unrelated.
    """
    if not isinstance(oserror_msg, str):
        return None
    for sym in _CANN_RUNTIME_ONLY_SYMBOLS:
        if sym in oserror_msg:
            return (
                f"Symbol '{sym}' is not available in the current CANN runtime.\n"
                f"The installed shmem wheel was compiled with a newer CANN toolkit "
                f"whose CCE compiler (-xasc mode) injects references to this API.\n"
                f"The older CANN runtime on this system does not export it.\n"
                f"Options:\n"
                f"  1) Upgrade the CANN runtime to match the wheel's build environment.\n"
                f"  2) Rebuild the shmem wheel against the currently installed CANN version."
            )
    return None


def _load_native():
    """Load native .so files.

    Failures are demoted to warnings in debug builds so that
    ``shmem-config --diagnose`` always runs.

    When a CANN runtime version mismatch is detected (new-compiler symbols
    missing from an older CANN installation), the error message includes a
    targeted diagnostic explaining the root cause.
    """
    required_so_files = [
        "libshmem_utils.so",
        "aclshmem_bootstrap_config_store.so",
        "libshmem.so",
    ]
    backend_so_dir = str(_get_backend_so_dir())
    for lib in required_so_files:
        lib_path = os.path.join(backend_so_dir, lib)
        try:
            ctypes.CDLL(lib_path)
        except FileNotFoundError:
            raise RuntimeError(f"Shared library file not found: {lib_path}")
        except OSError as e:
            hint = _diagnose_cann_mismatch(str(e))
            msg = f"Failed to load shared library {lib_path}: {e}"
            if hint:
                msg += f"{os.linesep}{hint}"
            raise RuntimeError(msg)

    _post_load_guard()


def _import_bindings():
    """Import pybind11 bindings.  Requires _load_native() to have succeeded."""
    from ._pyshmem import (  # noqa: E402
        aclshmem_init, aclshmem_get_unique_id, aclshmem_init_using_unique_id,
        aclshmem_finalize, aclshmem_malloc, aclshmem_free,
        aclshmem_ptr, aclshmemx_get_heap_base, my_pe, pe_count, set_conf_store_tls_key, team_split_strided,
        team_split_2d, team_translate_pe,
        team_destroy, InitAttr, OpEngineType,
        InitStatus, aclshmem_calloc, aclshmem_align, aclshmemx_init_status, get_ffts_config,
        team_my_pe, team_n_pes,
        aclshmem_putmem_nbi, aclshmem_getmem_nbi, aclshmem_putmem, aclshmem_getmem,
        aclshmemx_putmem_signal,
        aclshmemx_putmem_signal_nbi,
        aclshmem_info_get_version, aclshmem_info_get_name,
        aclshmem_team_get_config, OptionalAttr, aclshmem_global_exit, set_conf_store_tls,
        set_log_level, set_extern_logger, aclshmem_signal_wait_until,
    )

    globals().update({
        k: v for k, v in locals().items()
        if k not in ("__builtins__",) and not k.startswith("_")
    })

    # deprecated alias
    globals()["aclshmem_finialize"] = aclshmem_finalize


_NATIVE_LOADED = False
_NATIVE_LOAD_ERROR = None

# Public API surface — always defined so that introspection tools (help(), dir(),
# shmem-config --diagnose) work without native libraries being loaded.
# Accessing any exported name triggers lazy native loading via __getattr__.
__all__ = [
    'aclshmem_init',
    'aclshmem_get_unique_id',
    'aclshmem_init_using_unique_id',
    'aclshmem_finalize',
    'aclshmem_malloc',
    'aclshmem_free',
    'aclshmem_ptr',
    'aclshmemx_get_heap_base',
    'my_pe',
    'pe_count',
    'set_conf_store_tls_key',
    'set_conf_store_tls',
    'team_my_pe',
    'team_n_pes',
    'team_split_strided',
    'team_split_2d',
    'team_translate_pe',
    'team_destroy',
    'InitAttr',
    'InitStatus',
    'OpEngineType',
    'aclshmem_calloc',
    'aclshmem_align',
    'aclshmemx_init_status',
    'get_ffts_config',
    'aclshmem_global_exit',
    'aclshmem_putmem_nbi',
    'aclshmem_getmem_nbi',
    'aclshmemx_putmem_signal',
    'aclshmemx_putmem_signal_nbi',
    'aclshmem_putmem',
    'aclshmem_getmem',
    'aclshmem_info_get_version',
    'aclshmem_info_get_name',
    'aclshmem_team_get_config',
    'set_log_level',
    'set_extern_logger',
    'aclshmem_create_tensor',
    'aclshmem_free_tensor',
    'aclshmem_signal_wait_until',
]


def _ensure_native():
    """Lazily load native .so files and import pybind11 bindings.

    Called automatically on first access to any exported SHMEM API name.
    Subsequent calls are no-ops once loading succeeds.

    Raises:
        RuntimeError: Native libraries cannot be loaded (includes a CANN
            version-mismatch diagnostic when the CCE-compiler-injected
            symbols (aclrtGetFuncBySymbol etc.) are missing from the runtime).
    """
    global _NATIVE_LOADED, _NATIVE_LOAD_ERROR
    if _NATIVE_LOADED:
        return
    try:
        _load_native()
        _import_bindings()
    except Exception as exc:
        _NATIVE_LOAD_ERROR = exc
        raise
    from . import core  # noqa: E402
    from .construct_tensor import calc_nbytes, construct_tensor_from_ptr  # noqa: E402
    globals().update({
        'core': core,
        'calc_nbytes': calc_nbytes,
        'construct_tensor_from_ptr': construct_tensor_from_ptr,
    })
    _NATIVE_LOADED = True


def _get_native_load_error():
    """Return the exception from the last failed native-load attempt, or None."""
    return _NATIVE_LOAD_ERROR


def __getattr__(name):
    if name in __all__:
        _ensure_native()
        return globals()[name]
    raise AttributeError(f"module 'shmem' has no attribute {name!r}")


def aclshmem_create_tensor(shape, dtype: torch.dtype = torch.float32, device_id=0) -> torch.Tensor:
    _ensure_native()
    nbytes = calc_nbytes(shape, dtype)
    data_ptr = aclshmem_malloc(nbytes)

    if data_ptr == 0:
        raise RuntimeError("aclshmem_malloc failed")

    device = torch.device(f"npu:{device_id}")
    tensor = construct_tensor_from_ptr(data_ptr, shape, dtype, device)
    return tensor


def aclshmem_free_tensor(tensor: torch.Tensor):
    _ensure_native()
    data_ptr = tensor.data_ptr()
    aclshmem_free(data_ptr)
