#!/usr/bin/env python3
"""Pod environment checks, torch guard, and cleanup.

Commands:
  probe    capture torch, then verify the selected CPU, NPU, or GPU stack
  check    fail if pip replaced the accelerator torch stack
  cleanup  remove leftovers after timeout or cancellation

A replaced torch still imports, but torch.npu disappears and tensors fall back
to CPU. Cleanup is best-effort: it must not fail a successful job.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

INFORMATIONAL_MODULES = ("fla_npu", "fla")


def warn(message: str) -> None:
    print(f"::warning::{message}")


def error(message: str) -> None:
    print(f"::error::{message}")


def log(message: str) -> None:
    print(f"[env] {message}", file=sys.stderr)


# ---------------------------------------------------------------- environment verification
def _verify_environment(device: str) -> int:
    """Validate the selected device and its stack."""
    if device == "cpu":
        return _verify_cpu_environment()
    if device == "gpu":
        return _verify_gpu_environment()
    nodes = list(Path("/dev").glob("davinci*"))
    if not nodes:
        error("no /dev/davinci*, NPUs not mapped into the pod")
        return 1
    print(f"davinci count: {len(nodes)}")

    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    import triton

    print("python:", sys.version.split()[0], "| machine:", platform.machine())
    print("torch :", torch.__version__, "| torch_npu:", torch_npu.__version__)
    # Runtime version is authoritative: triton-ascend changes files, not metadata.
    print("triton:", triton.__version__, "(runtime; pip metadata differs by design)")
    for name in INFORMATIONAL_MODULES:
        try:
            print(f"{name}: {__import__(name).__version__}")
        except Exception as exc:  # noqa: BLE001 - informational only
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")
    if "nomimalloc" not in torch.__version__:
        warn("torch version has no nomimalloc tag; image may not be built by image-ci")

    assert torch.npu.is_available(), "torch.npu.is_available() is False: no NPU assigned"
    count = torch.npu.device_count()
    assert count > 0, "torch.npu.device_count() == 0: no NPU assigned"
    print("npu   : True | devices:", count)
    return 0


def _verify_cpu_environment() -> int:
    """Validate the portable CPU stack without touching NPU APIs."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - CPU install must provide torch
        error(f"torch is not importable: {type(exc).__name__}: {exc}")
        return 1
    print("python:", sys.version.split()[0], "| machine:", platform.machine())
    print("torch :", torch.__version__)
    print("cpu   : True | threads:", torch.get_num_threads())
    return 0


def _verify_gpu_environment() -> int:
    """Validate that the selected CUDA stack can see at least one device."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - accelerator install must provide torch
        error(f"torch is not importable: {type(exc).__name__}: {exc}")
        return 1
    print("python:", sys.version.split()[0], "| machine:", platform.machine())
    print("torch :", torch.__version__)
    if not torch.cuda.is_available():
        error("torch.cuda.is_available() is False: no GPU assigned")
        return 1
    count = torch.cuda.device_count()
    if count < 1:
        error("torch.cuda.device_count() == 0: no GPU assigned")
        return 1
    print("gpu   : True | devices:", count)
    return 0


# ---------------------------------------------------------------- install guard
# probe writes the baseline before pip; check compares it after pip.
def _torch_state() -> dict:
    """Return the shared torch snapshot used by capture and check."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - absence is an observation
        return {"torch_ok": False, "torch_error": f"{type(exc).__name__}: {exc}"}
    doc = {"torch_ok": True, "torch_version": torch.__version__, "torch_file": torch.__file__}
    try:
        import torch_npu

        doc["npu_version"] = getattr(torch_npu, "__version__", "unknown")
        doc["npu_file"] = torch_npu.__file__
    except Exception as exc:  # noqa: BLE001 - absence is an observation
        doc["npu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        doc["cuda_available"] = bool(torch.cuda.is_available())
        doc["cuda_devices"] = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001 - absence is an observation
        doc["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return doc


def _capture_torch(path: str) -> int:
    """Install guard, before pip: write the torch snapshot."""
    doc = _torch_state()
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if doc.get("torch_ok"):
        desc = (
            f"torch {doc.get('torch_version')} @ {doc.get('torch_file')}; "
            f"torch_npu {doc.get('npu_version', 'MISSING')}"
        )
    else:
        desc = f"torch unavailable ({doc.get('torch_error', 'not importable')})"
    log(f"captured: {desc}")
    return 0


def cmd_check(path: str, device: str) -> int:
    """Install guard, after pip: fail if torch or torch_npu changed."""
    saved = Path(path)
    if not saved.is_file():
        log(f"capture file not found: {saved}")
        return 2
    before = json.loads(saved.read_text(encoding="utf-8"))
    after = _torch_state()
    if not before.get("torch_ok") and not after.get("torch_ok"):
        log("torch is unavailable before and after install; nothing to compare")
        return 0
    if before.get("torch_ok") and not after.get("torch_ok"):
        error(f"torch is not importable after install: {after.get('torch_error', 'unknown')}")
        return 1
    old = (before.get("torch_version"), before.get("torch_file"))
    new = (after.get("torch_version"), after.get("torch_file"))
    if old != new:
        error(f"torch was replaced: {old[0]} ({old[1]}) -> {new[0]} ({new[1]})")
        error("pin the dependency or add an explicit torch constraint")
        return 1
    if device == "npu" and before.get("npu_version") and not after.get("npu_version"):
        error(f"torch_npu disappeared after install (was {before['npu_version']})")
        return 1
    if device == "gpu" and before.get("cuda_available") and not after.get("cuda_available"):
        error("torch.cuda disappeared after install")
        return 1
    log(f"ok: torch {after.get('torch_version')}; torch_npu {after.get('npu_version', 'n/a')}")
    return 0


def cmd_probe(capture_file: str, device: str) -> int:
    """Capture the torch baseline, then verify devices and stack."""
    _capture_torch(capture_file)
    return _verify_environment(device)


# ---------------------------------------------------------------- cleanup
TRAIN_PROCS = "run_lf\\.py|run_swift\\.py|torchrun|deepspeed"
TMP_PATTERNS = (
    "tt_*.yaml",
    "tt_pytest_files.txt",
    "train.log",
    "after.txt",
    "pytest.log",
    "pytest.xml",
    "turbo.log",
    "torch-before.json",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def cmd_cleanup(output_dir: str) -> int:
    print("=== env cleanup ===")
    left = _run(["pgrep", "-af", TRAIN_PROCS])
    if left.returncode == 0 and left.stdout.strip():
        print("--- leftover processes ---")
        print(left.stdout.strip())
        _run(["pkill", "-9", "-f", TRAIN_PROCS])
    else:
        print("no leftover training process")

    # Record memory usage; never kill another job's processes.
    if shutil.which("npu-smi"):
        mem = _run(["npu-smi", "info", "-t", "memory"])
        if mem.stdout:
            print("--- npu-smi memory ---")
            print("\n".join(mem.stdout.splitlines()[:20]))

    # Remove only files owned by this CI.
    for pattern in TMP_PATTERNS:
        for path in Path("/tmp").glob(pattern):
            _remove(path)

    out = Path(output_dir)
    if out.is_dir():
        if (out / ".tt-complete").is_file():
            print(f"checkpoint complete marker found, keeping {out}")
        else:
            print(f"no completeness marker, clearing partial output in {out}")
            for child in out.iterdir():
                _remove(child)
    print("=== env cleanup done ===")
    return 0


# ---------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Pod environment checks and cleanup")
    sub = ap.add_subparsers(dest="cmd", required=True)

    prb = sub.add_parser("probe", help="capture torch and verify devices and stack")
    prb.add_argument("--file", default="/tmp/torch-before.json")
    prb.add_argument("--device", choices=("cpu", "npu", "gpu"), default="npu")

    chk = sub.add_parser("check", help="fail if torch was replaced")
    chk.add_argument("--file", required=True)
    chk.add_argument("--device", choices=("cpu", "npu", "gpu"), default="npu")
    cln = sub.add_parser("cleanup", help="remove leftovers after timeout or cancellation")
    cln.add_argument("--output-dir", default="/tmp/tblf-sft")

    args = ap.parse_args(argv)
    if args.cmd == "probe":
        return cmd_probe(args.file, args.device)
    if args.cmd == "check":
        return cmd_check(args.file, args.device)
    return cmd_cleanup(args.output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
