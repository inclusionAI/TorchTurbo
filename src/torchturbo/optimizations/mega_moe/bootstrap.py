"""Prepare native libraries before importing Torch/CANN; safe to run by path."""

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def _boolean(value: str) -> bool:
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def activate(argv: list[str]) -> bool:
    """Resolve the same CLI/environment switch as EngineConfig without Torch."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tt_mega_moe", type=_boolean, nargs="?", const=True, default=False)
    args, _ = parser.parse_known_args(argv)
    enabled = args.tt_mega_moe
    if "TORCH_TURBO_MEGA_MOE" in os.environ:
        enabled = _boolean(os.environ["TORCH_TURBO_MEGA_MOE"])
    if not enabled:
        return False
    spec = importlib.util.find_spec("tt_megamoe")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "MegaMoE is enabled but tt-megamoe is not installed. "
            "Provision the MoonEP-enabled component before launching training."
        )
    entry = Path(spec.origin).resolve().parent / "bootstrap.py"
    if not entry.is_file():
        raise RuntimeError("The installed tt-megamoe component has no native bootstrap.")
    return runpy.run_path(str(entry))["activate"]()


if __name__ == "__main__":
    activate(sys.argv[1:])
