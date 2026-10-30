"""Activate packaged native libraries before Swift imports Torch or CANN.

This file is executed by path so importing torchturbo is unnecessary.
"""
import argparse
import runpy
from pathlib import Path
import sys


def activate(argv: list[str]) -> bool:
    """Prepare native paths when the requested YAML explicitly enables MegaMoE."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tt_config")
    args, _ = parser.parse_known_args(argv)
    if args.tt_config is None:
        return False
    import yaml

    with open(args.tt_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    enabled = (config.get("engine_config") or {}).get("mega_moe", False)
    if not isinstance(enabled, bool):
        raise TypeError("engine_config.mega_moe must be a YAML boolean.")
    if not enabled:
        return False
    entry = Path(__file__).resolve().parent / "runtime/bootstrap.py"
    return runpy.run_path(str(entry))["activate"]()


if __name__ == "__main__":
    activate(sys.argv[1:])
