#!/usr/bin/env python
"""Run DeepSeekV4 alignment checks from the torchturbo repo.

This is a convenience launcher for the pytest-based DeepSeekV4 checks in this
directory.  The actual assertions stay in the pytest files.

Examples:
    python tests/models/deepseek_v4/run_alignment.py
    python tests/models/deepseek_v4/run_alignment.py --mode moe
    python tests/models/deepseek_v4/run_alignment.py --mode post-hc
    python tests/models/deepseek_v4/run_alignment.py --mode checkpoint \\
        --model /a3_inference/itask/workdir/zss02308789/610/small_dsv4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

TEST_TARGETS = {
    "all": ["tests/models/deepseek_v4"],
    "modeling-vs-infer": ["tests/models/deepseek_v4/test_alignment.py"],
    "ops": ["tests/models/deepseek_v4/test_ops_precision.py"],
    "turbo-vs-modeling": ["tests/models/deepseek_v4/test_turbo_vs_base_precision.py"],
    "checkpoint": ["tests/models/deepseek_v4/test_checkpoint_smoke.py"],
    "moe": ["tests/models/deepseek_v4/test_moe_ops.py"],
    "post-hc": ["tests/models/deepseek_v4/test_post_attention_hc_ops.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeekV4 alignment pytest checks")
    parser.add_argument(
        "--mode",
        choices=sorted(TEST_TARGETS),
        default="all",
        help="Which DeepSeekV4 alignment check set to run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to small_dsv4 or another DeepSeekV4 checkpoint for --mode checkpoint.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Pass -s to pytest so test output is not captured.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to pytest. Put this option last.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    env.setdefault("TORCH_TURBO_DISABLE", "1")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if args.model:
        env["TT_DSV4_SMOKE_MODEL"] = args.model

    cmd = [sys.executable, "-m", "pytest", *TEST_TARGETS[args.mode], "-q"]
    if args.verbose:
        cmd.append("-s")
    cmd.extend(args.pytest_args)

    print(f"[DeepSeekV4 alignment] repo: {REPO_ROOT}", flush=True)
    print(f"[DeepSeekV4 alignment] mode: {args.mode}", flush=True)
    if args.model:
        print(f"[DeepSeekV4 alignment] checkpoint: {args.model}", flush=True)
    print("[DeepSeekV4 alignment] command:", " ".join(cmd), flush=True)

    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
