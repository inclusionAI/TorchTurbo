# SPDX-License-Identifier: Apache-2.0
"""Source-build MegaMoE + MoonEP precision tests; stdout is the report."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run_phase(name: str, command: list[str], env: dict[str, str]) -> bool:
    print("\nPHASE_START " + name, flush=True)
    print("COMMAND " + shlex.join(command), flush=True)
    result = subprocess.run(command, env=env, stderr=subprocess.STDOUT)
    print(
        f"PHASE_RESULT {'PASS' if result.returncode == 0 else 'FAIL'} {name} exit_code={result.returncode}",
        flush=True,
    )
    return result.returncode == 0


def run_npu_precision(
    root: Path,
    env: dict[str, str],
    checkpoint_only: bool = False,
    compatibility_only: bool = False,
) -> bool:
    cases = []
    for mode, limit in [
        ("none", "0"),
        ("pre_activation", ".25"),
        ("post_activation", ".25"),
    ]:
        for routing in ["uniform", "hot"]:
            for checkpoint in [False, True]:
                name = f"{mode}_{routing}_checkpoint{int(checkpoint)}"
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(root / "tests/npu/mega_moe_precision.py"),
                    "--mode",
                    mode,
                    "--limit",
                    limit,
                    "--routing",
                    routing,
                ]
                if checkpoint:
                    command.extend(["--identity", "--checkpoint"])
                cases.append((name, command))
        cases.append(
            (
                mode + "_ep8_checkpoint",
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=8",
                    str(root / "tests/npu/mega_moe_precision.py"),
                    "--mode",
                    mode,
                    "--limit",
                    limit,
                    "--routing",
                    "hot",
                    "--identity",
                    "--checkpoint",
                ],
            )
        )
    # Larger hot inputs trigger real expert copies instead of the small-load path.
    for checkpoint_mode in ["none", "reentrant", "nonreentrant"]:
        for layers in [1, 2]:
            name = f"hot_copies_{checkpoint_mode}_layers{layers}"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(root / "tests/npu/mega_moe_precision.py"),
                "--routing",
                "hot",
                "--tokens",
                "2048",
                "--layers",
                str(layers),
                "--checkpoint-mode",
                checkpoint_mode,
                "--require-copies",
            ]
            cases.append((name, command))
    for early_stop in [True, False]:
        name = f"hot_copies_nonreentrant_stack_early_stop{int(early_stop)}"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests/npu/mega_moe_precision.py"),
            "--routing",
            "hot",
            "--tokens",
            "2048",
            "--layers",
            "2",
            "--checkpoint-mode",
            "nonreentrant",
            "--checkpoint-scope",
            "stack",
            "--require-copies",
        ]
        if not early_stop:
            command.append("--disable-early-stop")
        cases.append((name, command))
    if checkpoint_only:
        cases = [case for case in cases if case[0].startswith("hot_copies_")]
    if compatibility_only:
        entry = str(root / "tests/npu/mega_moe_compatibility.py")
        prefix = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            entry,
        ]
        cases = [
            (
                "compatibility_full_adamw",
                prefix + ["--offload", "full", "--require-copies"],
            ),
            (
                "compatibility_optimizer_adamw",
                prefix
                + [
                    "--offload",
                    "optimizer",
                    "--checkpoint",
                    "reentrant",
                    "--require-copies",
                ],
            ),
            (
                "compatibility_full_muon",
                prefix
                + [
                    "--offload",
                    "full",
                    "--checkpoint",
                    "reentrant",
                    "--optimizer",
                    "muon",
                    "--require-copies",
                ],
            ),
        ]
    failures = []
    for index, (name, command) in enumerate(cases, 1):
        print(f"\nCASE_START {index}/{len(cases)} {name}", flush=True)
        result = subprocess.run(command, env=env, stderr=subprocess.STDOUT, timeout=300)
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"CASE_RESULT {status} {name} exit_code={result.returncode}", flush=True)
        if result.returncode:
            failures.append(name)
    print(
        f"\nCASE_SUMMARY total={len(cases)} passed={len(cases) - len(failures)} failed={len(failures)}",
        flush=True,
    )
    for name in failures:
        print("FAILED_CASE " + name, flush=True)
    return not failures


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only the configuration, hook and semantic checks; no multi-rank jobs",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Run only the eight borrowed-expert checkpoint NPU regressions after CPU checks",
    )
    parser.add_argument(
        "--compatibility-only",
        action="store_true",
        help="Run FSDP CPU-offload and optimizer combinations after CPU checks",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    }
    results = [
        run_phase(
            "unit_checks",
            [
                sys.executable,
                "-m",
                "pytest",
                "-vv",
                "--color=no",
                "-p",
                "no:cacheprovider",
                str(root / "tests/models/test_routed_experts.py"),
            ],
            env,
        )
    ]
    if not args.unit_only and all(results):
        results.append(
            run_npu_precision(root, env, args.checkpoint_only, args.compatibility_only)
        )
    passed = all(results) and len(results) == (1 if args.unit_only else 2)
    print(
        f"TEST_SUMMARY {'PASS' if passed else 'FAIL'} phases={len(results)}", flush=True
    )
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
