#!/usr/bin/env python3
"""Render non-blocking timing information for the CI job summary."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _summary(markdown: str) -> None:
    path = _env("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(markdown)
    else:
        print(markdown)


def timing() -> int:
    """Report queue and execution time without changing the CI verdict."""
    repo, run_id = _env("GITHUB_REPOSITORY"), _env("GITHUB_RUN_ID")
    if not (repo and run_id):
        print("::warning::missing GITHUB_REPOSITORY / GITHUB_RUN_ID; skip timing")
        return 0
    proc = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/runs/{run_id}",
            "--jq",
            r'"\(.created_at) \(.run_started_at)"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stamps = proc.stdout.strip()
    if proc.returncode != 0 or not stamps:
        print("::warning::cannot read run timestamps (actions:read?)")
        return 0
    created_text, started_text = stamps.split()
    created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
    started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    _summary(
        "\n| Timing | Seconds |\n|---|---:|\n"
        f"| queue | {int((started - created).total_seconds())} |\n"
        f"| run | {int((now - started).total_seconds())} |\n"
        f"| end-to-end | {int((now - created).total_seconds())} |\n"
    )
    return 0


def main(argv: list[str]) -> int:
    if argv == ["timing"]:
        return timing()
    print("usage: python -m scripts.ci.tools.report timing", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
