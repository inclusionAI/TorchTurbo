#!/usr/bin/env python3
"""Render CI timing and model-E2E facts for the job summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}", "--jq",
         r'"\(.created_at) \(.run_started_at)"'],
        capture_output=True, text=True, check=False,
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


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pairs(items: list[str]) -> dict[str, str]:
    values = {}
    for item in items:
        key, separator, value = item.partition("=")
        if separator:
            values[key] = value
        else:
            print(f"[report] ignoring '{item}': not key=value", file=sys.stderr)
    return values


def _fmt_seconds(value) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "n/a"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def _tail(path: Path, lines: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _getter(doc: dict):
    def get(key: str, default: str = "?") -> str:
        value = doc.get(key, "")
        return value if value not in ("", None) else default

    return get


def _phase_lines(phases: list[dict]) -> list[str]:
    total = 0
    for phase in phases:
        try:
            total += int(float(phase.get("seconds", 0)))
        except (TypeError, ValueError):
            pass
    lines = ["", "| phase | seconds | human | rc |", "|---|---|---|---|"]
    lines.extend(
        f"| {phase.get('name', '?')} | `{phase.get('seconds', 'n/a')}` | "
        f"{_fmt_seconds(phase.get('seconds'))} | `{phase.get('rc', 'n/a')}` |"
        for phase in phases
    )
    lines.append(f"| **total** | `{total}` | **{_fmt_seconds(total)}** | |")
    return lines


def render(doc: dict, *, log: Path, versions: Path) -> str:
    get = _getter(doc)
    lines = [
        f"## {get('model_type')} / {get('framework')} on `{get('runner_label')}`",
        "",
        "| item | value |",
        "|---|---|",
        f"| image | `{get('image')}` |",
        f"| run_dir | `{get('run_dir')}` |",
        f"| install / train | `{get('install_script')}` / `{get('train_script')}` |",
        f"| model cache | **{get('cache_hit')}** (sfs `{get('sfs_uuid')}`) |",
        f"| layers | `{get('layers', '0')}` (0 = 不裁剪) |",
        f"| dataset | `{get('dataset')}` |",
        f"| shards / size | {get('shards')} / {get('model_size')} |",
        f"| parallel (sp/ep) | `{get('sp')}/{get('ep')}` |",
        f"| output_dir | `{get('output_dir')}` |",
        *_phase_lines(doc.get("phases", [])),
        "",
        "### Versions after install",
        "",
        "```",
        _tail(versions, 40) or "(probe failed)",
        "```",
        "",
        "### 训练尾部",
        "",
        "```",
        _tail(log, 40) or "(no train log)",
        "```",
    ]
    return "\n".join(lines) + "\n"


def notice(doc: dict, model_type: str) -> str:
    phase = next(
        (item for item in doc.get("phases", []) if "train" in (item.get("name") or "")),
        {},
    )
    rc = phase.get("rc", "n/a")
    fields = [
        f"sfs={doc.get('sfs_uuid', '?')}",
        f"cache_hit={doc.get('cache_hit', '?')}",
        f"shards={doc.get('shards', '?')}",
        f"size={doc.get('model_size', '?')}",
        f"layers={doc.get('layers', '?')}",
        f"sp_ep={doc.get('sp', '?')}/{doc.get('ep', '?')}",
        f"train_s={phase.get('seconds', 'n/a')}",
        f"train_rc={rc}",
    ]
    lines = [f"::notice title=smoke {model_type}::{' '.join(fields)}"]
    if str(rc) not in ("0", "None", ""):
        lines.append(
            f"::error title=train failed::rc={rc}; see the train tail in the job summary"
        )
    return "\n".join(lines)


def _model_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "set", "phase", "report", "notice"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--model-type", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--seconds", default="")
    parser.add_argument("--rc", default="")
    parser.add_argument("--train-log", default="/tmp/train.log")
    parser.add_argument("--versions", default="/tmp/after.txt")
    parser.add_argument("values", nargs="*")
    # Python 3.11's argparse rejects positional values after options. The CI
    # emits KEY=value tails, so collect only well-formed pairs from the remainder.
    args, extra = parser.parse_known_args(argv)
    for token in extra:
        if "=" in token and not token.startswith("-"):
            args.values.append(token)
        else:
            parser.error(f"unrecognized argument: {token}")

    path = Path(args.file)
    if args.command == "init":
        _save(path, _pairs(args.values))
    elif args.command == "set":
        doc = _load(path)
        doc.update(_pairs(args.values))
        _save(path, doc)
    elif args.command == "phase":
        doc = _load(path)
        doc.setdefault("phases", []).append(
            {"name": args.name, "seconds": args.seconds, "rc": args.rc}
        )
        _save(path, doc)
    elif args.command == "notice":
        doc = _load(path)
        print(notice(doc, args.model_type or str(doc.get("model_type", "?"))))
    else:
        sys.stdout.write(
            render(_load(path), log=Path(args.train_log), versions=Path(args.versions))
        )
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "timing":
        return timing()
    return _model_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
