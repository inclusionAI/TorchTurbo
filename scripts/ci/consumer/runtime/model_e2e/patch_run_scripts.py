#!/usr/bin/env python3
"""Rewrite the three files a model smoke run needs, in place.

A run is an install script, a launcher, and an engine config. All three describe
a person's machine -- 16 cards, a persistent disk -- and CI has neither:

* install: ends with ``rm -rf`` on a cluster-wide NFS cache. Another pod writing
  into it turns the delete into ENOTEMPTY; ``set -e`` makes that a failed
  install, which then gets reported as a failed training run.
* train: saves checkpoints and logs to tensorboard. A smoke run wants neither.
* config: hard-codes ``sp_size``/``ep_size``, which must follow the real cards.

Editing the checkout is safe (it is disposable) and necessary: launcher paths are
relative to the script's own location, so copying the file elsewhere breaks
``REPO_ROOT`` and ``source ../deps/*.sh``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Command names CI refuses to run, plus the wrappers that can hide them.
DESTRUCTIVE = frozenset({"rm", "rmdir", "mv", "dd", "truncate", "shred"})
WRAPPERS = frozenset({"sudo", "command", "nohup", "time", "exec"})
# `a && rm -rf b` is one line and two commands, so split before looking.
SPLIT = re.compile(r"&&|\|\||[;|]")


def destructive(line: str) -> bool:
    for part in SPLIT.split(line):
        words = part.split()
        while words and words[0] in WRAPPERS:
            words = words[1:]
        if words and words[0] in DESTRUCTIVE:
            return True
    return False


def patch_install(script: Path) -> list[str]:
    """Drop destructive lines; keep every other command as written."""
    kept, removed = [], []
    for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(keepends=True), 1):
        body = line.strip()
        if body and not body.startswith("#") and destructive(body):
            removed.append(f"  line {number}: removed {body}")
        else:
            kept.append(line)
    if removed:
        script.write_text("".join(kept), encoding="utf-8")
    return removed


# Only the defaults that make a smoke run unsafe, and only where the flag exists.
REPLACE = (
    ("    --report_to tensorboard \\", "    --report_to none \\"),
    ("    --overwrite_cache \\", "    --overwrite_cache False \\"),
    ('    --save_strategy "steps" \\', '    --save_strategy "no" \\'),
)
# Dead once saving is off, and a leftover `--save_steps 200000000` reads as if
# saving were configured.
DROP = (
    "    --save_steps 200000000 \\\n",
    "    --save_total_limit 1 \\\n",
    "    --plot_loss \\\n",
)


def patch_train(script: Path) -> list[str]:
    """Apply the smoke defaults, asserting every anchor appears exactly once."""
    text = script.read_text(encoding="utf-8")
    notes: list[str] = []
    for old, new in REPLACE:
        found = text.count(old)
        if not found:
            if text.count(new) != 1:
                raise SystemExit(f"{script}: {old.strip()!r} is gone; the launcher changed")
            notes.append(f"  already applied: {new.strip()}")
            continue
        if found > 1:
            raise SystemExit(f"{script}: {old.strip()!r} appears {found} times, expected 1")
        text = text.replace(old, new)
        notes.append(f"  {old.strip()} -> {new.strip()}")
    for old in DROP:
        found = text.count(old)
        if found > 1:
            raise SystemExit(f"{script}: {old.strip()!r} appears {found} times, expected 1")
        if found:
            text = text.replace(old, "")
            notes.append(f"  removed {old.strip()}")
    if notes:
        script.write_text(text, encoding="utf-8")
    return notes


# The launcher names its own engine config, so rewriting that file keeps every
# downstream path intact -- no /tmp copy, no TT_CONFIG override.
TT_CONFIG_LINE = re.compile(r'^TT_CONFIG=\$\{TT_CONFIG:-(.*)\}$', re.M)


def config_source(launcher: Path) -> Path:
    match = TT_CONFIG_LINE.search(launcher.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"{launcher}: no TT_CONFIG= line; cannot find the engine config")
    raw = match.group(1).strip('"').replace('$(dirname "$0")', str(launcher.parent))
    source = Path(raw.replace("${SCRIPT_DIR}", str(launcher.parent)))
    if not source.is_file():
        raise SystemExit(f"{launcher}: TT_CONFIG points at a missing file: {source}")
    return source


def patch_config(launcher: Path, cards: int, sp: int, ep: int,
                 layers: int) -> tuple[Path, int, int]:
    """Write the declared plan into the engine config; refuse an impossible one."""
    import yaml

    # Declared, not derived: scaling silently would turn a wrong plan into a
    # different run that still reports green.
    if sp < 1 or ep < 1 or cards % sp or (cards // sp) % ep:
        raise SystemExit(f"{launcher}: cards={cards} sp={sp} ep={ep} cannot be laid out")

    source = config_source(launcher)
    doc = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    engine = doc.setdefault("engine_config", {}) or {}
    plan = engine.setdefault("parallel_plan", {}) or {}
    old = (int(plan.get("sp_size", 1) or 1), int(plan.get("ep_size", 1) or 1))
    plan["sp_size"], plan["ep_size"] = sp, ep
    if layers > 0:
        engine["mtp"] = {**(engine.get("mtp") or {}), "enabled": False}
    doc["engine_config"] = engine
    source.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                      encoding="utf-8")
    return source, old[0], old[1]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = root.add_subparsers(dest="profile", required=True)
    for name, text in (("install", "drop destructive commands"),
                       ("train", "replace smoke-unsafe defaults"),
                       ("config", "fit sp/ep to the card count")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--src", required=True, type=Path, help="file to patch")
        if name == "config":
            command.add_argument("--cards", required=True, type=int)
            command.add_argument("--sp", required=True, type=int)
            command.add_argument("--ep", required=True, type=int)
            command.add_argument("--layers", type=int, default=0)
    return root


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    if not args.src.is_file():
        raise SystemExit(f"missing script: {args.src}")

    if args.profile != "config":
        notes = (patch_install if args.profile == "install" else patch_train)(args.src)
        print(f"[patch] {args.src}: {args.profile}"
              + ("\n" + "\n".join(notes) if notes else ", nothing to change"))
        return 0

    # stdout stays KEY=value only so run_train.sh can use parse_kv on it; the
    # human-readable line goes to stderr.
    source, old_sp, old_ep = patch_config(
        args.src, args.cards, args.sp, args.ep, args.layers)
    print(f"[patch] {source}: sp {old_sp}->{args.sp}, ep {old_ep}->{args.ep} "
          f"(cards={args.cards})", file=sys.stderr)
    print(f"CARDS={args.cards}\nSP={args.sp}\nEP={args.ep}\nTT_CONFIG={source}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
