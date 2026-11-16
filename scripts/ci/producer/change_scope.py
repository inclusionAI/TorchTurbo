#!/usr/bin/env python3
"""Classify a changed-file list as documentation-only or code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

DOC_DIRS = frozenset({"docs"})
DOC_NAMES = frozenset({"LICENSE", "NOTICE"})
DOC_SUFFIXES = frozenset({".adoc", ".md", ".rst"})


def is_docs_path(path: str) -> bool:
    """Return whether one repository-relative path is documentation-only."""
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        return False
    parsed = PurePosixPath(normalized)
    if parsed.parts and parsed.parts[0] in DOC_DIRS:
        return True
    if parsed.name in DOC_NAMES:
        return True
    return parsed.suffix.lower() in DOC_SUFFIXES


def classify_paths(paths: Iterable[str]) -> str:
    """Return ``docs_only`` when every changed path is documentation."""
    values = [path.strip() for path in paths if path.strip()]
    if not values:
        return "code"
    return "docs_only" if all(is_docs_path(path) for path in values) else "code"


def _read_paths(path: str) -> list[str] | None:
    if path == "-":
        return sys.stdin.read().splitlines()
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"cannot read changed-file list: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed-files",
        default="-",
        help="newline-separated changed paths; '-' reads stdin",
    )
    args = parser.parse_args(argv)
    paths = _read_paths(args.changed_files)
    scope = "code" if paths is None else classify_paths(paths)
    print(f"change_scope={scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
