#!/usr/bin/env python3
"""Resolve an E2E model plan and prepare its weights.

Small models use the full checkpoint. Large models are trimmed to their first N
transformer layers so an 8-card (at most 16-card) job can run them.

Subcommands:

    fetch  obtain and validate a model directory
    probe  validate one existing model directory
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

MARKER = ".tt-cache-marker"
MARKER_VERSION = 2
INDEX = "model.safetensors.index.json"
SMALL_FILES = ["*.json", "*.py", "*.jinja", "*.model", "*.txt"]

# Per-layer config fields that must follow num_hidden_layers after a trim.
LAYER_LIST_FIELDS = ("layer_types", "mlp_layer_types", "indexer_types", "compress_ratios")
LAYER_ID_FIELDS = ("ple_layer_ids",)
TEXT_CONFIG_FIELDS = ("text_config", "llm_config", "language_config", "language_model_config")
MTP_FIELDS = ("num_nextn_predict_layers", "mtp_num_hidden_layers")


def validate_layer_count(layers: int) -> None:
    """Reject negative layer counts before they reach cache or trim logic."""
    if layers < 0:
        raise SystemExit(f"layers must be >= 0, got {layers}")


@contextmanager
def lock_trim_target(target: Path) -> Iterator[None]:
    """Serialize trims of the same target across processes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read_json_object(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return doc


def measure_directory(path: Path) -> int:
    result = subprocess.run(["du", "-sbL", str(path)], capture_output=True, text=True)
    try:
        return int(result.stdout.split()[0]) if result.returncode == 0 else 0
    except (IndexError, ValueError):
        return 0


def should_keep_tensor(name: str, layers: int) -> bool:
    """Keep non-MTP tensors below ``layers``; layers=0 keeps everything."""
    if layers <= 0:
        return True
    parts = name.split(".")
    if "mtp" in parts:
        return False
    for position, part in enumerate(parts[:-1]):
        if part == "layers" and parts[position + 1].isdigit():
            return int(parts[position + 1]) < layers
    return True


def get_text_configs(config: dict) -> list[dict]:
    """Config objects that describe the text transformer."""
    nodes = [config] if isinstance(config.get("num_hidden_layers"), int) else []
    for field in TEXT_CONFIG_FIELDS:
        child = config.get(field)
        if isinstance(child, dict):
            nodes += get_text_configs(child)
    return nodes


def count_hidden_layers(config: dict) -> int:
    return max((node["num_hidden_layers"] for node in get_text_configs(config)), default=0)


def trim_model_config(config: dict, layers: int) -> int:
    """Trim text configs in place; return the original layer count."""
    total = count_hidden_layers(config)
    if not 0 < layers < total:
        raise SystemExit(f"cannot trim to {layers} layers: model has {total}; use 0 for full")

    for node in get_text_configs(config):
        if node["num_hidden_layers"] != total:
            continue
        node["num_hidden_layers"] = layers
        for field in LAYER_LIST_FIELDS:
            values = node.get(field)
            if isinstance(values, list):
                if len(values) not in (total, layers):
                    raise SystemExit(
                        f"cannot trim {field}: {len(values)} entries for {total} layers"
                    )
                node[field] = values[:layers]
        for field in LAYER_ID_FIELDS:
            values = node.get(field)
            if isinstance(values, list):
                node[field] = [value for value in values if value <= layers]
        group = node.get("layer_group_size")
        if isinstance(group, int) and group > 0 and layers % group:
            raise SystemExit(
                f"cannot trim to {layers} layers: layer_group_size={group} "
                f"requires a multiple of {group}"
            )
        if isinstance(group, list):
            node["layer_group_size"] = group[:layers]
        for field in MTP_FIELDS:
            if field in node:
                node[field] = 0

    for field in MTP_FIELDS:
        if field in config:
            config[field] = 0
    return total


def validate_model_dir(directory: str | Path, layers: int = 0) -> tuple[int, int] | None:
    """Return (bytes, shards) when config, index, and shard files agree."""
    validate_layer_count(layers)
    root = Path(directory)
    try:
        config = read_json_object(root / "config.json")
    except ValueError as exc:
        log(f"probe {root}: {exc}")
        return None
    if layers > 0:
        actual = count_hidden_layers(config)
        if actual != layers:
            log(f"probe {root}: config has {actual} layers, expected {layers}")
            return None

    if not (root / INDEX).is_file():
        if layers > 0:
            log(f"probe {root}: trimmed model has no {INDEX}")
            return None
        shards = sorted(path for path in root.glob("*.safetensors") if path.stat().st_size)
        if not shards:
            log(f"probe {root}: no non-empty *.safetensors")
            return None
        return measure_directory(root), len(shards)

    try:
        index = read_json_object(root / INDEX)
        weights = index["weight_map"]
    except (ValueError, KeyError) as exc:
        log(f"probe {root}: index unreadable: {exc}")
        return None
    if not isinstance(weights, dict) or not weights:
        log(f"probe {root}: index has no weight_map")
        return None
    if layers > 0 and any(not should_keep_tensor(name, layers) for name in weights):
        log(f"probe {root}: index names tensors beyond the {layers}-layer trim")
        return None

    shards = sorted(set(weights.values()))
    missing = [name for name in shards
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        log(f"probe {root}: missing {len(missing)} indexed shards: {missing[:5]}")
        return None

    return measure_directory(root), len(shards)


def find_cached_model(repo: str, home: str, cache_root: str, target: Path,
                      marker: str | None, layers: int) -> tuple[Path, int, int] | None:
    """Search known cache roots; trims must also match our marker."""
    escaped = repo.replace(".", "___")
    candidates = [
        target,
        Path(cache_root) / repo,
        Path(cache_root) / escaped,
        Path(f"/root/.cache/modelscope/hub/models/{repo}"),
        Path(f"/root/.cache/modelscope/hub/models/{escaped}"),
        Path(f"/root/.cache/modelscope/hub/{repo}"),
    ]
    candidates += sorted(Path(home).glob(f"hub/models--{repo.replace('/', '--')}/snapshots/*"))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_dir():
            continue
        if marker is not None:
            stamp = candidate / MARKER
            if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != marker:
                log(f"stale {candidate} (not our {marker!r} trim)")
                continue
        result = validate_model_dir(candidate, layers)
        if result is None:
            log(f"miss {candidate}")
            continue
        log(f"HIT {candidate} bytes={result[0]} shards={result[1]} layers={layers}")
        return candidate, *result
    return None


def download_repo(repo: str, directory: str, patterns: list[str] | None, attempts: int) -> None:
    """Fetch files, retrying transient mirror failures."""
    from huggingface_hub import snapshot_download

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        log(f"--- download attempt {attempt}/{attempts} ---")
        try:
            snapshot_download(repo_id=repo, local_dir=directory, allow_patterns=patterns)
            return
        except Exception as exc:  # noqa: BLE001 - retry every download failure
            last = exc
            log(f"attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                wait = attempt * 30
                log(f"retrying in {wait}s")
                time.sleep(wait)
    raise SystemExit(f"download {repo} failed after {attempts} attempts") from last


def create_trimmed_model(repo: str, target: Path, layers: int, attempts: int) -> None:
    """Download only the shards and tensors needed by the first ``layers`` layers."""
    target.mkdir(parents=True, exist_ok=True)
    download_repo(repo, str(target), SMALL_FILES, attempts)
    try:
        config = read_json_object(target / "config.json")
        index = read_json_object(target / INDEX)
        weights = index["weight_map"]
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"cannot trim {repo}: {exc}") from exc
    if not isinstance(weights, dict) or not weights:
        raise SystemExit(f"cannot trim {repo}: {INDEX} has no weight_map")

    total = trim_model_config(config, layers)
    kept = {name: shard for name, shard in weights.items()
            if should_keep_tensor(name, layers)}
    if len(kept) == len(weights):
        raise SystemExit(f"cannot trim {repo}: no 'layers.<n>' tensors matched")
    shards = sorted(set(kept.values()))
    log(f"trim {total}->{layers} layers: keep {len(kept)}/{len(weights)} tensors, "
        f"{len(shards)}/{len(set(weights.values()))} shards")
    download_repo(repo, str(target), shards, attempts)

    index["weight_map"] = kept
    index.setdefault("metadata", {}).update(
        tt_trimmed_layers=layers, tt_original_layers=total
    )
    (target / INDEX).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (target / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def prepare_model_dir(repo: str, hint: str, layers: int, cache_root: str,
                      home: str, attempts: int) -> dict[str, object]:
    """Return a validated model directory, using hint, cache, trim, or download."""
    validate_layer_count(layers)
    if hint:
        result = validate_model_dir(hint, layers)
        if result is None:
            raise SystemExit(f"explicit model is incomplete: {hint}")
        return {"MODEL": hint, "BYTES": result[0], "SHARDS": result[1], "HIT": "yes"}

    target = Path(cache_root) / (
        repo if layers <= 0 else f"{repo.rsplit('/', 1)[-1]}-L{layers}"
    )
    marker = f"v{MARKER_VERSION} repo={repo} layers={layers}" if layers > 0 else None
    lock = lock_trim_target(target) if layers > 0 else nullcontext()
    with lock:
        found = find_cached_model(repo, home, cache_root, target, marker, layers)
        if found is not None:
            return {"MODEL": found[0], "BYTES": found[1], "SHARDS": found[2], "HIT": "yes"}

        if layers > 0:
            log(f"trim {repo} -> {target} ({layers} layers)")
            create_trimmed_model(repo, target, layers, attempts)
        else:
            log(f"fetch {repo} -> {target} (full)")
            target.mkdir(parents=True, exist_ok=True)
            download_repo(repo, str(target), None, attempts)

        result = validate_model_dir(target, layers)
        if result is None:
            raise SystemExit(f"prepared model is incomplete: {target}")
        if marker is not None:
            (target / MARKER).write_text(marker + "\n", encoding="utf-8")
        return {"MODEL": target, "BYTES": result[0], "SHARDS": result[1], "HIT": "no"}


def log(message: str) -> None:
    print(f"[model] {message}", file=sys.stderr, flush=True)


def emit_key_values(destination: str, **values: object) -> None:
    """Write KEY=value lines for parse_kv."""
    text = "".join(f"{key}={value}\n" for key, value in values.items())
    if destination:
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


def run_command(args: argparse.Namespace) -> int:
    """Dispatch one CLI command; each branch is a thin adapter only."""
    if args.command == "fetch":
        model = prepare_model_dir(args.repo, args.hint, args.layers,
                                  args.cache_root, args.home, args.attempts)
        log(f"model: {model['MODEL']}")
        emit_key_values(args.output, **model)
        return 0
    if args.command == "probe":
        result = validate_model_dir(args.directory, args.layers)
        if result is None:
            return 1
        print(*result)
        return 0

    raise SystemExit(f"unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="obtain a validated model directory")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--hint", default="")
    fetch.add_argument("--layers", type=int, default=0,
                       help="keep the first N layers; 0 downloads the full model")
    fetch.add_argument("--output", default="")
    fetch.add_argument("--home", default=os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    fetch.add_argument("--cache-root", default="/root/.cache/models")
    fetch.add_argument("--attempts", type=int, default=3)

    probe = sub.add_parser("probe", help="validate one model directory")
    probe.add_argument("--dir", dest="directory", required=True)
    probe.add_argument("--layers", type=int, default=0)

    return root


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
