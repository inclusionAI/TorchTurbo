#!/usr/bin/env python3
"""Probe model and dataset sources without downloading their payloads."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL_REPO = "inclusionAI/Ling-3.0-tiny"
MODEL_REVISION = "9a98e35fe1c9ee255f78dd64771c7ae15a799481"
DATASET_REPO = "a-m-team/AM-Thinking-v1-Distilled"
DATASET_REVISION = "186b575b5cc2113f8ba81d4f4ac9ae2494633829"
DATASET_FILE = "science.jsonl"
DEFAULT_SOURCES = ("https://hf-mirror.com", "https://huggingface.co")
MAX_METADATA_BYTES = 5 * 1024 * 1024


def _request(
    url: str, *, method: str = "GET", timeout: float = 30.0
) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "torchturbo-ci-asset-probe/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_METADATA_BYTES + 1) if method != "HEAD" else b""
            if len(body) > MAX_METADATA_BYTES:
                body = body[:MAX_METADATA_BYTES]
            result = {
                "ok": 200 <= response.status < 400,
                "status": response.status,
                "url": url,
                "final_url": response.geturl(),
                "content_type": response.headers.get("content-type", ""),
                "content_length": response.headers.get("content-length", ""),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
            if method != "HEAD":
                result["body"] = body
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read(1024) if method != "HEAD" else b""
        return {
            "ok": False,
            "status": exc.code,
            "url": url,
            "final_url": exc.geturl(),
            "content_type": exc.headers.get("content-type", ""),
            "content_length": exc.headers.get("content-length", ""),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "error": body.decode("utf-8", "replace")[:500],
        }
    except Exception as exc:  # noqa: BLE001 - probe reports transport failures
        return {
            "ok": False,
            "status": None,
            "url": url,
            "final_url": None,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _json_tree(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    body = result.pop("body", b"")
    if not result["ok"] or not body:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["ok"] = False
        result["error"] = "response is not valid JSON"
        return None
    if not isinstance(value, list):
        result["ok"] = False
        result["error"] = "tree response is not a JSON list"
        return None
    result["file_count"] = len(value)
    return [item for item in value if isinstance(item, dict)]


def _model_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {str(item.get("path", "")) for item in files}
    weights = sorted(path for path in paths if path.endswith(".safetensors"))
    sizes = {
        item.get("path"): item.get("size")
        for item in files
        if item.get("path") in {"config.json", "tokenizer_config.json", DATASET_FILE}
    }
    return {
        "shards": len(weights),
        "has_config": "config.json" in paths,
        "has_tokenizer": any(
            path in paths
            for path in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        ),
        "sizes": sizes,
    }


def probe_source(base: str) -> dict[str, Any]:
    base = base.rstrip("/")
    model_tree = _request(
        f"{base}/api/models/{MODEL_REPO}/tree/{MODEL_REVISION}"
        "?recursive=true&expand=false"
    )
    model_files = _json_tree(model_tree)

    dataset_tree = _request(
        f"{base}/api/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}"
        "?recursive=true&expand=false"
    )
    dataset_files = _json_tree(dataset_tree)
    dataset_summary = None
    if dataset_files is not None:
        match = next(
            (item for item in dataset_files if item.get("path") == DATASET_FILE),
            None,
        )
        dataset_summary = {
            "has_science_jsonl": match is not None,
            "size": match.get("size") if match else None,
        }

    model_file = _request(
        f"{base}/{MODEL_REPO}/resolve/{MODEL_REVISION}/config.json",
        method="HEAD",
    )
    dataset_file = _request(
        f"{base}/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{DATASET_FILE}",
        method="HEAD",
    )
    return {
        "base_url": base,
        "model_tree": model_tree,
        "model_summary": _model_summary(model_files) if model_files is not None else None,
        "dataset_tree": dataset_tree,
        "dataset_summary": dataset_summary,
        "model_config_head": model_file,
        "dataset_file_head": dataset_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
        },
        "sources": [probe_source(base) for base in args.sources or DEFAULT_SOURCES],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if any(source["model_tree"]["ok"] for source in report["sources"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
