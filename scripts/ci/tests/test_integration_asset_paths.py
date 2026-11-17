from __future__ import annotations

import importlib.util
from pathlib import Path


def _asset_path_module():
    path = Path(__file__).resolve().parents[3] / "tests/integration_tests/scripts/gen_init_checkpoint.py"
    spec = importlib.util.spec_from_file_location("ci_gen_init_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_path_preserves_named_symlink(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_text("{}", encoding="utf-8")
    named = tmp_path / "science.jsonl"
    named.symlink_to(blob)

    resolved = _asset_path_module().asset_path(str(named))

    assert resolved == named
    assert resolved.name == "science.jsonl"
