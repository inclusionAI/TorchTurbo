"""Stage-selection contracts for the ms-swift launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _launcher_module():
    path = Path(__file__).parents[3] / "scripts" / "runs" / "run_swift.py"
    spec = importlib.util.spec_from_file_location("torchturbo_test_run_swift", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_stage_supports_positional_form_and_default():
    launcher = _launcher_module()
    assert launcher._extract_stage(["dpo", "--model", "x"]) == (
        "dpo",
        ["--model", "x"],
    )
    assert launcher._extract_stage(["--model", "x"]) == (
        "sft",
        ["--model", "x"],
    )
