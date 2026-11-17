from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_script(
    "convert_text_dpo_to_swift",
    REPO_ROOT / "scripts/dataset/convert_text_dpo_to_swift.py",
)


def test_convert_dpo_row_adds_system_to_both_responses():
    converted = converter.convert_row(
        {
            "system": "Answer accurately.",
            "question": "What is 2 + 2?",
            "chosen": "4",
            "rejected": "5",
        }
    )

    assert converted == {
        "query": "What is 2 + 2?",
        "messages": [
            {"role": "system", "content": "Answer accurately."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected_messages": [
            {"role": "system", "content": "Answer accurately."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "5"},
        ],
    }


def test_convert_dpo_row_without_system_starts_with_user():
    converted = converter.convert_row(
        {"query": "Prompt", "chosen": "Preferred", "rejected": "Other"}
    )

    assert converted["messages"][0] == {"role": "user", "content": "Prompt"}
    assert converted["rejected_messages"][0] == {
        "role": "user",
        "content": "Prompt",
    }


def test_convert_dpo_row_drops_incomplete_preferences():
    assert converter.convert_row({"query": "Prompt", "chosen": "Preferred"}) is None
