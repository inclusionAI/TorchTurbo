from __future__ import annotations

from pathlib import Path

from scripts.ci.producer.change_scope import classify_paths, is_docs_path, main


def test_docs_paths_are_documentation_only() -> None:
    assert is_docs_path("docs/design.md")
    assert is_docs_path("docs/assets/overview.png")
    assert is_docs_path("README.md")
    assert is_docs_path("CHANGELOG.rst")
    assert is_docs_path("LICENSE")
    assert is_docs_path("NOTICE")


def test_code_and_config_paths_are_not_documentation_only() -> None:
    assert not is_docs_path("requirements.txt")
    assert not is_docs_path("pyproject.toml")
    assert not is_docs_path(".github/workflows/ascend-pr.yml")
    assert not is_docs_path("scripts/ci/producer/planner.py")
    assert not is_docs_path("pytest.ini")


def test_classify_paths_fails_open() -> None:
    assert classify_paths([]) == "code"
    assert classify_paths(["docs/design.md", "README.md"]) == "docs_only"
    assert classify_paths(["docs/design.md", "scripts/ci/producer/planner.py"]) == "code"
    assert classify_paths(["requirements.txt"]) == "code"


def test_cli_reads_changed_files(tmp_path: Path, capsys) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("docs/design.md\nREADME.md\n", encoding="utf-8")
    assert main(["--changed-files", str(changed)]) == 0
    assert capsys.readouterr().out.strip() == "change_scope=docs_only"


def test_cli_missing_file_fails_open(tmp_path: Path, capsys) -> None:
    assert main(["--changed-files", str(tmp_path / "missing.txt")]) == 0
    assert capsys.readouterr().out.strip() == "change_scope=code"
