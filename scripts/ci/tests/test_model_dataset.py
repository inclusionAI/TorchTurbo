from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.consumer.runtime import model_dataset
from scripts.ci.consumer.runtime.model_dataset import (
    ModelDatasetError,
    allow_download_from_env,
    resolve_model_dataset_args,
)


def _model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("", encoding="utf-8")
    return path


def test_local_paths_are_resolved_and_input_is_not_mutated(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data.json"
    data.write_text("{}", encoding="utf-8")
    args = {
        "model": None,
        "tokenizer": "tokenizer",
        "dataset": ["data.json"],
    }

    resolved, evidence = resolve_model_dataset_args(
        args,
        framework="swift",
        root=tmp_path,
    )

    assert resolved["model"] is None
    assert resolved["tokenizer"] == str(tokenizer)
    assert resolved["dataset"] == [str(data)]
    assert args["tokenizer"] == "tokenizer"
    assert [item["source"] for item in evidence] == ["local", "local"]


def test_ci_source_resolves_below_asset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "assets"
    model = _model(root / "models" / "tiny")
    monkeypatch.setenv("CI_ASSET_ROOT", str(root))

    resolved, evidence = resolve_model_dataset_args(
        {"model": {"source": "ci", "name": "models/tiny"}},
        framework="swift",
        root=tmp_path,
    )

    assert resolved["model"] == str(model)
    assert evidence == ({"field": "model", "source": "ci", "path": str(model)},)


def test_ci_source_requires_asset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI_ASSET_ROOT", raising=False)

    with pytest.raises(ModelDatasetError, match="CI_ASSET_ROOT"):
        resolve_model_dataset_args(
            {"model": {"source": "ci", "name": "models/tiny"}},
            framework="swift",
            root=tmp_path,
        )


def test_huggingface_dataset_uses_pinned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = tmp_path / "blob"
    blob.write_text("{}", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    dataset = snapshot / "science.jsonl"
    dataset.symlink_to(blob)
    calls: list[dict[str, object]] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    args = {
        "dataset": {
            "source": "huggingface",
            "repo": "a-m-team/AM-Thinking-v1-Distilled",
            "revision": "186b575b5cc2113f8ba81d4f4ac9ae2494633829",
            "file": "science.jsonl",
        }
    }

    resolved, _ = resolve_model_dataset_args(
        args,
        framework="swift",
        root=tmp_path,
        allow_download=True,
    )

    assert resolved["dataset"] == str(dataset)
    assert Path(resolved["dataset"]).name == "science.jsonl"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "186b575b5cc2113f8ba81d4f4ac9ae2494633829"
    assert calls[0]["allow_patterns"] == ["science.jsonl"]
    assert calls[0]["local_files_only"] is False


def test_huggingface_model_uses_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path / "model")
    calls: list[dict[str, object]] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(model)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    resolved, _ = resolve_model_dataset_args(
        {
            "model": {
                "source": "huggingface",
                "repo": "inclusionAI/Ling-3.0-tiny",
                "revision": "0123456789abcdef",
            }
        },
        framework="swift",
        root=tmp_path,
    )

    assert resolved["model"] == str(model)
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["local_files_only"] is True


def test_modelscope_dataset_uses_file_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data = dataset / "science.jsonl"
    data.write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(dataset)

    import modelscope

    monkeypatch.setattr(modelscope, "snapshot_download", fake_snapshot)
    resolved, _ = resolve_model_dataset_args(
        {
            "dataset": {
                "source": "modelscope",
                "repo": "example/dataset",
                "revision": "v1",
                "file": "science.jsonl",
            }
        },
        framework="swift",
        root=tmp_path,
    )

    assert resolved["dataset"] == str(data)
    assert calls[0]["allow_file_pattern"] == "science.jsonl"
    assert calls[0]["local_files_only"] is True


def test_remote_source_uses_configured_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data = dataset / "science.jsonl"
    data.write_text("{}", encoding="utf-8")

    def unavailable(*args: object, **kwargs: object) -> Path:
        raise ModelDatasetError("mirror unavailable")

    monkeypatch.setattr(model_dataset, "_huggingface_path", unavailable)
    monkeypatch.setattr(
        model_dataset,
        "_modelscope_path",
        lambda *args, **kwargs: data,
    )

    resolved, evidence = resolve_model_dataset_args(
        {
            "dataset": {
                "source": "huggingface",
                "repo": "example/dataset",
                "revision": "hf-revision",
                "file": "science.jsonl",
                "fallback": {
                    "source": "modelscope",
                    "repo": "example/dataset",
                    "revision": "ms-revision",
                    "file": "science.jsonl",
                },
            }
        },
        framework="swift",
        root=tmp_path,
        allow_download=True,
    )

    assert resolved["dataset"] == str(data)
    assert evidence[0]["source"] == "modelscope"
    assert evidence[0]["fallback"] == "true"


def test_fallback_cannot_be_nested(tmp_path: Path) -> None:
    with pytest.raises(ModelDatasetError, match="cannot contain another fallback"):
        resolve_model_dataset_args(
            {
                "dataset": {
                    "source": "huggingface",
                    "repo": "example/dataset",
                    "revision": "hf-revision",
                    "fallback": {
                        "source": "modelscope",
                        "repo": "example/dataset",
                        "revision": "ms-revision",
                        "fallback": {},
                    },
                }
            },
            framework="swift",
            root=tmp_path,
        )


def test_llamafactory_dataset_name_is_not_treated_as_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resolved, _ = resolve_model_dataset_args(
        {"dataset_dir": "dataset", "dataset": "data"},
        framework="llamafactory",
        root=tmp_path,
    )

    assert resolved["dataset_dir"] == str(dataset)
    assert resolved["dataset"] == "data"


def test_download_policy_defaults_to_offline_for_pr_and_local() -> None:
    assert allow_download_from_env({"GITHUB_EVENT_NAME": "pull_request"}) is False
    assert allow_download_from_env({"GITHUB_EVENT_NAME": "schedule"}) is True
    assert allow_download_from_env({"GITHUB_EVENT_NAME": "workflow_dispatch"}) is True
    assert allow_download_from_env({"CI_MODEL_DATASET_DOWNLOAD": "1"}) is True
    assert allow_download_from_env({"CI_MODEL_DATASET_DOWNLOAD": "0"}) is False


def test_model_requires_config_and_weights(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ModelDatasetError, match="config.json"):
        resolve_model_dataset_args(
            {"model": "model"},
            framework="swift",
            root=tmp_path,
        )
