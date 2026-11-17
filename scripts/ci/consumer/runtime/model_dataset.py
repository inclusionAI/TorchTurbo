"""Resolve model and dataset inputs for an integration case."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

CI_ROOT_ENV = "CI_ASSET_ROOT"
DOWNLOAD_ENV = "CI_MODEL_DATASET_DOWNLOAD"
REMOTE_SOURCES = {"huggingface", "modelscope"}
SOURCES = {"local", "ci", *REMOTE_SOURCES}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_MODEL_WEIGHTS = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class ModelDatasetError(ValueError):
    """Raised when a model, tokenizer, or dataset cannot be resolved."""


def allow_download_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether a job may download missing remote inputs.

    ``CI_MODEL_DATASET_DOWNLOAD`` is an explicit override. Without it, only
    scheduled and manually dispatched CI jobs may download. PR jobs and local
    runs remain cache-only.
    """
    values = os.environ if environ is None else environ
    override = values.get(DOWNLOAD_ENV)
    if override is not None:
        return override.strip().lower() in _TRUE_VALUES
    return values.get("GITHUB_EVENT_NAME") in {"schedule", "workflow_dispatch"}


def resolve_model_dataset_args(
    args: Mapping[str, Any],
    *,
    framework: str,
    root: Path,
    allow_download: bool = False,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Return a copy of ``args`` with model and dataset paths made local.

    The case YAML remains the source of truth. Values may be plain local paths,
    ``None`` for a generated model, or mappings with a ``source`` field. This
    function is the only place that knows how to find or download those inputs;
    the training frameworks receive ordinary local paths.
    """
    if framework not in {"swift", "llamafactory"}:
        raise ModelDatasetError(f"unsupported integration framework {framework!r}")
    values = os.environ if environ is None else environ
    resolved = copy.deepcopy(dict(args))
    evidence: list[dict[str, str]] = []

    model_field = "model" if framework == "swift" else "model_name_or_path"
    if model_field in resolved:
        resolved[model_field] = _resolve_value(
            resolved[model_field],
            field=model_field,
            role="model",
            root=root,
            allow_download=allow_download,
            environ=values,
            evidence=evidence,
            resolve_string=True,
        )
    if "tokenizer" in resolved:
        resolved["tokenizer"] = _resolve_value(
            resolved["tokenizer"],
            field="tokenizer",
            role="tokenizer",
            root=root,
            allow_download=allow_download,
            environ=values,
            evidence=evidence,
            resolve_string=True,
        )

    if framework == "swift":
        if "dataset" in resolved:
            resolved["dataset"] = _resolve_value(
                resolved["dataset"],
                field="dataset",
                role="dataset",
                root=root,
                allow_download=allow_download,
                environ=values,
                evidence=evidence,
                resolve_string=True,
            )
    elif "dataset_dir" in resolved:
        resolved["dataset_dir"] = _resolve_value(
            resolved["dataset_dir"],
            field="dataset_dir",
            role="dataset",
            root=root,
            allow_download=allow_download,
            environ=values,
            evidence=evidence,
            resolve_string=True,
        )
    return resolved, tuple(evidence)


def _resolve_value(
    value: Any,
    *,
    field: str,
    role: str,
    root: Path,
    allow_download: bool,
    environ: Mapping[str, str],
    evidence: list[dict[str, str]],
    resolve_string: bool,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        errors: list[str] = []
        candidates = _candidate_specs(value, field=field)
        for index, candidate in enumerate(candidates):
            try:
                path, source, revision = _resolve_spec(
                    candidate,
                    field=field,
                    role=role,
                    root=root,
                    allow_download=allow_download,
                    environ=environ,
                )
                _validate(path, role=role, spec=candidate, field=field)
            except ModelDatasetError as exc:
                errors.append(str(exc))
                continue
            item = {
                "field": field,
                "source": source,
                "path": str(path),
                **({"revision": revision} if revision else {}),
            }
            if index:
                item["fallback"] = "true"
            evidence.append(item)
            return str(path)
        raise ModelDatasetError(
            f"{field}: all configured sources failed: " + " | ".join(errors)
        )
    if isinstance(value, list):
        return [
            _resolve_value(
                item,
                field=f"{field}[{index}]",
                role=role,
                root=root,
                allow_download=allow_download,
                environ=environ,
                evidence=evidence,
                resolve_string=resolve_string,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and resolve_string:
        path = _local_path(value, root=root, field=field)
        _validate(path, role=role, spec={}, field=field)
        evidence.append({"field": field, "source": "local", "path": str(path)})
        return str(path)
    if not isinstance(value, str):
        raise ModelDatasetError(
            f"{field}: expected a path, null, mapping, or list; got {type(value).__name__}"
        )
    return value


def _candidate_specs(
    spec: Mapping[str, Any], *, field: str
) -> list[Mapping[str, Any]]:
    """Return the primary source followed by its single optional fallback."""
    primary = dict(spec)
    fallback = primary.pop("fallback", None)
    candidates: list[Mapping[str, Any]] = [primary]
    if fallback is None:
        return candidates
    if not isinstance(fallback, Mapping):
        raise ModelDatasetError(f"{field}.fallback must be a mapping")
    if "fallback" in fallback:
        raise ModelDatasetError(f"{field}.fallback cannot contain another fallback")
    candidates.append(dict(fallback))
    return candidates


def _resolve_spec(
    spec: Mapping[str, Any],
    *,
    field: str,
    role: str,
    root: Path,
    allow_download: bool,
    environ: Mapping[str, str],
) -> tuple[Path, str, str | None]:
    raw_source = spec.get("source")
    if not isinstance(raw_source, str) or not raw_source:
        raise ModelDatasetError(f"{field}.source is required")
    source = raw_source.lower()
    if source not in SOURCES:
        raise ModelDatasetError(
            f"{field}.source must be one of {sorted(SOURCES)}; got {raw_source!r}"
        )
    if source == "local":
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ModelDatasetError(f"{field}.path is required for source=local")
        return _local_path(raw_path, root=root, field=field), source, None
    if source == "ci":
        return _ci_path(spec, field=field, environ=environ), source, _revision(spec, field)
    if source == "huggingface":
        return _huggingface_path(
            spec,
            field=field,
            role=role,
            allow_download=allow_download,
            environ=environ,
        ), source, _revision(spec, field)
    return _modelscope_path(
        spec,
        field=field,
        role=role,
        allow_download=allow_download,
        environ=environ,
    ), source, _revision(spec, field)


def _local_path(value: str, *, root: Path, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelDatasetError(f"{field}: path does not exist: {path}") from exc


def _ci_path(
    spec: Mapping[str, Any], *, field: str, environ: Mapping[str, str]
) -> Path:
    raw_root = environ.get(CI_ROOT_ENV)
    if not raw_root:
        raise ModelDatasetError(
            f"{field}: {CI_ROOT_ENV} is required for source=ci"
        )
    root = Path(raw_root).expanduser().resolve()
    raw_name = spec.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        raise ModelDatasetError(f"{field}.name is required for source=ci")
    relative = Path(raw_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelDatasetError(
            f"{field}.name must stay below {CI_ROOT_ENV}: {raw_name!r}"
        )
    return _local_path(str(root / relative), root=root, field=field)


def _revision(spec: Mapping[str, Any], field: str) -> str | None:
    revision = spec.get("revision")
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision:
        raise ModelDatasetError(f"{field}.revision must be a non-empty string")
    return revision


def _remote_args(
    spec: Mapping[str, Any], *, field: str, environ: Mapping[str, str]
) -> tuple[str, str, str | None]:
    repo = spec.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ModelDatasetError(f"{field}.repo is required")
    revision = _revision(spec, field)
    if revision is None:
        raise ModelDatasetError(f"{field}.revision is required for a remote source")
    raw_file = spec.get("file")
    if raw_file is not None and (not isinstance(raw_file, str) or not raw_file):
        raise ModelDatasetError(f"{field}.file must be a non-empty string")
    return repo, revision, raw_file


def _huggingface_path(
    spec: Mapping[str, Any],
    *,
    field: str,
    role: str,
    allow_download: bool,
    environ: Mapping[str, str],
) -> Path:
    repo, revision, filename = _remote_args(spec, field=field, environ=environ)
    repo_type = "dataset" if role == "dataset" else "model"
    cache_dir = environ.get("HF_HUB_CACHE") or None
    try:
        from huggingface_hub import snapshot_download

        if filename:
            path = snapshot_download(
                repo_id=repo,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[filename],
                local_files_only=not allow_download,
            )
            candidate = Path(path) / filename
            if candidate.is_file():
                return candidate
            if Path(path).is_file():
                return Path(path)
            raise ModelDatasetError(f"{field}: {filename!r} is absent from {repo!r}")
        else:
            path = snapshot_download(
                repo_id=repo,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=not allow_download,
            )
        return Path(path)
    except Exception as exc:  # noqa: BLE001 - add input context to provider errors
        mode = "download" if allow_download else "cache"
        raise ModelDatasetError(
            f"{field}: cannot resolve Hugging Face {repo_type} {repo!r}@{revision} "
            f"from {mode}: {exc}"
        ) from exc


def _modelscope_path(
    spec: Mapping[str, Any],
    *,
    field: str,
    role: str,
    allow_download: bool,
    environ: Mapping[str, str],
) -> Path:
    repo, revision, filename = _remote_args(spec, field=field, environ=environ)
    cache_dir = environ.get("MODELSCOPE_CACHE") or None
    try:
        from modelscope import snapshot_download

        kwargs: dict[str, Any] = {
            "model_id": repo,
            "revision": revision,
            "local_files_only": not allow_download,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if role == "dataset":
            kwargs["repo_type"] = "dataset"
        if filename:
            kwargs["allow_file_pattern"] = filename
        result = Path(snapshot_download(**kwargs))
        if filename:
            candidate = result / filename
            if candidate.is_file():
                return candidate
            if result.is_file():
                return result
            raise ModelDatasetError(f"{field}: {filename!r} is absent from {repo!r}")
        return result
    except ModelDatasetError:
        raise
    except Exception as exc:  # noqa: BLE001 - add input context to provider errors
        mode = "download" if allow_download else "cache"
        raise ModelDatasetError(
            f"{field}: cannot resolve ModelScope {repo!r}@{revision} "
            f"from {mode}: {exc}"
        ) from exc


def _validate(path: Path, *, role: str, spec: Mapping[str, Any], field: str) -> None:
    if role == "model":
        _validate_model(path, spec=spec, field=field)
    elif role == "tokenizer":
        if not path.is_dir():
            raise ModelDatasetError(f"{field}: tokenizer must be a directory: {path}")
    elif not path.exists():
        raise ModelDatasetError(f"{field}: dataset does not exist: {path}")


def _validate_model(path: Path, *, spec: Mapping[str, Any], field: str) -> None:
    if not path.is_dir():
        raise ModelDatasetError(f"{field}: model must be a directory: {path}")
    required = spec.get("required_files")
    if required is not None:
        if isinstance(required, str):
            required = [required]
        if not isinstance(required, list) or not all(
            isinstance(item, str) and item for item in required
        ):
            raise ModelDatasetError(f"{field}.required_files must be a list of paths")
        for name in required:
            if not (path / name).is_file():
                raise ModelDatasetError(f"{field}: required model file is missing: {path / name}")
        return
    if not (path / "config.json").is_file():
        raise ModelDatasetError(f"{field}: config.json is missing from {path}")
    if any((path / name).is_file() for name in _MODEL_WEIGHTS):
        return
    if any(path.glob("*.safetensors")) or any(path.glob("*.bin")):
        return
    raise ModelDatasetError(f"{field}: no model weights found in {path}")
