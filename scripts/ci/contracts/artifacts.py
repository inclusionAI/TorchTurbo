#!/usr/bin/env python3
"""Serializable contracts shared by the CI planner, executor, and gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1
JOB_RESULT_FILENAME = "job-result.json"
SUITES = ("core", "model", "adapter")
JOB_KINDS = ("pytest", "model_e2e")
DEVICES = ("cpu", "npu", "gpu")
JOB_STATUSES = ("passed", "failed", "skipped", "cancelled", "infra_error")
PLAN_ARTIFACT_STATUSES = ("ready", "failed")


class ContractError(ValueError):
    """Raised when a CI contract is malformed or internally inconsistent."""


class Contract:
    """Dataclass mixin that keeps the public ``to_dict`` API."""

    def to_dict(self) -> dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class Issue(Contract):
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class Source(Contract):
    repository: str
    ref: str
    sha: str


@dataclass(frozen=True)
class ImageRef(Contract):
    name: str
    kind: str
    ref: str
    digest: str = ""


@dataclass(frozen=True)
class VenvSpec(Contract):
    name: str
    python: str
    requirements: tuple[str, ...]
    overrides: tuple[str, ...]
    extends: str | None
    constraints: str
    digest: str


@dataclass(frozen=True)
class Runtime(Contract):
    device: str
    ranks: int
    lane: str
    image: ImageRef
    venv: VenvSpec


@dataclass(frozen=True)
class PytestSpec(Contract):
    suite: str
    frameworks: tuple[str, ...]
    model_families: tuple[str, ...]
    marker: str
    process_group: str
    tags: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec(Contract):
    script: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelE2ESpec(Contract):
    case_id: str
    model: str
    framework: str
    run_dir: str
    workdir: str
    install: CommandSpec
    train: CommandSpec
    hf_repo: str
    layers: int
    dataset: str
    ranks: int
    parallelism_sp: int
    parallelism_ep: int
    seq_len: int
    finetuning_type: str
    fetch_timeout_seconds: int
    case_timeout_seconds: int


@dataclass(frozen=True)
class Step(Contract):
    id: str
    kind: str
    ranks: int
    timeout_seconds: int
    spec: PytestSpec | ModelE2ESpec


@dataclass(frozen=True)
class Job(Contract):
    id: str
    kind: str
    allowed_skip_reasons: tuple[str, ...]
    timeout_minutes: int
    runtime: Runtime
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Plan(Contract):
    schema_version: int
    plan_id: str
    policy: str
    source: Source
    inputs: dict[str, str]
    coverage: dict[str, Any]
    jobs: tuple[Job, ...]


@dataclass(frozen=True)
class PlanArtifact(Contract):
    schema_version: int
    artifact_id: str
    status: str
    policy: str
    source: Source
    errors: tuple[Issue, ...] = ()
    plan: Plan | None = None


@dataclass(frozen=True)
class PytestEvidence(Contract):
    executed_files: int
    tests_total: int
    tests_failed: int
    tests_errors: int
    tests_skipped: int


@dataclass(frozen=True)
class ModelE2EEvidence(Contract):
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepResult(Contract):
    id: str
    status: str
    exit_code: int
    spec_digest: str
    case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobResult(Contract):
    schema_version: int
    plan_id: str
    job_id: str
    kind: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: int
    runtime: Runtime
    items: PytestEvidence | ModelE2EEvidence
    steps: tuple[StepResult, ...]
    skip_reason: str | None = None
    skip_message: str | None = None
    failure: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Result(Contract):
    schema_version: int
    plan_id: str
    jobs: tuple[JobResult, ...]


@dataclass(frozen=True)
class Decision(Contract):
    schema_version: int
    plan_id: str
    status: str
    summary: dict[str, int]
    checks: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]


def _dump(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _dump(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _dump(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_dump(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Return deterministic JSON for hashing and artifact comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_payload(plan: Plan) -> dict[str, Any]:
    value = plan.to_dict()
    value.pop("plan_id")
    return value


def plan_digest(plan: Plan) -> str:
    """Return the digest that must match ``plan.plan_id``."""
    return digest(_plan_payload(plan))


def _artifact_payload(artifact: PlanArtifact) -> dict[str, Any]:
    value = artifact.to_dict()
    value.pop("artifact_id")
    return value


def build_plan(
    *,
    policy: str,
    source: Source,
    inputs: dict[str, str],
    coverage: dict[str, Any],
    jobs: list[Job] | tuple[Job, ...],
) -> Plan:
    plan = Plan(
        schema_version=SCHEMA_VERSION,
        plan_id="",
        policy=policy,
        source=source,
        inputs=dict(inputs),
        coverage=coverage,
        jobs=tuple(jobs),
    )
    return replace(plan, plan_id=plan_digest(plan))


def build_plan_artifact(
    *,
    policy: str,
    source: Source,
    errors: list[Issue] | tuple[Issue, ...] = (),
    plan: Plan | None = None,
) -> PlanArtifact:
    status = "ready" if plan is not None and not errors else "failed"
    artifact = PlanArtifact(
        schema_version=SCHEMA_VERSION,
        artifact_id="",
        status=status,
        policy=policy,
        source=source,
        errors=tuple(errors),
        plan=plan,
    )
    return replace(artifact, artifact_id=digest(_artifact_payload(artifact)))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path} {message}")


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _decode(tp: Any, value: Any, path: str) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        candidates = [item for item in args if item is not type(None)]
        if value is None and type(None) in args:
            return None
        for candidate in candidates:
            try:
                return _decode(candidate, value, path)
            except ContractError:
                pass
        _fail(path, "has an invalid value")
    if origin is tuple:
        if not isinstance(value, list):
            _fail(path, "must be a list")
        if args and args[-1] is Ellipsis:
            return tuple(_decode(args[0], item, f"{path}[{index}]")
                         for index, item in enumerate(value))
        if len(value) != len(args):
            _fail(path, f"must contain {len(args)} items")
        return tuple(_decode(item_tp, item, f"{path}[{index}]")
                     for index, (item_tp, item) in enumerate(zip(args, value)))
    if origin is list:
        if not isinstance(value, list):
            _fail(path, "must be a list")
        item_type = args[0] if args else Any
        return [_decode(item_type, item, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            _fail(path, "must be a mapping")
        key_type, value_type = args if len(args) == 2 else (str, Any)
        return {
            _decode(key_type, key, f"{path} key"):
            _decode(value_type, item, f"{path}.{key}")
            for key, item in value.items()
        }
    if tp is str:
        if not isinstance(value, str):
            _fail(path, "must be a string")
        return value
    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            _fail(path, "must be an integer")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            _fail(path, "must be a boolean")
        return value
    if isinstance(tp, type) and is_dataclass(tp):
        return _decode_dataclass(tp, value, path)
    return value


def _decode_spec(kind: Any, value: Any, path: str) -> PytestSpec | ModelE2ESpec:
    if not isinstance(kind, str):
        _fail(path, "has no kind")
    if kind == "pytest":
        return _decode(PytestSpec, value, path)
    if kind == "model_e2e":
        return _decode(ModelE2ESpec, value, path)
    _fail(f"{path}.kind", f"has unknown kind {kind!r}")


def _decode_dataclass(cls: type, value: Any, path: str) -> Any:
    if not isinstance(value, dict):
        _fail(path, "must be a mapping")
    allowed = {item.name for item in fields(cls)}
    defaults = {
        item.name for item in fields(cls)
        if item.default is not MISSING or item.default_factory is not MISSING
    }
    missing = sorted(allowed - set(value) - defaults)
    unknown = sorted(set(value) - allowed)
    if missing:
        _fail(path, f"is missing fields: {', '.join(missing)}")
    if unknown:
        _fail(path, f"has unknown fields: {', '.join(unknown)}")

    hints = _hints(cls)
    data: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in value:
            continue
        if cls is Step and item.name == "spec":
            data[item.name] = _decode_spec(value.get("kind"), value[item.name], f"{path}.spec")
        elif cls is JobResult and item.name == "items":
            evidence = PytestEvidence if value.get("kind") == "pytest" else ModelE2EEvidence
            data[item.name] = _decode(evidence, value[item.name], f"{path}.items")
        else:
            data[item.name] = _decode(hints[item.name], value[item.name], f"{path}.{item.name}")
    result = cls(**data)
    _validate(result, path)
    return result


def _validate(value: Any, path: str) -> None:
    if isinstance(value, ImageRef) and value.kind not in {"host", "container"}:
        _fail(f"{path}.kind", "must be host or container")
    if isinstance(value, Runtime):
        if value.device not in DEVICES:
            _fail(f"{path}.device", f"has unknown device {value.device!r}")
        if value.ranks < 1:
            _fail(f"{path}.ranks", "must be >= 1")
    if isinstance(value, PytestSpec) and value.suite not in SUITES:
        _fail(f"{path}.suite", f"has unknown suite {value.suite!r}")
    if isinstance(value, ModelE2ESpec):
        for name in (
            "ranks", "parallelism_sp", "parallelism_ep", "seq_len",
            "fetch_timeout_seconds", "case_timeout_seconds",
        ):
            if getattr(value, name) < 1:
                _fail(f"{path}.{name}", "must be >= 1")
        if value.layers < 0:
            _fail(f"{path}.layers", "must be >= 0")
    if isinstance(value, Step):
        if value.ranks < 1 or value.timeout_seconds < 1:
            _fail(path, "ranks and timeout_seconds must be >= 1")
        if (value.kind == "pytest") != isinstance(value.spec, PytestSpec):
            _fail(path, "kind does not match spec")
    if isinstance(value, Job):
        if value.kind not in JOB_KINDS:
            _fail(f"{path}.kind", f"has unknown kind {value.kind!r}")
        if value.timeout_minutes < 1:
            _fail(f"{path}.timeout_minutes", "must be >= 1")
        if any(step.kind != value.kind for step in value.steps):
            _fail(path, "contains a step with a different kind")
    if isinstance(value, Plan):
        if value.schema_version != SCHEMA_VERSION:
            _fail(f"{path}.schema_version",
                  f"is {value.schema_version}, expected {SCHEMA_VERSION}")
        expected = plan_digest(value)
        if value.plan_id != expected:
            _fail(f"{path}.plan_id", f"does not match plan contents; expected {expected}")
    if isinstance(value, PlanArtifact):
        if value.schema_version != SCHEMA_VERSION:
            _fail(f"{path}.schema_version",
                  f"is {value.schema_version}, expected {SCHEMA_VERSION}")
        if value.status not in PLAN_ARTIFACT_STATUSES:
            _fail(f"{path}.status", f"has unknown status {value.status!r}")
        if value.status == "ready" and (value.plan is None or value.errors):
            _fail(path, "status=ready requires a plan and no errors")
        if value.status == "failed" and not value.errors:
            _fail(path, "status=failed requires at least one error")
        expected = digest(_artifact_payload(value))
        if value.artifact_id != expected:
            _fail(f"{path}.artifact_id", f"does not match contents; expected {expected}")
    if isinstance(value, PytestEvidence):
        if min(value.executed_files, value.tests_total, value.tests_failed,
               value.tests_errors, value.tests_skipped) < 0:
            _fail(path, "counts must be >= 0")
    if isinstance(value, ModelE2EEvidence) and not value.case_ids:
        _fail(f"{path}.case_ids", "must not be empty")
    if isinstance(value, StepResult):
        if value.status not in JOB_STATUSES:
            _fail(f"{path}.status", f"has unknown status {value.status!r}")
        if value.exit_code < -255:
            _fail(f"{path}.exit_code", "must be >= -255")
    if isinstance(value, JobResult):
        if value.schema_version != SCHEMA_VERSION:
            _fail(f"{path}.schema_version",
                  f"is {value.schema_version}, expected {SCHEMA_VERSION}")
        if value.kind not in JOB_KINDS:
            _fail(f"{path}.kind", f"has unknown kind {value.kind!r}")
        if value.status not in JOB_STATUSES:
            _fail(f"{path}.status", f"has unknown status {value.status!r}")
        if value.status == "skipped" and value.skip_reason is None:
            _fail(path, "status=skipped requires skip.reason")
        if value.status != "skipped" and value.skip_reason is not None:
            _fail(path, "skip is only valid for skipped status")
        if (value.kind == "pytest") != isinstance(value.items, PytestEvidence):
            _fail(f"{path}.items", "does not match JobResult kind")


def load_plan(path: Path) -> Plan:
    """Load a Plan and verify its schema, fields, and content digest."""
    try:
        return _decode(Plan, read_json(path), "plan")
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read plan {path}: {exc}") from exc


def load_plan_artifact(path: Path) -> PlanArtifact:
    """Load and verify a Planner artifact, including failed-plan output."""
    try:
        return _decode(PlanArtifact, read_json(path), "artifact")
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read plan artifact {path}: {exc}") from exc


def load_job_result(path: Path) -> JobResult:
    """Load one JobResult JSON file."""
    try:
        return _decode(JobResult, read_json(path), "job_result")
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read job result {path}: {exc}") from exc


def load_result(paths: list[Path]) -> Result:
    """Load JobResult files and aggregate them into a Result."""
    if not paths:
        raise ContractError("at least one job result is required")
    jobs = [load_job_result(path) for path in paths]
    plan_ids = {job.plan_id for job in jobs}
    if len(plan_ids) != 1:
        raise ContractError("job results have different plan_id values")
    job_ids = [job.job_id for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise ContractError("job results contain duplicate job_id values")
    return Result(schema_version=SCHEMA_VERSION, plan_id=next(iter(plan_ids)), jobs=tuple(jobs))
