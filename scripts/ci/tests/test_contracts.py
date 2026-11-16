from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.contracts.artifacts import (
    ContractError,
    JobResult,
    ModelE2EEvidence,
    PytestEvidence,
    StepResult,
    digest,
    load_job_result,
    load_plan,
    load_result,
    write_json,
)


def test_plan_round_trip(tmp_path: Path, pr_plan) -> None:
    path = tmp_path / "plan.json"
    write_json(path, pr_plan.to_dict())
    loaded = load_plan(path)
    assert loaded.to_dict() == pr_plan.to_dict()


def test_plan_rejects_unknown_field(tmp_path: Path, pr_plan) -> None:
    value = pr_plan.to_dict()
    value["unexpected"] = True
    path = tmp_path / "plan.json"
    write_json(path, value)
    with pytest.raises(ContractError, match="unknown fields"):
        load_plan(path)


def test_plan_rejects_tampered_plan_id(tmp_path: Path, pr_plan) -> None:
    value = pr_plan.to_dict()
    value["plan_id"] = "sha256:bad"
    path = tmp_path / "plan.json"
    write_json(path, value)
    with pytest.raises(ContractError, match="plan_id does not match"):
        load_plan(path)


def _pytest_result(plan) -> JobResult:
    job = next(item for item in plan.jobs if item.kind == "pytest")
    steps = tuple(
        StepResult(
            id=step.id,
            status="passed",
            exit_code=0,
            spec_digest=digest(step.spec.to_dict()),
        )
        for step in job.steps
    )
    executed_files = sum(len(step.spec.files) for step in job.steps)
    return JobResult(
        schema_version=1,
        plan_id=plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status="passed",
        started_at="2026-11-16T00:00:00Z",
        finished_at="2026-11-16T00:01:00Z",
        duration_seconds=60,
        runtime=job.runtime,
        items=PytestEvidence(
            executed_files=executed_files,
            tests_total=100,
            tests_failed=0,
            tests_errors=0,
            tests_skipped=1,
        ),
        steps=steps,
        artifacts={"log": "turbo-ci.log", "junit": "pytest.xml"},
    )


def test_job_result_round_trip_and_aggregate(tmp_path: Path, pr_plan) -> None:
    result = _pytest_result(pr_plan)
    path = tmp_path / "result.json"
    write_json(path, result.to_dict())
    loaded = load_job_result(path)
    assert loaded.to_dict() == result.to_dict()

    aggregate = load_result([path])
    assert aggregate.plan_id == pr_plan.plan_id
    assert aggregate.jobs == (loaded,)


def test_result_rejects_unknown_status(tmp_path: Path, pr_plan) -> None:
    value = _pytest_result(pr_plan).to_dict()
    value["status"] = "not_started"
    path = tmp_path / "result.json"
    write_json(path, value)
    with pytest.raises(ContractError, match="unknown status"):
        load_job_result(path)


def test_skipped_result_requires_reason(tmp_path: Path, pr_plan) -> None:
    value = _pytest_result(pr_plan).to_dict()
    value["status"] = "skipped"
    path = tmp_path / "result.json"
    write_json(path, value)
    with pytest.raises(ContractError, match="requires skip.reason"):
        load_job_result(path)


def test_e2e_result_requires_case_evidence(tmp_path: Path, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "model_e2e")
    result = JobResult(
        schema_version=1,
        plan_id=pr_plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status="passed",
        started_at="2026-11-16T00:00:00Z",
        finished_at="2026-11-16T00:01:00Z",
        duration_seconds=60,
        runtime=job.runtime,
        items=ModelE2EEvidence(case_ids=(job.steps[0].spec.case_id,)),
        steps=(StepResult(
            id=job.steps[0].id,
            status="passed",
            exit_code=0,
            spec_digest=digest(job.steps[0].spec.to_dict()),
            case_ids=(job.steps[0].spec.case_id,),
        ),),
    )
    path = tmp_path / "result.json"
    write_json(path, result.to_dict())
    assert load_job_result(path).to_dict() == result.to_dict()

    bad = result.to_dict()
    bad["items"]["case_ids"] = []
    write_json(path, bad)
    with pytest.raises(ContractError, match="case_ids must not be empty"):
        load_job_result(path)
