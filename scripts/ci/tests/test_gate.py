from __future__ import annotations

from dataclasses import replace

from scripts.ci.contracts.artifacts import (
    Issue,
    JobResult,
    ModelE2EEvidence,
    PytestEvidence,
    StepResult,
    build_plan,
    build_plan_artifact,
    digest,
    read_json,
    write_json,
)
from scripts.ci.consumer import gate
from scripts.ci.consumer.gate import decide, decide_artifact
from scripts.ci.producer.planner import build


def _result_for(plan, job, status: str = "passed") -> JobResult:
    steps = tuple(
        StepResult(
            id=step.id,
            status="passed" if status == "passed" else "failed",
            exit_code=0 if status == "passed" else 1,
            spec_digest=digest(step.spec.to_dict()),
            case_ids=(step.spec.case_id,) if hasattr(step.spec, "case_id") else (),
        )
        for step in job.steps
    )
    if job.kind == "pytest":
        files = {path for step in job.steps for path in step.spec.files}
        items = PytestEvidence(
            executed_files=len(files),
            tests_total=max(1, len(files)),
            tests_failed=0,
            tests_errors=0,
            tests_skipped=0,
        )
    else:
        items = ModelE2EEvidence(case_ids=tuple(
            step.spec.case_id for step in job.steps
        ))
    return JobResult(
        schema_version=1,
        plan_id=plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status=status,
        started_at="2026-11-16T00:00:00Z",
        finished_at="2026-11-16T00:01:00Z",
        duration_seconds=60,
        runtime=job.runtime,
        items=items,
        steps=steps,
        failure={"code": "step-failed", "step_id": job.steps[0].id} if status == "failed" else None,
    )


def _results(plan) -> list[JobResult]:
    return [_result_for(plan, job) for job in plan.jobs]


def _plan_with_allowed_skip(plan, job_id: str, reason: str):
    jobs = tuple(
        replace(job, allowed_skip_reasons=(reason,)) if job.id == job_id else job
        for job in plan.jobs
    )
    return build_plan(
        policy=plan.policy,
        source=plan.source,
        inputs=plan.inputs,
        coverage=plan.coverage,
        jobs=jobs,
    )


def test_all_jobs_pass(pr_plan) -> None:
    decision = decide(pr_plan, _results(pr_plan))
    assert decision.status == "passed"
    assert decision.summary["expected_jobs"] == 3
    assert decision.summary["passed"] == 3
    assert decision.failures == ()


def test_docs_only_plan_passes_without_results(catalog, pr_policy, source) -> None:
    plan = build(catalog, pr_policy, source, change_scope="docs_only")
    decision = decide(plan, [])

    assert decision.status == "passed"
    assert decision.summary == {
        "expected_jobs": 0,
        "reported_jobs": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "unexpected": 0,
    }
    assert all(check["status"] == "passed" for check in decision.checks)


def test_missing_result_fails(pr_plan) -> None:
    results = _results(pr_plan)[:-1]
    decision = decide(pr_plan, results)
    assert decision.status == "failed"
    assert "missing-result" in {failure["code"] for failure in decision.failures}


def test_incomplete_executor_result_is_infra_error(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(
        i for i, job in enumerate(pr_plan.jobs)
        if job.id == "pytest-core-npu-npu-default-default"
    )
    results[index] = replace(
        results[index],
        status="infra_error",
        steps=(),
        failure={
            "code": "executor-incomplete",
            "message": "executor did not finalize JobResult",
        },
    )

    decision = decide(pr_plan, results)
    codes = {failure["code"] for failure in decision.failures}
    assert decision.status == "failed"
    assert "job-infra_error" in codes
    assert "missing-result" not in codes
    assert decision.summary["failed"] == 1


def test_duplicate_result_fails(pr_plan) -> None:
    results = _results(pr_plan)
    decision = decide(pr_plan, [*results, results[0]])
    assert decision.status == "failed"
    assert "duplicate-result" in {failure["code"] for failure in decision.failures}


def test_unexpected_result_fails(pr_plan) -> None:
    result = _results(pr_plan)[0]
    decision = decide(pr_plan, [*_results(pr_plan), replace(result, job_id="unexpected-job")])
    assert decision.status == "failed"
    assert "unexpected-result" in {failure["code"] for failure in decision.failures}


def test_plan_id_mismatch_fails(pr_plan) -> None:
    results = _results(pr_plan)
    results[0] = replace(results[0], plan_id="sha256:bad")
    decision = decide(pr_plan, results)
    assert "result-plan-id" in {failure["code"] for failure in decision.failures}


def test_runtime_mismatch_fails(pr_plan) -> None:
    results = _results(pr_plan)
    results[0] = replace(results[0], runtime=replace(results[0].runtime, ranks=999))
    decision = decide(pr_plan, results)
    assert "result-runtime" in {failure["code"] for failure in decision.failures}


def test_allowed_skip_passes(pr_plan) -> None:
    index = next(
        i for i, result in enumerate(_results(pr_plan))
        if result.runtime.device == "npu" and result.kind == "pytest"
    )
    job = pr_plan.jobs[index]
    plan = _plan_with_allowed_skip(pr_plan, job.id, "maintenance")
    results = _results(plan)
    results[index] = replace(
        results[index],
        status="skipped",
        steps=(),
        failure=None,
        skip_reason="maintenance",
        skip_message="NPU maintenance window",
    )
    decision = decide(plan, results)
    assert decision.status == "passed"
    assert decision.summary["skipped"] == 1


def test_skip_reason_outside_allowlist_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(
        i for i, result in enumerate(results)
        if result.runtime.device == "npu" and result.kind == "pytest"
    )
    results[index] = replace(
        results[index],
        status="skipped",
        steps=(),
        failure=None,
        skip_reason="maintenance",
        skip_message="not allowed for npu",
    )
    decision = decide(pr_plan, results)
    assert decision.status == "failed"
    assert "skip-reason" in {failure["code"] for failure in decision.failures}


def test_pytest_without_tests_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    items = replace(results[index].items, tests_total=0)
    results[index] = replace(results[index], items=items)
    decision = decide(pr_plan, results)
    assert "pytest-no-tests" in {failure["code"] for failure in decision.failures}


def test_pytest_failures_fail(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    items = replace(results[index].items, tests_failed=1)
    results[index] = replace(results[index], items=items)
    decision = decide(pr_plan, results)
    assert "pytest-failures" in {failure["code"] for failure in decision.failures}


def test_e2e_case_mismatch_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "model_e2e")
    results[index] = replace(results[index], items=ModelE2EEvidence(case_ids=("wrong-case",)))
    decision = decide(pr_plan, results)
    assert "e2e-case-ids" in {failure["code"] for failure in decision.failures}


def test_coverage_mismatch_fails(pr_plan) -> None:
    coverage = {**pr_plan.coverage, "jobs": 99}
    broken = replace(pr_plan, coverage=coverage)
    decision = decide(broken, _results(pr_plan))
    assert "coverage-jobs" in {failure["code"] for failure in decision.failures}


def test_failed_planner_artifact_fails(pr_plan) -> None:
    artifact = build_plan_artifact(
        policy="pull_request",
        source=pr_plan.source,
        errors=(),
    )
    decision = decide_artifact(artifact)
    assert decision.status == "failed"
    assert decision.checks[0]["id"] == "planner.artifact"
    assert decision.failures[0]["code"] == "artifact-not-ready"


def test_planner_issue_path_is_not_a_step_id(pr_plan) -> None:
    artifact = build_plan_artifact(
        policy="pull_request",
        source=pr_plan.source,
        errors=(Issue(code="lint", path="scripts/ci/x.py", message="bad"),),
    )
    decision = decide_artifact(artifact)
    failure = decision.failures[0]
    assert failure["path"] == "scripts/ci/x.py"
    assert "step_id" not in failure


def _write_cli_inputs(tmp_path, plan, results):
    plan_path = tmp_path / "plan.json"
    results_dir = tmp_path / "results"
    write_json(plan_path, build_plan_artifact(
        policy=plan.policy, source=plan.source, plan=plan
    ).to_dict())
    for result in results:
        write_json(results_dir / result.job_id / "job-result.json", result.to_dict())
    write_json(results_dir / "diagnostics" / "torch-before.json", {"torch_ok": True})
    return plan_path, results_dir


def test_cli_aggregates_results_and_writes_decision(tmp_path, capsys, pr_plan) -> None:
    plan_path, results_dir = _write_cli_inputs(tmp_path, pr_plan, _results(pr_plan))
    decision_path = tmp_path / "decision.json"
    summary_path = tmp_path / "summary.md"

    assert gate.main([
        "--plan", str(plan_path),
        "--results", str(results_dir),
        "--output", str(decision_path),
        "--summary", str(summary_path),
    ]) == 0

    value = read_json(decision_path)
    assert value["status"] == "passed"
    assert value["summary"]["reported_jobs"] == len(pr_plan.jobs)
    summary = summary_path.read_text(encoding="utf-8")
    assert "## CI Gate: passed" in summary
    assert capsys.readouterr().out == ""


def test_cli_failed_result_returns_one(tmp_path, pr_plan) -> None:
    results = _results(pr_plan)
    results[0] = _result_for(pr_plan, pr_plan.jobs[0], status="failed")
    plan_path, results_dir = _write_cli_inputs(tmp_path, pr_plan, results)
    decision_path = tmp_path / "decision.json"

    assert gate.main([
        "--plan", str(plan_path),
        "--results", str(results_dir),
        "--output", str(decision_path),
    ]) == 1

    value = read_json(decision_path)
    assert value["status"] == "failed"
    assert "job-failed" in {item["code"] for item in value["failures"]}
