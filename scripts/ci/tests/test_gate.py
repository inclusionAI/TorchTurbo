from __future__ import annotations

from dataclasses import replace

from scripts.ci.contracts.artifacts import (
    SCHEMA_VERSION,
    IntegrationEvidence,
    Issue,
    JobResult,
    PytestEvidence,
    PytestSpec,
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
        spec = job.steps[0].spec
        items = IntegrationEvidence(suite=spec.suite, case_ids=(spec.case_id,))
    return JobResult(
        schema_version=SCHEMA_VERSION,
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
        scope=plan.scope,
        source=plan.source,
        inputs=plan.inputs,
        coverage=plan.coverage,
        jobs=jobs,
    )


def test_all_jobs_pass(pr_plan) -> None:
    decision = decide(pr_plan, _results(pr_plan))
    assert decision.status == "passed"
    assert decision.summary["expected_jobs"] == 7
    assert decision.summary["passed"] == 7
    assert decision.failures == ()


def test_docs_only_plan_passes_without_results(config, facts, source) -> None:
    plan = build(config, facts, "pull_request", source, change_scope="docs_only")
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
    decision = decide(pr_plan, _results(pr_plan)[:-1])
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
    results.append(results[0])
    decision = decide(pr_plan, results)
    assert decision.status == "failed"
    assert "duplicate-result" in {failure["code"] for failure in decision.failures}


def test_extra_result_fails(pr_plan) -> None:
    results = _results(pr_plan)
    foreign = replace(results[0], job_id="unexpected-job")
    results.append(foreign)
    decision = decide(pr_plan, results)
    assert "unexpected-result" in {failure["code"] for failure in decision.failures}


def test_skip_reason_must_be_allowed(pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.runtime.device == "npu")
    allowed = _plan_with_allowed_skip(pr_plan, job.id, "maintenance")
    results = _results(allowed)
    index = next(i for i, result in enumerate(results) if result.job_id == job.id)
    results[index] = replace(
        results[index],
        status="skipped",
        steps=(),
        failure=None,
        skip_reason="maintenance",
        skip_message="window",
    )
    assert decide(allowed, results).status == "passed"

    results[index] = replace(results[index], skip_reason="free-form")
    decision = decide(allowed, results)
    assert decision.status == "failed"
    assert "skip-reason" in {failure["code"] for failure in decision.failures}


def test_pytest_without_tests_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    results[index] = replace(
        results[index],
        items=replace(results[index].items, tests_total=0),
    )
    decision = decide(pr_plan, results)
    assert "pytest-no-tests" in {failure["code"] for failure in decision.failures}


def test_pytest_failures_fail(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    results[index] = replace(
        results[index],
        items=replace(results[index].items, tests_failed=1),
    )
    decision = decide(pr_plan, results)
    assert "pytest-failures" in {failure["code"] for failure in decision.failures}


def test_integration_case_mismatch_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "integration")
    results[index] = replace(
        results[index],
        items=IntegrationEvidence(
            suite=results[index].items.suite,
            case_ids=("wrong-case",),
        ),
    )
    decision = decide(pr_plan, results)
    assert "integration-case-ids" in {failure["code"] for failure in decision.failures}


def test_integration_suite_mismatch_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "integration")
    results[index] = replace(
        results[index],
        items=replace(results[index].items, suite="wrong-suite"),
    )
    decision = decide(pr_plan, results)
    assert "integration-suite" in {failure["code"] for failure in decision.failures}


def test_coverage_mismatch_fails(pr_plan) -> None:
    coverage = {**pr_plan.coverage, "jobs": 99}
    broken = replace(pr_plan, coverage=coverage)
    decision = decide(broken, _results(pr_plan))
    assert "coverage-jobs" in {failure["code"] for failure in decision.failures}


def test_unsupported_route_cannot_also_be_executed(pr_plan) -> None:
    coverage = {
        **pr_plan.coverage,
        "unsupported_routes": [{
            "path": "tests/unit_tests/ops/test_packed_params.py",
            "device": "cpu",
        }],
    }
    broken = replace(pr_plan, coverage=coverage)
    decision = decide(broken, _results(pr_plan))
    assert "coverage-routes" in {failure["code"] for failure in decision.failures}


def test_same_pytest_file_on_distinct_targets_is_allowed(pr_plan) -> None:
    source_job = next(
        job for job in pr_plan.jobs
        if job.id == "pytest-core-npu-npu-default-default"
    )
    source_step = source_job.steps[0]
    assert isinstance(source_step.spec, PytestSpec)

    def target_job(target: str, suffix: str):
        runtime = replace(
            source_job.runtime,
            target=target,
            image=replace(source_job.runtime.image, name=f"{target}-image"),
        )
        return replace(
            source_job,
            id=f"{source_job.id}-{suffix}",
            runtime=runtime,
            steps=(replace(source_step, id=f"{source_job.id}-{suffix}/run"),),
        )

    jobs = (
        target_job("npu-a3", "a3"),
        target_job("npu-a5", "a5"),
    )
    files = len(source_step.spec.files)
    plan = build_plan(
        scope=pr_plan.scope,
        source=pr_plan.source,
        inputs=pr_plan.inputs,
        coverage={
            "pytest": {
                "core": {"mode": "all", "files": files, "routes": files * 2},
                "model": {"mode": "none", "files": 0, "routes": 0},
                "adapter": {"mode": "none", "files": 0, "routes": 0},
            },
            "integration": {"base_suite": {"mode": "none", "cases": []}},
            "unsupported_routes": [],
            "jobs": 2,
        },
        jobs=jobs,
    )
    decision = decide(plan, _results(plan))
    assert decision.status == "passed"
    assert "duplicate-route" not in {failure["code"] for failure in decision.failures}


def test_failed_planner_artifact_fails(pr_plan) -> None:
    artifact = build_plan_artifact(
        scope="pull_request",
        source=pr_plan.source,
        errors=(),
    )
    decision = decide_artifact(artifact)
    assert decision.status == "failed"
    assert decision.checks[0]["id"] == "planner.artifact"
    assert decision.failures[0]["code"] == "artifact-missing-plan"


def test_planner_issue_path_is_not_a_step_id(pr_plan) -> None:
    artifact = build_plan_artifact(
        scope="pull_request",
        source=pr_plan.source,
        errors=(Issue(code="lint", path="scripts/ci/x.py", message="bad"),),
    )
    failure = decide_artifact(artifact).failures[0]
    assert failure["path"] == "scripts/ci/x.py"
    assert "step_id" not in failure


def _write_cli_inputs(tmp_path, plan, results):
    plan_path = tmp_path / "plan.json"
    results_dir = tmp_path / "results"
    write_json(plan_path, build_plan_artifact(
        scope=plan.scope, source=plan.source, plan=plan
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
    assert "## CI Gate: passed" in summary_path.read_text(encoding="utf-8")
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
