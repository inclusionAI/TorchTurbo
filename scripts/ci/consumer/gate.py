#!/usr/bin/env python3
"""Judge a Plan against JobResult artifacts and write a Decision."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..contracts.artifacts import (
    ContractError,
    Decision,
    JOB_RESULT_FILENAME,
    Job,
    JobResult,
    ModelE2EEvidence,
    ModelE2ESpec,
    Plan,
    PlanArtifact,
    PytestEvidence,
    PytestSpec,
    canonical_json,
    digest,
    load_job_result,
    load_plan_artifact,
    write_json,
)


def _failure(
    code: str,
    detail: str,
    *,
    job_id: str = "",
    step_id: str = "",
    path: str = "",
) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "detail": detail}
    if job_id:
        value["job_id"] = job_id
    if step_id:
        value["step_id"] = step_id
    if path:
        value["path"] = path
    return value


def _plan_failures(plan: Plan) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if plan.schema_version != 1:
        failures.append(_failure("plan-schema", f"unsupported schema {plan.schema_version}"))
    job_ids: set[str] = set()
    step_ids: set[str] = set()
    test_routes: set[tuple[str, str]] = set()
    pytest_files = {"core": set(), "model": set(), "adapter": set()}
    pytest_routes: Counter[str] = Counter()
    e2e_cases: list[str] = []
    for job in plan.jobs:
        if job.id in job_ids:
            failures.append(_failure("duplicate-job", "duplicate job id", job_id=job.id))
        job_ids.add(job.id)
        if not job.steps:
            failures.append(_failure("empty-job", "job has no steps", job_id=job.id))
        for step in job.steps:
            if step.id in step_ids:
                failures.append(_failure("duplicate-step", "duplicate step id", step_id=step.id))
            step_ids.add(step.id)
            if isinstance(step.spec, PytestSpec):
                if step.spec.suite not in pytest_files:
                    failures.append(_failure("pytest-suite", "unknown pytest suite", step_id=step.id))
                    continue
                for path in step.spec.files:
                    route = (path, job.runtime.device)
                    if route in test_routes:
                        failures.append(_failure("duplicate-route", f"duplicate route {route}", step_id=step.id))
                    test_routes.add(route)
                    pytest_files[step.spec.suite].add(path)
                    pytest_routes[step.spec.suite] += 1
            else:
                e2e_cases.append(step.spec.case_id)

    coverage = plan.coverage
    if coverage.get("jobs") != len(plan.jobs):
        failures.append(_failure("coverage-jobs", "coverage job count differs from jobs"))
    pytest_coverage = coverage.get("pytest")
    if not isinstance(pytest_coverage, dict) or set(pytest_coverage) != set(pytest_files):
        failures.append(_failure("coverage-pytest", "coverage pytest suites are invalid"))
    else:
        for suite, files in pytest_files.items():
            value = pytest_coverage[suite]
            expected = {"files": len(files), "routes": pytest_routes[suite]}
            if not isinstance(value, dict) or any(value.get(key) != count for key, count in expected.items()):
                failures.append(_failure(
                    "coverage-pytest",
                    f"coverage for suite {suite} does not match jobs",
                ))
    e2e_coverage = coverage.get("model_e2e")
    if not isinstance(e2e_coverage, dict):
        failures.append(_failure("coverage-e2e", "coverage model_e2e is invalid"))
    elif sorted(e2e_coverage.get("cases") or []) != sorted(e2e_cases):
        failures.append(_failure("coverage-e2e", "coverage E2E cases differ from jobs"))
    return failures


def _expected_step_digests(job: Job) -> dict[str, str]:
    return {step.id: digest(step.spec.to_dict()) for step in job.steps}


def _expected_pytest_files(job: Job) -> set[str]:
    return {
        path
        for step in job.steps
        if isinstance(step.spec, PytestSpec)
        for path in step.spec.files
    }


def _expected_cases(job: Job) -> set[str]:
    return {
        step.spec.case_id
        for step in job.steps
        if isinstance(step.spec, ModelE2ESpec)
    }


def _result_failures(plan: Plan, job: Job, result: JobResult) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if result.plan_id != plan.plan_id:
        failures.append(_failure("result-plan-id", "result plan_id differs from Plan", job_id=job.id))
    if result.kind != job.kind:
        failures.append(_failure("result-kind", "result kind differs from Plan", job_id=job.id))
    if result.runtime.to_dict() != job.runtime.to_dict():
        failures.append(_failure("result-runtime", "result runtime differs from Plan", job_id=job.id))
    if result.status not in {"passed", "failed", "skipped", "cancelled", "infra_error"}:
        failures.append(_failure("result-status", f"unknown status {result.status!r}", job_id=job.id))
        return failures

    expected_steps = _expected_step_digests(job)
    seen_steps: set[str] = set()
    for step in result.steps:
        if step.id in seen_steps:
            failures.append(_failure("duplicate-step-result", "duplicate step result", job_id=job.id, step_id=step.id))
        seen_steps.add(step.id)
        if step.id not in expected_steps:
            failures.append(_failure("unknown-step-result", "step result is not in Plan", job_id=job.id, step_id=step.id))
        elif step.spec_digest != expected_steps[step.id]:
            failures.append(_failure("result-spec-digest", "step spec digest differs from Plan", job_id=job.id, step_id=step.id))

    if result.status == "passed":
        if result.skip_reason is not None:
            failures.append(_failure("unexpected-skip", "passed result has a skip reason", job_id=job.id))
        if seen_steps != set(expected_steps):
            failures.append(_failure("result-steps", "passed result does not cover all planned steps", job_id=job.id))
        if any(step.status != "passed" or step.exit_code != 0 for step in result.steps):
            failures.append(_failure("result-step-status", "passed job has a non-passing step", job_id=job.id))
        if isinstance(result.items, PytestEvidence):
            if result.items.tests_total == 0:
                failures.append(_failure("pytest-no-tests", "pytest collected no tests", job_id=job.id))
            if result.items.tests_failed or result.items.tests_errors:
                failures.append(_failure("pytest-failures", "pytest evidence contains failures or errors", job_id=job.id))
            expected_files = _expected_pytest_files(job)
            if result.items.executed_files != len(expected_files):
                failures.append(_failure(
                    "pytest-file-count",
                    f"expected {len(expected_files)} files, got {result.items.executed_files}",
                    job_id=job.id,
                ))
        elif isinstance(result.items, ModelE2EEvidence):
            if set(result.items.case_ids) != _expected_cases(job):
                failures.append(_failure("e2e-case-ids", "E2E case evidence differs from Plan", job_id=job.id))
        else:
            failures.append(_failure("result-items", "result evidence type is invalid", job_id=job.id))
    elif result.status == "skipped":
        if result.skip_reason is None:
            failures.append(_failure("skip-reason", "skipped result has no reason", job_id=job.id))
        elif result.skip_reason not in job.allowed_skip_reasons:
            failures.append(_failure("skip-reason", "skip reason is not allowed by Plan", job_id=job.id))
        if result.steps:
            failures.append(_failure("skip-steps", "skipped result contains step results", job_id=job.id))
    elif result.status == "failed":
        if result.failure is None and not any(step.status == "failed" for step in result.steps):
            failures.append(_failure("failed-detail", "failed result has no failure evidence", job_id=job.id))
    return failures


def _check(identifier: str, failures: Iterable[dict[str, Any]], detail: str = "") -> dict[str, Any]:
    values = list(failures)
    value: dict[str, Any] = {
        "id": identifier,
        "status": "failed" if values else "passed",
    }
    if values:
        value["detail"] = detail or "; ".join(item["detail"] for item in values[:3])
    return value


def decide(plan: Plan, results: list[JobResult]) -> Decision:
    """Judge one Plan and its JobResults without reading Catalog or Policy."""
    failures: list[dict[str, Any]] = []
    plan_failures = _plan_failures(plan)
    failures.extend(plan_failures)

    by_id: dict[str, list[JobResult]] = {}
    for result in results:
        by_id.setdefault(result.job_id, []).append(result)
    expected = {job.id: job for job in plan.jobs}
    for job_id, values in sorted(by_id.items()):
        if len(values) > 1:
            failures.append(_failure("duplicate-result", "multiple results for one job", job_id=job_id))
        if job_id not in expected:
            failures.append(_failure("unexpected-result", "result job is not in Plan", job_id=job_id))

    missing = sorted(set(expected) - set(by_id))
    for job_id in missing:
        failures.append(_failure("missing-result", "planned job has no result", job_id=job_id))

    job_contract_failures: list[dict[str, Any]] = []
    status_failures: list[dict[str, Any]] = []
    for job_id in sorted(set(expected) & set(by_id)):
        values = by_id[job_id]
        if len(values) != 1:
            continue
        job = expected[job_id]
        result = values[0]
        current = _result_failures(plan, job, result)
        job_contract_failures.extend(
            item for item in current
            if item["code"] not in {"pytest-no-tests", "pytest-failures"}
        )
        status_failures.extend(
            item for item in current
            if item["code"] in {"pytest-no-tests", "pytest-failures"}
        )
        if result.status in {"failed", "infra_error", "cancelled"}:
            status_failures.append(_failure(
                f"job-{result.status}",
                f"job finished with status {result.status}",
                job_id=job_id,
            ))

    failures.extend(job_contract_failures)
    failures.extend(status_failures)
    status_counts = Counter(
        values[0].status for job_id, values in by_id.items()
        if job_id in expected and len(values) == 1
    )
    summary = {
        "expected_jobs": len(plan.jobs),
        "reported_jobs": len(results),
        "passed": status_counts["passed"],
        "failed": status_counts["failed"] + status_counts["infra_error"] + status_counts["cancelled"],
        "skipped": status_counts["skipped"],
        "missing": len(missing),
        "unexpected": sum(1 for job_id in by_id if job_id not in expected),
    }
    checks = (
        _check("plan.contract", plan_failures, "Plan structure is invalid"),
        _check(
            "results.complete",
            [
                item for item in failures
                if item["code"] in {
                    "missing-result", "unexpected-result", "duplicate-result",
                    "result-plan-id",
                }
            ],
            "Result set is incomplete or inconsistent",
        ),
        _check("jobs.contract", job_contract_failures, "Job results do not match Plan"),
        _check("jobs.status", status_failures, "One or more jobs did not pass"),
    )
    return Decision(
        schema_version=1,
        plan_id=plan.plan_id,
        status="passed" if not failures else "failed",
        summary=summary,
        checks=checks,
        failures=tuple(failures),
    )


def decide_artifact(artifact: PlanArtifact) -> Decision:
    """Turn a failed Planner artifact into a failed Decision."""
    if artifact.status == "ready" and artifact.plan is not None:
        return decide(artifact.plan, [])
    failures = tuple(
        _failure(f"artifact-{issue.code}", issue.message, path=issue.path)
        for issue in artifact.errors
    )
    if not failures:
        failures = (
            _failure(
                "artifact-not-ready",
                f"planner artifact status is {artifact.status!r}, not 'ready'",
            ),
        )
    return Decision(
        schema_version=1,
        plan_id=artifact.plan.plan_id if artifact.plan is not None else "",
        status="failed",
        summary={
            "expected_jobs": 0,
            "reported_jobs": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "missing": 0,
            "unexpected": 0,
        },
        checks=({
            "id": "planner.artifact",
            "status": "failed",
            "detail": "Planner did not produce a ready Plan",
        },),
        failures=failures,
    )


def _load_results(path: Path | None) -> list[JobResult]:
    if path is None:
        return []
    if path.is_file():
        return [load_job_result(path)]
    if path.is_dir():
        return [
            load_job_result(item)
            for item in sorted(path.rglob(JOB_RESULT_FILENAME))
        ]
    raise ContractError(f"results path does not exist: {path}")


def render_summary(decision: Decision) -> str:
    lines = [
        f"## CI Gate: {decision.status}",
        "",
        "| expected | reported | passed | failed | skipped | missing | unexpected |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {expected_jobs} | {reported_jobs} | {passed} | {failed} | {skipped} | "
        "{missing} | {unexpected} |".format(**decision.summary),
    ]
    if decision.failures:
        lines += ["", "### Failures", ""]
        for failure in decision.failures:
            target = (
                failure.get("job_id")
                or failure.get("step_id")
                or failure.get("path")
                or ""
            )
            prefix = f"`{target}`: " if target else ""
            lines.append(f"- `{failure['code']}`: {prefix}{failure['detail']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Plan or PlanArtifact JSON")
    parser.add_argument("--results", default="", help="JobResult file or directory")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)
    try:
        artifact = load_plan_artifact(Path(args.plan))
        if artifact.status != "ready":
            decision = decide_artifact(artifact)
        else:
            if artifact.plan is None:
                raise ContractError("ready plan artifact has no plan")
            decision = decide(
                artifact.plan,
                _load_results(Path(args.results) if args.results else None),
            )
    except (ContractError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    if args.output:
        write_json(Path(args.output), decision.to_dict())
    else:
        print(canonical_json(decision.to_dict()))
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as output:
            output.write(render_summary(decision))
    return 0 if decision.status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
