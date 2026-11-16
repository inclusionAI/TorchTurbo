from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.ci.contracts.artifacts import (
    IntegrationSpec,
    Issue,
    Plan,
    PytestSpec,
    load_plan_artifact,
    plan_digest,
    write_json,
)
from scripts.ci.producer.ci_config import CiConfig, ScopeSuite
from scripts.ci.producer.facts import TestFacts
from scripts.ci.producer.planner import (
    PlanError,
    artifact_for,
    build,
    main,
    matrix_outputs,
    validate_plan,
)


def _with_plan_id(plan: Plan) -> Plan:
    return replace(plan, plan_id=plan_digest(plan))


def test_pull_request_scope(config: CiConfig, facts: TestFacts, pr_plan: Plan) -> None:
    assert Counter(job.kind for job in pr_plan.jobs) == {
        "pytest": 2,
        "integration": 2,
    }
    assert pr_plan.coverage == {
        "pytest": {
            "core": {"mode": "all", "files": 86, "routes": 88},
            "model": {"mode": "none", "files": 0, "routes": 0},
            "adapter": {"mode": "none", "files": 0, "routes": 0},
        },
        "integration": {
            "base_suite": {
                "mode": "explicit",
                "cases": ["qwen3_5_moe/sft_2cpu", "qwen3_5_moe/sft_2npu"],
            }
        },
        "unsupported_routes": [{
            "path": "tests/unit_tests/ops/test_packed_params.py",
            "device": "gpu",
        }],
        "jobs": 4,
    }
    assert validate_plan(pr_plan, config, facts) == []
    assert [job.id for job in pr_plan.jobs] == [
        "integration-base-suite-qwen3-5-moe-sft-2cpu",
        "integration-base-suite-qwen3-5-moe-sft-2npu",
        "pytest-core-cpu-cpu-default-default",
        "pytest-core-npu-npu-default-default",
    ]
    for job in pr_plan.jobs:
        if job.kind == "integration":
            assert len(job.steps) == 1
            assert isinstance(job.steps[0].spec, IntegrationSpec)
            assert job.steps[0].ranks == job.runtime.ranks


def test_docs_only_scope_has_no_jobs(
    config: CiConfig, facts: TestFacts, source
) -> None:
    plan = build(config, facts, "pull_request", source, change_scope="docs_only")
    assert plan.jobs == ()
    assert plan.inputs["change_scope"] == "docs_only"
    assert plan.coverage["jobs"] == 0
    assert plan.coverage["unsupported_routes"] == []
    assert plan.coverage["integration"] == {
        "base_suite": {"mode": "none", "cases": []}
    }
    assert validate_plan(plan, config, facts) == []
    matrices = matrix_outputs(plan, config)
    assert {key: len(value) for key, value in matrices.items()} == {
        "cpu_jobs": 0,
        "npu_jobs": 0,
        "gpu_jobs": 0,
    }


def test_nightly_scope(config: CiConfig, facts: TestFacts, nightly_plan: Plan) -> None:
    assert Counter(job.kind for job in nightly_plan.jobs) == {
        "pytest": 9,
        "integration": 3,
    }
    pytest_suites = {
        step.spec.suite
        for job in nightly_plan.jobs
        if job.kind == "pytest"
        for step in job.steps
        if isinstance(step.spec, PytestSpec)
    }
    assert pytest_suites == {"core", "model", "adapter"}
    assert nightly_plan.coverage["integration"]["base_suite"]["cases"] == [
        "bailing_v3/sft_16npu",
        "qwen3_5_moe/sft_2cpu",
        "qwen3_5_moe/sft_2npu",
    ]
    assert validate_plan(nightly_plan, config, facts) == []


def test_target_integration_job(
    config: CiConfig, facts: TestFacts, pr_plan: Plan, source
) -> None:
    target = "integration-base-suite-qwen3-5-moe-sft-2cpu"
    artifact = artifact_for(config, facts, "pull_request", source, (), target_job=target)
    assert artifact.status == "ready"
    assert artifact.plan is not None
    assert [job.id for job in artifact.plan.jobs] == [target]
    assert artifact.plan.inputs["target_job"] == target
    assert artifact.plan.coverage["integration"]["base_suite"] == {
        "mode": "explicit",
        "cases": ["qwen3_5_moe/sft_2cpu"],
    }
    assert validate_plan(artifact.plan, config, facts) == []


def test_unknown_target_job_is_rejected(
    config: CiConfig, facts: TestFacts, source
) -> None:
    artifact = artifact_for(config, facts, "pull_request", source, (), target_job="missing")
    assert artifact.status == "failed"
    assert [issue.code for issue in artifact.errors] == ["unknown-job"]


def test_explicit_unknown_integration_case_is_rejected(
    config: CiConfig, facts: TestFacts, source
) -> None:
    scope = config.scope("pull_request")
    broken_scope = replace(
        scope,
        integration={
            **scope.integration,
            "base_suite": ScopeSuite(mode="explicit", cases=("missing/case",)),
        },
    )
    broken = replace(config, scopes={**config.scopes, "pull_request": broken_scope})
    with pytest.raises(PlanError, match="unknown integration case"):
        build(broken, facts, "pull_request", source)


def test_planner_failure_artifact_round_trip(
    tmp_path: Path, config: CiConfig, source
) -> None:
    artifact = artifact_for(
        config,
        None,
        "pull_request",
        source,
        (Issue(code="missing-facts", path="tests", message="facts missing"),),
    )
    assert artifact.status == "failed"
    assert artifact.plan is None

    path = tmp_path / "plan-artifact.json"
    write_json(path, artifact.to_dict())
    loaded = load_plan_artifact(path)
    assert loaded.to_dict() == artifact.to_dict()


def test_ready_artifact_round_trip(
    tmp_path: Path, config: CiConfig, facts: TestFacts, source
) -> None:
    artifact = artifact_for(config, facts, "pull_request", source, ())
    assert artifact.status == "ready"
    assert artifact.plan is not None

    path = tmp_path / "plan-artifact.json"
    write_json(path, artifact.to_dict())
    loaded = load_plan_artifact(path)
    assert loaded.status == "ready"
    assert loaded.plan is not None
    assert loaded.plan.to_dict() == artifact.plan.to_dict()


def test_missing_job_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: Plan
) -> None:
    missing = "pytest-core-npu-npu-default-default"
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(job for job in pr_plan.jobs if job.id != missing),
    ))
    codes = {issue.code for issue in validate_plan(broken, config, facts)}
    assert "plan-mismatch" in codes


def test_tampered_integration_spec_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: Plan
) -> None:
    job = next(job for job in pr_plan.jobs if job.kind == "integration")
    spec = job.steps[0].spec
    assert isinstance(spec, IntegrationSpec)
    bad_spec = replace(spec, golden_digest="sha256:bad")
    bad_job = replace(job, steps=(replace(job.steps[0], spec=bad_spec),))
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(bad_job if item.id == job.id else item for item in pr_plan.jobs),
    ))
    codes = {issue.code for issue in validate_plan(broken, config, facts)}
    assert "plan-mismatch" in codes


def test_tampered_coverage_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: Plan
) -> None:
    coverage = {
        **pr_plan.coverage,
        "pytest": {
            **pr_plan.coverage["pytest"],
            "core": {"mode": "all", "files": 1, "routes": 1},
        },
    }
    broken = _with_plan_id(replace(pr_plan, coverage=coverage))
    codes = {issue.code for issue in validate_plan(broken, config, facts)}
    assert "plan-mismatch" in codes


def test_cli_uses_scope_and_writes_matrix(
    tmp_path: Path, config: CiConfig
) -> None:
    output = tmp_path / "plan.json"
    matrix = tmp_path / "matrix.txt"
    assert main([
        "--root", str(config.root),
        "--scope", "pull_request",
        "--output", str(output),
        "--github-output", str(matrix),
    ]) == 0
    assert output.is_file()
    assert "cpu_job_count=2" in matrix.read_text(encoding="utf-8")
    assert "npu_job_count=2" in matrix.read_text(encoding="utf-8")
