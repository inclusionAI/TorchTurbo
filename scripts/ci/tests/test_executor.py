from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.ci.consumer import executor
from scripts.ci.contracts.artifacts import (
    ModelE2ESpec,
    PytestSpec,
    build_plan,
    build_plan_artifact,
    load_job_result,
    write_json,
)
from scripts.ci.consumer.runtime.venv import Venv


def _fake_venv(tmp_path: Path, job) -> Venv:
    path = tmp_path / "venv"
    return Venv(path=path, python=path / "bin" / "python", spec=job.runtime.venv)


def _patch_execution(monkeypatch, tmp_path: Path, job) -> list[tuple]:
    class FakeManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ensure(self, env_name: str) -> Venv:
            return _fake_venv(tmp_path, job)

    monkeypatch.setattr(executor, "VenvManager", FakeManager)
    monkeypatch.setattr(executor, "_write_versions", lambda *args: None)
    monkeypatch.setattr(executor, "_probe_environment", lambda *args, **kwargs: (0, False))
    cleanup_calls: list[tuple] = []
    monkeypatch.setattr(
        executor,
        "_cleanup_environment",
        lambda *args: cleanup_calls.append(args),
    )
    return cleanup_calls


def _write_junit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="1"></testsuite>\n',
        encoding="utf-8",
    )


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


def test_execute_pytest_job_writes_passing_result(
    tmp_path: Path, monkeypatch, pr_plan
) -> None:
    job = next(item for item in pr_plan.jobs if item.id == "pytest-core-cpu-cpu-default-default")

    _patch_execution(monkeypatch, tmp_path, job)

    calls = 0

    def fake_run(command, *, cwd, environment, log_path, timeout_seconds):
        nonlocal calls
        calls += 1
        junit = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        _write_junit(Path(junit))
        log_path.write_text("pytest output\n", encoding="utf-8")
        return 0, False

    monkeypatch.setattr(executor, "_run_stream", fake_run)
    output_dir = tmp_path / "result"
    result = executor.execute(pr_plan, job.id, output_dir=output_dir, root=tmp_path)

    assert result.status == "passed"
    assert result.job_id == job.id
    assert calls == len(job.steps)
    assert len(result.steps) == len(job.steps)
    assert all(step.status == "passed" for step in result.steps)
    assert result.items.tests_total == 2 * len(job.steps)
    assert result.items.executed_files == len({
        path for step in job.steps for path in step.spec.files
    })
    loaded = load_job_result(output_dir / "job-result.json")
    assert loaded.to_dict() == result.to_dict()


def test_execute_leaves_provisional_result_if_terminated(
    tmp_path: Path, monkeypatch, pr_plan
) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "pytest")
    _patch_execution(monkeypatch, tmp_path, job)
    output_dir = tmp_path / "result"
    observed: list[tuple[str, str]] = []

    def fake_run(command, *, cwd, environment, log_path, timeout_seconds):
        provisional = load_job_result(output_dir / "job-result.json")
        observed.append((provisional.status, provisional.failure["code"]))
        raise SystemExit(137)

    monkeypatch.setattr(executor, "_run_stream", fake_run)
    with pytest.raises(SystemExit, match="137"):
        executor.execute(pr_plan, job.id, output_dir=output_dir, root=tmp_path)

    assert observed == [("infra_error", "executor-incomplete")]
    result = load_job_result(output_dir / "job-result.json")
    assert result.status == "infra_error"
    assert result.failure["code"] == "executor-incomplete"
    assert result.steps == ()


def test_execute_stops_after_first_failed_step(tmp_path: Path, monkeypatch, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "pytest")

    _patch_execution(monkeypatch, tmp_path, job)

    calls = 0

    def fake_run(command, *, cwd, environment, log_path, timeout_seconds):
        nonlocal calls
        calls += 1
        return 1, False

    monkeypatch.setattr(executor, "_run_stream", fake_run)
    result = executor.execute(pr_plan, job.id, output_dir=tmp_path / "result", root=tmp_path)

    assert calls == 1
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure["code"] == "step-failed"
    assert len(result.steps) == 1
    assert result.steps[0].status == "failed"


def test_execute_records_timeout(tmp_path: Path, monkeypatch, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "pytest")

    _patch_execution(monkeypatch, tmp_path, job)
    monkeypatch.setattr(
        executor,
        "_run_stream",
        lambda *args, **kwargs: (124, True),
    )
    result = executor.execute(pr_plan, job.id, output_dir=tmp_path / "result", root=tmp_path)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure["code"] == "step-timeout"
    assert result.steps[0].exit_code == 124


def test_environment_failure_still_writes_result(tmp_path: Path, monkeypatch, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "pytest")

    class FakeManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ensure(self, env_name: str) -> Venv:
            raise RuntimeError("venv install failed")

    monkeypatch.setattr(executor, "VenvManager", FakeManager)
    result = executor.execute(pr_plan, job.id, output_dir=tmp_path / "result", root=tmp_path)

    assert result.status == "infra_error"
    assert result.failure is not None
    assert result.failure["code"] == "environment-setup"
    assert load_job_result(tmp_path / "result" / "job-result.json").status == "infra_error"


def test_record_skip_enforces_allowed_reason(tmp_path: Path, pr_plan) -> None:
    npu = next(
        item for item in pr_plan.jobs
        if item.kind == "pytest" and item.runtime.device == "npu"
    )
    plan = _plan_with_allowed_skip(pr_plan, npu.id, "maintenance")
    result = executor.record_skip(
        plan,
        npu.id,
        "maintenance",
        output_dir=tmp_path / "skip",
    )
    assert result.status == "skipped"
    assert result.skip_reason == "maintenance"
    assert result.steps == ()

    with pytest.raises(executor.ExecutorError, match="does not allow"):
        executor.record_skip(
            plan,
            npu.id,
            "free-form-reason",
            output_dir=tmp_path / "skip",
        )


def test_e2e_environment_comes_from_plan(tmp_path: Path, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "model_e2e")
    spec = job.steps[0].spec
    assert isinstance(spec, ModelE2ESpec)
    venv = _fake_venv(tmp_path, job)
    environment = executor._e2e_environment(
        venv, spec, job, output_dir=tmp_path / "result"
    )
    assert environment["CASE_ID"] == spec.case_id
    assert environment["MODEL_TYPE"] == spec.model
    assert environment["PLAN_CARDS"] == str(spec.ranks)
    assert environment["PLAN_SP"] == str(spec.parallelism_sp)
    assert environment["PLAN_EP"] == str(spec.parallelism_ep)
    assert environment["INSTALL_SCRIPT"] == spec.install.script
    assert environment["TRAIN_SCRIPT"] == spec.train.script
    assert environment["TORCH_BEFORE"] == str(tmp_path / "result" / "torch-before.json")


def test_npu_job_cleans_up_environment(tmp_path: Path, monkeypatch, pr_plan) -> None:
    job = next(
        item for item in pr_plan.jobs
        if item.runtime.device == "npu" and item.kind == "pytest"
    )
    cleanup_calls = _patch_execution(monkeypatch, tmp_path, job)

    def fake_run(command, *, cwd, environment, log_path, timeout_seconds):
        junit = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        _write_junit(Path(junit))
        return 0, False

    monkeypatch.setattr(executor, "_run_stream", fake_run)
    result = executor.execute(pr_plan, job.id, output_dir=tmp_path / "result", root=tmp_path)

    assert result.status == "passed"
    assert len(cleanup_calls) == 1


def test_timeout_budget_caps_at_job_deadline(monkeypatch) -> None:
    monkeypatch.setattr(executor.time, "monotonic", lambda: 100.0)
    assert executor._timeout_budget(110.0, 30) == (10, True)
    assert executor._timeout_budget(90.0, 30) == (0, True)
    assert executor._timeout_budget(200.0, 30) == (30, False)


def test_merge_junit_ignores_truncated_xml(tmp_path: Path) -> None:
    good = tmp_path / "good.xml"
    _write_junit(good)
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuite", encoding="utf-8")
    output = tmp_path / "merged.xml"

    executor._merge_junit([bad, good], output)

    assert "<testsuite" in output.read_text(encoding="utf-8")


def test_pytest_command_uses_only_plan_evidence(tmp_path: Path, pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.kind == "pytest")
    spec = job.steps[0].spec
    assert isinstance(spec, PytestSpec)
    command = executor._pytest_command(_fake_venv(tmp_path, job), spec, tmp_path / "junit.xml")
    assert command[-len(spec.files):] == list(spec.files)
    assert "-m" in command
    assert spec.marker in command


def test_cli_record_skip_writes_result_file(tmp_path: Path, pr_plan) -> None:
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "artifacts" / "job-result.json"
    summary_path = tmp_path / "summary.md"
    job = next(
        item for item in pr_plan.jobs
        if item.runtime.device == "npu" and item.kind == "pytest"
    )
    plan = _plan_with_allowed_skip(pr_plan, job.id, "maintenance")
    write_json(plan_path, build_plan_artifact(
        policy=plan.policy, source=plan.source, plan=plan
    ).to_dict())

    assert executor.main([
        "--plan", str(plan_path),
        "--job-id", job.id,
        "--record-skip", "maintenance",
        "--skip-message", "NPU maintenance window",
        "--output", str(output_path),
        "--summary", str(summary_path),
    ]) == 0

    result = load_job_result(output_path)
    assert result.status == "skipped"
    assert result.skip_reason == "maintenance"
    assert result.skip_message == "NPU maintenance window"
    assert (output_path.parent / "job-result.json").is_file()
    assert "## CI Job: skipped" in summary_path.read_text(encoding="utf-8")
