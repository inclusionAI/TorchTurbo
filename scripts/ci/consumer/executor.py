#!/usr/bin/env python3
"""Execute one planned CI job and write a self-contained JobResult."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts.artifacts import (
    ContractError,
    Job,
    JobResult,
    ModelE2EEvidence,
    ModelE2ESpec,
    Plan,
    PytestEvidence,
    PytestSpec,
    Step,
    StepResult,
    JOB_RESULT_FILENAME,
    canonical_json,
    digest,
    load_plan_artifact,
    write_json,
)
from .runtime.venv import Venv, VenvManager


class ExecutorError(ValueError):
    """Raised when a requested job cannot be executed."""


PROBE_TIMEOUT_SECONDS = 300
CLEANUP_TIMEOUT_SECONDS = 60


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_ready_plan(path: Path) -> Plan:
    try:
        artifact = load_plan_artifact(path)
    except ContractError as exc:
        raise ExecutorError(str(exc)) from exc
    if artifact.status != "ready" or artifact.plan is None:
        details = "; ".join(issue.message for issue in artifact.errors)
        raise ExecutorError(f"planner artifact is not ready: {details}")
    return artifact.plan


def _job(plan: Plan, job_id: str) -> Job:
    for job in plan.jobs:
        if job.id == job_id:
            return job
    raise ExecutorError(f"job {job_id!r} is not in plan {plan.plan_id}")


def _terminate_process_group(process: subprocess.Popen[str], sig: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _run_stream(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    print("+ " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = threading.Event()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def terminate() -> None:
            timed_out.set()
            _terminate_process_group(process, signal.SIGTERM)

            def kill() -> None:
                _terminate_process_group(process, signal.SIGKILL)

            timer = threading.Timer(10, kill)
            timer.daemon = True
            timer.start()

        timer = threading.Timer(timeout_seconds, terminate)
        timer.daemon = True
        timer.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = process.wait()
        except KeyboardInterrupt:
            _terminate_process_group(process, signal.SIGTERM)
            process.wait()
            raise
        finally:
            timer.cancel()
    if timed_out.is_set():
        return 124, True
    return rc, False


def _probe_environment(
    venv: Venv,
    job: Job,
    *,
    root: Path,
    output_dir: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    return _run_stream(
        [
            str(venv.python),
            str(root / "scripts/ci/consumer/runtime/env.py"),
            "probe",
            "--device",
            job.runtime.device,
            "--file",
            str(output_dir / "torch-before.json"),
        ],
        cwd=root,
        environment=venv.environment(),
        log_path=log_path,
        timeout_seconds=timeout_seconds,
    )


def _cleanup_environment(venv: Venv, root: Path, log_path: Path) -> None:
    """Best-effort cleanup after an NPU job, including failed or timed out runs."""
    try:
        result = subprocess.run(
            [
                str(venv.python),
                str(root / "scripts/ci/consumer/runtime/env.py"),
                "cleanup",
            ],
            cwd=root,
            env=venv.environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=CLEANUP_TIMEOUT_SECONDS,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(result.stdout)
            log.write(result.stderr)
        if result.returncode != 0:
            print(
                f"::warning::environment cleanup exited with {result.returncode}",
                file=sys.stderr,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"::warning::environment cleanup failed: {exc}", file=sys.stderr)


def _prepare_environment(
    job: Job,
    *,
    root: Path,
    output_dir: Path,
    venv_root: Path,
    log_path: Path,
    deadline: float,
) -> tuple[Venv | None, str, dict[str, Any] | None]:
    """Install the planned venv and probe the planned runtime."""
    try:
        specs = {job.runtime.venv.name: job.runtime.venv}
        manager = VenvManager(root, venv_root, job.runtime.device, log_path, specs)
        venv = manager.ensure(job.runtime.venv.name)
        probe_limit, probe_limited = _timeout_budget(deadline, PROBE_TIMEOUT_SECONDS)
        if probe_limit == 0:
            return venv, "failed", _job_timeout_failure(job)
        probe_rc, probe_timeout = _probe_environment(
            venv, job, root=root, output_dir=output_dir,
            log_path=log_path, timeout_seconds=probe_limit,
        )
        if probe_rc == 0:
            return venv, "passed", None
        if probe_timeout and probe_limited:
            return venv, "failed", _job_timeout_failure(job)
        return venv, "infra_error", {
            "code": "environment-probe-timeout" if probe_timeout else "environment-probe",
            "message": f"environment probe failed with rc={probe_rc}",
        }
    except Exception as exc:  # noqa: BLE001 - every setup failure must produce a result
        return None, "infra_error", {
            "code": "environment-setup",
            "message": f"{type(exc).__name__}: {exc}",
        }


def _timeout_budget(deadline: float, requested_seconds: int) -> tuple[int, bool]:
    """Return the timeout allowed by both the step and the remaining job budget."""
    remaining = max(0, int(deadline - time.monotonic()))
    return min(requested_seconds, remaining), remaining < requested_seconds


def _pytest_command(venv: Venv, spec: PytestSpec, junit: Path) -> list[str]:
    args = shlex.split(os.environ.get("PYTEST_ARGS", "-v --durations=20"))
    return [
        str(venv.python),
        "-m",
        "pytest",
        *args,
        "-m",
        spec.marker,
        "--continue-on-collection-errors",
        f"--junitxml={junit}",
        *spec.files,
    ]


def _step_command(
    venv: Venv,
    job: Job,
    step: Step,
    *,
    index: int,
    root: Path,
    output_dir: Path,
) -> tuple[list[str], dict[str, str], Path | None]:
    if isinstance(step.spec, PytestSpec):
        junit = output_dir / "junit" / f"step-{index:02d}.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        return _pytest_command(venv, step.spec, junit), venv.environment(), junit
    if isinstance(step.spec, ModelE2ESpec):
        script = root / "scripts/ci/consumer/runtime/model_e2e/run_train.sh"
        environment = _e2e_environment(venv, step.spec, job, output_dir=output_dir)
        return ["bash", str(script), "case"], environment, None
    raise ExecutorError(f"unsupported step spec for {step.id}")


def _step_result(
    step: Step,
    status: str,
    exit_code: int,
    case_ids: tuple[str, ...] = (),
) -> StepResult:
    return StepResult(
        id=step.id,
        status=status,
        exit_code=exit_code,
        spec_digest=digest(step.spec.to_dict()),
        case_ids=case_ids,
    )


def _step_failure(
    job: Job,
    step: Step,
    *,
    exit_code: int,
    timed_out: bool,
    limit: int,
    deadline: float,
) -> dict[str, Any]:
    if timed_out and time.monotonic() >= deadline:
        return _job_timeout_failure(job, step.id)
    code = "step-timeout" if timed_out else "step-failed"
    message = f"step timed out after {limit}s" if timed_out else f"step exited with code {exit_code}"
    return {"code": code, "step_id": step.id, "message": message}


def _execute_steps(
    venv: Venv,
    job: Job,
    *,
    root: Path,
    output_dir: Path,
    log_path: Path,
    deadline: float,
) -> tuple[str, dict[str, Any] | None, list[StepResult], list[Path], int]:
    status: str = "passed"
    failure: dict[str, Any] | None = None
    results: list[StepResult] = []
    junit_paths: list[Path] = []
    executed_files = 0
    for index, step in enumerate(job.steps):
        case_ids = (step.spec.case_id,) if isinstance(step.spec, ModelE2ESpec) else ()
        limit, _ = _timeout_budget(deadline, step.timeout_seconds)
        if limit == 0:
            results.append(_step_result(step, "failed", 124, case_ids))
            return "failed", _job_timeout_failure(job, step.id), results, junit_paths, executed_files
        step_started = time.monotonic()
        try:
            command, environment, junit = _step_command(
                venv, job, step, index=index, root=root, output_dir=output_dir
            )
            if junit is not None:
                junit_paths.append(junit)
                executed_files += len(step.spec.files)
            exit_code, timed_out = _run_stream(
                command,
                cwd=root,
                environment=environment,
                log_path=log_path,
                timeout_seconds=limit,
            )
        except KeyboardInterrupt:
            status, exit_code = "cancelled", 130
            failure = {"code": "cancelled", "step_id": step.id, "message": "cancelled"}
        except Exception as exc:  # noqa: BLE001 - report step infrastructure failure
            status, exit_code, timed_out = "infra_error", 2, False
            failure = {
                "code": "step-setup",
                "step_id": step.id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        step_status = status if status != "passed" else (
            "passed" if exit_code == 0 else "failed"
        )
        results.append(_step_result(step, step_status, exit_code, case_ids))
        if status == "passed" and exit_code != 0:
            status = "failed"
            failure = _step_failure(
                job, step, exit_code=exit_code, timed_out=timed_out,
                limit=limit, deadline=deadline,
            )
        if status != "passed":
            elapsed = int(time.monotonic() - step_started)
            print(f"step {step.id} finished in {elapsed}s with status={step_status}", flush=True)
            break
    return status, failure, results, junit_paths, executed_files


def _job_timeout_failure(job: Job, step_id: str = "") -> dict[str, Any]:
    failure = {
        "code": "job-timeout",
        "message": f"job budget exhausted after {job.timeout_minutes} minutes",
    }
    if step_id:
        failure["step_id"] = step_id
    return failure


def _e2e_environment(
    venv: Venv,
    spec: ModelE2ESpec,
    job: Job,
    *,
    output_dir: Path,
) -> dict[str, str]:
    environment = venv.environment()
    environment.update({
        "MODEL_TYPE": spec.model,
        "CASE_ID": spec.case_id,
        "FRAMEWORK": spec.framework,
        "ENV_NAME": job.runtime.venv.name,
        "RUN_DIR": spec.run_dir,
        "INSTALL_SCRIPT": spec.install.script,
        "INSTALL_ARGS": " ".join(spec.install.args),
        "TRAIN_SCRIPT": spec.train.script,
        "TRAIN_ARGS": " ".join(spec.train.args),
        "HF_REPO": spec.hf_repo,
        "LAYERS": str(spec.layers),
        "DATASET_IN": spec.dataset,
        "SEQ_LEN": str(spec.seq_len),
        "FINETUNING_TYPE": spec.finetuning_type,
        "PLAN_CARDS": str(spec.ranks),
        "PLAN_SP": str(spec.parallelism_sp),
        "PLAN_EP": str(spec.parallelism_ep),
        "FETCH_TIMEOUT": str(spec.fetch_timeout_seconds),
        "CASE_TIMEOUT": str(spec.case_timeout_seconds),
        "TORCH_BEFORE": str(output_dir / "torch-before.json"),
        "RUNNER_LABEL": os.environ.get("RUNNER_LABEL", ""),
        "IMAGE": job.runtime.image.ref,
        "RESULT_JSON": str(output_dir / "e2e-facts.json"),
        "GITHUB_OUTPUT": os.environ.get("GITHUB_OUTPUT", os.devnull),
        "GITHUB_STEP_SUMMARY": os.environ.get(
            "GITHUB_STEP_SUMMARY", os.devnull
        ),
    })
    return environment


def _junit_suites(path: Path) -> list[ET.Element]:
    if not path.is_file():
        return []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    return [root] if root.tag == "testsuite" else root.findall("testsuite")


def _merge_junit(paths: list[Path], output: Path) -> None:
    merged = ET.Element("testsuites")
    for path in paths:
        merged.extend(_junit_suites(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(output, encoding="utf-8", xml_declaration=True)


def _junit_evidence(path: Path, executed_files: int) -> PytestEvidence:
    suites = _junit_suites(path)

    def count(key: str) -> int:
        return sum(int(suite.get(key, 0)) for suite in suites)

    return PytestEvidence(
        executed_files=executed_files,
        tests_total=count("tests"),
        tests_failed=count("failures"),
        tests_errors=count("errors"),
        tests_skipped=count("skipped"),
    )


def _write_versions(venv: Venv, output: Path, root: Path) -> None:
    code = (
        "import importlib.metadata as m\n"
        "for name in ('torch', 'transformers', 'peft', 'numpy', "
        "'llamafactory', 'swift'):\n"
        "    try: print(f'{name} = {m.version(name)}')\n"
        "    except m.PackageNotFoundError: pass\n"
    )
    result = subprocess.run(
        [str(venv.python), "-c", code],
        cwd=root,
        env=venv.environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout.strip() + "\n", encoding="utf-8")


def _evidence(job: Job, executed_files: int, merged_junit: Path) -> PytestEvidence | ModelE2EEvidence:
    if job.kind == "pytest":
        return _junit_evidence(merged_junit, executed_files)
    case_ids = tuple(
        step.spec.case_id for step in job.steps if isinstance(step.spec, ModelE2ESpec)
    )
    return ModelE2EEvidence(case_ids=case_ids)


def _write_result(result: JobResult, output_dir: Path) -> None:
    write_json(output_dir / JOB_RESULT_FILENAME, result.to_dict())


def _new_result(
    plan: Plan,
    job: Job,
    *,
    status: str,
    started_at: str,
    finished_at: str,
    duration_seconds: int,
    items: PytestEvidence | ModelE2EEvidence,
    **values: Any,
) -> JobResult:
    values.setdefault("steps", ())
    values.setdefault("artifacts", {})
    return JobResult(
        schema_version=1,
        plan_id=plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        runtime=job.runtime,
        items=items,
        **values,
    )


def _provisional_result(
    plan: Plan,
    job: Job,
    *,
    started_at: str,
    log_path: Path,
    merged_junit: Path,
) -> JobResult:
    """Write a parseable result before execution can be killed externally."""
    return _new_result(
        plan,
        job,
        status="infra_error",
        started_at=started_at,
        finished_at=started_at,
        duration_seconds=0,
        items=_evidence(job, 0, merged_junit),
        failure={
            "code": "executor-incomplete",
            "message": "executor did not finalize JobResult",
        },
        artifacts={"log": log_path.name} if log_path.is_file() else {},
    )


def execute(
    plan: Plan,
    job_id: str,
    *,
    output_dir: Path,
    root: Path | None = None,
) -> JobResult:
    """Execute one Job from a ready Plan without reading Catalog or Policy."""
    job = _job(plan, job_id)
    root = root or Path(__file__).resolve().parents[3]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "turbo-ci.log"
    merged_junit = output_dir / "pytest.xml"
    versions_path = output_dir / "versions.txt"
    venv_root = Path(os.environ.get(
        "CI_VENV_ROOT", Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "turbo-ci-venv"
    ))
    log_path.write_text("", encoding="utf-8")
    started_at = _now()
    started = time.monotonic()
    _write_result(
        _provisional_result(
            plan,
            job,
            started_at=started_at,
            log_path=log_path,
            merged_junit=merged_junit,
        ),
        output_dir,
    )
    deadline = started + job.timeout_minutes * 60
    step_results: list[StepResult] = []
    junit_paths: list[Path] = []
    executed_files = 0

    venv, status, failure = _prepare_environment(
        job,
        root=root,
        output_dir=output_dir,
        venv_root=venv_root,
        log_path=log_path,
        deadline=deadline,
    )

    if status == "passed" and venv is not None:
        status, failure, step_results, junit_paths, executed_files = _execute_steps(
            venv,
            job,
            root=root,
            output_dir=output_dir,
            log_path=log_path,
            deadline=deadline,
        )

    if venv is not None and job.runtime.device == "npu":
        _cleanup_environment(venv, root, log_path)

    _merge_junit(junit_paths, merged_junit)
    if venv is not None:
        try:
            _write_versions(venv, versions_path, root)
        except OSError as exc:
            print(f"::warning::could not write versions: {exc}", file=sys.stderr)

    artifacts = {"log": log_path.name}
    if job.kind == "pytest":
        artifacts["junit"] = merged_junit.name
    e2e_facts = output_dir / "e2e-facts.json"
    if job.kind == "model_e2e" and e2e_facts.is_file():
        artifacts["e2e"] = e2e_facts.name
    torch_before = output_dir / "torch-before.json"
    if torch_before.is_file():
        artifacts["torch-before"] = torch_before.name
    if versions_path.is_file():
        artifacts["versions"] = versions_path.name
    finished_at = _now()
    result = _new_result(
        plan,
        job,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=int(time.monotonic() - started),
        items=_evidence(job, executed_files, merged_junit),
        steps=tuple(step_results),
        failure=failure,
        artifacts=artifacts,
    )
    _write_result(result, output_dir)
    return result


def record_skip(
    plan: Plan,
    job_id: str,
    reason: str,
    *,
    output_dir: Path,
    message: str = "",
) -> JobResult:
    """Write a structured skipped result for a platform-level skip."""
    job = _job(plan, job_id)
    if reason not in job.allowed_skip_reasons:
        raise ExecutorError(f"job {job.id!r} does not allow skip reason {reason!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _now()
    result = _new_result(
        plan,
        job,
        status="skipped",
        started_at=timestamp,
        finished_at=timestamp,
        duration_seconds=0,
        items=_evidence(job, 0, output_dir / "pytest.xml"),
        skip_reason=reason,
        skip_message=message or reason,
    )
    _write_result(result, output_dir)
    return result


def render_summary(result: JobResult) -> str:
    lines = [
        f"## CI Job: {result.status}",
        "",
        f"- job: `{result.job_id}`",
        f"- plan: `{result.plan_id}`",
        f"- duration: {result.duration_seconds}s",
    ]
    if isinstance(result.items, PytestEvidence):
        lines.append(
            f"- pytest: {result.items.tests_total} tests, "
            f"{result.items.tests_failed} failed, {result.items.tests_errors} errors, "
            f"{result.items.tests_skipped} skipped"
        )
    elif isinstance(result.items, ModelE2EEvidence):
        lines.append(f"- E2E cases: {', '.join(result.items.case_ids)}")
    if result.failure:
        step = f" ({result.failure['step_id']})" if result.failure.get("step_id") else ""
        lines.append(
            f"- failure: `{result.failure.get('code', 'unknown')}`{step}: "
            f"{result.failure.get('message', '')}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Plan or ready PlanArtifact JSON")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", default="", help="JobResult JSON path")
    parser.add_argument("--root", default=None)
    parser.add_argument("--summary", default="")
    parser.add_argument(
        "--record-skip",
        default="",
        metavar="REASON",
        help="write a skipped JobResult instead of executing",
    )
    parser.add_argument("--skip-message", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    try:
        plan = _load_ready_plan(Path(args.plan))
        output_dir = (
            Path(args.output).resolve().parent
            if args.output
            else Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "ci-job"
        )
        if args.record_skip:
            result = record_skip(
                plan,
                args.job_id,
                args.record_skip,
                output_dir=output_dir,
                message=args.skip_message,
            )
        else:
            result = execute(plan, args.job_id, output_dir=output_dir, root=root)
    except (ContractError, ExecutorError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    if args.output:
        write_json(Path(args.output), result.to_dict())
    else:
        print(canonical_json(result.to_dict()))
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as output:
            output.write(render_summary(result))
    return 0 if result.status in {"passed", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
