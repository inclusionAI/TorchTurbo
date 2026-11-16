#!/usr/bin/env python3
"""Compile Catalog and Policy into a self-contained CI PlanArtifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .catalog import (
    Catalog,
    CatalogError,
    CatalogImage,
    CatalogLane,
    ModelE2ECase,
    TestEntry,
    load as load_catalog,
)
from .policy import Policy, PolicyError, load as load_policy, validate as validate_policy
from ..contracts.artifacts import (
    SCHEMA_VERSION,
    CommandSpec,
    Issue,
    Job,
    ModelE2ESpec,
    Plan,
    PlanArtifact,
    PytestSpec,
    Runtime,
    Source,
    Step,
    build_plan,
    build_plan_artifact,
    canonical_json,
    digest,
    plan_digest,
    write_json,
)

POLICIES = Path("scripts/ci/config/policies.yml")
SUITES = ("core", "model", "adapter")


class PlanError(ValueError):
    """Raised when selected work cannot be compiled into a valid plan."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass
class _PytestGroup:
    suite: str
    device: str
    image: CatalogImage
    venv_name: str
    entries: list[TestEntry] = field(default_factory=list)
    max_ranks: int = 1


def _issue(path: str, message: str, code: str = "invalid-plan") -> Issue:
    return Issue(code=code, path=path, message=message)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "job"


def _marker(device: str) -> str:
    return "cpu" if device == "cpu" else f"{device} or multi_{device}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_for(root: Path, *, repository: str = "", ref: str = "", sha: str = "") -> Source:
    return Source(
        repository=repository or _git(root, "config", "--get", "remote.origin.url"),
        ref=ref or _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        sha=sha or _git(root, "rev-parse", "HEAD"),
    )


def _runtime(
    catalog: Catalog, device: str, ranks: int, image: CatalogImage,
    venv_name: str, lane: CatalogLane,
) -> Runtime:
    return Runtime(
        device=device,
        ranks=ranks,
        lane=lane.name,
        image=image.as_ref(),
        venv=catalog.venvs[venv_name],
    )


def _build_pytest_jobs(
    catalog: Catalog, policy: Policy, issues: list[Issue]
) -> tuple[list[Job], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], _PytestGroup] = {}
    selected_files = {suite: set() for suite in SUITES}
    selected_routes = {suite: set() for suite in SUITES}

    for entry in catalog.tests:
        if not policy.selects_suite(entry.suite):
            continue
        selected_files[entry.suite].add(entry.path)
        route_errors: list[Issue] = []
        route_count = 0
        for device in entry.devices:
            ranks = entry.ranks_by_device.get(device, 1)
            try:
                image = catalog.image_for(device, entry.image_name)
                catalog.lane_for(device, image.name, ranks)
            except CatalogError as exc:
                route_errors.extend(exc.issues)
                continue
            route_count += 1
            selected_routes[entry.suite].add((entry.path, device))
            key = (entry.suite, device, image.name, entry.venv_name)
            group = groups.setdefault(
                key,
                _PytestGroup(entry.suite, device, image, entry.venv_name),
            )
            group.entries.append(entry)
            group.max_ranks = max(group.max_ranks, ranks)
        if route_count == 0:
            issues.extend(route_errors or [_issue(entry.path, "selected test has no executable route")])

    jobs: list[Job] = []
    for key in sorted(groups):
        group = groups[key]
        job_id = _slug(f"pytest-{'-'.join(key)}")
        lane = catalog.lane_for(group.device, group.image.name, group.max_ranks)
        steps_by_key: dict[tuple[str, str], list[TestEntry]] = {}
        for entry in group.entries:
            steps_by_key.setdefault((entry.process_group, _marker(group.device)), []).append(entry)
        steps: list[Step] = []
        for (process_group, marker), entries in sorted(steps_by_key.items()):
            files = tuple(sorted({entry.path for entry in entries}))
            steps.append(Step(
                id=f"{job_id}/{_slug(process_group)}-{_slug(marker)}",
                kind="pytest",
                ranks=max(entry.ranks_by_device.get(group.device, 1) for entry in entries),
                timeout_seconds=catalog.pytest.step_timeout_seconds[group.device],
                spec=PytestSpec(
                    suite=group.suite,
                    frameworks=tuple(sorted({value for entry in entries for value in entry.frameworks})),
                    model_families=tuple(sorted({
                        value for entry in entries for value in entry.model_families
                    })),
                    marker=marker,
                    process_group=process_group,
                    tags=tuple(sorted({value for entry in entries for value in entry.tags})),
                    files=files,
                ),
            ))
        jobs.append(Job(
            id=job_id,
            kind="pytest",
            allowed_skip_reasons=policy.allow_skip.get(group.device, ()),
            timeout_minutes=catalog.pytest.job_timeout_minutes[group.device],
            runtime=_runtime(
                catalog, group.device, group.max_ranks, group.image, group.venv_name, lane
            ),
            steps=tuple(steps),
        ))

    coverage = {
        suite: {
            "mode": policy.pytest[suite],
            "files": len(selected_files[suite]),
            "routes": len(selected_routes[suite]),
        }
        for suite in SUITES
    }
    for suite in SUITES:
        if policy.pytest[suite] == "all" and not selected_files[suite]:
            issues.append(_issue(
                f"policies.{policy.name}.pytest.{suite}", "selected suite contains no tests"
            ))
    return jobs, coverage


def _model_spec(case: ModelE2ECase, defaults: Any) -> ModelE2ESpec:
    return ModelE2ESpec(
        case_id=case.case_id,
        model=case.model,
        framework=case.framework,
        run_dir=case.run_dir,
        workdir=case.workdir,
        install=CommandSpec(script=case.install_script, args=case.install_args),
        train=CommandSpec(script=case.train_script, args=case.train_args),
        hf_repo=case.hf_repo,
        layers=case.layers,
        dataset=case.dataset,
        ranks=case.ranks,
        parallelism_sp=case.sp,
        parallelism_ep=case.ep,
        seq_len=case.seq_len,
        finetuning_type=case.finetuning_type,
        fetch_timeout_seconds=defaults.fetch_timeout_seconds,
        case_timeout_seconds=case.timeout_seconds,
    )


def _build_model_e2e_jobs(
    catalog: Catalog, policy: Policy, issues: list[Issue]
) -> tuple[list[Job], dict[str, Any]]:
    selected = [case for case in catalog.cases if policy.selects_case(case.case_id)]
    jobs: list[Job] = []
    for case in selected:
        try:
            image = catalog.image_for(case.device, case.image_name)
            lane = catalog.lane_for(case.device, image.name, case.ranks)
        except CatalogError as exc:
            issues.extend(exc.issues)
            continue
        job_id = _slug(f"model-e2e-{case.case_id}")
        jobs.append(Job(
            id=job_id,
            kind="model_e2e",
            allowed_skip_reasons=policy.allow_skip.get(case.device, ()),
            timeout_minutes=catalog.model_e2e_defaults.job_timeout_minutes,
            runtime=_runtime(catalog, case.device, case.ranks, image, case.venv_name, lane),
            steps=(Step(
                id=f"{job_id}/run",
                kind="model_e2e",
                ranks=case.ranks,
                timeout_seconds=case.timeout_seconds,
                spec=_model_spec(case, catalog.model_e2e_defaults),
            ),),
        ))
    if policy.model_e2e_mode == "all" and not selected:
        issues.append(_issue(f"policies.{policy.name}.model_e2e", "all mode selected no E2E cases"))
    return jobs, {"mode": policy.model_e2e_mode, "cases": [case.case_id for case in selected]}


def _plan_inputs(catalog: Catalog, policy: Policy, change_scope: str) -> dict[str, str]:
    return {
        "catalog_digest": catalog.digest,
        "policy_digest": policy.digest,
        "image_refs_digest": digest({
            name: image.ref for name, image in sorted(catalog.images.items())
        }),
        "change_scope": change_scope,
    }


def _empty_coverage() -> dict[str, Any]:
    return {
        "pytest": {suite: {"mode": "none", "files": 0, "routes": 0} for suite in SUITES},
        "model_e2e": {"mode": "none", "cases": []},
        "jobs": 0,
    }


def build(
    catalog: Catalog, policy: Policy, source: Source, *, change_scope: str = "code"
) -> Plan:
    """Build a ready plan or raise PlanError with structured issues."""
    if change_scope not in {"code", "docs_only"}:
        raise ValueError(f"unknown change scope {change_scope!r}")
    issues = validate_policy(policy, catalog)
    if change_scope == "docs_only":
        if issues:
            raise PlanError(issues)
        return build_plan(
            policy=policy.name,
            source=source,
            inputs=_plan_inputs(catalog, policy, change_scope),
            coverage=_empty_coverage(),
            jobs=(),
        )

    pytest_jobs, pytest_coverage = _build_pytest_jobs(catalog, policy, issues)
    e2e_jobs, e2e_coverage = _build_model_e2e_jobs(catalog, policy, issues)
    jobs = sorted((*pytest_jobs, *e2e_jobs), key=lambda job: job.id)
    if issues:
        raise PlanError(issues)
    return build_plan(
        policy=policy.name,
        source=source,
        inputs=_plan_inputs(catalog, policy, change_scope),
        coverage={"pytest": pytest_coverage, "model_e2e": e2e_coverage, "jobs": len(jobs)},
        jobs=jobs,
    )


def _target_plan(plan: Plan, target_job: str) -> Plan:
    """Reduce a validated Plan to one job for an explicit manual run."""
    jobs = tuple(job for job in plan.jobs if job.id == target_job)
    if not jobs:
        raise PlanError([_issue(
            target_job, f"target job {target_job!r} is not in the full plan", "unknown-job"
        )])
    pytest_files = {suite: set() for suite in SUITES}
    pytest_routes = {suite: 0 for suite in SUITES}
    e2e_cases: list[str] = []
    for job in jobs:
        for step in job.steps:
            if isinstance(step.spec, PytestSpec):
                pytest_files[step.spec.suite].update(step.spec.files)
                pytest_routes[step.spec.suite] += len(step.spec.files)
            else:
                e2e_cases.append(step.spec.case_id)
    return build_plan(
        policy=plan.policy,
        source=plan.source,
        inputs={**plan.inputs, "target_job": target_job},
        coverage={
            "pytest": {
                suite: {
                    "mode": "explicit" if files else "none",
                    "files": len(files),
                    "routes": pytest_routes[suite],
                }
                for suite, files in pytest_files.items()
            },
            "model_e2e": {
                "mode": "explicit" if e2e_cases else "none",
                "cases": sorted(e2e_cases),
            },
            "jobs": len(jobs),
        },
        jobs=jobs,
    )


def validate_plan(plan: Plan, catalog: Catalog, policy: Policy) -> list[Issue]:
    """Check a plan by rebuilding the one Catalog + Policy must produce."""
    issues: list[Issue] = []
    if plan.schema_version != SCHEMA_VERSION:
        issues.append(_issue("plan.schema_version", f"expected {SCHEMA_VERSION}"))
    if plan.policy != policy.name:
        issues.append(_issue("plan.policy", f"expected {policy.name!r}"))
    expected_id = plan_digest(plan)
    if plan.plan_id != expected_id:
        issues.append(_issue("plan.plan_id", f"does not match plan contents; expected {expected_id}"))
    scope = plan.inputs.get("change_scope", "code")
    try:
        expected = build(catalog, policy, plan.source, change_scope=scope)
        if target := plan.inputs.get("target_job"):
            expected = _target_plan(expected, target)
    except PlanError as exc:
        return issues + list(exc.issues)
    if plan.to_dict() != expected.to_dict():
        issues.append(_issue("plan", "does not match Catalog and Policy", "plan-mismatch"))
    return issues


def artifact_for(
    catalog: Catalog | None,
    policy_name: str,
    source: Source,
    issues: Iterable[Issue],
    target_job: str = "",
    change_scope: str = "code",
) -> PlanArtifact:
    values = tuple(issues)
    if catalog is None or values:
        return build_plan_artifact(policy=policy_name, source=source, errors=values)
    try:
        policy = load_policy(catalog.root / POLICIES, policy_name)
        plan = build(catalog, policy, source, change_scope=change_scope)
        if target_job:
            plan = _target_plan(plan, target_job)
    except (PlanError, PolicyError) as exc:
        return build_plan_artifact(policy=policy_name, source=source, errors=exc.issues)
    return build_plan_artifact(policy=policy_name, source=source, plan=plan)


def matrix_outputs(plan: Plan, catalog: Catalog) -> dict[str, list[dict[str, Any]]]:
    """Build platform matrix entries without embedding full job payloads."""
    output = {f"{device}_jobs": [] for device in ("cpu", "npu", "gpu")}
    for job in plan.jobs:
        lane = catalog.lanes[job.runtime.lane]
        output[f"{job.runtime.device}_jobs"].append({
            "job_id": job.id,
            "image_ref": job.runtime.image.ref,
            "runner_variable": lane.runner_variable,
            "runner": lane.runner,
            "timeout_minutes": job.timeout_minutes,
        })
    return output


def _write_github_output(path: Path, key: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{key}={value}\n")


def _render(issues: Iterable[Issue]) -> None:
    for issue in issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)


def _image_refs(values: list[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for value in values:
        name, separator, ref = value.partition("=")
        if not separator or not name or not ref:
            raise ValueError(f"bad --image-ref {value!r}; expected NAME=REF")
        refs[name] = ref
    return refs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--policy", default="pull_request")
    parser.add_argument("--output", default="")
    parser.add_argument("--github-output", default="")
    parser.add_argument("--image-ref", action="append", default=[], metavar="NAME=REF")
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--fail-code", default="planner-precondition")
    parser.add_argument("--fail-path", default="planner")
    parser.add_argument("--fail-message", default="")
    parser.add_argument("--change-scope", choices=("code", "docs_only"), default="code")
    parser.add_argument("--target-job", default="")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify and args.target_job:
        parser.error("--target-job cannot be combined with --verify")
    if args.target_job and args.change_scope == "docs_only":
        parser.error("--target-job cannot be combined with a docs-only plan")

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    source = source_for(
        root, repository=args.source_repository, ref=args.source_ref, sha=args.source_sha
    )
    try:
        image_refs = _image_refs(args.image_ref)
    except ValueError as exc:
        parser.error(str(exc))

    if args.fail_message:
        artifact = artifact_for(None, args.policy, source, (Issue(
            code=args.fail_code, path=args.fail_path, message=args.fail_message
        ),))
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(artifact.errors)
        return 1

    try:
        catalog = load_catalog(root, image_refs=image_refs)
    except CatalogError as exc:
        artifact = artifact_for(None, args.policy, source, exc.issues)
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(exc.issues)
        return 1

    if args.verify:
        failures: list[Issue] = []
        for name in ("pull_request", "nightly"):
            try:
                policy = load_policy(root / POLICIES, name)
                plan = build(catalog, policy, source)
                docs = build(catalog, policy, source, change_scope="docs_only")
                failures.extend(validate_plan(plan, catalog, policy))
                failures.extend(validate_plan(docs, catalog, policy))
                print(f"{name}: {len(plan.jobs)} jobs / {len(catalog.tests)} tests")
                print(f"{name} docs-only: {len(docs.jobs)} jobs")
            except (PolicyError, PlanError) as exc:
                failures.extend(exc.issues)
        if failures:
            _render(failures)
            return 1
        print(f"planner OK ({len(catalog.cases)} E2E cases / {len(catalog.tests)} tests)")
        return 0

    artifact = artifact_for(
        catalog, args.policy, source, (), target_job=args.target_job,
        change_scope=args.change_scope,
    )
    if args.output:
        write_json(Path(args.output), artifact.to_dict())
    if artifact.status != "ready" or artifact.plan is None:
        _render(artifact.errors)
        return 1
    if args.github_output:
        output_path = Path(args.github_output)
        for key, value in matrix_outputs(artifact.plan, catalog).items():
            _write_github_output(
                output_path, key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
            _write_github_output(output_path, f"{key[:-1]}_count", str(len(value)))
    if not args.output:
        print(canonical_json(artifact.plan.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
