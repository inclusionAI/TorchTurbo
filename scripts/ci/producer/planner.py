#!/usr/bin/env python3
"""Compile CiConfig and TestFacts into a self-contained CI PlanArtifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .ci_config import (
    CiConfig,
    CiConfigError,
    CiImage,
    CiLane,
    CiTarget,
    IntegrationSuite,
    Scope,
    load as load_config,
)
from .facts import (
    FactsError,
    IntegrationFact,
    PytestFact,
    TestFacts,
    load as load_facts,
)
from ..contracts.artifacts import (
    SCHEMA_VERSION,
    SUITES,
    CommandSpec,
    IntegrationSpec,
    Issue,
    Job,
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


class PlanError(ValueError):
    """Raised when selected work cannot be compiled into a valid plan."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass
class _PytestGroup:
    suite: str
    device: str
    target: CiTarget
    image: CiImage
    venv_name: str
    entries: list[PytestFact] = field(default_factory=list)
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


def source_for(
    root: Path, *, repository: str = "", ref: str = "", sha: str = ""
) -> Source:
    return Source(
        repository=repository or _git(root, "config", "--get", "remote.origin.url"),
        ref=ref or _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        sha=sha or _git(root, "rev-parse", "HEAD"),
    )


def _runtime(
    config: CiConfig,
    device: str,
    target: str,
    ranks: int,
    image: CiImage,
    venv_name: str,
    lane: CiLane,
) -> Runtime:
    return Runtime(
        device=device,
        target=target,
        ranks=ranks,
        lane=lane.name,
        image=image.as_ref(),
        venv=config.venvs[venv_name],
    )


def _build_pytest_jobs(
    config: CiConfig,
    facts: TestFacts,
    scope: Scope,
    issues: list[Issue],
) -> tuple[list[Job], dict[str, Any], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], _PytestGroup] = {}
    selected_files = {suite: set() for suite in SUITES}
    selected_routes = {suite: set() for suite in SUITES}
    unsupported: set[tuple[str, str]] = set()

    for fact in facts.pytest:
        if scope.pytest.get(fact.suite) != "all":
            continue
        selected_files[fact.suite].add(fact.path)
        venv_name, venv_issues = config.resolve_pytest_venv(
            fact.path, fact.model_families, fact.frameworks
        )
        issues.extend(venv_issues)
        if venv_issues:
            continue
        if venv_name not in config.venvs:
            issues.append(_issue(
                fact.path,
                f"resolved venv {venv_name!r} is not defined",
                "unknown-venv",
            ))
            continue
        override = config.pytest.overrides.get(fact.path)
        image_name = override.image if override is not None else None
        required_targets = override.targets if override is not None else ()
        if image_name in config.images and not required_targets:
            required_targets = (config.images[image_name].target,)
        invalid_targets = [
            name for name in required_targets
            if config.targets[name].device not in fact.devices
        ]
        if invalid_targets:
            issues.append(_issue(
                fact.path,
                f"target override {invalid_targets[0]!r} has no matching test device",
                "target-device",
            ))
            continue
        for device in fact.devices:
            ranks = fact.ranks_by_device.get(device, 1)
            if not config.device_is_configured(device):
                unsupported.add((fact.path, device))
                continue
            required_for_device = tuple(
                name for name in required_targets
                if config.targets[name].device == device
            )
            if required_targets and not required_for_device:
                continue
            try:
                targets = config.targets_for(
                    device,
                    scope.targets,
                    required_for_device,
                )
            except CiConfigError as exc:
                issues.extend(exc.issues)
                continue
            if not targets:
                continue
            for target in targets:
                try:
                    image = config.resolve_image(
                        target,
                        explicit=image_name,
                        scope_images=scope.images,
                        workload_images=config.pytest.images,
                    )
                    config.lane_for(target, ranks)
                except CiConfigError as exc:
                    issues.extend(exc.issues)
                    continue
                key = (fact.suite, target.name, image.name, venv_name)
                group = groups.setdefault(
                    key,
                    _PytestGroup(
                        suite=fact.suite,
                        device=device,
                        target=target,
                        image=image,
                        venv_name=venv_name,
                    ),
                )
                group.entries.append(fact)
                group.max_ranks = max(group.max_ranks, ranks)
                selected_routes[fact.suite].add((fact.path, target.name))

    jobs: list[Job] = []
    for key in sorted(groups):
        group = groups[key]
        job_timeout = config.pytest.job_timeout_minutes.get(group.device)
        step_timeout = config.pytest.step_timeout_minutes.get(group.device)
        if job_timeout is None or step_timeout is None:
            issues.append(_issue(
                f"workloads.pytest.defaults.{group.device}",
                "missing pytest timeout for selected device",
            ))
            continue
        job_id = _slug(
            f"pytest-{group.suite}-{group.device}-{group.image.name}-{group.venv_name}"
        )
        lane = config.lane_for(group.target, group.max_ranks)
        steps_by_key: dict[tuple[str, str], list[PytestFact]] = {}
        for entry in group.entries:
            steps_by_key.setdefault(
                (entry.process_group, _marker(group.device)), []
            ).append(entry)
        steps: list[Step] = []
        for (process_group, marker), entries in sorted(steps_by_key.items()):
            files = tuple(sorted({entry.path for entry in entries}))
            steps.append(Step(
                id=f"{job_id}/{_slug(process_group)}-{_slug(marker)}",
                kind="pytest",
                ranks=max(
                    entry.ranks_by_device.get(group.device, 1) for entry in entries
                ),
                timeout_minutes=step_timeout,
                spec=PytestSpec(
                    suite=group.suite,
                    frameworks=tuple(sorted({
                        value for entry in entries for value in entry.frameworks
                    })),
                    model_families=tuple(sorted({
                        value for entry in entries for value in entry.model_families
                    })),
                    marker=marker,
                    process_group=process_group,
                    tags=tuple(sorted({
                        value for entry in entries for value in entry.tags
                    })),
                    files=files,
                ),
            ))
        jobs.append(Job(
            id=job_id,
            kind="pytest",
            allowed_skip_reasons=scope.allow_skip.get(group.device, ()),
            timeout_minutes=job_timeout,
            runtime=_runtime(
                config,
                group.device,
                group.target.name,
                group.max_ranks,
                group.image,
                group.venv_name,
                lane,
            ),
            steps=tuple(steps),
        ))

    coverage = {
        suite: {
            "mode": scope.pytest.get(suite, "none"),
            "files": len(selected_files[suite]),
            "routes": len(selected_routes[suite]),
        }
        for suite in SUITES
    }
    for suite in SUITES:
        if scope.pytest.get(suite) == "all" and not selected_files[suite]:
            issues.append(_issue(
                f"scopes.{scope.name}.pytest.{suite}",
                "selected suite contains no tests",
                "empty-suite",
            ))
    unsupported_routes = [
        {"path": path, "device": device} for path, device in sorted(unsupported)
    ]
    return jobs, coverage, unsupported_routes


def _integration_spec(suite: IntegrationSuite, case: IntegrationFact) -> IntegrationSpec:
    return IntegrationSpec(
        suite=case.suite,
        case_id=case.case_id,
        framework=case.framework,
        config_digest=case.config_digest,
        golden_digest=case.golden_digest,
        runner=CommandSpec(script=suite.runner),
    )


def _build_integration_jobs(
    config: CiConfig,
    facts: TestFacts,
    scope: Scope,
    issues: list[Issue],
) -> tuple[list[Job], dict[str, Any]]:
    jobs: list[Job] = []
    coverage: dict[str, Any] = {}
    for suite_name, suite in sorted(config.integration.suites.items()):
        selection = scope.integration.get(suite_name)
        mode = selection.mode if selection is not None else "none"
        available = facts.integration_for(suite_name)
        by_id = {case.case_id: case for case in available}
        selected: list[IntegrationFact]
        if mode == "none":
            selected = []
        elif mode == "all":
            selected = list(available)
            if not selected:
                issues.append(_issue(
                    f"scopes.{scope.name}.integration.{suite_name}",
                    "all mode selected no integration cases",
                    "empty-case-suite",
                ))
        else:
            selected = []
            for case_id in selection.cases if selection is not None else ():
                case = by_id.get(case_id)
                if case is None:
                    issues.append(_issue(
                        f"scopes.{scope.name}.integration.{suite_name}.cases",
                        f"unknown integration case {case_id!r}",
                        "unknown-case",
                    ))
                    continue
                selected.append(case)
        coverage[suite_name] = {
            "mode": mode,
            "cases": [case.case_id for case in selected],
        }
        for case in selected:
            if not case.compare:
                issues.append(_issue(
                    case.case_id,
                    "CI integration cases must compare metrics",
                    "integration-compare-disabled",
                ))
                continue
            venv_name = suite.framework_venvs.get(case.framework)
            if venv_name is None:
                issues.append(_issue(
                    f"workloads.integration.suites.{suite_name}.framework_venvs",
                    f"no venv for framework {case.framework!r}",
                    "missing-framework-venv",
                ))
                continue
            if venv_name not in config.venvs:
                issues.append(_issue(
                    f"workloads.integration.suites.{suite_name}.framework_venvs.{case.framework}",
                    f"unknown venv {venv_name!r}",
                ))
                continue
            if not config.device_is_configured(case.platform):
                issues.append(_issue(
                    case.case_id,
                    f"platform {case.platform!r} has no CI image or lane",
                    "missing-integration-resource",
                ))
                continue
            override = suite.overrides.get(case.case_id)
            explicit_image = override.image if override is not None else None
            required_targets = (
                override.targets if override is not None and override.targets
                else case.targets
            )
            if explicit_image in config.images and not required_targets:
                required_targets = (config.images[explicit_image].target,)
            try:
                targets = config.targets_for(
                    case.platform,
                    scope.targets,
                    required_targets,
                )
            except CiConfigError as exc:
                issues.extend(exc.issues)
                continue
            if not targets:
                continue
            try:
                job_timeout = suite.job_timeout_minutes_for(case.platform, case.case_id)
                step_timeout = suite.step_timeout_minutes_for(case.platform, case.case_id)
            except KeyError:
                issues.append(_issue(
                    f"workloads.integration.suites.{suite_name}.defaults.{case.platform}",
                    "missing integration timeout for selected platform",
                    "missing-timeout",
                ))
                continue
            for target in targets:
                try:
                    image = config.resolve_image(
                        target,
                        explicit=explicit_image,
                        scope_images=scope.images,
                        workload_images=suite.images,
                    )
                    lane = config.lane_for(target, case.ranks)
                except CiConfigError as exc:
                    issues.extend(exc.issues)
                    continue
                base_id = _slug(f"integration-{suite_name}-{case.case_id}")
                job_id = (
                    _slug(f"{base_id}-{target.name}")
                    if len(targets) > 1 else base_id
                )
                jobs.append(Job(
                    id=job_id,
                    kind="integration",
                    allowed_skip_reasons=scope.allow_skip.get(case.platform, ()),
                    timeout_minutes=job_timeout,
                    runtime=_runtime(
                        config,
                        case.platform,
                        target.name,
                        case.ranks,
                        image,
                        venv_name,
                        lane,
                    ),
                    steps=(Step(
                        id=f"{job_id}/run",
                        kind="integration",
                        ranks=case.ranks,
                        timeout_minutes=step_timeout,
                        spec=_integration_spec(suite, case),
                    ),),
                ))
    return jobs, coverage


def _plan_inputs(config: CiConfig, facts: TestFacts, change_scope: str) -> dict[str, str]:
    return {
        "config_digest": config.digest,
        "facts_digest": facts.digest,
        "image_refs_digest": digest({
            name: image.ref for name, image in sorted(config.images.items())
        }),
        "change_scope": change_scope,
    }


def _empty_coverage(config: CiConfig) -> dict[str, Any]:
    return {
        "pytest": {
            suite: {"mode": "none", "files": 0, "routes": 0}
            for suite in SUITES
        },
        "integration": {
            suite: {"mode": "none", "cases": []}
            for suite in sorted(config.integration.suites)
        },
        "unsupported_routes": [],
        "jobs": 0,
    }


def build(
    config: CiConfig,
    facts: TestFacts,
    scope_name: str,
    source: Source,
    *,
    change_scope: str = "code",
) -> Plan:
    """Build a ready plan or raise PlanError with structured issues."""
    if change_scope not in {"code", "docs_only"}:
        raise ValueError(f"unknown change scope {change_scope!r}")
    try:
        scope = config.scope(scope_name)
    except CiConfigError as exc:
        raise PlanError(exc.issues) from exc
    if change_scope == "docs_only":
        return build_plan(
            scope=scope.name,
            source=source,
            inputs=_plan_inputs(config, facts, change_scope),
            coverage=_empty_coverage(config),
            jobs=(),
        )

    issues: list[Issue] = []
    pytest_jobs, pytest_coverage, unsupported_routes = _build_pytest_jobs(
        config, facts, scope, issues
    )
    integration_jobs, integration_coverage = _build_integration_jobs(
        config, facts, scope, issues
    )
    jobs = sorted((*pytest_jobs, *integration_jobs), key=lambda job: job.id)
    if len({job.id for job in jobs}) != len(jobs):
        issues.append(_issue("plan.jobs", "job ids are not unique", "duplicate-job"))
    if issues:
        raise PlanError(issues)
    return build_plan(
        scope=scope.name,
        source=source,
        inputs=_plan_inputs(config, facts, change_scope),
        coverage={
            "pytest": pytest_coverage,
            "integration": integration_coverage,
            "unsupported_routes": unsupported_routes,
            "jobs": len(jobs),
        },
        jobs=jobs,
    )


def _target_plan(plan: Plan, target_jobs: str) -> Plan:
    """Reduce a validated Plan to explicit jobs for one manual run."""
    targets = tuple(dict.fromkeys(
        part for part in re.split(r"[,\s]+", target_jobs.strip()) if part
    ))
    if not targets:
        raise PlanError([_issue(
            "target_job",
            "target job list is empty",
            "unknown-job",
        )])
    known = {job.id for job in plan.jobs}
    unknown = [target for target in targets if target not in known]
    if unknown:
        raise PlanError([
            _issue(
                target,
                f"target job {target!r} is not in the full plan",
                "unknown-job",
            )
            for target in unknown
        ])
    selected = set(targets)
    jobs = tuple(job for job in plan.jobs if job.id in selected)
    pytest_files = {suite: set() for suite in plan.coverage.get("pytest", {})}
    pytest_routes = {suite: 0 for suite in pytest_files}
    integration_cases = {
        suite: [] for suite in plan.coverage.get("integration", {})
    }
    for job in jobs:
        for step in job.steps:
            if isinstance(step.spec, PytestSpec):
                pytest_files[step.spec.suite].update(step.spec.files)
                pytest_routes[step.spec.suite] += len(step.spec.files)
            elif isinstance(step.spec, IntegrationSpec):
                integration_cases.setdefault(step.spec.suite, []).append(step.spec.case_id)
    return build_plan(
        scope=plan.scope,
        source=plan.source,
        inputs={**plan.inputs, "target_job": ",".join(targets)},
        coverage={
            "pytest": {
                suite: {
                    "mode": "explicit" if files else "none",
                    "files": len(files),
                    "routes": pytest_routes[suite],
                }
                for suite, files in pytest_files.items()
            },
            "integration": {
                suite: {
                    "mode": "explicit" if cases else "none",
                    "cases": sorted(cases),
                }
                for suite, cases in integration_cases.items()
            },
            "unsupported_routes": [],
            "jobs": len(jobs),
        },
        jobs=jobs,
    )


def validate_plan(plan: Plan, config: CiConfig, facts: TestFacts) -> list[Issue]:
    """Check a plan by rebuilding the Plan its inputs must produce."""
    issues: list[Issue] = []
    if plan.schema_version != SCHEMA_VERSION:
        issues.append(_issue("plan.schema_version", f"expected {SCHEMA_VERSION}"))
    expected_id = plan_digest(plan)
    if plan.plan_id != expected_id:
        issues.append(_issue(
            "plan.plan_id",
            f"does not match plan contents; expected {expected_id}",
        ))
    scope = plan.inputs.get("change_scope", "code")
    try:
        expected = build(config, facts, plan.scope, plan.source, change_scope=scope)
        if target := plan.inputs.get("target_job"):
            expected = _target_plan(expected, target)
    except PlanError as exc:
        return issues + list(exc.issues)
    if plan.to_dict() != expected.to_dict():
        issues.append(_issue(
            "plan",
            "does not match CiConfig and TestFacts",
            "plan-mismatch",
        ))
    return issues


def artifact_for(
    config: CiConfig | None,
    facts: TestFacts | None,
    scope_name: str,
    source: Source,
    issues: Iterable[Issue],
    target_job: str = "",
    change_scope: str = "code",
) -> PlanArtifact:
    values = tuple(issues)
    if not values and (config is None or facts is None):
        values = (_issue(
            "planner",
            "CiConfig and TestFacts are required",
            "missing-input",
        ),)
    if config is None or facts is None or values:
        return build_plan_artifact(scope=scope_name, source=source, errors=values)
    try:
        plan = build(config, facts, scope_name, source, change_scope=change_scope)
        if target_job:
            plan = _target_plan(plan, target_job)
    except PlanError as exc:
        return build_plan_artifact(scope=scope_name, source=source, errors=exc.issues)
    return build_plan_artifact(scope=scope_name, source=source, plan=plan)


def matrix_outputs(plan: Plan, config: CiConfig) -> dict[str, list[dict[str, Any]]]:
    """Build platform matrix entries without embedding full job payloads."""
    output = {f"{device}_jobs": [] for device in ("cpu", "npu", "gpu")}
    for job in plan.jobs:
        lane = config.lanes[job.runtime.lane]
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
    parser.add_argument("--scope", default="pull_request")
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
    parser.add_argument(
        "--target-job",
        default="",
        help="exact Plan job id(s), separated by commas or whitespace",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify and args.target_job:
        parser.error("--target-job cannot be combined with --verify")
    if args.target_job and args.change_scope == "docs_only":
        parser.error("--target-job cannot be combined with a docs-only plan")

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    source = source_for(
        root,
        repository=args.source_repository,
        ref=args.source_ref,
        sha=args.source_sha,
    )
    try:
        image_refs = _image_refs(args.image_ref)
    except ValueError as exc:
        parser.error(str(exc))

    if args.fail_message:
        artifact = artifact_for(
            None,
            None,
            args.scope,
            source,
            (Issue(code=args.fail_code, path=args.fail_path, message=args.fail_message),),
        )
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(artifact.errors)
        return 1

    try:
        config = load_config(root, image_refs=image_refs)
        facts = load_facts(root, config)
    except (CiConfigError, FactsError) as exc:
        artifact = artifact_for(None, None, args.scope, source, exc.issues)
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(exc.issues)
        return 1

    if args.verify:
        failures: list[Issue] = []
        for name in sorted(config.scopes):
            try:
                plan = build(config, facts, name, source)
                docs = build(config, facts, name, source, change_scope="docs_only")
                failures.extend(validate_plan(plan, config, facts))
                failures.extend(validate_plan(docs, config, facts))
                print(f"{name}: {len(plan.jobs)} jobs / {len(facts.pytest)} tests")
                print(f"{name} docs-only: {len(docs.jobs)} jobs")
            except PlanError as exc:
                failures.extend(exc.issues)
        if failures:
            _render(failures)
            return 1
        print(
            f"planner OK ({len(facts.integration)} integration cases / "
            f"{len(facts.pytest)} tests)"
        )
        return 0

    artifact = artifact_for(
        config,
        facts,
        args.scope,
        source,
        (),
        target_job=args.target_job,
        change_scope=args.change_scope,
    )
    if args.output:
        write_json(Path(args.output), artifact.to_dict())
    if artifact.status != "ready" or artifact.plan is None:
        _render(artifact.errors)
        return 1
    if args.github_output:
        output_path = Path(args.github_output)
        for key, value in matrix_outputs(artifact.plan, config).items():
            _write_github_output(
                output_path,
                key,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            )
            _write_github_output(output_path, f"{key[:-1]}_count", str(len(value)))
    if not args.output:
        print(canonical_json(artifact.plan.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
