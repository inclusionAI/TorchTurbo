#!/usr/bin/env python3
"""Derive test facts from pytest sources and integration suite manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from . import scan as test_scan
from .ci_config import CiConfig
from ..contracts.artifacts import (
    DEVICES,
    SUITES,
    Issue,
    canonical_json,
    digest,
    file_digest,
    write_json,
)

PYTEST_ROOT = Path("tests/unit_tests")
MANIFEST_SCHEMA_VERSION = 1


class FactsError(ValueError):
    """Raised when repository test facts are missing or inconsistent."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class PytestFact:
    path: str
    suite: str
    frameworks: tuple[str, ...]
    model_families: tuple[str, ...]
    devices: tuple[str, ...]
    ranks_by_device: dict[str, int]
    tags: tuple[str, ...]
    process_group: str


@dataclass(frozen=True)
class IntegrationFact:
    suite: str
    case_id: str
    framework: str
    platform: str
    targets: tuple[str, ...]
    ranks: int
    compare: bool
    update_golden: bool
    config: str
    golden: str
    config_digest: str
    golden_digest: str


@dataclass(frozen=True)
class TestFacts:
    __test__ = False

    pytest: tuple[PytestFact, ...]
    integration: tuple[IntegrationFact, ...]
    registered_models: tuple[str, ...]
    digest: str

    def dump_payload(self) -> dict[str, Any]:
        return {
            "pytest": [asdict(item) for item in self.pytest],
            "integration": [asdict(item) for item in self.integration],
            "registered_models": list(self.registered_models),
        }

    def integration_for(self, suite: str) -> tuple[IntegrationFact, ...]:
        return tuple(item for item in self.integration if item.suite == suite)


def _issue(path: str, message: str, code: str = "invalid-test-facts") -> Issue:
    return Issue(code=code, path=path, message=message)


def classify_suite(path: str, frameworks: Iterable[str], families: Iterable[str]) -> str:
    """Classify a pytest file by responsibility, not by device or framework."""
    framework_set = set(frameworks)
    family_set = set(families)
    parts = test_scan.test_parts(path)
    if parts[:1] == ("adapters",) or framework_set:
        return "adapter"
    if parts[:1] == ("models",) and len(parts) >= 2:
        if len(parts) == 2 or parts[1] == "modules":
            return "core"
        if len(parts) >= 3 and parts[1] not in test_scan.MODEL_PACKAGES:
            return "model"
    if family_set:
        return "model"
    return "core"


def _pytest_facts(root: Path, models: set[str]) -> tuple[list[PytestFact], list[Issue]]:
    scanned, errors = test_scan.scan_tests(root, models)
    issues = [_issue(error.path, error.message) for error in errors]
    facts: list[PytestFact] = []
    for path, fact in sorted(scanned.items()):
        suite = classify_suite(path, fact.frameworks, fact.families)
        if suite not in SUITES:
            issues.append(_issue(path, f"unknown pytest suite {suite!r}"))
        facts.append(PytestFact(
            path=path,
            suite=suite,
            frameworks=tuple(sorted(fact.frameworks)),
            model_families=tuple(sorted(fact.families)),
            devices=tuple(sorted(fact.devices)),
            ranks_by_device={
                device: fact.processes_for(device) for device in sorted(fact.devices)
            },
            tags=("integration",) if fact.integration else (),
            process_group=fact.process,
        ))
    return facts, issues


def _manifest(root: Path, suite_name: str, runner: str) -> tuple[dict[str, Any] | None, Issue | None]:
    command = [
        sys.executable,
        str(root / runner),
        "--suite",
        suite_name,
        "--list-cases",
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return None, _issue(
            runner,
            f"integration suite {suite_name!r} manifest failed with rc="
            f"{result.returncode}: {detail}",
            "integration-manifest",
        )
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        return None, _issue(
            runner,
            f"integration suite {suite_name!r} manifest is not JSON: {exc}",
            "integration-manifest",
        )
    if not isinstance(payload, dict):
        return None, _issue(
            runner,
            f"integration suite {suite_name!r} manifest must be a mapping",
            "integration-manifest",
        )
    return payload, None


def _repo_file(root: Path, value: str, where: str, issues: list[Issue]) -> Path | None:
    if not value:
        issues.append(_issue(where, "path is required"))
        return None
    path = Path(value)
    if path.is_absolute():
        issues.append(_issue(where, "path must be repository-relative"))
        return None
    absolute = (root / path).resolve()
    try:
        absolute.relative_to(root.resolve())
    except ValueError:
        issues.append(_issue(where, "path escapes the repository"))
        return None
    if not absolute.is_file():
        issues.append(_issue(where, f"file is missing: {value}"))
        return None
    return absolute


def _integration_facts(
    root: Path,
    config: CiConfig,
) -> tuple[list[IntegrationFact], list[Issue]]:
    issues: list[Issue] = []
    facts: list[IntegrationFact] = []
    seen: set[str] = set()
    for suite_name, suite in sorted(config.integration.suites.items()):
        payload, error = _manifest(root, suite_name, suite.runner)
        if error is not None:
            issues.append(error)
            continue
        assert payload is not None
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            issues.append(_issue(
                suite.runner,
                f"suite {suite_name!r} manifest schema must be "
                f"{MANIFEST_SCHEMA_VERSION}",
                "integration-manifest",
            ))
        if payload.get("suite") != suite_name:
            issues.append(_issue(
                suite.runner,
                f"manifest suite is {payload.get('suite')!r}, expected {suite_name!r}",
                "integration-manifest",
            ))
        cases = payload.get("cases")
        if not isinstance(cases, list):
            issues.append(_issue(
                suite.runner,
                f"suite {suite_name!r} manifest cases must be a list",
                "integration-manifest",
            ))
            continue
        for index, item in enumerate(cases):
            where = f"{suite.runner}.cases[{index}]"
            if not isinstance(item, dict):
                issues.append(_issue(where, "case must be a mapping", "integration-manifest"))
                continue
            case_id = item.get("case_id")
            framework = item.get("framework")
            platform = item.get("platform")
            target = item.get("target")
            targets_value = item.get("targets")
            ranks = item.get("ranks")
            compare = item.get("compare")
            update_golden = item.get("update_golden")
            config_path = item.get("config")
            golden_path = item.get("golden")
            if not isinstance(case_id, str) or not case_id:
                issues.append(_issue(f"{where}.case_id", "case_id is required", "integration-manifest"))
                continue
            if case_id in seen:
                issues.append(_issue(f"{where}.case_id", f"duplicate case {case_id!r}", "integration-manifest"))
            seen.add(case_id)
            if not isinstance(framework, str) or not framework:
                issues.append(_issue(f"{where}.framework", "framework is required", "integration-manifest"))
                framework = ""
            elif framework not in suite.framework_venvs:
                issues.append(_issue(
                    f"{where}.framework",
                    f"framework {framework!r} has no venv mapping",
                    "unknown-integration-framework",
                ))
            if not isinstance(platform, str) or platform not in DEVICES:
                issues.append(_issue(f"{where}.platform", "unknown platform", "integration-manifest"))
                platform = ""
            if target is not None and targets_value is not None:
                issues.append(_issue(
                    f"{where}.targets",
                    "declare target or targets, not both",
                    "integration-manifest",
                ))
            if target is not None and (not isinstance(target, str) or not target):
                issues.append(_issue(
                    f"{where}.target",
                    "target must be a non-empty string",
                    "integration-manifest",
                ))
            if targets_value is not None and (
                not isinstance(targets_value, list)
                or not all(isinstance(value, str) and value for value in targets_value)
            ):
                issues.append(_issue(
                    f"{where}.targets",
                    "targets must be a list of non-empty strings",
                    "integration-manifest",
                ))
                targets_value = []
            targets = (
                (target,) if isinstance(target, str) and target
                else tuple(targets_value or ())
            )
            if len(set(targets)) != len(targets):
                issues.append(_issue(
                    f"{where}.targets",
                    "contains duplicates",
                    "integration-manifest",
                ))
            for target_name in targets:
                configured = config.targets.get(target_name)
                if configured is None:
                    issues.append(_issue(
                        f"{where}.targets",
                        f"unknown target {target_name!r}",
                        "integration-manifest",
                    ))
                elif platform and configured.device != platform:
                    issues.append(_issue(
                        f"{where}.targets",
                        f"target {target_name!r} is for {configured.device}, not {platform}",
                        "integration-manifest",
                    ))
            if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
                issues.append(_issue(f"{where}.ranks", "ranks must be an integer >= 1", "integration-manifest"))
                ranks = 1
            if not isinstance(compare, bool):
                issues.append(_issue(f"{where}.compare", "compare must be a boolean", "integration-manifest"))
                compare = False
            if not isinstance(update_golden, bool):
                issues.append(_issue(
                    f"{where}.update_golden",
                    "update_golden must be a boolean",
                    "integration-manifest",
                ))
                update_golden = False
            if update_golden:
                issues.append(_issue(
                    f"{where}.update_golden",
                    "CI must not update golden values",
                    "golden-update-forbidden",
                ))
            if not isinstance(config_path, str):
                issues.append(_issue(f"{where}.config", "config path is required", "integration-manifest"))
                config_path = ""
            if not isinstance(golden_path, str):
                issues.append(_issue(f"{where}.golden", "golden path is required", "integration-manifest"))
                golden_path = ""
            config_file = _repo_file(root, config_path, f"{where}.config", issues)
            golden_file = _repo_file(root, golden_path, f"{where}.golden", issues)
            if compare and golden_file is None:
                issues.append(_issue(
                    f"{where}.golden",
                    "a compared case needs an existing golden file",
                    "missing-golden",
                ))
            facts.append(IntegrationFact(
                suite=suite_name,
                case_id=case_id,
                framework=framework,
                platform=platform,
                targets=targets,
                ranks=ranks,
                compare=compare,
                update_golden=update_golden,
                config=config_path,
                golden=golden_path,
                config_digest=file_digest(config_file) if config_file is not None else "",
                golden_digest=file_digest(golden_file) if golden_file is not None else "",
            ))
    return facts, issues


def load(root: Path, config: CiConfig) -> TestFacts:
    """Scan pytest files and load every configured integration suite manifest."""
    models = test_scan.registered_models(root)
    pytest, pytest_issues = _pytest_facts(root, models)
    integration, integration_issues = _integration_facts(root, config)
    issues = [*pytest_issues, *integration_issues]
    if not pytest:
        issues.append(_issue(PYTEST_ROOT.as_posix(), "no pytest facts found", "empty-pytest"))
    if issues:
        raise FactsError(issues)
    facts = TestFacts(
        pytest=tuple(pytest),
        integration=tuple(sorted(integration, key=lambda item: (item.suite, item.case_id))),
        registered_models=tuple(sorted(models)),
        digest="",
    )
    return TestFacts(
        pytest=facts.pytest,
        integration=facts.integration,
        registered_models=facts.registered_models,
        digest=digest(facts.dump_payload()),
    )


def counts(facts: TestFacts) -> dict[str, int]:
    result = {suite: 0 for suite in SUITES}
    for fact in facts.pytest:
        result[fact.suite] += 1
    result["total"] = len(facts.pytest)
    result["integration"] = len(facts.integration)
    return result


def _render(issues: Iterable[Issue]) -> None:
    for issue in issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "dump"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--name", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    from .ci_config import CiConfigError, load as load_config

    try:
        config = load_config(root)
        facts = load(root, config)
    except (CiConfigError, FactsError) as exc:
        _render(exc.issues)
        return 1
    if args.command == "dump":
        payload = facts.dump_payload()
        if args.name:
            payload = payload[args.name]
        if args.output:
            write_json(Path(args.output), payload)
        else:
            print(canonical_json(payload))
        return 0
    value = counts(facts)
    print(
        f"test facts OK ({value['total']} pytest files; core={value['core']} "
        f"model={value['model']} adapter={value['adapter']}; "
        f"integration={value['integration']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
