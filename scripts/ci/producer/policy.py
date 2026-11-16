#!/usr/bin/env python3
"""Load declarative pull-request and nightly CI policies."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .catalog import Catalog, CatalogError, load as load_catalog
from ..contracts.artifacts import DEVICES, SCHEMA_VERSION, SUITES, Issue, canonical_json, digest

POLICIES = Path("scripts/ci/config/policies.yml")


class PolicyError(ValueError):
    """Raised when a policy file is malformed or references unknown work."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class Policy:
    name: str
    pytest: dict[str, str]
    model_e2e_mode: str
    model_e2e_cases: tuple[str, ...]
    allow_skip: dict[str, tuple[str, ...]]
    digest: str

    def selects_suite(self, suite: str) -> bool:
        return self.pytest.get(suite) == "all"

    def selects_case(self, case_id: str) -> bool:
        return self.model_e2e_mode == "all" or (
            self.model_e2e_mode == "explicit" and case_id in self.model_e2e_cases
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pytest": dict(sorted(self.pytest.items())),
            "model_e2e": {"mode": self.model_e2e_mode, "cases": list(self.model_e2e_cases)},
            "allow_skip": {key: list(value) for key, value in sorted(self.allow_skip.items())},
        }


def _issue(path: str, message: str, code: str = "invalid-policy") -> Issue:
    return Issue(code=code, path=path, message=message)


def _map(value: Any, path: str, issues: list[Issue]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be a mapping"))
        return {}
    return value


def _keys(value: dict[str, Any], allowed: set[str], path: str, issues: list[Issue]) -> None:
    for key in sorted(set(value) - allowed):
        issues.append(_issue(f"{path}.{key}", "unknown field"))


def _strings(value: Any, path: str, issues: list[Issue]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(_issue(path, "must be a string list"))
        return ()
    return tuple(value)


def load(path: Path, name: str) -> Policy:
    """Load one named policy and validate its local schema."""
    issues: list[Issue] = []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError([_issue(str(path), str(exc))]) from exc
    document = _map(raw, "policies", issues)
    _keys(document, {"schema_version", "policies"}, "policies", issues)
    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("schema_version", f"expected {SCHEMA_VERSION}"))
    policies = _map(document.get("policies"), "policies", issues)
    if name not in policies:
        raise PolicyError([*issues, _issue("policies", f"unknown policy {name!r}")])
    row = _map(policies[name], f"policies.{name}", issues)
    _keys(row, {"pytest", "model_e2e", "allow_skip"}, f"policies.{name}", issues)

    pytest_row = _map(row.get("pytest"), f"policies.{name}.pytest", issues)
    _keys(pytest_row, set(SUITES), f"policies.{name}.pytest", issues)
    pytest_modes: dict[str, str] = {}
    for suite in SUITES:
        mode = str(pytest_row.get(suite) or "none")
        if mode not in {"all", "none"}:
            issues.append(_issue(f"policies.{name}.pytest.{suite}", "must be all or none"))
            mode = "none"
        pytest_modes[suite] = mode

    model_row = _map(row.get("model_e2e"), f"policies.{name}.model_e2e", issues)
    _keys(model_row, {"mode", "cases"}, f"policies.{name}.model_e2e", issues)
    model_mode = str(model_row.get("mode") or "none")
    if model_mode not in {"all", "none", "explicit"}:
        issues.append(_issue(f"policies.{name}.model_e2e.mode", "must be all, none, or explicit"))
        model_mode = "none"
    cases = _strings(model_row.get("cases") or [], f"policies.{name}.model_e2e.cases", issues)
    if model_mode == "explicit" and not cases:
        issues.append(_issue(f"policies.{name}.model_e2e.cases", "explicit mode needs at least one case"))
    if len(set(cases)) != len(cases):
        issues.append(_issue(f"policies.{name}.model_e2e.cases", "contains duplicates"))

    allow_row = _map(row.get("allow_skip") or {}, f"policies.{name}.allow_skip", issues)
    allow_skip: dict[str, tuple[str, ...]] = {}
    for device, reasons in allow_row.items():
        if device not in DEVICES:
            issues.append(_issue(f"policies.{name}.allow_skip.{device}", "unknown device"))
            continue
        allow_skip[device] = _strings(reasons, f"policies.{name}.allow_skip.{device}", issues)
    if name != "pull_request" and allow_skip:
        issues.append(_issue(f"policies.{name}.allow_skip", "allow_skip is only supported for pull_request"))

    if issues:
        raise PolicyError(issues)
    policy = Policy(name, pytest_modes, model_mode, cases, allow_skip, "")
    return replace(policy, digest=digest(policy.to_dict()))


def validate(policy: Policy, catalog: Catalog) -> list[Issue]:
    """Validate references from a loaded policy into the catalog."""
    issues: list[Issue] = []
    present = {entry.suite for entry in catalog.tests}
    for suite, mode in policy.pytest.items():
        if mode == "all" and suite not in present:
            issues.append(_issue(
                f"policies.{policy.name}.pytest.{suite}",
                f"suite {suite!r} selects no tests",
                "empty-suite",
            ))
    known = {case.case_id for case in catalog.cases}
    for case_id in policy.model_e2e_cases:
        if case_id not in known:
            issues.append(_issue(
                f"policies.{policy.name}.model_e2e.cases",
                f"unknown E2E case {case_id!r}",
                "unknown-case",
            ))
    if policy.model_e2e_mode == "all" and not catalog.cases:
        issues.append(_issue(
            f"policies.{policy.name}.model_e2e", "all mode selects no E2E cases", "empty-cases"
        ))
    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--policy", action="append", default=[])
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    try:
        catalog = load_catalog(root)
    except CatalogError as exc:
        for issue in exc.issues:
            print(f"{issue.path}: {issue.message}", file=sys.stderr)
        return 1
    failures: list[Issue] = []
    for name in args.policy or ["pull_request", "nightly"]:
        try:
            policy = load(root / POLICIES, name)
        except PolicyError as exc:
            failures.extend(exc.issues)
            continue
        failures.extend(validate(policy, catalog))
        print(f"policy {name}: {canonical_json(policy.to_dict())}")
    if failures:
        for issue in failures:
            print(f"{issue.path}: {issue.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
