#!/usr/bin/env python3
"""Load the single CI configuration: resources, workloads, and scopes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..contracts.artifacts import (
    DEVICES,
    SUITES,
    ImageRef,
    Issue,
    VenvSpec,
    canonical_json,
    digest,
    file_digest,
    write_json,
)

CONFIG_PATH = Path("scripts/ci/config/ci.yml")
SCHEMA_VERSION = 1
SCOPES = ("pull_request", "nightly")


class CiConfigError(ValueError):
    """Raised when CI configuration or a cross-reference is invalid."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class CiImage:
    name: str
    device: str
    kind: str
    ref: str
    default: bool

    def as_ref(self) -> ImageRef:
        return ImageRef(name=self.name, kind=self.kind, ref=self.ref)


@dataclass(frozen=True)
class CiLane:
    name: str
    device: str
    image_name: str
    capacity_slots: int
    runner_variable: str
    runner: str


@dataclass(frozen=True)
class PytestOverride:
    venv: str
    image: str | None = None


@dataclass(frozen=True)
class PytestWorkload:
    job_timeout_minutes: dict[str, int]
    step_timeout_seconds: dict[str, int]
    model_venvs: dict[str, str]
    framework_venvs: dict[str, str]
    combo_venvs: dict[str, str]
    overrides: dict[str, PytestOverride]


@dataclass(frozen=True)
class IntegrationOverride:
    job_timeout_minutes: int | None = None
    step_timeout_seconds: int | None = None


@dataclass(frozen=True)
class IntegrationSuite:
    name: str
    runner: str
    job_timeout_minutes: dict[str, int]
    step_timeout_seconds: dict[str, int]
    framework_venvs: dict[str, str]
    overrides: dict[str, IntegrationOverride]

    def timeout_minutes(self, device: str, case_id: str) -> int:
        override = self.overrides.get(case_id)
        if override is not None and override.job_timeout_minutes is not None:
            return override.job_timeout_minutes
        return self.job_timeout_minutes[device]

    def timeout_seconds(self, device: str, case_id: str) -> int:
        override = self.overrides.get(case_id)
        if override is not None and override.step_timeout_seconds is not None:
            return override.step_timeout_seconds
        return self.step_timeout_seconds[device]


@dataclass(frozen=True)
class IntegrationWorkload:
    suites: dict[str, IntegrationSuite]


@dataclass(frozen=True)
class ScopeSuite:
    mode: str
    cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scope:
    name: str
    pytest: dict[str, str]
    integration: dict[str, ScopeSuite]
    allow_skip: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class CiConfig:
    schema_version: int
    root: Path
    images: dict[str, CiImage]
    venvs: dict[str, VenvSpec]
    lanes: dict[str, CiLane]
    pytest: PytestWorkload
    integration: IntegrationWorkload
    scopes: dict[str, Scope]
    digest: str

    def image_for(self, device: str, name: str | None = None) -> CiImage:
        if name:
            image = self.images.get(name)
            if image is None:
                raise CiConfigError([Issue("unknown-image", name, f"unknown image {name!r}")])
            if image.device != device:
                raise CiConfigError([Issue(
                    "image-device",
                    name,
                    f"image {name!r} is for {image.device}, not {device}",
                )])
            return image
        defaults = [
            image for image in self.images.values()
            if image.device == device and image.default
        ]
        if len(defaults) != 1:
            raise CiConfigError([Issue(
                "default-image",
                device,
                f"device {device!r} needs exactly one default image",
            )])
        return defaults[0]

    def lane_for(self, device: str, image_name: str, ranks: int) -> CiLane:
        matches = [
            lane for lane in self.lanes.values()
            if lane.device == device
            and lane.image_name == image_name
            and lane.capacity_slots >= ranks
        ]
        if not matches:
            raise CiConfigError([Issue(
                "missing-lane",
                f"{device}/{image_name}",
                f"no lane for device={device} image={image_name} ranks={ranks}",
            )])
        return min(matches, key=lambda lane: (lane.capacity_slots, lane.name))

    def scope(self, name: str) -> Scope:
        scope = self.scopes.get(name)
        if scope is None:
            raise CiConfigError([Issue("unknown-scope", name, f"unknown CI scope {name!r}")])
        return scope

    def device_is_configured(self, device: str) -> bool:
        return any(image.device == device for image in self.images.values()) or any(
            lane.device == device for lane in self.lanes.values()
        )

    def integration_suite(self, name: str) -> IntegrationSuite:
        suite = self.integration.suites.get(name)
        if suite is None:
            raise CiConfigError([Issue(
                "unknown-integration-suite",
                name,
                f"unknown integration suite {name!r}",
            )])
        return suite

    def resolve_pytest_venv(
        self,
        path: str,
        families: Iterable[str],
        frameworks: Iterable[str],
    ) -> tuple[str, list[Issue]]:
        override = self.pytest.overrides.get(path)
        if override is not None:
            return override.venv, []
        family_set = set(families)
        framework_set = set(frameworks)
        if len(framework_set) > 1:
            return "", [Issue(
                "multi-framework-venv",
                path,
                "a multi-framework test needs an explicit override",
            )]
        family_venvs = {
            self.pytest.model_venvs.get(family, "default") for family in family_set
        }
        if len(family_venvs) > 1:
            return "", [Issue(
                "multi-family-venv",
                path,
                "a test cannot use multiple model-family venvs",
            )]
        family_venv = next(iter(family_venvs), "default")
        framework = next(iter(framework_set), "")
        if family_set and framework and family_venv != "default":
            family = next(iter(family_set))
            combo = self.pytest.combo_venvs.get(f"{family}+{framework}")
            if combo is None:
                return "", [Issue(
                    "missing-combo-venv",
                    path,
                    f"no venv for family={family} framework={framework}",
                )]
            return combo, []
        if framework:
            venv = self.pytest.framework_venvs.get(framework)
            if venv is None:
                return "", [Issue(
                    "missing-framework-venv",
                    path,
                    f"no venv for framework={framework}",
                )]
            return venv, []
        return family_venv, []

    def dump_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resources": {
                "images": {
                    key: asdict(value) for key, value in sorted(self.images.items())
                },
                "venvs": {
                    key: value.to_dict() for key, value in sorted(self.venvs.items())
                },
                "lanes": {
                    key: asdict(value) for key, value in sorted(self.lanes.items())
                },
            },
            "workloads": {
                "pytest": asdict(self.pytest),
                "integration": {
                    key: asdict(value)
                    for key, value in sorted(self.integration.suites.items())
                },
            },
            "scopes": {
                key: {
                    "pytest": value.pytest,
                    "integration": {
                        suite: asdict(selection)
                        for suite, selection in sorted(value.integration.items())
                    },
                    "allow_skip": {
                        device: list(reasons)
                        for device, reasons in sorted(value.allow_skip.items())
                    },
                }
                for key, value in sorted(self.scopes.items())
            },
        }


def _issue(path: str, message: str, code: str = "invalid-ci-config") -> Issue:
    return Issue(code=code, path=path, message=message)


def _mapping(value: Any, path: str, issues: list[Issue]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be a mapping"))
        return {}
    return value


def _unknown_keys(
    value: dict[str, Any], allowed: set[str], path: str, issues: list[Issue]
) -> None:
    for key in sorted(set(value) - allowed):
        issues.append(_issue(f"{path}.{key}", "unknown field"))


def _row(value: Any, path: str, allowed: set[str], issues: list[Issue]) -> dict[str, Any]:
    row = _mapping(value, path, issues)
    _unknown_keys(row, allowed, path, issues)
    return row


def _strings(value: Any, path: str, issues: list[Issue]) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(_issue(path, "must be a string or list of strings"))
        return ()
    return tuple(value)


def _int(value: Any, path: str, issues: list[Issue], minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        issues.append(_issue(path, f"must be an integer >= {minimum}"))
        return minimum
    return value


def _timeouts(raw: Any, path: str, issues: list[Issue]) -> dict[str, int]:
    values = _mapping(raw, path, issues)
    result: dict[str, int] = {}
    for device, value in values.items():
        if device not in DEVICES:
            issues.append(_issue(f"{path}.{device}", "unknown device"))
            continue
        result[device] = _int(value, f"{path}.{device}", issues, 1)
    return result


def _load_images(
    raw: Any,
    root: Path,
    image_refs: dict[str, str],
    issues: list[Issue],
) -> dict[str, CiImage]:
    rows = _mapping(raw, "resources.images", issues)
    images: dict[str, CiImage] = {}
    for name, value in rows.items():
        where = f"resources.images.{name}"
        row = _row(value, where, {"device", "kind", "ref", "default"}, issues)
        device = str(row.get("device") or "")
        kind = str(row.get("kind") or "")
        ref = str(image_refs.get(name) or row.get("ref") or "")
        if device not in DEVICES:
            issues.append(_issue(f"{where}.device", "unknown device"))
        if kind not in {"host", "container"}:
            issues.append(_issue(f"{where}.kind", "must be host or container"))
        if not ref:
            issues.append(_issue(f"{where}.ref", "image ref is required"))
        images[name] = CiImage(
            name=name,
            device=device,
            kind=kind,
            ref=ref,
            default=bool(row.get("default", False)),
        )
    for device in DEVICES:
        defaults = [
            image for image in images.values()
            if image.device == device and image.default
        ]
        if len(defaults) > 1:
            issues.append(_issue(
                "resources.images",
                f"device {device} has multiple default images",
            ))
        if any(image.device == device for image in images.values()) and not defaults:
            issues.append(_issue(
                "resources.images",
                f"device {device} has no default image",
            ))
    return images


def _venv_digest(
    root: Path,
    python: str,
    requirements: tuple[str, ...],
    overrides: tuple[str, ...],
    extends: str | None,
    constraints: str,
) -> str:
    files: dict[str, str] = {}
    for relative in (*requirements, *overrides):
        path = root / relative
        if path.is_file():
            files[relative] = file_digest(path)
    return digest({
        "python": python,
        "requirements": list(requirements),
        "overrides": list(overrides),
        "extends": extends,
        "constraints": constraints,
        "files": files,
    })


def _load_venvs(raw: Any, root: Path, issues: list[Issue]) -> dict[str, VenvSpec]:
    rows = _mapping(raw, "resources.venvs", issues)
    local: dict[str, VenvSpec] = {}
    for name, value in rows.items():
        where = f"resources.venvs.{name}"
        row = _row(
            value,
            where,
            {"python", "requirements", "overrides", "extends", "constraints"},
            issues,
        )
        python = str(row.get("python") or "")
        requirements = _strings(row.get("requirements"), f"{where}.requirements", issues)
        overrides = _strings(row.get("overrides"), f"{where}.overrides", issues)
        extends = str(row.get("extends") or "") or None
        constraints = str(row.get("constraints") or "auto")
        if not python:
            issues.append(_issue(f"{where}.python", "python version is required"))
        for relative in (*requirements, *overrides):
            if not (root / relative).is_file():
                issues.append(_issue(where, f"missing file {relative!r}"))
        local[name] = VenvSpec(
            name=name,
            python=python,
            requirements=requirements,
            overrides=overrides,
            extends=extends,
            constraints=constraints,
            digest="",
        )
    if "default" not in local:
        issues.append(_issue("resources.venvs", "default venv is required"))
    for venv in local.values():
        if venv.extends and venv.extends not in local:
            issues.append(_issue(
                f"resources.venvs.{venv.name}.extends",
                "unknown venv",
            ))

    resolved: dict[str, VenvSpec] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> VenvSpec:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            issues.append(_issue(f"resources.venvs.{name}.extends", "environment cycle"))
            return local[name]
        resolving.add(name)
        spec = local[name]
        base = resolve(spec.extends) if spec.extends in local else None
        if base is not None:
            requirements = tuple(dict.fromkeys((*base.requirements, *spec.requirements)))
            overrides = tuple(dict.fromkeys((*base.overrides, *spec.overrides)))
        else:
            requirements, overrides = spec.requirements, spec.overrides
        result = replace(
            spec,
            requirements=requirements,
            overrides=overrides,
            digest=_venv_digest(
                root,
                spec.python,
                requirements,
                overrides,
                spec.extends,
                spec.constraints,
            ),
        )
        resolving.remove(name)
        resolved[name] = result
        return result

    for name in local:
        resolve(name)
    return resolved


def _load_lanes(
    raw: Any,
    images: dict[str, CiImage],
    issues: list[Issue],
) -> dict[str, CiLane]:
    rows = _mapping(raw, "resources.lanes", issues)
    lanes: dict[str, CiLane] = {}
    for name, value in rows.items():
        where = f"resources.lanes.{name}"
        row = _row(
            value,
            where,
            {"device", "image", "capacity_slots", "runner_variable", "runner"},
            issues,
        )
        device = str(row.get("device") or "")
        image_name = str(row.get("image") or "")
        image = images.get(image_name)
        if device not in DEVICES:
            issues.append(_issue(f"{where}.device", "unknown device"))
        if image is None:
            issues.append(_issue(f"{where}.image", "unknown image"))
        elif image.device != device:
            issues.append(_issue(where, "image and lane devices differ"))
        lane = CiLane(
            name=name,
            device=device,
            image_name=image_name,
            capacity_slots=_int(
                row.get("capacity_slots"), f"{where}.capacity_slots", issues, 1
            ),
            runner_variable=str(row.get("runner_variable") or ""),
            runner=str(row.get("runner") or ""),
        )
        if not lane.runner_variable or not lane.runner:
            issues.append(_issue(where, "runner_variable and runner are required"))
        lanes[name] = lane
    for device in DEVICES:
        defaults = [
            image for image in images.values()
            if image.device == device and image.default
        ]
        if defaults and not any(
            lane.device == device and lane.image_name == defaults[0].name
            for lane in lanes.values()
        ):
            issues.append(_issue(
                "resources.lanes",
                f"device {device} has no lane for its default image",
            ))
    return lanes


def _load_pytest(
    raw: Any,
    venvs: dict[str, VenvSpec],
    images: dict[str, CiImage],
    root: Path,
    issues: list[Issue],
) -> PytestWorkload:
    row = _row(
        raw,
        "workloads.pytest",
        {"defaults", "model_venvs", "framework_venvs", "combo_venvs", "overrides"},
        issues,
    )
    defaults = _row(
        row.get("defaults"),
        "workloads.pytest.defaults",
        {"job_timeout_minutes", "step_timeout_seconds"},
        issues,
    )
    model_venvs = _mapping(
        row.get("model_venvs"), "workloads.pytest.model_venvs", issues
    )
    framework_venvs = _mapping(
        row.get("framework_venvs"), "workloads.pytest.framework_venvs", issues
    )
    combo_venvs = _mapping(
        row.get("combo_venvs"), "workloads.pytest.combo_venvs", issues
    )
    overrides_raw = _mapping(
        row.get("overrides"), "workloads.pytest.overrides", issues
    )

    for key, value in (
        *model_venvs.items(),
        *framework_venvs.items(),
        *combo_venvs.items(),
    ):
        if str(value) not in venvs:
            issues.append(_issue(
                f"workloads.pytest.{key}", f"unknown venv {value!r}"
            ))
    overrides: dict[str, PytestOverride] = {}
    for path, value in overrides_raw.items():
        where = f"workloads.pytest.overrides.{path}"
        override = _row(value, where, {"venv", "image"}, issues)
        venv = str(override.get("venv") or "")
        image = str(override.get("image") or "") or None
        if venv not in venvs:
            issues.append(_issue(f"{where}.venv", "unknown venv"))
        if image and image not in images:
            issues.append(_issue(f"{where}.image", "unknown image"))
        if not (root / path).is_file():
            issues.append(_issue(where, "test file is missing"))
        overrides[path] = PytestOverride(venv=venv, image=image)

    return PytestWorkload(
        job_timeout_minutes=_timeouts(
            defaults.get("job_timeout_minutes"),
            "workloads.pytest.defaults.job_timeout_minutes",
            issues,
        ),
        step_timeout_seconds=_timeouts(
            defaults.get("step_timeout_seconds"),
            "workloads.pytest.defaults.step_timeout_seconds",
            issues,
        ),
        model_venvs={key: str(value) for key, value in model_venvs.items()},
        framework_venvs={key: str(value) for key, value in framework_venvs.items()},
        combo_venvs={key: str(value) for key, value in combo_venvs.items()},
        overrides=overrides,
    )


def _load_integration(
    raw: Any,
    venvs: dict[str, VenvSpec],
    root: Path,
    issues: list[Issue],
) -> IntegrationWorkload:
    row = _row(raw, "workloads.integration", {"suites"}, issues)
    raw_suites = _mapping(row.get("suites"), "workloads.integration.suites", issues)
    suites: dict[str, IntegrationSuite] = {}
    for name, value in raw_suites.items():
        where = f"workloads.integration.suites.{name}"
        suite = _row(
            value,
            where,
            {"runner", "defaults", "framework_venvs", "overrides"},
            issues,
        )
        runner = str(suite.get("runner") or "")
        if not runner or not (root / runner).is_file():
            issues.append(_issue(f"{where}.runner", "runner script is missing"))
        defaults = _row(
            suite.get("defaults"),
            f"{where}.defaults",
            {"job_timeout_minutes", "step_timeout_seconds"},
            issues,
        )
        frameworks = _mapping(
            suite.get("framework_venvs"), f"{where}.framework_venvs", issues
        )
        for framework, venv in frameworks.items():
            if str(venv) not in venvs:
                issues.append(_issue(
                    f"{where}.framework_venvs.{framework}",
                    f"unknown venv {venv!r}",
                ))
        overrides: dict[str, IntegrationOverride] = {}
        raw_overrides = _mapping(suite.get("overrides"), f"{where}.overrides", issues)
        for case_id, item in raw_overrides.items():
            override_where = f"{where}.overrides.{case_id}"
            override = _row(
                item,
                override_where,
                {"job_timeout_minutes", "step_timeout_seconds"},
                issues,
            )
            job_timeout = override.get("job_timeout_minutes")
            step_timeout = override.get("step_timeout_seconds")
            overrides[case_id] = IntegrationOverride(
                job_timeout_minutes=(
                    _int(job_timeout, f"{override_where}.job_timeout_minutes", issues, 1)
                    if job_timeout is not None else None
                ),
                step_timeout_seconds=(
                    _int(step_timeout, f"{override_where}.step_timeout_seconds", issues, 1)
                    if step_timeout is not None else None
                ),
            )
        suites[name] = IntegrationSuite(
            name=name,
            runner=runner,
            job_timeout_minutes=_timeouts(
                defaults.get("job_timeout_minutes"), f"{where}.defaults.job_timeout_minutes", issues
            ),
            step_timeout_seconds=_timeouts(
                defaults.get("step_timeout_seconds"), f"{where}.defaults.step_timeout_seconds", issues
            ),
            framework_venvs={
                framework: str(venv) for framework, venv in frameworks.items()
            },
            overrides=overrides,
        )
    return IntegrationWorkload(suites=suites)


def _load_scopes(
    raw: Any,
    suites: dict[str, IntegrationSuite],
    issues: list[Issue],
) -> dict[str, Scope]:
    rows = _mapping(raw, "scopes", issues)
    scopes: dict[str, Scope] = {}
    for name, value in rows.items():
        where = f"scopes.{name}"
        row = _row(value, where, {"pytest", "integration", "allow_skip"}, issues)
        pytest_row = _mapping(row.get("pytest"), f"{where}.pytest", issues)
        _unknown_keys(pytest_row, set(SUITES), f"{where}.pytest", issues)
        pytest_modes: dict[str, str] = {}
        for suite in SUITES:
            mode = str(pytest_row.get(suite) or "none")
            if mode not in {"all", "none"}:
                issues.append(_issue(
                    f"{where}.pytest.{suite}", "must be all or none"
                ))
                mode = "none"
            pytest_modes[suite] = mode

        integration_row = _mapping(
            row.get("integration") or {}, f"{where}.integration", issues
        )
        for suite_name in set(integration_row) - set(suites):
            issues.append(_issue(
                f"{where}.integration.{suite_name}",
                "unknown integration suite",
            ))
        integration: dict[str, ScopeSuite] = {}
        for suite_name in suites:
            value_row = integration_row.get(suite_name)
            if value_row is None:
                integration[suite_name] = ScopeSuite(mode="none")
                continue
            selection = _row(
                value_row,
                f"{where}.integration.{suite_name}",
                {"mode", "cases"},
                issues,
            )
            mode = str(selection.get("mode") or "none")
            if mode not in {"all", "explicit"}:
                issues.append(_issue(
                    f"{where}.integration.{suite_name}.mode",
                    "must be all or explicit",
                ))
                mode = "none"
            cases = _strings(
                selection.get("cases"),
                f"{where}.integration.{suite_name}.cases",
                issues,
            )
            if mode == "explicit" and not cases:
                issues.append(_issue(
                    f"{where}.integration.{suite_name}.cases",
                    "explicit mode needs at least one case",
                ))
            if len(set(cases)) != len(cases):
                issues.append(_issue(
                    f"{where}.integration.{suite_name}.cases",
                    "contains duplicates",
                ))
            integration[suite_name] = ScopeSuite(mode=mode, cases=cases)

        allow_row = _mapping(
            row.get("allow_skip") or {}, f"{where}.allow_skip", issues
        )
        allow_skip: dict[str, tuple[str, ...]] = {}
        for device, reasons in allow_row.items():
            if device not in DEVICES:
                issues.append(_issue(f"{where}.allow_skip.{device}", "unknown device"))
                continue
            allow_skip[device] = _strings(
                reasons, f"{where}.allow_skip.{device}", issues
            )
        scopes[name] = Scope(
            name=name,
            pytest=pytest_modes,
            integration=integration,
            allow_skip=allow_skip,
        )
    for name in SCOPES:
        if name not in scopes:
            issues.append(_issue("scopes", f"required scope {name!r} is missing"))
    return scopes


def load(root: Path, *, image_refs: dict[str, str] | None = None) -> CiConfig:
    """Load ci.yml and validate all configuration-local cross references."""
    issues: list[Issue] = []
    path = root / CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CiConfigError([_issue(CONFIG_PATH.as_posix(), str(exc))]) from exc
    document = _row(
        raw, "ci", {"schema_version", "resources", "workloads", "scopes"}, issues
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue(
            "schema_version", f"expected {SCHEMA_VERSION}", "unsupported-schema"
        ))
    resources = _row(
        document.get("resources"), "resources", {"images", "venvs", "lanes"}, issues
    )
    images = _load_images(
        resources.get("images"), root, image_refs or {}, issues
    )
    venvs = _load_venvs(resources.get("venvs"), root, issues)
    lanes = _load_lanes(resources.get("lanes"), images, issues)
    workloads = _row(
        document.get("workloads"), "workloads", {"pytest", "integration"}, issues
    )
    pytest = _load_pytest(workloads.get("pytest"), venvs, images, root, issues)
    integration = _load_integration(workloads.get("integration"), venvs, root, issues)
    scopes = _load_scopes(document.get("scopes"), integration.suites, issues)
    if issues:
        raise CiConfigError(issues)
    config = CiConfig(
        schema_version=SCHEMA_VERSION,
        root=root,
        images=images,
        venvs=venvs,
        lanes=lanes,
        pytest=pytest,
        integration=integration,
        scopes=scopes,
        digest="",
    )
    return replace(config, digest=digest(config.dump_payload()))


def _render(issues: Iterable[Issue]) -> None:
    for issue in issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "dump"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--image-ref", action="append", default=[], metavar="NAME=REF")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    image_refs: dict[str, str] = {}
    for value in args.image_ref:
        name, separator, ref = value.partition("=")
        if not separator or not name or not ref:
            parser.error(f"bad --image-ref {value!r}; expected NAME=REF")
        image_refs[name] = ref
    try:
        config = load(root, image_refs=image_refs)
    except CiConfigError as exc:
        _render(exc.issues)
        return 1
    if args.command == "dump":
        payload = config.dump_payload()
        if args.output:
            write_json(Path(args.output), payload)
        else:
            print(canonical_json(payload))
        return 0
    print(
        f"ci config OK ({len(config.images)} images, {len(config.venvs)} venvs, "
        f"{len(config.lanes)} lanes, {len(config.integration.suites)} integration suites)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
