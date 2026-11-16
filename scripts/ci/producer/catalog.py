#!/usr/bin/env python3
"""Load the CI catalog and derive pytest facts without importing tests."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import scan as test_scan
from ..contracts.artifacts import (
    DEVICES,
    SCHEMA_VERSION,
    ImageRef,
    Issue,
    VenvSpec,
    canonical_json,
    digest,
    file_digest,
    write_json,
)

MODEL_PACKAGES = test_scan.MODEL_PACKAGES
FRAMEWORKS = ("lf", "swift")


class CatalogError(ValueError):
    """Raised when catalog configuration or derived facts are invalid."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class CatalogImage:
    name: str
    device: str
    kind: str
    ref: str
    default: bool

    def as_ref(self) -> ImageRef:
        return ImageRef(name=self.name, kind=self.kind, ref=self.ref)


@dataclass(frozen=True)
class CatalogLane:
    name: str
    device: str
    image_name: str
    capacity_slots: int
    runner_variable: str
    runner: str

@dataclass(frozen=True)
class TestOverride:
    venv: str
    image: str | None = None


@dataclass(frozen=True)
class PytestWorkload:
    job_timeout_minutes: dict[str, int]
    step_timeout_seconds: dict[str, int]
    model_venvs: dict[str, str]
    framework_venvs: dict[str, str]
    combo_venvs: dict[str, str]
    overrides: dict[str, TestOverride]

@dataclass(frozen=True)
class ModelE2EDefaults:
    job_timeout_minutes: int
    fetch_timeout_seconds: int
    case_timeout_seconds: int
    seq_len: int
    finetuning_type: str

@dataclass(frozen=True)
class TestEntry:
    path: str
    suite: str
    frameworks: tuple[str, ...]
    model_families: tuple[str, ...]
    tags: tuple[str, ...]
    devices: tuple[str, ...]
    ranks_by_device: dict[str, int]
    image_name: str | None
    venv_name: str
    process_group: str

@dataclass(frozen=True)
class ModelE2ECase:
    case_id: str
    model: str
    framework: str
    device: str
    venv_name: str
    image_name: str | None
    run_dir: str
    workdir: str
    install_script: str
    train_script: str
    install_args: tuple[str, ...]
    train_args: tuple[str, ...]
    hf_repo: str
    layers: int
    dataset: str
    ranks: int
    sp: int
    ep: int
    timeout_seconds: int
    seq_len: int
    finetuning_type: str

@dataclass(frozen=True)
class Catalog:
    schema_version: int
    root: Path
    images: dict[str, CatalogImage]
    venvs: dict[str, VenvSpec]
    lanes: dict[str, CatalogLane]
    pytest: PytestWorkload
    model_e2e_defaults: ModelE2EDefaults
    tests: tuple[TestEntry, ...]
    cases: tuple[ModelE2ECase, ...]
    registered_models: tuple[str, ...]
    digest: str

    def image_for(self, device: str, name: str | None = None) -> CatalogImage:
        if name:
            image = self.images.get(name)
            if image is None:
                raise CatalogError([Issue("unknown-image", name, f"unknown image {name!r}")])
            if image.device != device:
                raise CatalogError([
                    Issue("image-device", name, f"image {name!r} is for {image.device}, not {device}")
                ])
            return image
        defaults = [image for image in self.images.values()
                    if image.device == device and image.default]
        if len(defaults) != 1:
            raise CatalogError([
                Issue("default-image", device, f"device {device!r} needs exactly one default image")
            ])
        return defaults[0]

    def lane_for(self, device: str, image_name: str, ranks: int) -> CatalogLane:
        matches = [lane for lane in self.lanes.values()
                   if lane.device == device and lane.image_name == image_name
                   and lane.capacity_slots >= ranks]
        if not matches:
            raise CatalogError([
                Issue(
                    "missing-lane",
                    f"{device}/{image_name}",
                    f"no lane for device={device} image={image_name} ranks={ranks}",
                )
            ])
        return min(matches, key=lambda lane: (lane.capacity_slots, lane.name))

    def case(self, case_id: str) -> ModelE2ECase:
        case = next((item for item in self.cases if item.case_id == case_id), None)
        if case is None:
            raise CatalogError([
                Issue("unknown-case", case_id, f"unknown E2E case {case_id!r}")
            ])
        return case

    def dump_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "images": {key: asdict(value) for key, value in sorted(self.images.items())},
            "venvs": {key: value.to_dict() for key, value in sorted(self.venvs.items())},
            "lanes": {key: asdict(value) for key, value in sorted(self.lanes.items())},
            "pytest": asdict(self.pytest),
            "model_e2e": {
                "defaults": asdict(self.model_e2e_defaults),
                "cases": [asdict(case) for case in self.cases],
            },
            "tests": [asdict(entry) for entry in self.tests],
            "registered_models": list(self.registered_models),
        }


def _issue(path: str, message: str, code: str = "invalid") -> Issue:
    return Issue(code=code, path=path, message=message)


def _mapping(value: Any, path: str, issues: list[Issue]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be a mapping"))
        return {}
    return value


def _unknown_keys(value: dict[str, Any], allowed: set[str], path: str,
                  issues: list[Issue]) -> None:
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


def _load_images(raw: Any, image_refs: dict[str, str], root: Path,
                 issues: list[Issue]) -> dict[str, CatalogImage]:
    rows = _mapping(raw, "resources.images", issues)
    images: dict[str, CatalogImage] = {}
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
        images[name] = CatalogImage(
            name=name,
            device=device,
            kind=kind,
            ref=ref,
            default=bool(row.get("default", False)),
        )
    for device in DEVICES:
        defaults = [image for image in images.values() if image.device == device and image.default]
        if len(defaults) > 1:
            issues.append(_issue("resources.images", f"device {device} has multiple default images"))
    return images


def _venv_digest(root: Path, python: str, requirements: tuple[str, ...],
                 overrides: tuple[str, ...], extends: str | None,
                 constraints: str) -> str:
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
    local_venvs: dict[str, VenvSpec] = {}
    for name, value in rows.items():
        where = f"resources.venvs.{name}"
        row = _row(value, where, {"python", "requirements", "overrides", "extends", "constraints"}, issues)
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
        local_venvs[name] = VenvSpec(
            name=name,
            python=python,
            requirements=requirements,
            overrides=overrides,
            extends=extends,
            constraints=constraints,
            digest="",
        )
    for venv in local_venvs.values():
        if venv.extends and venv.extends not in local_venvs:
            issues.append(_issue(f"resources.venvs.{venv.name}.extends", "unknown venv"))

    resolved: dict[str, VenvSpec] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> VenvSpec:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            issues.append(_issue(f"resources.venvs.{name}.extends", "environment cycle"))
            return local_venvs[name]
        resolving.add(name)
        spec = local_venvs[name]
        base = resolve(spec.extends) if spec.extends in local_venvs else None
        if base is not None:
            requirements = tuple(dict.fromkeys((*base.requirements, *spec.requirements)))
            overrides = tuple(dict.fromkeys((*base.overrides, *spec.overrides)))
        else:
            requirements = spec.requirements
            overrides = spec.overrides
        value = replace(
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
        resolved[name] = value
        return value

    for name in local_venvs:
        resolve(name)
    return resolved


def _load_lanes(raw: Any, images: dict[str, CatalogImage], issues: list[Issue]) -> dict[str, CatalogLane]:
    rows = _mapping(raw, "resources.lanes", issues)
    lanes: dict[str, CatalogLane] = {}
    for name, value in rows.items():
        where = f"resources.lanes.{name}"
        row = _row(value, where, {"device", "image", "capacity_slots", "runner_variable", "runner"}, issues)
        device = str(row.get("device") or "")
        image_name = str(row.get("image") or "")
        image = images.get(image_name)
        if device not in DEVICES:
            issues.append(_issue(f"{where}.device", "unknown device"))
        if image is None:
            issues.append(_issue(f"{where}.image", "unknown image"))
        elif image.device != device:
            issues.append(_issue(where, "image and lane devices differ"))
        lane = CatalogLane(
            name=name,
            device=device,
            image_name=image_name,
            capacity_slots=_int(row.get("capacity_slots"), f"{where}.capacity_slots", issues, 1),
            runner_variable=str(row.get("runner_variable") or ""),
            runner=str(row.get("runner") or ""),
        )
        if not lane.runner_variable or not lane.runner:
            issues.append(_issue(where, "runner_variable and runner are required"))
        lanes[name] = lane
    return lanes


def _load_pytest(raw: Any, venvs: dict[str, VenvSpec], root: Path,
                 issues: list[Issue]) -> PytestWorkload:
    row = _row(
        raw, "workloads.pytest",
        {"defaults", "model_venvs", "framework_venvs", "combo_venvs", "overrides"}, issues,
    )
    defaults = _row(
        row.get("defaults"), "workloads.pytest.defaults",
        {"job_timeout_minutes", "step_timeout_seconds"}, issues,
    )
    job_timeout = _mapping(defaults.get("job_timeout_minutes"), "workloads.pytest.defaults.job_timeout_minutes", issues)
    step_timeout = _mapping(defaults.get("step_timeout_seconds"), "workloads.pytest.defaults.step_timeout_seconds", issues)
    job_timeout_map = {
        device: _int(value, f"workloads.pytest.defaults.job_timeout_minutes.{device}", issues, 1)
        for device, value in job_timeout.items()
    }
    step_timeout_map = {
        device: _int(value, f"workloads.pytest.defaults.step_timeout_seconds.{device}", issues, 1)
        for device, value in step_timeout.items()
    }

    model_venvs = _mapping(row.get("model_venvs"), "workloads.pytest.model_venvs", issues)
    framework_venvs = _mapping(row.get("framework_venvs"), "workloads.pytest.framework_venvs", issues)
    combo_venvs = _mapping(row.get("combo_venvs"), "workloads.pytest.combo_venvs", issues)
    overrides_raw = _mapping(row.get("overrides"), "workloads.pytest.overrides", issues)

    for key, value in (*model_venvs.items(), *framework_venvs.items(), *combo_venvs.items()):
        if str(value) not in venvs:
            issues.append(_issue(f"workloads.pytest.{key}", f"unknown venv {value!r}"))
    if "default" not in venvs:
        issues.append(_issue("resources.venvs", "default venv is required"))

    overrides: dict[str, TestOverride] = {}
    for path, value in overrides_raw.items():
        where = f"workloads.pytest.overrides.{path}"
        override = _row(value, where, {"venv", "image"}, issues)
        venv = str(override.get("venv") or "")
        image = str(override.get("image") or "") or None
        if venv not in venvs:
            issues.append(_issue(f"{where}.venv", "unknown venv"))
        if not (root / path).is_file():
            issues.append(_issue(where, "test file is missing"))
        overrides[path] = TestOverride(venv=venv, image=image)

    return PytestWorkload(
        job_timeout_minutes=job_timeout_map,
        step_timeout_seconds=step_timeout_map,
        model_venvs={key: str(value) for key, value in model_venvs.items()},
        framework_venvs={key: str(value) for key, value in framework_venvs.items()},
        combo_venvs={key: str(value) for key, value in combo_venvs.items()},
        overrides=overrides,
    )


def _load_model_e2e_defaults(raw: Any, issues: list[Issue]) -> ModelE2EDefaults:
    row = _row(
        raw, "workloads.model_e2e.defaults",
        {"job_timeout_minutes", "fetch_timeout", "case_timeout", "seq_len", "finetuning_type"}, issues,
    )
    fetch_timeout = _int(row.get("fetch_timeout"), "workloads.model_e2e.defaults.fetch_timeout", issues, 1)
    case_timeout = _int(row.get("case_timeout"), "workloads.model_e2e.defaults.case_timeout", issues, 1)
    if case_timeout <= fetch_timeout:
        issues.append(_issue(
            "workloads.model_e2e.defaults",
            "case_timeout must be greater than fetch_timeout",
        ))
    return ModelE2EDefaults(
        job_timeout_minutes=_int(row.get("job_timeout_minutes"), "workloads.model_e2e.defaults.job_timeout_minutes", issues, 1),
        fetch_timeout_seconds=fetch_timeout,
        case_timeout_seconds=case_timeout,
        seq_len=_int(row.get("seq_len"), "workloads.model_e2e.defaults.seq_len", issues, 1),
        finetuning_type=str(row.get("finetuning_type") or ""),
    )


def _load_model_e2e_cases(raw: Any, root: Path, defaults: ModelE2EDefaults,
                          venvs: dict[str, VenvSpec], images: dict[str, CatalogImage],
                          issues: list[Issue]) -> tuple[ModelE2ECase, ...]:
    models = _mapping(raw, "workloads.model_e2e.models", issues)
    cases: list[ModelE2ECase] = []
    seen: set[str] = set()
    for model, value in models.items():
        where = f"workloads.model_e2e.models.{model}"
        row = _row(value, where, {"run_dir", "cases"}, issues)
        run_dir = str(row.get("run_dir") or "")
        workdir = f"scripts/runs/{run_dir}"
        if not run_dir or not (root / workdir).is_dir():
            issues.append(_issue(f"{where}.run_dir", "run_dir is missing"))
        raw_cases = row.get("cases") or []
        if not isinstance(raw_cases, list):
            issues.append(_issue(f"{where}.cases", "must be a list"))
            continue
        for index, item in enumerate(raw_cases):
            case_where = f"{where}.cases[{index}]"
            case_row = _row(
                item,
                case_where,
                {
                    "id", "device", "framework", "venv", "image", "install", "train",
                    "install_args", "train_args", "download", "dataset", "ranks",
                    "parallelism", "timeout_seconds", "seq_len", "finetuning_type",
                },
                issues,
            )
            case_id = str(case_row.get("id") or "")
            if not case_id:
                issues.append(_issue(f"{case_where}.id", "case id is required"))
                continue
            if case_id in seen:
                issues.append(_issue(f"{case_where}.id", f"duplicate case id {case_id!r}"))
            seen.add(case_id)
            device = str(case_row.get("device") or "npu")
            framework = str(case_row.get("framework") or "")
            venv = str(case_row.get("venv") or "")
            image_name = str(case_row.get("image") or "") or None
            if device not in DEVICES:
                issues.append(_issue(f"{case_where}.device", "unknown device"))
            if framework not in FRAMEWORKS:
                issues.append(_issue(f"{case_where}.framework", "must be lf or swift"))
            if venv not in venvs:
                issues.append(_issue(f"{case_where}.venv", "unknown venv"))
            if image_name and image_name not in images:
                issues.append(_issue(f"{case_where}.image", "unknown image"))
            install_name = str(case_row.get("install") or "")
            train_name = str(case_row.get("train") or "")
            install_script = f"{workdir}/{install_name}" if install_name else ""
            train_script = f"{workdir}/{train_name}" if train_name else ""
            if not install_script or not (root / install_script).is_file():
                issues.append(_issue(f"{case_where}.install", "install script is missing"))
            if not train_script or not (root / train_script).is_file():
                issues.append(_issue(f"{case_where}.train", "train script is missing"))

            download = _row(case_row.get("download"), f"{case_where}.download", {"hf_repo", "layers"}, issues)
            hf_repo = str(download.get("hf_repo") or "")
            layers = _int(download.get("layers", 0), f"{case_where}.download.layers", issues)
            if not hf_repo:
                issues.append(_issue(f"{case_where}.download.hf_repo", "hf_repo is required"))

            ranks = _int(case_row.get("ranks"), f"{case_where}.ranks", issues, 1)
            parallelism = _row(case_row.get("parallelism"), f"{case_where}.parallelism", {"sp", "ep"}, issues)
            sp = _int(parallelism.get("sp"), f"{case_where}.parallelism.sp", issues, 1)
            ep = _int(parallelism.get("ep"), f"{case_where}.parallelism.ep", issues, 1)
            if ranks % sp or (ranks // sp) % ep:
                issues.append(_issue(case_where, "ranks/sp/ep do not form a valid parallel plan"))

            dataset = str(case_row.get("dataset") or "")
            if not dataset:
                issues.append(_issue(f"{case_where}.dataset", "dataset is required"))
            cases.append(ModelE2ECase(
                case_id=case_id,
                model=model,
                framework=framework,
                device=device,
                venv_name=venv,
                image_name=image_name,
                run_dir=run_dir,
                workdir=workdir,
                install_script=install_script,
                train_script=train_script,
                install_args=_strings(case_row.get("install_args"), f"{case_where}.install_args", issues),
                train_args=_strings(case_row.get("train_args"), f"{case_where}.train_args", issues),
                hf_repo=hf_repo,
                layers=layers,
                dataset=dataset,
                ranks=ranks,
                sp=sp,
                ep=ep,
                timeout_seconds=_int(
                    case_row.get("timeout_seconds", defaults.case_timeout_seconds),
                    f"{case_where}.timeout_seconds",
                    issues,
                    1,
                ),
                seq_len=_int(
                    case_row.get("seq_len", defaults.seq_len),
                    f"{case_where}.seq_len",
                    issues,
                    1,
                ),
                finetuning_type=str(
                    case_row.get("finetuning_type") or defaults.finetuning_type
                ),
            ))
    return tuple(cases)


def classify_test(path: str, frameworks: Iterable[str], families: Iterable[str],
                  registered_models: set[str]) -> str:
    """Classify a pytest file using the suite rules from the design."""
    framework_set = set(frameworks)
    family_set = set(families)
    parts = test_scan.test_parts(path)
    if parts[:1] == ("adapters",) or framework_set:
        return "adapter"
    if parts[:1] == ("models",) and len(parts) >= 2:
        if len(parts) == 2 or parts[1] == "modules":
            return "core"
        if len(parts) >= 3 and parts[1] in registered_models and parts[1] not in MODEL_PACKAGES:
            return "model"
    if family_set:
        return "model"
    return "core"


def resolve_venv(path: str, families: Iterable[str], frameworks: Iterable[str],
                 workload: PytestWorkload, issues: list[Issue]) -> str:
    override = workload.overrides.get(path)
    if override is not None:
        return override.venv
    family_set = set(families)
    framework_set = set(frameworks)
    if len(framework_set) > 1:
        issues.append(_issue(path, "a multi-framework test needs an explicit override"))
        return ""
    family_venvs = {workload.model_venvs.get(family, "default") for family in family_set}
    if len(family_venvs) > 1:
        issues.append(_issue(path, "a test cannot use multiple model-family venvs"))
        return ""
    family_venv = next(iter(family_venvs), "default")
    framework = next(iter(framework_set), "")
    if family_set and framework and family_venv != "default":
        family = next(iter(family_set))
        combo = workload.combo_venvs.get(f"{family}+{framework}")
        if combo is None:
            issues.append(_issue(path, f"no venv for family={family} framework={framework}"))
            return ""
        return combo
    if framework:
        return workload.framework_venvs.get(framework, "")
    return family_venv


def _scan_tests(root: Path, registered: set[str], venvs: dict[str, VenvSpec],
                workload: PytestWorkload,
                issue_list: list[Issue]) -> tuple[TestEntry, ...]:
    scanned, errors = test_scan.scan_tests(root, registered)
    issue_list.extend(_issue(error.path, error.message) for error in errors)

    entries: list[TestEntry] = []
    for path, fact in sorted(scanned.items()):
        suite = classify_test(path, fact.frameworks, fact.families, registered)
        venv_name = resolve_venv(path, fact.families, fact.frameworks, workload, issue_list)
        if venv_name and venv_name not in venvs:
            issue_list.append(_issue(path, f"resolved venv {venv_name!r} is not defined"))
        override = workload.overrides.get(path)
        entries.append(TestEntry(
            path=path,
            suite=suite,
            frameworks=tuple(sorted(fact.frameworks)),
            model_families=tuple(sorted(fact.families)),
            tags=("integration",) if fact.integration else (),
            devices=tuple(sorted(fact.devices)),
            ranks_by_device={device: fact.processes_for(device) for device in sorted(fact.devices)},
            image_name=override.image if override is not None else None,
            venv_name=venv_name,
            process_group=fact.process,
        ))
    return tuple(entries)


def load(root: Path, *, image_refs: dict[str, str] | None = None) -> Catalog:
    """Load catalog.yml, scan tests, and validate all cross references."""
    issues: list[Issue] = []
    path = root / "scripts" / "ci" / "config" / "catalog.yml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError([_issue("scripts/ci/config/catalog.yml", str(exc))]) from exc
    document = _row(raw, "catalog", {"schema_version", "resources", "workloads"}, issues)
    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("schema_version", f"expected {SCHEMA_VERSION}"))

    resources = _row(document.get("resources"), "resources", {"images", "venvs", "lanes"}, issues)
    images = _load_images(resources.get("images"), image_refs or {}, root, issues)
    venvs = _load_venvs(resources.get("venvs"), root, issues)
    lanes = _load_lanes(resources.get("lanes"), images, issues)
    for device in DEVICES:
        if any(lane.device == device for lane in lanes.values()):
            try:
                image = next(item for item in images.values()
                             if item.device == device and item.default)
            except StopIteration:
                issues.append(_issue("resources.images", f"device {device} has no default image"))
            else:
                if not any(lane.device == device and lane.image_name == image.name
                           for lane in lanes.values()):
                    issues.append(_issue("resources.lanes", f"device {device} has no lane for its default image"))

    workloads = _row(document.get("workloads"), "workloads", {"pytest", "model_e2e"}, issues)
    pytest_workload = _load_pytest(workloads.get("pytest"), venvs, root, issues)
    model_row = _row(workloads.get("model_e2e"), "workloads.model_e2e", {"defaults", "models"}, issues)
    defaults = _load_model_e2e_defaults(model_row.get("defaults"), issues)
    cases = _load_model_e2e_cases(
        model_row.get("models"), root, defaults, venvs, images, issues
    )

    registered = test_scan.registered_models(root)
    for model in sorted({case.model for case in cases} - registered):
        issues.append(_issue("workloads.model_e2e.models", f"model {model!r} is not registered"))
    tests = _scan_tests(root, registered, venvs, pytest_workload, issues)
    for entry in tests:
        if not entry.devices:
            issues.append(_issue(entry.path, "test has no device"))
        for device in entry.devices:
            if device not in pytest_workload.job_timeout_minutes:
                issues.append(_issue(entry.path, f"missing pytest job timeout for {device}"))
            if device not in pytest_workload.step_timeout_seconds:
                issues.append(_issue(entry.path, f"missing pytest step timeout for {device}"))

    if issues:
        raise CatalogError(issues)
    base = Catalog(
        schema_version=SCHEMA_VERSION,
        root=root,
        images=images,
        venvs=venvs,
        lanes=lanes,
        pytest=pytest_workload,
        model_e2e_defaults=defaults,
        tests=tests,
        cases=cases,
        registered_models=tuple(sorted(registered)),
        digest="",
    )
    return replace(base, digest=digest(base.dump_payload()))


def counts(catalog: Catalog) -> dict[str, int]:
    value = {"core": 0, "model": 0, "adapter": 0}
    for entry in catalog.tests:
        value[entry.suite] += 1
    value["total"] = len(catalog.tests)
    return value


def _render_errors(issues: Iterable[Issue]) -> None:
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
        catalog = load(root, image_refs=image_refs)
    except CatalogError as exc:
        _render_errors(exc.issues)
        return 1
    if args.command == "dump":
        payload = catalog.dump_payload()
        if args.output:
            write_json(Path(args.output), payload)
        else:
            print(canonical_json(payload))
        return 0
    value = counts(catalog)
    print(
        f"catalog OK ({len(catalog.cases)} E2E cases / {value['total']} test files; "
        f"core={value['core']} model={value['model']} adapter={value['adapter']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
