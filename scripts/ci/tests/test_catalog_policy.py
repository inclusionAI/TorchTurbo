from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.ci.producer.catalog import Catalog, counts
from scripts.ci.producer.policy import (
    PolicyError,
    load as load_policy,
    validate as validate_policy,
)


def test_catalog_snapshot(catalog: Catalog) -> None:
    assert counts(catalog) == {
        "core": 86,
        "model": 60,
        "adapter": 16,
        "total": 162,
    }
    assert len(catalog.cases) == 11
    assert [image.name for image in catalog.images.values() if image.device == "cpu" and image.default] == [
        "cpu-default"
    ]
    assert [image.name for image in catalog.images.values() if image.device == "npu" and image.default] == [
        "npu-default"
    ]
    assert not [image for image in catalog.images.values() if image.device == "gpu" and image.default]


def test_catalog_routes_resolve_to_images_and_lanes(catalog: Catalog) -> None:
    for entry in catalog.tests:
        assert entry.suite in {"core", "model", "adapter"}
        assert entry.devices
        for device in entry.devices:
            if device == "gpu":
                assert not [lane for lane in catalog.lanes.values() if lane.device == "gpu"]
                continue
            image = catalog.image_for(device, entry.image_name)
            lane = catalog.lane_for(device, image.name, entry.ranks_by_device[device])
            assert lane.device == device
            assert lane.image_name == image.name


@pytest.mark.parametrize(
    ("ranks", "lane_name"),
    (
        (1, "npu/default-2"),
        (2, "npu/default-2"),
        (4, "npu/default-4"),
        (8, "npu/default-8"),
        (16, "npu/default-16"),
    ),
)
def test_npu_lane_picks_smallest_sufficient_capacity(
    catalog: Catalog, ranks: int, lane_name: str
) -> None:
    assert catalog.lane_for("npu", "npu-default", ranks).name == lane_name


def test_catalog_model_venv_distribution(catalog: Catalog) -> None:
    model_venvs = Counter(
        entry.venv_name for entry in catalog.tests if entry.suite == "model"
    )
    assert model_venvs == {"default": 42, "qwen4_exp": 18}


def test_extends_is_expanded_for_plan_execution(catalog: Catalog) -> None:
    venv = catalog.venvs["qwen4_exp_swift"]
    assert venv.extends == "swift"
    assert venv.requirements == (
        "scripts/ci/config/envs/swift.txt",
    )
    assert venv.overrides == ("scripts/ci/config/envs/swift_overrides.txt",)


def test_pull_request_policy_is_fixed_scope(pr_policy) -> None:
    assert pr_policy.pytest == {"core": "all", "model": "none", "adapter": "none"}
    assert pr_policy.model_e2e_mode == "explicit"
    assert pr_policy.model_e2e_cases == (
        "bailing_hybrid-lf-sft",
    )
    assert pr_policy.allow_skip == {}


def test_nightly_policy_selects_all(nightly_policy) -> None:
    assert nightly_policy.pytest == {"core": "all", "model": "all", "adapter": "all"}
    assert nightly_policy.model_e2e_mode == "all"
    assert nightly_policy.allow_skip == {}


def test_policy_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "policies.yml"
    path.write_text(
        yaml.safe_dump({
            "schema_version": 1,
            "unexpected": True,
            "policies": {
                "pull_request": {
                    "pytest": {"core": "all", "model": "none", "adapter": "none"},
                    "model_e2e": {"mode": "none"},
                }
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="unknown field"):
        load_policy(path, "pull_request")


def test_policy_rejects_unknown_e2e_case(tmp_path: Path, catalog: Catalog) -> None:
    path = tmp_path / "policies.yml"
    path.write_text(
        yaml.safe_dump({
            "schema_version": 1,
            "policies": {
                "pull_request": {
                    "pytest": {"core": "none", "model": "none", "adapter": "none"},
                    "model_e2e": {"mode": "explicit", "cases": ["missing-case"]},
                }
            },
        }),
        encoding="utf-8",
    )
    policy = load_policy(path, "pull_request")
    issues = validate_policy(policy, catalog)
    assert [issue.code for issue in issues] == ["unknown-case"]
