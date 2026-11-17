from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from scripts.ci.producer.ci_config import (
    CiConfig,
    CiConfigError,
    CiImage,
    CiLane,
    CiTarget,
)
from scripts.ci.producer.facts import TestFacts, counts


def test_config_resources(config: CiConfig) -> None:
    assert config.targets["cpu"].device == "cpu"
    assert config.targets["npu-a3"].device == "npu"
    assert config.targets["npu-a3"].hardware == "a3"
    assert [
        image.name for image in config.images.values() if image.target == "cpu"
    ] == [
        "cpu-default"
    ]
    assert [
        image.name for image in config.images.values() if image.target == "npu-a3"
    ] == [
        "npu-default"
    ]
    assert not [image for image in config.images.values() if image.target == "gpu"]
    assert set(config.venvs) == {
        "default",
        "qwen4_exp",
        "lf",
        "swift",
        "qwen4_exp_swift",
    }
    assert config.pytest.job_timeout_minutes == {"cpu": 40, "npu": 90, "gpu": 90}
    assert config.pytest.step_timeout_minutes == {"cpu": 40, "npu": 90, "gpu": 90}
    suite = config.integration.suites["base_suite"]
    assert suite.job_timeout_minutes == {"cpu": 40, "npu": 90}
    assert suite.step_timeout_minutes == {"cpu": 40, "npu": 90}


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
    config: CiConfig, ranks: int, lane_name: str
) -> None:
    assert config.lane_for("npu-a3", ranks).name == lane_name


def test_scope_selects_targets_not_images(config: CiConfig) -> None:
    pr = config.scope("pull_request")
    assert pr.targets == ("cpu", "npu-a3")
    assert config.targets_for("npu", pr.targets) == (config.targets["npu-a3"],)
    assert config.targets_for("cpu", pr.targets) == (config.targets["cpu"],)


def test_image_resolution_precedence(config: CiConfig) -> None:
    target = config.targets["npu-a3"]
    alternative = replace(
        config.images["npu-default"],
        name="npu-a3-alt",
        ref="swr.example/torchturbo_npu:a3-alt",
        default=False,
    )
    candidate = replace(
        config,
        images={**config.images, alternative.name: alternative},
    )
    scope_image = {target.name: alternative.name}
    workload_image = {target.name: "npu-default"}

    assert candidate.resolve_image(target).name == "npu-default"
    assert candidate.resolve_image(
        target, workload_images=workload_image
    ).name == "npu-default"
    assert candidate.resolve_image(
        target,
        scope_images=scope_image,
        workload_images=workload_image,
    ).name == alternative.name
    assert candidate.resolve_image(
        target,
        explicit="npu-default",
        scope_images=scope_image,
        workload_images=workload_image,
    ).name == "npu-default"


def test_image_must_match_target(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="is for target"):
        config.image_for("npu-a3", "cpu-default")


def test_unknown_target_fails_closed(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="unknown target"):
        config.target_for("npu-a5")


def _with_a5(config: CiConfig) -> CiConfig:
    target = CiTarget(name="npu-a5", device="npu", hardware="a5")
    image = CiImage(
        name="npu-a5-default",
        target=target.name,
        kind="container",
        ref="swr.example/torchturbo_npu:a5",
        default=True,
    )
    lane = CiLane(
        name="npu-a5/default-2",
        target=target.name,
        capacity_slots=2,
        runner_variable="CI_NPU_A5_2_RUNNER",
        runner="linux-aarch64-a5-800i-2",
    )
    scopes = {
        name: replace(
            scope,
            targets=(*scope.targets, target.name),
            images={**scope.images, target.name: image.name},
        )
        for name, scope in config.scopes.items()
    }
    return replace(
        config,
        targets={**config.targets, target.name: target},
        images={**config.images, image.name: image},
        lanes={**config.lanes, lane.name: lane},
        scopes=scopes,
    )


def test_a5_target_is_explicit_and_lane_is_target_scoped(config: CiConfig) -> None:
    candidate = _with_a5(config)
    target = candidate.target_for("npu-a5")
    assert candidate.targets_for("npu", ("cpu", "npu-a3", "npu-a5"), ("npu-a5",)) == (
        target,
    )
    assert candidate.lane_for(target, 2).name == "npu-a5/default-2"
    assert candidate.resolve_image(target).name == "npu-a5-default"


def test_venv_extends_is_expanded(config: CiConfig) -> None:
    venv = config.venvs["qwen4_exp_swift"]
    assert venv.extends == "swift"
    assert venv.requirements == ("scripts/ci/config/envs/swift.txt",)
    assert venv.overrides == ("scripts/ci/config/envs/swift_overrides.txt",)


def test_scope_selection_is_declarative(config: CiConfig) -> None:
    pr = config.scope("pull_request")
    nightly = config.scope("nightly")
    assert pr.pytest == {"core": "all", "model": "none", "adapter": "all"}
    assert pr.integration["base_suite"].mode == "explicit"
    assert pr.integration["base_suite"].cases == (
        "qwen3_5_moe/sft_2cpu",
        "qwen3_5_moe/sft_2npu",
    )
    assert nightly.pytest == {"core": "all", "model": "all", "adapter": "all"}
    assert nightly.integration["base_suite"].mode == "all"


def test_pytest_venv_resolution(config: CiConfig) -> None:
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/qwen4_exp/test_model.py",
        ("qwen4_exp",),
        (),
    )
    assert (venv, issues) == ("qwen4_exp", [])
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/qwen4_exp/test_swift.py",
        ("qwen4_exp",),
        ("swift",),
    )
    assert (venv, issues) == ("qwen4_exp_swift", [])
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/bailing_moe_v3_vl/test_lf_image_parity.py",
        ("bailing_moe_v3_vl",),
        ("lf",),
    )
    assert (venv, issues) == ("lf", [])


def test_unknown_scope_is_rejected(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="unknown CI scope"):
        config.scope("missing")


def test_facts_snapshot(facts: TestFacts) -> None:
    assert counts(facts) == {
        "core": 86,
        "model": 60,
        "adapter": 16,
        "total": 162,
        "integration": 3,
    }
    assert [case.case_id for case in facts.integration] == [
        "bailing_v3/sft_16npu",
        "qwen3_5_moe/sft_2cpu",
        "qwen3_5_moe/sft_2npu",
    ]
    assert all(case.compare for case in facts.integration)
    assert not any(case.update_golden for case in facts.integration)


def test_model_venv_distribution(config: CiConfig, facts: TestFacts) -> None:
    distribution = Counter()
    for fact in facts.pytest:
        if fact.suite != "model":
            continue
        venv, issues = config.resolve_pytest_venv(
            fact.path, fact.model_families, fact.frameworks
        )
        assert not issues
        distribution[venv] += 1
    assert distribution == {"default": 42, "qwen4_exp": 18}
