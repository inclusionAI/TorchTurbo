from __future__ import annotations

from collections import Counter

import pytest

from scripts.ci.producer.ci_config import CiConfig, CiConfigError
from scripts.ci.producer.facts import TestFacts, counts


def test_config_resources(config: CiConfig) -> None:
    assert [image.name for image in config.images.values() if image.device == "cpu"] == [
        "cpu-default"
    ]
    assert [image.name for image in config.images.values() if image.device == "npu"] == [
        "npu-default"
    ]
    assert not [image for image in config.images.values() if image.device == "gpu"]
    assert set(config.venvs) == {
        "default",
        "qwen4_exp",
        "lf",
        "swift",
        "qwen4_exp_swift",
    }


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
    assert config.lane_for("npu", "npu-default", ranks).name == lane_name


def test_venv_extends_is_expanded(config: CiConfig) -> None:
    venv = config.venvs["qwen4_exp_swift"]
    assert venv.extends == "swift"
    assert venv.requirements == ("scripts/ci/config/envs/swift.txt",)
    assert venv.overrides == ("scripts/ci/config/envs/swift_overrides.txt",)


def test_scope_selection_is_declarative(config: CiConfig) -> None:
    pr = config.scope("pull_request")
    nightly = config.scope("nightly")
    assert pr.pytest == {"core": "all", "model": "none", "adapter": "none"}
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
