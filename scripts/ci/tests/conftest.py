from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.contracts.artifacts import Source  # noqa: E402
from scripts.ci.producer.ci_config import CiConfig, Scope, load as load_config  # noqa: E402
from scripts.ci.producer.facts import TestFacts, load as load_facts  # noqa: E402
from scripts.ci.producer.planner import build, source_for  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config() -> CiConfig:
    return load_config(ROOT)


@pytest.fixture(scope="session")
def facts(config: CiConfig) -> TestFacts:
    return load_facts(ROOT, config)


@pytest.fixture(scope="session")
def source(repo_root: Path) -> Source:
    return source_for(
        repo_root,
        repository="example/torchturbo",
        ref="refs/pull/123/head",
        sha="0123456789abcdef0123456789abcdef01234567",
    )


@pytest.fixture(scope="session")
def pr_scope(config: CiConfig) -> Scope:
    return config.scope("pull_request")


@pytest.fixture(scope="session")
def nightly_scope(config: CiConfig) -> Scope:
    return config.scope("nightly")


@pytest.fixture(scope="session")
def pr_plan(config: CiConfig, facts: TestFacts, source: Source):
    return build(config, facts, "pull_request", source)


@pytest.fixture(scope="session")
def nightly_plan(config: CiConfig, facts: TestFacts, source: Source):
    return build(config, facts, "nightly", source)
