from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.contracts.artifacts import Source  # noqa: E402
from scripts.ci.producer.catalog import Catalog, load as load_catalog  # noqa: E402
from scripts.ci.producer.planner import build, source_for  # noqa: E402
from scripts.ci.producer.policy import Policy, load as load_policy  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog(ROOT)


@pytest.fixture(scope="session")
def source(repo_root: Path) -> Source:
    return source_for(
        repo_root,
        repository="example/torchturbo",
        ref="refs/pull/123/head",
        sha="0123456789abcdef0123456789abcdef01234567",
    )


def _policy(name: str) -> Policy:
    return load_policy(ROOT / "scripts" / "ci" / "config" / "policies.yml", name)


@pytest.fixture(scope="session")
def pr_policy() -> Policy:
    return _policy("pull_request")


@pytest.fixture(scope="session")
def nightly_policy() -> Policy:
    return _policy("nightly")


@pytest.fixture(scope="session")
def pr_plan(catalog: Catalog, pr_policy: Policy, source: Source):
    return build(catalog, pr_policy, source)


@pytest.fixture(scope="session")
def nightly_plan(catalog: Catalog, nightly_policy: Policy, source: Source):
    return build(catalog, nightly_policy, source)
