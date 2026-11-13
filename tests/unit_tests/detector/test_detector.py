"""Tests for FrameworkDetector."""

import pytest

from torchturbo.detector import FrameworkDetector
from torchturbo.registry import BaseAdapter, clear_adapters, register_adapter
from torchturbo.registry.registry import _adapters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AlwaysDetect(BaseAdapter):
    def detect(self) -> bool:
        return True

    def activate(self, config=None):
        pass

    def get_framework_version(self) -> str:
        return "1.0.0"


class NeverDetect(BaseAdapter):
    def detect(self) -> bool:
        return False

    def activate(self, config=None):
        pass

    def get_framework_version(self) -> str:
        return "1.0.0"


class BrokenDetect(BaseAdapter):
    def detect(self) -> bool:
        raise RuntimeError("boom")

    def activate(self, config=None):
        pass

    def get_framework_version(self) -> str:
        return "1.0.0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_adapters)
    clear_adapters()
    yield
    clear_adapters()
    _adapters.update(saved)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetect:
    def test_detect_returns_first_match(self):
        register_adapter("a", priority=10)(AlwaysDetect)
        register_adapter("b", priority=5)(AlwaysDetect)

        result = FrameworkDetector.detect()
        assert result == "a"

    def test_detect_skips_non_matching(self):
        register_adapter("nope", priority=10)(NeverDetect)
        register_adapter("yes", priority=5)(AlwaysDetect)

        result = FrameworkDetector.detect()
        assert result == "yes"

    def test_detect_returns_none_when_nothing_matches(self):
        register_adapter("nope")(NeverDetect)
        assert FrameworkDetector.detect() is None

    def test_detect_returns_none_with_empty_registry(self):
        assert FrameworkDetector.detect() is None

    def test_detect_handles_broken_adapter(self):
        register_adapter("broken", priority=10)(BrokenDetect)
        register_adapter("ok", priority=5)(AlwaysDetect)

        result = FrameworkDetector.detect()
        assert result == "ok"


class TestDetectAll:
    def test_detect_all_returns_all_matches(self):
        register_adapter("a", priority=10)(AlwaysDetect)
        register_adapter("b", priority=5)(AlwaysDetect)
        register_adapter("c", priority=1)(NeverDetect)

        result = FrameworkDetector.detect_all()
        assert result == ["a", "b"]

    def test_detect_all_empty(self):
        register_adapter("nope")(NeverDetect)
        assert FrameworkDetector.detect_all() == []


class TestGetAdapter:
    def test_get_existing(self):
        register_adapter("test")(AlwaysDetect)
        adapter = FrameworkDetector.get_adapter("test")
        assert adapter is not None
        assert isinstance(adapter, AlwaysDetect)

    def test_get_nonexistent(self):
        assert FrameworkDetector.get_adapter("nope") is None
