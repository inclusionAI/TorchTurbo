"""Tests for adapter registry and BaseAdapter."""

import pytest

from torchturbo.registry import (
    BaseAdapter,
    clear_adapters,
    get_adapter,
    get_adapters_by_priority,
    get_all_adapters,
    register_adapter,
)
from torchturbo.registry.registry import _adapters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self._activated = False
        self._config = None

    def detect(self) -> bool:
        return True

    def activate(self, config=None):
        self._activated = True
        self._config = config

    def deactivate(self):
        self._activated = False

    def get_framework_version(self) -> str:
        return "1.0.0"


class NeverDetectAdapter(BaseAdapter):
    def detect(self) -> bool:
        return False

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
# Tests: registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_decorator(self):
        @register_adapter("test_framework")
        class TestAdapter(BaseAdapter):
            def detect(self) -> bool:
                return True
            def activate(self, config=None):
                pass
            def get_framework_version(self) -> str:
                return "1.0.0"

        adapter = get_adapter("test_framework")
        assert adapter is not None
        assert adapter.framework_name == "test_framework"
        assert isinstance(adapter, TestAdapter)

    def test_get_nonexistent(self):
        assert get_adapter("does_not_exist") is None

    def test_all(self):
        register_adapter("a")(DummyAdapter)
        register_adapter("b")(NeverDetectAdapter)

        all_a = get_all_adapters()
        assert set(all_a.keys()) == {"a", "b"}

    def test_duplicate_overwrites(self):
        register_adapter("dup")(DummyAdapter)
        register_adapter("dup")(NeverDetectAdapter)

        adapter = get_adapter("dup")
        assert isinstance(adapter, NeverDetectAdapter)

    def test_register_non_adapter_raises(self):
        with pytest.raises(TypeError, match="must subclass BaseAdapter"):
            register_adapter("bad")(str)  # type: ignore


# ---------------------------------------------------------------------------
# Tests: priority ordering
# ---------------------------------------------------------------------------

class TestPriority:
    def test_sorted_by_priority(self):
        register_adapter("low", priority=1)(DummyAdapter)
        register_adapter("high", priority=100)(NeverDetectAdapter)
        register_adapter("mid", priority=50)(DummyAdapter)

        ordered = get_adapters_by_priority()
        names = [name for name, _ in ordered]
        assert names == ["high", "mid", "low"]

    def test_default_priority_is_zero(self):
        register_adapter("default")(DummyAdapter)
        assert _adapters["default"][1] == 0


# ---------------------------------------------------------------------------
# Tests: BaseAdapter interface
# ---------------------------------------------------------------------------

class TestBaseAdapter:
    def test_activate_deactivate(self):
        adapter = DummyAdapter()
        assert not adapter._activated

        adapter.activate({"key": "value"})
        assert adapter._activated
        assert adapter._config == {"key": "value"}

        adapter.deactivate()
        assert not adapter._activated

    def test_repr(self):
        adapter = DummyAdapter()
        adapter.framework_name = "test"
        assert "DummyAdapter" in repr(adapter)
        assert "test" in repr(adapter)
