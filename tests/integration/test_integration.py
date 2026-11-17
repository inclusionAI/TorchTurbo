"""Integration tests for torchturbo package."""

import sys
from unittest import mock

import pytest


class TestPackageImport:
    """Test that importing torchturbo discovers built-in adapters."""

    def test_import_registers_llamafactory_adapter(self):
        from torchturbo.registry import get_adapter

        adapter = get_adapter("llamafactory_v0")
        assert adapter is not None
        assert adapter.framework_name == "llamafactory_v0"

    def test_detect_function_exported(self):
        import torchturbo

        assert callable(torchturbo.detect)
        assert callable(torchturbo.activate)
        assert callable(torchturbo.detect_all)

    def test_version_exported(self):
        import torchturbo

        assert hasattr(torchturbo, "__version__")
        assert torchturbo.__version__ == "0.1.0"


class TestDetectWithMockedLlamaFactory:
    """Test detection behaviour when we mock llamafactory being in sys.modules."""

    def test_detect_returns_llamafactory_when_loaded(self):
        """Simulate llamafactory being imported."""
        import torchturbo

        # Create a mock llamafactory module with v0 version
        mock_lf = mock.MagicMock()
        mock_lf.__version__ = "0.9.2"

        with mock.patch.dict(sys.modules, {"llamafactory": mock_lf}):
            result = torchturbo.detect()
            assert result == "llamafactory_v0"

    def test_detect_returns_none_when_not_loaded(self):
        """When llamafactory is not installed at all, detect should return None."""
        import torchturbo

        # Remove llamafactory from sys.modules and make find_spec return None
        modules_to_remove = [k for k in sys.modules if k.startswith("llamafactory")]
        cleaned = {k: v for k, v in sys.modules.items() if not k.startswith("llamafactory")}

        with mock.patch.dict(sys.modules, cleaned, clear=True):
            with mock.patch("importlib.util.find_spec", return_value=None):
                result = torchturbo.detect()
                assert result is None


class TestActivate:
    """Test the activate() function."""

    def test_activate_with_no_framework_returns_none(self):
        import torchturbo

        modules_to_remove = [k for k in sys.modules if k.startswith("llamafactory")]
        cleaned = {k: v for k, v in sys.modules.items() if not k.startswith("llamafactory")}

        with mock.patch.dict(sys.modules, cleaned, clear=True):
            with mock.patch("importlib.util.find_spec", return_value=None):
                result = torchturbo.activate()
                assert result is None

    def test_activate_unknown_framework_raises(self):
        import torchturbo

        with pytest.raises(ValueError, match="No adapter registered"):
            torchturbo.activate("nonexistent_framework_xyz")

    def test_activate_patches_dynamic_module_loading(self, monkeypatch):
        import torchturbo
        from torchturbo.utils import transformers_utils

        dynamic_module_utils = pytest.importorskip("transformers.dynamic_module_utils")
        monkeypatch.setattr(
            dynamic_module_utils,
            "get_class_in_module",
            dynamic_module_utils.get_class_in_module,
        )
        monkeypatch.setattr(
            transformers_utils,
            "_patched_get_class_in_module",
            transformers_utils._patched_get_class_in_module,
        )
        adapter = mock.Mock()
        monkeypatch.setattr(torchturbo, "get_adapter", lambda name: adapter)

        assert torchturbo.activate("fake") is adapter
        assert (
            dynamic_module_utils.get_class_in_module
            is transformers_utils._patched_get_class_in_module
        )
        adapter.activate.assert_called_once_with()
