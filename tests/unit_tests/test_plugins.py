"""Plugin system tests; ``torchturbo.engine.plugins`` loaded with a stubbed logger (no torch).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGINS_PATH = os.path.join(_REPO_ROOT, "src", "torchturbo", "engine", "plugins.py")


def _load_plugins_module():
    tt = types.ModuleType("torchturbo")
    tt.__path__ = [os.path.join(_REPO_ROOT, "src", "torchturbo")]
    logging_mod = types.ModuleType("torchturbo.logging")

    class _StubLogger:
        def info(self, *a, **k):
            pass

        def info_rank0(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def warning_rank0(self, *a, **k):
            pass

    logging_mod.get_logger = lambda *a, **k: _StubLogger()
    sys.modules.setdefault("torchturbo", tt)
    sys.modules.setdefault("torchturbo.logging", logging_mod)
    spec = importlib.util.spec_from_file_location("torchturbo.engine.plugins", _PLUGINS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["torchturbo.engine.plugins"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _cleanup_torchturbo_plugin_modules():
    yield
    for name in list(sys.modules):
        if name.startswith("torchturbo_plugin."):
            del sys.modules[name]


@pytest.fixture
def module():
    return _load_plugins_module()


@pytest.fixture
def manager(module):
    return module.PluginManager()


_counter = 0


def _write_plugin(tmp_path, body: str, name: str | None = None) -> str:
    global _counter
    _counter += 1
    name = name or f"plug_{_counter}.py"
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def test_register_plugin_rejects_non_plugin_subclass(module, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import Plugin, register_plugin\n"
        "@register_plugin('bad')\nclass Bad:\n"
        "    pass\n",
    )
    with pytest.raises(TypeError):
        module.PluginManager([plugin_file])


def test_load_populates_dataset_plugin(module, manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import DatasetPlugin, register_plugin\n"
        "@register_plugin('ds')\nclass Ds(DatasetPlugin):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        self.names_ = ['a', 'b']\n"
        "    def dataset_names(self): return list(self.names_)\n"
        "    def dataset_info(self, name): return {}\n",
    )
    manager.load([plugin_file])
    assert len(manager.plugins) == 1
    assert isinstance(manager.plugins[0], module.DatasetPlugin)
    assert manager.plugins[0].dataset_names() == ["a", "b"]
    assert manager.plugins[0].dataset_info("a") == {}


def test_plugin_config_binding_passed_to_init(manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import DatasetPlugin, register_plugin\n"
        "RAN=[]\n"
        "@register_plugin('ds')\nclass Ds(DatasetPlugin):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        RAN.append(self.config)\n",
    )
    cfg = tmp_path / "datasets.json"
    cfg.write_text("{}")
    manager.load([f"{plugin_file}:{cfg}"])
    assert manager.plugins[0].config == str(cfg)


def test_load_only_does_not_register_modules_without_decorator(manager, tmp_path):
    plugin_file = _write_plugin(tmp_path, "MARKER = 'loaded'\n")
    manager.load([plugin_file])
    assert manager.plugins == []
    mod_name = f"torchturbo_plugin.{os.path.splitext(os.path.basename(plugin_file))[0]}"
    assert sys.modules[mod_name].MARKER == "loaded"


def test_missing_file_raises(manager, tmp_path):
    with pytest.raises(FileNotFoundError):
        manager.load([str(tmp_path / "nope.py")])


def test_try_consume_runs_consumer_only_on_matching_plugins(module, manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import DatasetPlugin, register_plugin\n"
        "@register_plugin('ds')\nclass Ds(DatasetPlugin):\n"
        "    def dataset_names(self): return ['a']\n"
        "    def dataset_info(self, name): return {}\n",
    )
    manager.load([plugin_file])

    seen: list = []

    class _C(module.PluginConsumer):
        def matches(self, plugin):
            return isinstance(plugin, module.DatasetPlugin)

        def consume(self, plugin):
            seen.append(plugin.dataset_names())

    class _NonMatching(module.PluginConsumer):
        def matches(self, plugin):
            return False

        def consume(self, plugin):
            seen.append("nope")

    manager.try_consume(_C())
    manager.try_consume(_NonMatching())
    assert seen == [["a"]]

def test_checkpoint_export_plugin_loads_without_platform_sdk(module, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import CheckpointExportPlugin, register_plugin\n"
        "@register_plugin('export')\n"
        "class Export(CheckpointExportPlugin):\n"
        "    def on_hf_export(self, export_dir):\n"
        "        self.export_dir = export_dir\n",
    )
    manager = module.PluginManager([plugin_file])
    assert len(manager.plugins) == 1
    plugin = manager.plugins[0]
    assert isinstance(plugin, module.CheckpointExportPlugin)
    plugin.on_hf_export(str(tmp_path))
    assert plugin.export_dir == str(tmp_path)
