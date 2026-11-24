"""Runtime lookup and loading contracts for image-installed AscendC libraries."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from torchturbo.ops.providers.npu.ascendc import _config, _library


@pytest.fixture(autouse=True)
def clear_library_cache():
    _library.load_library.cache_clear()
    yield
    _library.load_library.cache_clear()


def test_image_library_directory_is_default(monkeypatch):
    monkeypatch.delenv("TORCHTURBO_ASCENDC_LIB_DIR", raising=False)
    assert _library.library_path("gated_activation") == Path(
        "/opt/torchturbo/ascendc/lib/libtorchturbo_gated_activation.so"
    )


@pytest.mark.parametrize("override", [None, "/tmp/custom ascendc"])
def test_install_config_runs_without_site_packages(override):
    environment = os.environ.copy()
    environment.pop("TORCHTURBO_ASCENDC_LIB_DIR", None)
    if override is not None:
        environment["TORCHTURBO_ASCENDC_LIB_DIR"] = override
    result = subprocess.run(
        [sys.executable, "-I", "-S", _config.__file__],
        env=environment, check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == (override or "/opt/torchturbo/ascendc/lib")


def test_library_directory_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTURBO_ASCENDC_LIB_DIR", str(tmp_path))
    assert _library.library_path("gated_activation") == tmp_path / "libtorchturbo_gated_activation.so"


def test_missing_override_fails_without_compiling(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTURBO_ASCENDC_LIB_DIR", str(tmp_path))
    monkeypatch.setattr(_library.torch.ops, "load_library", lambda path: pytest.fail("Missing library loaded"))
    with pytest.raises(RuntimeError, match="AscendC library missing"):
        _library.load_library("gated_activation")
    assert not list(tmp_path.iterdir())


def test_existing_library_is_loaded_once(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTURBO_ASCENDC_LIB_DIR", str(tmp_path))
    library = tmp_path / "libtorchturbo_gated_activation.so"
    library.touch()
    calls = []
    monkeypatch.setattr(_library.torch.ops, "load_library", calls.append)
    _library.load_library("gated_activation")
    _library.load_library("gated_activation")
    assert calls == [str(library.resolve())]
