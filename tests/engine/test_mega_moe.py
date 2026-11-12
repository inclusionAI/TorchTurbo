"""Keep the existing MegaMoE switch aligned with main's config precedence."""

import pytest

from torchturbo.engine.args import parse_torchturbo_args
from torchturbo.engine.config import EngineConfig, build_from_args
from torchturbo.optimizations.mega_moe import bootstrap


@pytest.mark.parametrize("cli_value, expected", [(None, False), ("true", True), ("false", False)])
def test_mega_moe_config_and_early_bootstrap_agree(tmp_path, monkeypatch, cli_value, expected):
    monkeypatch.setenv("TORCH_TURBO_MEGA_MOE", "true")
    path = tmp_path / "config.yaml"
    path.write_text("engine_config:\n  mega_moe: false\n")
    argv = ["--tt_config", str(path)]
    if cli_value is not None:
        argv += ["--tt_mega_moe", cli_value]
    args, remaining = parse_torchturbo_args(argv)
    assert remaining == []
    config = build_from_args(args)
    assert config.mega_moe is expected
    restored = EngineConfig.from_dict(config.to_dict())
    assert restored.mega_moe is expected
    (tmp_path / "bootstrap.py").write_text("def activate():\n    return True\n")
    from types import SimpleNamespace

    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py")))
    assert bootstrap.activate(argv) is expected
