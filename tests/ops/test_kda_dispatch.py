"""chunk_kda ops dispatch: fla preferred, naive fallback, per-call override."""

from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpRegistry
from torchturbo.ops.providers.native.kda import FlaChunkKDA, NaiveChunkKDA
from torchturbo.ops.registry import registry as global_registry
from torchturbo.runtime.cpu import CpuRuntime


def _fresh_dispatcher() -> OpDispatcher:
    registry = OpRegistry()
    registry.register(FlaChunkKDA.op_spec())
    registry.register(NaiveChunkKDA.op_spec())
    return OpDispatcher(registry=registry, device_runtime=CpuRuntime())


def _candidate(dispatcher, impl=None):
    return dispatcher._candidate_impl(
        "chunk_kda", dispatcher.device_runtime.turbo_device, impl=impl
    )


def test_chunk_kda_registered_with_two_impls():
    specs = {
        s.impl_name: s
        for s in global_registry.list(
            spec_filter=lambda s: s.op_name == "chunk_kda"
        )
    }
    assert specs["fla_chunk_kda"].priority == 10
    assert specs["naive_chunk_kda"].priority == 0
    assert specs["fla_chunk_kda"].devices == ()
    assert specs["naive_chunk_kda"].devices == ()


def test_auto_prefers_fla_when_available(monkeypatch):
    monkeypatch.setattr(FlaChunkKDA, "available", lambda self: True)
    monkeypatch.setattr(NaiveChunkKDA, "available", lambda self: True)
    assert isinstance(_candidate(_fresh_dispatcher()), FlaChunkKDA)


def test_auto_falls_back_to_naive_when_fla_missing(monkeypatch):
    monkeypatch.setattr(FlaChunkKDA, "available", lambda self: False)
    assert isinstance(_candidate(_fresh_dispatcher()), NaiveChunkKDA)


def test_forced_naive_override(monkeypatch):
    monkeypatch.setattr(FlaChunkKDA, "available", lambda self: True)
    assert isinstance(
        _candidate(_fresh_dispatcher(), impl="naive_chunk_kda"), NaiveChunkKDA
    )


def test_naive_impl_sets_transpose_state_layout(monkeypatch):
    # NOTE: ops/kda/__init__ re-exports the function name `chunk_kda_naive`,
    # which shadows the submodule on the package — resolve via sys.modules.
    naive_mod = sys.modules["torchturbo.ops.kda.chunk_kda_naive"]

    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(naive_mod, "chunk_kda_naive", spy)
    NaiveChunkKDA()(torch.zeros(1))
    assert captured["transpose_state_layout"] is True
    captured.clear()
    NaiveChunkKDA()(torch.zeros(1), transpose_state_layout=False)
    assert captured["transpose_state_layout"] is False


def test_fla_impl_forwards_kwargs_verbatim(monkeypatch):
    fla_kda = sys.modules["fla.ops.kda"]

    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(fla_kda, "chunk_kda", spy)
    FlaChunkKDA()(torch.zeros(1), transpose_state_layout=False, marker=7)
    assert captured["transpose_state_layout"] is False
    assert captured["marker"] == 7


def test_entry_standalone_forced_naive(monkeypatch):
    import torchturbo.ops.kda as kda_pkg
    ops_mod = sys.modules["torchturbo.ops.ops"]

    monkeypatch.setattr(
        ops_mod, "get_ops", Mock(side_effect=RuntimeError("not configured"))
    )
    spy = Mock(return_value=(None, None))
    monkeypatch.setattr(kda_pkg, "chunk_kda_naive", spy)
    kda_pkg.chunk_kda(torch.zeros(1), impl="naive_chunk_kda")
    assert spy.call_count == 1


def test_entry_standalone_falls_back_when_fla_missing(monkeypatch):
    import torchturbo.ops.kda as kda_pkg
    ops_mod = sys.modules["torchturbo.ops.ops"]

    monkeypatch.setitem(sys.modules, "fla.ops.kda", None)
    monkeypatch.setattr(
        ops_mod, "get_ops", Mock(side_effect=RuntimeError("not configured"))
    )
    spy = Mock(return_value=(None, None))
    monkeypatch.setattr(kda_pkg, "chunk_kda_naive", spy)
    warn = Mock()
    monkeypatch.setattr(kda_pkg, "logger", Mock(warning=warn))
    kda_pkg.chunk_kda(torch.zeros(1))
    assert spy.call_count == 1
    assert warn.called
    assert "naive" in warn.call_args[0][0]


def test_entry_standalone_forced_fla_raises_when_missing(monkeypatch):
    import torchturbo.ops.kda as kda_pkg
    ops_mod = sys.modules["torchturbo.ops.ops"]

    monkeypatch.setitem(sys.modules, "fla.ops.kda", None)
    monkeypatch.setattr(
        ops_mod, "get_ops", Mock(side_effect=RuntimeError("not configured"))
    )
    with pytest.raises(ImportError):
        kda_pkg.chunk_kda(torch.zeros(1), impl="fla_chunk_kda")


def test_entry_naive_matches_direct_naive(monkeypatch):
    import torchturbo.ops.kda as kda_pkg
    ops_mod = sys.modules["torchturbo.ops.ops"]
    from torchturbo.ops.kda.chunk_kda_naive import chunk_kda_naive

    monkeypatch.setattr(
        ops_mod, "get_ops", Mock(side_effect=RuntimeError("not configured"))
    )
    torch.manual_seed(0)
    b, t, h, d = 1, 8, 2, 4
    q, k, v, g = (torch.randn(b, t, h, d) for _ in range(4))
    beta = torch.randn(b, t, h)
    a_log = torch.randn(h)
    dt_bias = torch.randn(h * d)
    common = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    out_entry, state_entry = kda_pkg.chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, impl="naive_chunk_kda", **common
    )
    out_direct, state_direct = chunk_kda_naive(
        q=q, k=k, v=v, g=g, beta=beta, transpose_state_layout=True, **common
    )
    assert torch.allclose(out_entry, out_direct)
    assert torch.allclose(state_entry, state_direct)
