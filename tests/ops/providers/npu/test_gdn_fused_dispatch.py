"""Adaptive GDN dispatch tests: probe quadrants drive the wrapper decision.

These tests mock the ``fused_fwd_usable`` / ``fused_bwd_usable`` probes (both
``lru_cache``-wrapped) to cover the dispatch quadrants without running any
real fused kernel:

* both probes true     -> AdaptiveFunction with fused_backward=True
* fwd true, bwd false  -> AdaptiveFunction with fused_backward=False
                          (the fla-npu 26.9.1 quadrant)
* fwd false           -> composed ChunkGatedDeltaRuleFunction takes over,
                          and the bwd probe is never consulted
* both probes true but outside the fused envelope (V=256) -> composed chain
* ``_USE_FUSED_GDN`` escape hatch off -> composed chain

The Function classes themselves are replaced by recorders, so nothing
launches on device; only the wrapper's dispatch logic (argument passing +
the trailing ``fused_backward`` flag) is under test. A separate test locks
the probes' once-per-process caching semantics against ``_probe_pair``.

These tests import the vendored adapter (which needs ``fla_npu`` /
``torch_npu``), so they run on an NPU install; the importorskip guards make
them a no-op elsewhere.
"""

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

import torch  # noqa: E402

import torchturbo.ops.providers.npu._fla_gdn as gdn_pkg  # noqa: E402
import torchturbo.ops.providers.npu._fla_gdn.chunk_gated_delta_rule_fused_function as fused_mod  # noqa: E402


def _fake_tensors(batch=1, tokens=1057, value_dim=128):
    """Time-first inputs in the wrapper's fla-like calling convention."""
    q = torch.randn(batch, tokens, 16, 128)
    k = torch.randn(batch, tokens, 16, 128)
    v = torch.randn(batch, tokens, 32, value_dim)
    g = -torch.rand(batch, tokens, 32)
    beta = torch.rand(batch, tokens, 32)
    return q, k, v, g, beta


class _Recorder:
    """Mimics a torch.autograd.Function: ``.apply(...)`` records the call."""

    def __init__(self, result=(torch.zeros(1), None)):
        self.result = result
        self.calls = []

    def apply(self, *args, **kwargs):  # noqa: D102 (stands in for Function)
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def clean_probe_cache():
    """Clear the lru_cache on both probes around every test."""
    fused_mod.fused_fwd_usable.cache_clear()
    fused_mod.fused_bwd_usable.cache_clear()
    yield
    fused_mod.fused_fwd_usable.cache_clear()
    fused_mod.fused_bwd_usable.cache_clear()


def _run_wrapper(monkeypatch, tensors, use_fused=True, **extra):
    adaptive_rec = _Recorder()
    composed_rec = _Recorder()
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleAdaptiveFunction", adaptive_rec)
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFunction", composed_rec)
    if use_fused is not None:
        monkeypatch.setattr(gdn_pkg, "_USE_FUSED_GDN", use_fused)
    q, k, v, g, beta = tensors
    o, final_state = gdn_pkg.ascendc_chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        use_qk_l2norm_in_kernel=False,
        chunk_size=64,
        **extra,
    )
    return adaptive_rec, composed_rec, o, final_state


def test_both_probes_true_full_fusion(monkeypatch, clean_probe_cache):
    monkeypatch.setattr(gdn_pkg, "fused_fwd_usable", lambda: True)
    monkeypatch.setattr(gdn_pkg, "fused_bwd_usable", lambda: True)
    adaptive, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(adaptive.calls) == 1
    assert len(composed.calls) == 0
    args = adaptive.calls[0][0]
    assert len(args) == 15  # composed-chain 14 inputs + trailing fused flag
    # q/k/v were transposed to head-first [B,H,T,D] on the way in
    assert tuple(args[0].shape) == (1, 16, 1057, 128)
    assert tuple(args[2].shape) == (1, 32, 1057, 128)
    assert args[5] == pytest.approx(128 ** -0.5)  # scale defaults to dk ** -0.5
    assert args[13] == 64  # chunk_size
    assert args[14] is True  # fused_backward


def test_bwd_probe_failed_mixed_path(monkeypatch, clean_probe_cache):
    """fla-npu 26.9.1 quadrant: fused fwd usable, fused bwd kernel missing."""
    monkeypatch.setattr(gdn_pkg, "fused_fwd_usable", lambda: True)
    monkeypatch.setattr(gdn_pkg, "fused_bwd_usable", lambda: False)
    adaptive, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(adaptive.calls) == 1
    assert len(composed.calls) == 0
    assert adaptive.calls[0][0][14] is False  # composed backward inside


def test_fwd_probe_failed_full_fallback(monkeypatch, clean_probe_cache):
    """A failed fwd probe must fall to the composed chain without consulting
    the bwd probe (bwd implies fwd)."""
    monkeypatch.setattr(gdn_pkg, "fused_fwd_usable", lambda: False)

    def _must_not_run():
        raise AssertionError("bwd probe consulted although fwd failed")

    monkeypatch.setattr(gdn_pkg, "fused_bwd_usable", _must_not_run)
    adaptive, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(adaptive.calls) == 0
    assert len(composed.calls) == 1
    args = composed.calls[0][0]
    assert len(args) == 15  # original 14-input signature + save_intermediates
    assert isinstance(args[-1], bool)  # save_backward_intermediates slot


def test_outside_envelope_ignores_probes(monkeypatch, clean_probe_cache):
    """V=256 is in the fused fwd's own envelope but outside the stricter
    bwd envelope (K=V=128), so the wrapper must stay composed even with both
    probes green."""
    monkeypatch.setattr(gdn_pkg, "fused_fwd_usable", lambda: True)
    monkeypatch.setattr(gdn_pkg, "fused_bwd_usable", lambda: True)
    adaptive, composed, *_ = _run_wrapper(
        monkeypatch, _fake_tensors(value_dim=256)
    )
    assert len(adaptive.calls) == 0
    assert len(composed.calls) == 1


def test_escape_hatch_disables_fusion(monkeypatch, clean_probe_cache):
    """``_USE_FUSED_GDN`` (env TORCHTURBO_GDN_USE_FUSED=0) forced off keeps
    the wrapper on the composed chain regardless of the probes."""
    monkeypatch.setattr(gdn_pkg, "fused_fwd_usable", lambda: True)
    monkeypatch.setattr(gdn_pkg, "fused_bwd_usable", lambda: True)
    adaptive, composed, *_ = _run_wrapper(
        monkeypatch, _fake_tensors(), use_fused=False
    )
    assert len(adaptive.calls) == 0
    assert len(composed.calls) == 1


def test_probes_cached_once_per_process(monkeypatch, clean_probe_cache):
    """Each probe direction runs its tiny launch at most once, however often
    it is asked (lru_cache around ``_probe_pair``)."""
    probe_calls = []

    def fake_probe_pair(probe_bwd):
        probe_calls.append(probe_bwd)
        if probe_bwd:
            return False, "aclnn 161002"
        return True, ""

    monkeypatch.setattr(fused_mod, "_probe_pair", fake_probe_pair)
    assert fused_mod.fused_fwd_usable() is True
    assert fused_mod.fused_fwd_usable() is True
    assert fused_mod.fused_bwd_usable() is False
    assert fused_mod.fused_fwd_usable() is True
    assert probe_calls == [False, True]


def test_probe_failure_caches_unusable(monkeypatch, clean_probe_cache):
    """A launch failure (aclnn 161002-style) inside ``_probe_pair`` returns
    ``(False, reason)``; the cached result must mark the direction unusable,
    never crash the caller -- importability alone must not mark it usable."""
    def failing_probe_pair(probe_bwd):
        if probe_bwd:
            return False, "RuntimeError('aclnn 161002: launch failed')"
        return True, ""

    monkeypatch.setattr(fused_mod, "_probe_pair", failing_probe_pair)
    # a 26.9.1-like install: importable + fused fwd works, fused bwd does not
    assert fused_mod.fused_fwd_usable() is True
    assert fused_mod.fused_bwd_usable() is False
    assert fused_mod.fused_bwd_usable() is False


def test_fused_envelope_supported():
    assert fused_mod.fused_envelope_supported(128, 128, 64) is True
    assert fused_mod.fused_envelope_supported(128, 256, 64) is False
    assert fused_mod.fused_envelope_supported(128, 128, 128) is False
    assert fused_mod.fused_envelope_supported(64, 128, 64) is False
