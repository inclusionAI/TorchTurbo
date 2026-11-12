"""Check GC ownership independently of model and framework imports."""

import gc
import importlib.util
from pathlib import Path
import weakref

import pytest


_source = Path(__file__).parents[2] / "src/torchturbo/engine/garbage_collection.py"
_spec = importlib.util.spec_from_file_location("training_gc_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GarbageCollector = _module.GarbageCollector


class Node:
    pass


def cycle_reference():
    node = Node()
    node.self = node
    return weakref.ref(node)


def test_inactive_scope_does_not_change_gc(monkeypatch):
    def unexpected_call():
        raise AssertionError("GC changed before explicit activation")

    with monkeypatch.context() as patch:
        for name in ("collect", "freeze", "unfreeze"):
            patch.setattr(gc, name, unexpected_call)
        with GarbageCollector():
            pass


def test_new_cycles_are_collected_with_unchanged_settings():
    thresholds, enabled = gc.get_threshold(), gc.isenabled()
    with GarbageCollector() as collector:
        collector.freeze()
        assert gc.get_freeze_count() > 0
        assert gc.get_threshold() == thresholds
        assert gc.isenabled() == enabled
        reference = cycle_reference()
        gc.collect(2)
        assert reference() is None
    assert gc.get_freeze_count() == 0
    assert gc.get_threshold() == thresholds
    assert gc.isenabled() == enabled


def test_frozen_cycles_become_collectable_after_exit():
    node = Node()
    node.self = node
    reference = weakref.ref(node)
    with GarbageCollector() as collector:
        collector.freeze()
        del node
        gc.collect(2)
        assert reference() is not None
    gc.collect(2)
    assert reference() is None


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_exception_restores_owned_heap(error):
    with pytest.raises(error):
        with GarbageCollector() as collector:
            collector.freeze()
            raise error()
    assert gc.get_freeze_count() == 0


def test_repeated_freeze_collects_only_once(monkeypatch):
    calls = []
    original = gc.collect

    def collect():
        calls.append("collect")
        return original()

    monkeypatch.setattr(gc, "collect", collect)
    with GarbageCollector() as collector:
        collector.freeze()
        collector.freeze()
    assert calls == ["collect"]


def test_nested_collectors_preserve_outer_owner():
    with GarbageCollector() as outer:
        outer.freeze()
        with GarbageCollector() as inner:
            inner.freeze()
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0


def test_nested_same_collector_does_not_unfreeze_early():
    with GarbageCollector() as collector:
        collector.freeze()
        with collector:
            collector.freeze()
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0


def test_collector_can_be_reused_after_exception():
    collector = GarbageCollector()
    for _ in range(2):
        with pytest.raises(RuntimeError):
            with collector:
                assert gc.get_freeze_count() == 0
                collector.freeze()
                raise RuntimeError()
        assert gc.get_freeze_count() == 0


def test_existing_heap_is_left_to_the_caller():
    gc.collect()
    gc.freeze()
    try:
        with GarbageCollector() as collector:
            collector.freeze()
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()


def test_disabled_automatic_gc_stays_disabled():
    enabled = gc.isenabled()
    gc.disable()
    try:
        with GarbageCollector() as collector:
            collector.freeze()
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if enabled:
            gc.enable()


def test_freeze_requires_a_scope():
    with pytest.raises(RuntimeError, match="active GarbageCollector scope"):
        GarbageCollector().freeze()
