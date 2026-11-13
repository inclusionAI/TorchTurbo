"""Check GC scheduling, memory pressure, and process-wide state ownership."""

import gc
from types import SimpleNamespace
import weakref

import pytest

from torchturbo.engine import garbage_collection as training_gc
from torchturbo.engine.garbage_collection import GarbageCollector
from torchturbo.runtime import TurboDevice


_memory_pressure_probe = training_gc._under_memory_pressure


@pytest.fixture(autouse=True)
def restore_gc(monkeypatch):
    enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    gc.enable()
    monkeypatch.setattr(training_gc, "_under_memory_pressure", lambda: False)
    yield
    gc.set_threshold(*thresholds)
    if enabled:
        gc.enable()
    else:
        gc.disable()


class Node:
    pass


def test_inactive_scope_does_not_change_gc(monkeypatch):
    def unexpected_call():
        raise AssertionError("GC changed before explicit activation")

    with monkeypatch.context() as patch:
        for name in ("collect", "freeze", "unfreeze", "enable", "disable"):
            patch.setattr(gc, name, unexpected_call)
        with GarbageCollector():
            pass


def test_step_restores_gc_without_changing_thresholds_or_freezing(monkeypatch):
    def unexpected_collect():
        raise AssertionError("No full collection without memory pressure")

    monkeypatch.setattr(gc, "collect", unexpected_collect)
    thresholds = gc.get_threshold()
    frozen = gc.get_freeze_count()
    with GarbageCollector() as collector:
        for _ in range(2):
            collector.begin_step()
            assert not gc.isenabled()
            assert gc.get_threshold() == thresholds
            assert gc.get_freeze_count() == frozen
            collector.end_step()
            assert gc.isenabled()
    assert gc.get_threshold() == thresholds
    assert gc.get_freeze_count() == frozen


def test_pressure_collects_old_and_new_cycles_only_after_step(monkeypatch):
    monkeypatch.setattr(training_gc, "_under_memory_pressure", lambda: True)
    old = Node()
    old.self = old
    old_reference = weakref.ref(old)
    gc.collect(2)
    with GarbageCollector() as collector:
        collector.begin_step()
        new = Node()
        new.self = new
        new_reference = weakref.ref(new)
        del old, new
        objects = [[] for _ in range(gc.get_threshold()[0] + 1)]
        assert old_reference() is not None
        assert new_reference() is not None
        collector.end_step()
        assert gc.isenabled()
        assert old_reference() is None
        assert new_reference() is None
        assert objects


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_exception_restores_gc_in_unfinished_step(error):
    with pytest.raises(error):
        with GarbageCollector() as collector:
            collector.begin_step()
            raise error()
    assert gc.isenabled()


def test_failed_pressure_check_leaves_gc_enabled(monkeypatch):
    def failed_probe():
        raise RuntimeError("memory query failed")

    monkeypatch.setattr(training_gc, "_under_memory_pressure", failed_probe)
    with pytest.raises(RuntimeError, match="memory query failed"):
        with GarbageCollector() as collector:
            collector.begin_step()
            collector.end_step()
    assert gc.isenabled()


def test_nested_collectors_preserve_outer_disabled_state():
    with GarbageCollector() as outer:
        outer.begin_step()
        with GarbageCollector() as inner:
            inner.begin_step()
            inner.end_step()
        assert not gc.isenabled()
        outer.end_step()
        assert gc.isenabled()


def test_nested_same_collector_does_not_restore_early():
    with GarbageCollector() as collector:
        collector.begin_step()
        with collector:
            assert not gc.isenabled()
        assert not gc.isenabled()
        collector.end_step()
    assert gc.isenabled()


def test_duplicate_step_events_do_not_overwrite_gc_ownership(monkeypatch):
    calls = []
    monkeypatch.setattr(training_gc, "_under_memory_pressure", lambda: True)
    monkeypatch.setattr(gc, "collect", lambda: calls.append("collect"))
    with GarbageCollector() as collector:
        collector.begin_step()
        collector.begin_step()
        collector.end_step()
        collector.end_step()
    assert gc.isenabled()
    assert calls == ["collect"]


def test_collector_can_be_reused_after_exception():
    collector = GarbageCollector()
    for _ in range(2):
        with pytest.raises(RuntimeError):
            with collector:
                assert gc.isenabled()
                collector.begin_step()
                raise RuntimeError()
        assert gc.isenabled()


def test_existing_frozen_heap_is_left_to_the_caller():
    gc.collect()
    gc.freeze()
    frozen = gc.get_freeze_count()
    try:
        with GarbageCollector() as collector:
            collector.begin_step()
            collector.end_step()
        assert gc.get_freeze_count() == frozen
    finally:
        gc.unfreeze()


def test_caller_disabled_gc_stays_disabled_without_collection(monkeypatch):
    def unexpected_probe():
        raise AssertionError("Caller owns disabled GC")

    monkeypatch.setattr(training_gc, "_under_memory_pressure", unexpected_probe)
    gc.disable()
    with GarbageCollector() as collector:
        collector.begin_step()
        collector.end_step()
        assert not gc.isenabled()
        collector.begin_step()
    assert not gc.isenabled()


def test_begin_step_requires_a_scope():
    with pytest.raises(RuntimeError, match="active GarbageCollector scope"):
        GarbageCollector().begin_step()
    assert gc.isenabled()


@pytest.mark.parametrize("device", [TurboDevice.CUDA, TurboDevice.NPU, TurboDevice.CPU])
@pytest.mark.parametrize(
    ("host_free", "device_free", "reserved", "allocated", "expected"),
    [
        (99, 500, 500, 500, True),
        (500, 99, 901, 901, True),
        (500, 100, 900, 900, False),
        (500, 10, 990, 500, False),
        (100, 500, 500, 500, False),
    ],
)
def test_memory_pressure_accounts_for_host_device_and_reusable_cache(
    monkeypatch, device, host_free, device_free, reserved, allocated, expected
):
    monkeypatch.setattr(training_gc.CpuRuntime, "memory_info", lambda self: (host_free, 1000))

    def memory_info():
        assert device != TurboDevice.CPU
        return device_free, 1000

    runtime = SimpleNamespace(
        turbo_device=device,
        memory_info=memory_info,
        memory_reserved=lambda: reserved,
        memory_allocated=lambda: allocated,
    )
    monkeypatch.setattr(training_gc, "get_device_runtime", lambda: runtime)
    if device == TurboDevice.CPU:
        expected = host_free < 100
    assert _memory_pressure_probe() == expected
