"""Check the process-wide GC lifetime without importing device runtimes."""

import gc
import importlib.util
from pathlib import Path
import weakref

import pytest


_source = Path(__file__).parents[2] / "src/torchturbo/runtime/garbage_collection.py"
_spec = importlib.util.spec_from_file_location("training_gc_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
freeze_garbage_collection = _module.freeze_garbage_collection


class Node:
    pass


def cycle_reference():
    node = Node()
    node.self = node
    return weakref.ref(node)


def test_new_cycles_are_collected_with_unchanged_gc_settings():
    thresholds, enabled = gc.get_threshold(), gc.isenabled()
    with freeze_garbage_collection():
        assert gc.get_freeze_count() > 0
        assert gc.get_threshold() == thresholds
        assert gc.isenabled() == enabled
        reference = cycle_reference()
        gc.collect(2)
        assert reference() is None
    assert gc.get_freeze_count() == 0
    assert gc.get_threshold() == thresholds
    assert gc.isenabled() == enabled


def test_frozen_cycles_become_collectable_after_scope_exit():
    node = Node()
    node.self = node
    reference = weakref.ref(node)
    with freeze_garbage_collection():
        del node
        gc.collect(2)
        assert reference() is not None
    gc.collect(2)
    assert reference() is None


def test_exception_unfreezes_owned_heap():
    with pytest.raises(ValueError, match="training failed"):
        with freeze_garbage_collection():
            raise ValueError("training failed")
    assert gc.get_freeze_count() == 0


def test_nested_scope_preserves_outer_freeze():
    with freeze_garbage_collection():
        with freeze_garbage_collection():
            assert gc.get_freeze_count() > 0
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0


def test_existing_freeze_is_left_to_the_caller():
    gc.collect()
    gc.freeze()
    try:
        with freeze_garbage_collection():
            pass
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()


def test_disabled_automatic_gc_is_not_enabled_by_the_scope():
    enabled = gc.isenabled()
    gc.disable()
    try:
        with freeze_garbage_collection():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if enabled:
            gc.enable()
