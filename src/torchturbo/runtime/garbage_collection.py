"""Limit cyclic GC scanning to objects created during a training scope."""

from collections.abc import Iterator
from contextlib import contextmanager
import gc


@contextmanager
def freeze_garbage_collection() -> Iterator[None]:
    """Exclude initialization survivors without changing automatic GC settings.

    An existing frozen heap belongs to the caller and is left unchanged.
    Scopes must be nested, as with the engine's training lifecycle.
    """
    owns_freeze = gc.get_freeze_count() == 0
    if owns_freeze:
        gc.collect()
        gc.freeze()
    try:
        yield
    finally:
        if owns_freeze:
            gc.unfreeze()
