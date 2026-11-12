"""Limit cyclic GC scanning to objects created during a training scope."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import gc
from typing import Any


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


@contextmanager
def freeze_training_gc(trainer: Any) -> Iterator[None]:
    """Freeze after model and optimizer setup, then unfreeze on every exit."""
    from transformers import TrainerCallback

    with ExitStack() as scope:
        class FreezeOnTrainBegin(TrainerCallback):
            def on_train_begin(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> None:
                scope.enter_context(freeze_garbage_collection())

        callback = FreezeOnTrainBegin()
        trainer.add_callback(callback)
        try:
            yield
        finally:
            trainer.remove_callback(callback)
