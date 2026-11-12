"""Scope frozen initialization objects to a Transformers trainer invocation."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from ..runtime.garbage_collection import freeze_garbage_collection


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
