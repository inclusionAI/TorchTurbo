"""Own a process-wide GC freeze within a nested training lifetime."""

from __future__ import annotations

from contextlib import AbstractContextManager
import gc
from types import TracebackType


class GarbageCollector(AbstractContextManager["GarbageCollector"]):
    """Freeze initialization survivors when requested, then restore on exit.

    Entering a scope does not change GC behavior. Call ``freeze`` after model
    and optimizer setup. Automatic GC settings stay unchanged, and a frozen
    heap owned by the caller is left alone. Scopes must be nested because
    Python's GC state is process-wide.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._owns_freeze = False

    def __enter__(self) -> GarbageCollector:
        self._depth += 1
        return self

    def freeze(self) -> None:
        """Exclude existing objects from cyclic GC until the outer scope exits."""
        if self._depth == 0:
            raise RuntimeError("freeze() requires an active GarbageCollector scope")
        if self._owns_freeze or gc.get_freeze_count():
            return
        gc.collect()
        gc.freeze()
        self._owns_freeze = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if self._depth == 0 and self._owns_freeze:
            gc.unfreeze()
            self._owns_freeze = False
