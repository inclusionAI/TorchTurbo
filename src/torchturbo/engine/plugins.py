"""Plugin system: ``--tt_plugin`` files register ``Plugin`` subclasses; ``PluginManager``
loads them and hands each to the ``PluginConsumer`` that recognizes it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ..logging import get_logger

logger = get_logger(__name__)


class Plugin:
    """Base for every ``--tt_plugin``; ``self.config`` is the bound ``plugin:config``.
    A plugin's contract is its base class (e.g. :class:`DatasetPlugin`); consumers
    match plugins by ``isinstance``."""

    def __init__(self, config: str | None = None) -> None:
        self.config: str | None = config


class PluginConsumer:
    """A component that uses plugins: ``matches`` recognizes the plugins it consumes,
    ``consume`` puts a recognized plugin to use."""

    def matches(self, plugin: Plugin) -> bool:
        raise NotImplementedError

    def consume(self, plugin: Plugin) -> None:
        raise NotImplementedError


class PluginManager:
    """Loads ``--tt_plugin`` files and hands each loaded plugin to a :class:`PluginConsumer`."""

    def __init__(self, paths: list[str] | None = None) -> None:
        self.plugins: list[Plugin] = []
        self.load(paths or [])

    def load(self, paths: list[str]) -> None:
        """Import each plugin file and instantiate its registered plugin classes (deduped by class)."""
        for entry in paths:
            plugin_path, sep, config = entry.partition(":")
            plugin_path = os.path.abspath(os.path.expanduser(plugin_path))
            if not os.path.isfile(plugin_path):
                raise FileNotFoundError(f"--tt_plugin does not point to a file: {plugin_path}")
            bound = os.path.abspath(os.path.expanduser(config)) if sep and config.strip() else None
            module = self._import_plugin(plugin_path)

            for cls in vars(module).values():
                if not (isinstance(cls, type) and issubclass(cls, Plugin) and getattr(cls, "plugin_name", None)):
                    continue
                if any(type(x) is cls for x in self.plugins):
                    continue
                self.plugins.append(cls(config=bound))
                logger.info_rank0(
                    "Initialized plugin '%s' (%s)",
                    cls.plugin_name, cls.__name__,
                )

    def try_consume(self, consumer: PluginConsumer) -> None:
        """Hand each loaded plugin to the consumer if it ``matches``; failures are logged and skipped."""
        for plugin in self.plugins:
            if not consumer.matches(plugin):
                continue
            try:
                consumer.consume(plugin)
            except Exception:
                logger.warning_rank0(
                    "Plugin consumer %s failed on plugin '%s' (%s); skipping.",
                    type(consumer).__name__,
                    getattr(plugin, "plugin_name", "?"),
                    type(plugin).__name__,
                    exc_info=True,
                )

    @staticmethod
    def _import_plugin(path: str):
        """Import (once) the plugin .py at ``path`` and return its module object."""
        plugin_dir = os.path.dirname(path)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        module_name = f"torchturbo_plugin.{os.path.splitext(os.path.basename(path))[0]}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load plugin module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def register_plugin(name: str) -> Callable:
    """Decorator naming a ``Plugin`` subclass ``name``."""
    def decorator(cls: type[Plugin]) -> type[Plugin]:
        if not issubclass(cls, Plugin):
            raise TypeError(f"{cls.__name__} must subclass Plugin")
        cls.plugin_name = name
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Plugin API surface (consumers down-cast by plugin base class)
# ---------------------------------------------------------------------------


class CheckpointExportPlugin(Plugin):
    """Consume completed HF exports on global rank zero.

    Other ranks wait for the callback. Finish reading before returning, since
    later saves or rotation may remove the directory. Failures are logged and
    training continues.
    """

    def on_hf_export(self, export_dir: str) -> None:
        """Consume an export whose weights and metadata have been saved."""
        raise NotImplementedError


@dataclass(frozen=True)
class DatasetPrepareContext:
    """Context for DatasetPlugin"""

    encode_fn: Callable[[Mapping], dict]                         # encode function to produce input_ids
    preprocess: Callable[[Any], Any]                             # normalize datasets
    max_length: int                                              # length cap
    show_dataset: Optional[Callable[[Any, Any], Any]] = None     # auxiluary logging function
    cache_path: Optional[str] = None                             # cache dir
    num_proc: int = 1                                            # parallel encode workers
    seed: int = 42


class DatasetPlugin(Plugin):
    """Supplies a training dataset per name: build the complete dataset, or supply
    raw rows. Provided datasets are pack-grouped (default collator) or one packed
    row per item; the latter override :meth:`get_collate_fn`."""

    def get_collate_fn(self) -> Callable[[list], dict] | None:
        """Batch collator for the provided dataset; ``None`` keeps the framework
        collator. Override for one-packed-row-per-item plugins."""
        return None

    def dataset_names(self) -> list[str]:
        raise NotImplementedError

    def dataset_info(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def provides_dataset(self, name: str) -> bool:
        """Whether this plugin provides the complete dataset itself; default False."""
        return False

    def build_dataset(self, name: str, ctx: DatasetPrepareContext) -> tuple[Any, Any]:
        """Return ``(train_dataset, val_dataset)``; only called when ``provides_dataset`` is True."""
        raise NotImplementedError

    def samples(self, name: str, cache_dir: str | None = None) -> Any:
        """Return raw rows (HF Dataset); only used when ``provides_dataset`` is False."""
        raise NotImplementedError


__all__ = [
    "Plugin",
    "PluginConsumer",
    "PluginManager",
    "CheckpointExportPlugin",
    "DatasetPlugin",
    "DatasetPrepareContext",
    "register_plugin",
]
