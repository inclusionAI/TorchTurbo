"""CPU parameter interfaces and node-shared storage.

Shared parameters create native node groups on first materialization.
materialize_, load_, and save coordinate errors and completion across WORLD.
Readers and writers own file formats, names, and checkpoint metadata. Shared
parameters select the node containing rank zero to save.
"""

from .base import CPUResidentParameter
from .node_shared import NodeSharedCPUResidentParameter

__all__ = ["CPUResidentParameter", "NodeSharedCPUResidentParameter"]
