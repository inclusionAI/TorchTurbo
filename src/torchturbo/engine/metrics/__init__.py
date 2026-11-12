"""Engine metrics: running-mean tracker + online MFU/HFU estimation.

:class:`MFUEstimator` asks the live model (``model.get_flops_class()``) for a
forward-FLOP recipe, sums the components per micro-batch, and derives MFU/HFU
from the window wall-clock time. Models own their recipes (subclass
``BaseModelFlops`` / ``LLMFlops`` / ``VLMFlops`` in :mod:`flops`); :mod:`op_flops`
provides the FLOP primitives.
"""

from __future__ import annotations

from .tracker import MetricTracker
from .flops import (
    BaseModelFlops,
    Component,
    ComputeProfile,
    DenseLLMFlops,
    LLMFlops,
    MlaLLMFlops,
    MoELLMFlops,
    VLMFlops,
)
from .mfu_estimator import MFUEstimator

__all__ = [
    "MetricTracker",
    "MFUEstimator",
    # Flops data types + reusable recipe bases (models subclass these).
    "BaseModelFlops",
    "Component",
    "ComputeProfile",
    "DenseLLMFlops",
    "LLMFlops",
    "MlaLLMFlops",
    "MoELLMFlops",
    "VLMFlops",
]