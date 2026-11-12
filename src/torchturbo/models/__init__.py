"""torchturbo models — each model in its own directory.
"""

from ..engine.config import ParameterGroup
from .base import TurboModel, TurboVLMModel
from .registry import get_all_models, get_model, register_model, _register_builtins

# Auto-register built-in models
_register_builtins()

__all__ = [
    "ParameterGroup",
    "TurboModel",
    "TurboVLMModel",
    "get_all_models",
    "get_model",
    "register_model",
]
