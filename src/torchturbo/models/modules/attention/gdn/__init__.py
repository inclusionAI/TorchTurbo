"""Shared GatedDeltaNet sequence-parallel forwards."""

from .forward import build_gdn_sp_forward

__all__ = [
    "build_gdn_sp_forward",
]
