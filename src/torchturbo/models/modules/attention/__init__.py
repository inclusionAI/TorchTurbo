"""Shared attention execution and sequence-parallel wrappers."""

from .interface import attention_forward

__all__ = ["attention_forward", "ulysses"]
