"""Attention utilities."""

# TODO: The current model paths patch attention implementation directly, so the
# generic attention wrappers are not used. If we need to support more models or
# reduce per-model patching, this package should be reorganized around a proper
# attention factory that registers backends (for example FA2, SDPA, Ulysses
# variants) and creates them through one consistent interface.

__all__ = ["ulysses"]
