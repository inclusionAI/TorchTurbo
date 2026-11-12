"""Custom functools utilities for torch_turbo."""

import functools
import torch


def _same_tensor(a, b):
    """Check if two tensors share the same storage and view (O(1), no data transfer)."""
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.storage_offset() == b.storage_offset()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def _args_match(old_args, new_args, old_kwargs, new_kwargs):
    """Check if args/kwargs match, using same-storage check for tensors."""
    if len(old_args) != len(new_args) or len(old_kwargs) != len(new_kwargs):
        return False
    for a, b in zip(old_args, new_args):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not _same_tensor(a, b):
                return False
        elif a != b:
            return False
    for k in old_kwargs:
        if k not in new_kwargs:
            return False
        a, b = old_kwargs[k], new_kwargs[k]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not _same_tensor(a, b):
                return False
        elif a != b:
            return False
    return True


def cache(user_function):
    """
    Simple single-item cache decorator for functions with tensor arguments.

    Only cache-hits when tensors share the same storage and view (data_ptr +
    shape + stride + offset match), so no data transfer is needed.
    Suitable for deterministic functions like get_cu_seqlens_npu.

    Only keeps the most recent cache entry. When args change, old cache is cleared.

    Example:
        @cache
        def get_cu_seqlens_npu(attention_mask):
            ...

    Args:
        user_function: The function to cache.

    Returns:
        Wrapped function with single-item cache.
    """
    _cache_args = None
    _cache_kwargs = None
    _cache_value = None

    @functools.wraps(user_function)
    def wrapper(*args, **kwargs):
        nonlocal _cache_args, _cache_kwargs, _cache_value

        if _cache_args is not None and _args_match(_cache_args, args, _cache_kwargs, kwargs):
            return _cache_value

        result = user_function(*args, **kwargs)
        _cache_args = args
        _cache_kwargs = kwargs
        _cache_value = result

        return _cache_value

    def cache_clear():
        nonlocal _cache_args, _cache_kwargs, _cache_value
        _cache_args = None
        _cache_kwargs = None
        _cache_value = None

    wrapper.cache_clear = cache_clear

    return wrapper
