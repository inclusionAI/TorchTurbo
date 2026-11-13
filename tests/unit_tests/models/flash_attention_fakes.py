"""CPU stand-ins for the flash-attention pieces HF imports lazily."""

from collections.abc import Callable
from typing import Any

import transformers
from packaging.version import Version
from transformers import modeling_flash_attention_utils as flash_utils

# Transformers 5.4.0 inserted a KV-cache kernel after the varlen kernel.
_HAS_KV_CACHE_KERNEL = Version(transformers.__version__) >= Version("5.4.0")


def lazy_flash_attention_kernels(
    flash_fn: Callable | None, varlen_fn: Callable | None,
) -> tuple[Callable | None, ...]:
    """Build the kernel tuple returned by ``lazy_import_flash_attention``.

    Tests replace the dense and varlen kernels and keep HF's padding helpers,
    in the tuple layout of the installed transformers version.
    """
    if _HAS_KV_CACHE_KERNEL:
        return (flash_fn, varlen_fn, None, flash_utils._pad_input, flash_utils._unpad_input)
    return (flash_fn, varlen_fn, flash_utils._pad_input, flash_utils._unpad_input)


def process_flash_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Map HF's flash-attention kwargs onto the fake kernels' arguments.

    Since transformers 5.4.0, HF routes the varlen maxima through this hook
    rather than passing them to the kernel, so they are forwarded when given.
    """
    flash_kwargs = {"causal": kwargs["is_causal"], "softmax_scale": kwargs["softmax_scale"]}
    for key in ("max_seqlen_q", "max_seqlen_k"):
        if key in kwargs:
            flash_kwargs[key] = kwargs[key]
    return flash_kwargs
