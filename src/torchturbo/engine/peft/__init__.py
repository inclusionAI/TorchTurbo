"""PEFT (LoRA) integration for torchturbo.

This package provides LoRA support for MoE models that use
:class:`~torchturbo.models.modules.moe.experts.GroupedMLP`, which relies
on fused weight tensors and grouped GEMM kernels instead of standard
``nn.Linear`` layers.

Quick start
-----------
::

    from peft import LoraConfig
    from torchturbo.engine.peft import apply_peft, register_grouped_mlp_lora

    # Option 1: Use apply_peft (recommended — handles everything)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["experts"],  # targets GroupedMLP modules
        lora_dropout=0.0,
        bias="none",
    )
    model = apply_peft(model, lora_config)

    # Option 2: Manual registration before get_peft_model
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["experts"])
    register_grouped_mlp_lora(lora_config)
    model = get_peft_model(model, lora_config)

Public API
----------
.. autofunction:: apply_peft
.. autofunction:: register_grouped_mlp_lora
.. autofunction:: expand_all_linear_targets
.. autoclass:: torchturbo.engine.peft.lora.GroupedMLPLoRA
    :members:
"""

from __future__ import annotations

from typing import Any, Union

from ...logging import get_logger
from .weights import (
    apply_lora_delta,
    build_lora_delta_map,
    compute_lora_delta_linear,
    compute_lora_delta_grouped_mlp_w1,
    compute_lora_delta_grouped_mlp_w2,
)

logger = get_logger(__name__)

__all__ = [
    "apply_peft",
    "expand_all_linear_targets",
    "register_grouped_mlp_lora",
    "build_lora_delta_map",
    "apply_lora_delta",
    "compute_lora_delta_linear",
    "compute_lora_delta_grouped_mlp_w1",
    "compute_lora_delta_grouped_mlp_w2",
]


def register_grouped_mlp_lora(
    lora_config: Any,
) -> None:
    """Register :class:`GroupedMLPLoRA` as the custom LoRA handler for
    :class:`~torchturbo.models.modules.moe.experts.GroupedMLP` in the given
    ``LoraConfig``.

    This uses PEFT's experimental ``_register_custom_module`` API so that
    when :func:`peft.get_peft_model` encounters a ``GroupedMLP`` module
    that matches ``target_modules``, it will wrap it with
    ``GroupedMLPLoRA`` instead of raising an error.

    **Must be called before** :func:`peft.get_peft_model` or
    :func:`peft.inject_adapter_in_model`.

    Parameters
    ----------
    lora_config : LoraConfig
        The LoRA configuration to register the custom module with.
    """
    from .lora import GroupedMLPLoRA
    from torchturbo.models.modules.moe.experts import GroupedMLP

    lora_config._register_custom_module({GroupedMLP: GroupedMLPLoRA})
    logger.info_rank0("Registered GroupedMLPLoRA for GroupedMLP")


def expand_all_linear_targets(model: Any, peft_config: Any) -> None:
    """Expand ``target_modules="all-linear"`` to include custom module types.

    PEFT's built-in ``_maybe_include_all_linear_layers`` only expands
    ``"all-linear"`` to ``nn.Linear`` and ``Conv1D`` module names.  Custom
    modules registered via ``_register_custom_module`` (e.g.
    :class:`GroupedMLP`) are not ``nn.Linear`` subclasses, so they get
    missed.

    This function:

    1. Calls PEFT's ``_maybe_include_all_linear_layers`` to expand
       ``"all-linear"`` into the full set of linear module names.
    2. Walks the model to find instances of each custom module class
       registered in ``peft_config._custom_modules``, and appends their
       short names (e.g. ``"experts"``) to ``peft_config.target_modules``
       so that PEFT's suffix-matching logic picks them up.

    This is a **no-op** when ``target_modules`` is not ``"all-linear"``
    or when no custom modules are registered.

    Parameters
    ----------
    model : nn.Module
        The model to scan for custom module instances.
    peft_config : LoraConfig
        The LoRA config (modified in-place).
    """
    # Detect "all-linear" / "all" whether it's the sole string or an
    # element inside a list/set (e.g. target_modules=["all-linear", ...]).
    has_all_linear = False
    if isinstance(peft_config.target_modules, str):
        has_all_linear = peft_config.target_modules.lower() == "all-linear"
    elif isinstance(peft_config.target_modules, (list, set, frozenset)):
        has_all_linear = any(
            isinstance(m, str) and m.strip().lower() in ("all", "all-linear")
            for m in peft_config.target_modules
        )

    if not has_all_linear:
        return

    # Register GroupedMLP custom handler.
    register_grouped_mlp_lora(peft_config)

    from peft.tuners.tuners_utils import _maybe_include_all_linear_layers

    # If target_modules is a list/set containing "all-linear" alongside
    # other target names, save the extra targets so they can be merged
    # back after PEFT expands "all-linear".
    extra_targets: set[str] = set()
    if isinstance(peft_config.target_modules, (list, set, frozenset)):
        extra_targets = {
            m for m in peft_config.target_modules
            if not (isinstance(m, str) and m.strip().lower() in ("all", "all-linear"))
        }
        peft_config.target_modules = "all-linear"

    peft_config = _maybe_include_all_linear_layers(peft_config, model)
    # peft_config.target_modules is now a set of fully-qualified names
    # for all nn.Linear / Conv1D modules.  Merge back any pre-existing
    # extra targets, then append short names for custom module types so
    # they get matched by PEFT's suffix-matching logic
    # (key.endswith(f".{target_key}")).
    if extra_targets:
        peft_config.target_modules = set(peft_config.target_modules) | extra_targets

    # Add fully-qualified names for every custom module instance so that
    # PEFT matches the exact modules we found — not just a short suffix
    # that could accidentally match an unrelated module with the same name.
    for custom_cls in peft_config._custom_modules:
        for name, module in model.named_modules():
            if isinstance(module, custom_cls):
                if name not in peft_config.target_modules:
                    peft_config.target_modules.add(name)
                    logger.debug_rank0(
                        "expand_all_linear_targets: added custom module "
                        "target %r (from %s)",
                        name,
                        custom_cls.__name__,
                    )


def apply_peft(
    model: Any,
    peft_config: Union[Any, dict],
) -> Any:
    """Apply PEFT LoRA to a model, with built-in support for GroupedMLP.

    This is the recommended high-level entry point.  It:

    1. Converts a dict config to ``LoraConfig`` if needed.
    2. Registers the custom ``GroupedMLPLoRA`` handler for ``GroupedMLP``.
    3. Expands ``target_modules="all-linear"`` to include custom module
       names (e.g. ``"experts"``) that PEFT's built-in expansion would miss,
       since custom modules like GroupedMLP are not ``nn.Linear`` subclasses.
    4. Freezes all base model parameters.
    5. Calls :func:`peft.get_peft_model` to inject adapters.

    Parameters
    ----------
    model : nn.Module
        The model to apply LoRA to.  Should be the model *after* module
        replacement (i.e., after the parallel plan's ``replace_module``)
        but *before* FSDP2 sharding.
    peft_config : LoraConfig or dict
        LoRA configuration.  If a dict, it will be passed to
        ``LoraConfig(**peft_config)``.

    Returns
    -------
    nn.Module
        The model with LoRA adapters injected.
    """
    from peft import LoraConfig, get_peft_model

    if isinstance(peft_config, dict):
        peft_config = LoraConfig(**peft_config)

    # Register the GroupedMLP handler for every target format. Explicit
    # targets and the expanded targets saved in checkpoint metadata can name
    # experts without the "all-linear" marker.
    register_grouped_mlp_lora(peft_config)

    # Expand "all-linear" to include custom modules like GroupedMLP.
    expand_all_linear_targets(model, peft_config)

    # Freeze base model parameters (standard LoRA practice).
    for param in model.parameters():
        param.requires_grad = False

    # Apply PEFT.
    model = get_peft_model(model, peft_config)

    # Log frozen / trainable parameter statistics.
    frozen_count = 0
    lora_count = 0
    for param in model.parameters():
        n = param.numel()
        if param.requires_grad:
            lora_count += n
        else:
            frozen_count += n
    total_count = frozen_count + lora_count
    lora_ratio = lora_count / total_count if total_count > 0 else 0.0

    logger.info_rank0(
        "Applied PEFT LoRA to model — "
        "frozen params: %s, LoRA params: %s, LoRA ratio: %.4f",
        f"{frozen_count:,}",
        f"{lora_count:,}",
        lora_ratio,
    )
    return model
