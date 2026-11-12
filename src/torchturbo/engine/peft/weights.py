"""LoRA weight operations — initialization, delta computation, and merging.

This module collects all tensor-level operations on LoRA weights:

- **Initialization**: :func:`init_lora_tensor`, :func:`get_peft_init_config`
- **Delta computation**: :func:`compute_lora_delta_linear`,
  :func:`compute_lora_delta_grouped_mlp_w1`,
  :func:`compute_lora_delta_grouped_mlp_w2`, :func:`compute_dora_magnitude`
- **Delta map & merge**: :func:`build_lora_delta_map`,
  :func:`apply_lora_delta`
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor

from torchturbo.engine.utils import strip_peft_prefix
from torchturbo.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Gather helper
# ---------------------------------------------------------------------------


def _gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a possibly-sharded DTensor, staging CPU-offloaded shards."""
    if not isinstance(tensor, DTensor):
        return tensor

    compute_device = tensor.device_mesh.device_type
    local = tensor.to_local()
    if local.device.type == compute_device:
        return tensor.full_tensor()

    staged = DTensor.from_local(
        local.to(compute_device),
        tensor.device_mesh,
        tensor.placements,
        run_check=False,
        shape=tensor.shape,
        stride=tensor.stride(),
    )
    return staged.full_tensor()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def get_peft_init_config(model: torch.nn.Module) -> tuple[Any, int | None]:
    """Extract ``init_lora_weights`` and ``r`` from the model's peft_config.

    Returns ``(init_lora_weights, r)`` or ``(None, None)`` when PEFT is
    not active.
    """
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None:
        # PeftModel stores peft_config on base_model.
        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            peft_config = getattr(base_model, "peft_config", None)
    if not peft_config:
        return None, None
    # peft_config is a dict {adapter_name: LoraConfig}.  Use the first.
    cfg = next(iter(peft_config.values()))
    return cfg.init_lora_weights, cfg.r


def init_lora_tensor(
    tensor: torch.Tensor,
    is_a: bool,
    init_lora_weights: Any,
    r: int | None,
) -> None:
    """Initialize a LoRA tensor according to ``init_lora_weights``.

    Mirrors the logic in PEFT's ``LoraLayer.reset_lora_parameters`` and
    ``GroupedMLPLoRA.update_layer``.

    Parameters
    ----------
    tensor : Tensor
        The LoRA weight tensor to initialize in-place.
    is_a : bool
        ``True`` for lora_A / lora_A2 / lora_embedding_A; ``False`` for
        the corresponding B matrices.
    init_lora_weights : bool or str
        Value from ``LoraConfig.init_lora_weights``.
    r : int or None
        LoRA rank (used by ``"gaussian"`` init).
    """
    if init_lora_weights is False:
        # Leave as-is (PyTorch default from nn.Linear constructor).
        return
    if isinstance(init_lora_weights, str):
        if init_lora_weights.lower() == "gaussian":
            if is_a:
                nn.init.normal_(tensor, std=1 / r)
            else:
                nn.init.zeros_(tensor)
        elif init_lora_weights.lower() == "orthogonal":
            # Orthogonal init requires even rank; both A and B are
            # non-zero.  Fallback to default for embedding params
            # which have a different convention.
            if is_a:
                nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            else:
                nn.init.zeros_(tensor)
        elif init_lora_weights in ("pissa", "loftq", "eva", "olora", "corda"):
            # These modes require base weights to be already loaded or
            # data-driven init and are not compatible with the
            # meta-device → broadcast flow.  Fall back to default init.
            logger.warning_rank0_once(
                "init_lora_weights=%r is not supported in meta-device "
                "broadcast flow, falling back to default (kaiming_uniform / zeros)",
                init_lora_weights,
            )
            if is_a:
                nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            else:
                nn.init.zeros_(tensor)
        else:
            raise ValueError(
                f"Unknown init_lora_weights={init_lora_weights!r}"
            )
    else:
        # init_lora_weights is True (default).
        if is_a:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        else:
            nn.init.zeros_(tensor)


# ---------------------------------------------------------------------------
# Delta map
# ---------------------------------------------------------------------------


def build_lora_delta_map(model: torch.nn.Module) -> dict[str, tuple]:
    """Build a mapping from clean base-param names to LoRA modules.

    Scans ``model.named_modules()`` for LoRA-wrapped layers and returns
    a dict that :func:`apply_lora_delta` can use to merge deltas at save
    time.

    Returns
    -------
    dict
        ``{clean_param_name: (lora_module, weight_key)}``

        * For standard ``lora.Linear``:
          ``"model.layers.0.q_proj.weight" → (module, "weight")``
        * For ``GroupedMLPLoRA``:
          ``"model.layers.0.mlp.experts.weight1" → (module, "weight1")``
          ``"model.layers.0.mlp.experts.weight2" → (module, "weight2")``
    """
    delta_map: dict[str, tuple] = {}

    try:
        from torchturbo.engine.peft.lora import GroupedMLPLoRA
    except ImportError:
        GroupedMLPLoRA = None  # type: ignore[assignment,misc]

    for module_name, module in model.named_modules():
        # --- GroupedMLPLoRA (MoE expert layer) ---
        if GroupedMLPLoRA is not None and isinstance(module, GroupedMLPLoRA):
            clean_prefix = strip_peft_prefix(module_name)
            delta_map[f"{clean_prefix}.weight1"] = (module, "weight1")
            delta_map[f"{clean_prefix}.weight2"] = (module, "weight2")
            continue

        # --- Standard lora.Linear (duck-typing) ---
        # Has base_layer + lora_A/lora_B ModuleDicts, but NOT lora_A2
        # (which distinguishes GroupedMLPLoRA).
        if (
            hasattr(module, "base_layer")
            and hasattr(module, "lora_A")
            and isinstance(module.lora_A, torch.nn.ModuleDict)
            and hasattr(module, "lora_B")
            and isinstance(module.lora_B, torch.nn.ModuleDict)
            and not hasattr(module, "lora_A2")
        ):
            clean_prefix = strip_peft_prefix(module_name)
            for param_name, _ in module.base_layer.named_parameters(recurse=False):
                delta_map[f"{clean_prefix}.{param_name}"] = (module, param_name)

    return delta_map


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def _adapter_delta_linear(lora_module: Any, adapter_name: str) -> torch.Tensor:
    """Gather and scale one Linear adapter's weight update."""
    weight_a = _gather(lora_module.lora_A[adapter_name].weight)
    weight_b = _gather(lora_module.lora_B[adapter_name].weight)
    delta = weight_b @ weight_a
    if lora_module.fan_in_fan_out:
        delta = delta.T
    return delta * lora_module.scaling[adapter_name]


@torch.no_grad()
def compute_lora_delta_linear(lora_module: Any, weight_key: str) -> torch.Tensor | None:
    """Compute the LoRA delta for a standard ``lora.Linear`` layer.

    ``delta = sum( scaling * (B @ A) )`` over all non-merged adapters.

    Returns ``None`` when no delta applies (e.g. bias with lora_bias=False).
    """
    # Only weight deltas are merged; bias LoRA is rare and can be
    # added later if needed.
    if weight_key != "weight":
        return None

    delta: torch.Tensor | None = None
    merged = set(getattr(lora_module, "merged_adapters", []))

    for adapter_name in lora_module.r.keys():
        if adapter_name in merged:
            continue

        adapter_delta = _adapter_delta_linear(lora_module, adapter_name)

        delta = adapter_delta if delta is None else delta + adapter_delta

    return delta


def merge_lora_linear(
    base_tensor: torch.Tensor,
    lora_module: Any,
    weight_key: str,
) -> torch.Tensor:
    """Return a standard Linear weight with LoRA or DoRA adapters merged."""
    if weight_key != "weight":
        return base_tensor

    merged = set(lora_module.merged_adapters)
    result = base_tensor
    for adapter_name in lora_module.r:
        if adapter_name in merged:
            continue

        delta = _adapter_delta_linear(lora_module, adapter_name)
        result = result + delta

        if adapter_name in lora_module.lora_magnitude_vector:
            magnitude = _gather(
                lora_module.lora_magnitude_vector[adapter_name].weight
            )
            normalized = result.T if lora_module.fan_in_fan_out else result
            weight_norm = torch.linalg.vector_norm(normalized, dim=1)
            dora_factor = magnitude / weight_norm
            dora_factor = dora_factor.view(-1, 1)
            if lora_module.fan_in_fan_out:
                dora_factor = dora_factor.T
            result = dora_factor * result

    return result


@torch.no_grad()
def compute_dora_magnitude(
    lora_module: Any,
    adapter_name: str,
) -> torch.Tensor:
    """Return the initial DoRA magnitude after base and LoRA weights load."""
    base_weight = _gather(lora_module.base_layer.weight)
    delta = _adapter_delta_linear(lora_module, adapter_name)
    initialized_weight = base_weight + delta
    if lora_module.fan_in_fan_out:
        initialized_weight = initialized_weight.T
    return torch.linalg.vector_norm(initialized_weight, dim=1)


@torch.no_grad()
def compute_lora_delta_grouped_mlp_w1(
    lora_module: Any,
) -> torch.Tensor | None:
    """Compute the LoRA delta for weight1 of a ``GroupedMLPLoRA`` module.

    ``delta_w1 = sum( scaling * (lora_A @ lora_B) )`` over all non-merged
    adapters.

    Returns ``None`` if no non-merged adapters exist.
    """
    delta: torch.Tensor | None = None
    merged = set(getattr(lora_module, "merged_adapters", []))

    for adapter_name in lora_module.r.keys():
        if adapter_name in merged:
            continue

        scaling = lora_module.scaling[adapter_name]
        A = _gather(lora_module.lora_A[adapter_name].weight)
        B = _gather(lora_module.lora_B[adapter_name].weight)

        d = scaling * torch.matmul(A, B)  # (E, hidden, fc1_out)
        delta = d if delta is None else delta + d

    return delta


@torch.no_grad()
def compute_lora_delta_grouped_mlp_w2(
    lora_module: Any,
) -> torch.Tensor | None:
    """Compute the LoRA delta for weight2 of a ``GroupedMLPLoRA`` module.

    ``delta_w2 = sum( scaling * (lora_A2 @ lora_B2) )`` over all non-merged
    adapters.

    Returns ``None`` if no non-merged adapters exist.
    """
    delta: torch.Tensor | None = None
    merged = set(getattr(lora_module, "merged_adapters", []))

    for adapter_name in lora_module.r.keys():
        if adapter_name in merged:
            continue

        scaling = lora_module.scaling[adapter_name]
        A2 = _gather(lora_module.lora_A2[adapter_name].weight)
        B2 = _gather(lora_module.lora_B2[adapter_name].weight)

        d = scaling * torch.matmul(A2, B2)  # (E, ffn, hidden)
        delta = d if delta is None else delta + d

    return delta


# ---------------------------------------------------------------------------
# Apply delta
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply_lora_delta(
    base_tensor: torch.Tensor,
    clean_name: str,
    lora_delta_map: dict[str, tuple],
) -> torch.Tensor:
    """Merge LoRA deltas into *base_tensor* if applicable.

    Returns a **new** tensor (``base + delta``); the original model
    parameters are never modified in-place.

    Parameters
    ----------
    base_tensor : Tensor
        The gathered (full) base weight tensor.
    clean_name : str
        Cleaned parameter name (after :func:`strip_peft_prefix`).
    lora_delta_map : dict
        The mapping returned by :func:`build_lora_delta_map`.
        If empty, this function is a no-op and returns ``base_tensor``
        unchanged.
    """
    if not lora_delta_map:
        return base_tensor

    entry = lora_delta_map.get(clean_name)
    if entry is None:
        return base_tensor

    lora_module, weight_key = entry

    # --- GroupedMLPLoRA ---
    if hasattr(lora_module, "lora_A2"):
        if weight_key == "weight1":
            delta = compute_lora_delta_grouped_mlp_w1(lora_module)
        else:
            delta = compute_lora_delta_grouped_mlp_w2(lora_module)

        if delta is not None:
            return base_tensor + delta.view_as(base_tensor)
        return base_tensor

    # --- Standard lora.Linear ---
    return merge_lora_linear(base_tensor, lora_module, weight_key)
