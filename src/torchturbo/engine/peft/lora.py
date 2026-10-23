"""LoRA adapter for GroupedMLP (MoE experts with grouped matmul).

This module provides ``GroupedMLPLoRA``, a custom LoRA layer that wraps
:class:`torchturbo.models.modules.moe.experts.GroupedMLP` and integrates
with the PEFT library via ``LoraConfig._register_custom_module``.

Design
------
GroupedMLP stores all expert weights in two fused tensors (``weight1`` and
``weight2``) and uses the ``grouped_matmul`` op to perform all expert
matmuls in a single fused launch when available.  Standard PEFT LoRA
cannot be applied because:

1. GroupedMLP is not an ``nn.Linear`` — it has no ``in_features`` /
   ``out_features`` and its forward signature is different.
2. The weights use a non-standard layout ``(in, out)`` per expert (not
   the transposed ``(out, in)`` convention of ``nn.Linear``).
3. The ``gmm`` kernel requires reshaped 3-D weight tensors.

``GroupedMLPLoRA`` uses an **additive-residual** approach inspired by
ms-swift's ``mcore_bridge.LoraParallelLinear``: LoRA-A and LoRA-B are
themselves 3-D per-expert weight tensors that execute via the same
grouped GEMM kernel as the base layer.  The forward pass computes:

    result = base_gmm(x, w) + scaling * gmm(gmm(dropout(x), lora_A), lora_B)

This avoids materializing the full ``(E, in, out)`` LoRA delta tensor
(the old merge-weight approach) and instead only creates a small
``(tokens, r)`` intermediate per LoRA-A call.

Each expert gets its own LoRA A/B matrices, stored as 3-D parameters
of shape ``(num_local_experts, in, r)`` and ``(num_local_experts, r, out)``.
This ensures that different EP ranks (which may hold different numbers of
local experts) have independent per-expert LoRA configurations.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from torchturbo.ops import ops
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.logging import get_logger

logger = get_logger(__name__)


class GroupedLoRALinear(nn.Module):
    """Per-expert LoRA weight with grouped GEMM forward.

    Stores per-expert LoRA matrices as a 3-D parameter
    ``(num_local_experts, in_features, out_features)`` and executes the
    grouped GEMM kernel in forward.

    This enables LoRA forward to use the same efficient grouped matmul
    as the base ``GroupedMLP`` layer, avoiding the need to materialize
    the full ``B @ A`` delta tensor.

    Parameters
    ----------
    num_experts : int
        Number of local experts on this EP rank.
    in_features : int
        Input dimension per expert.
    out_features : int
        Output dimension per expert.
    dtype : torch.dtype
        Parameter dtype.  Defaults to ``torch.float32`` to prevent
        overflow in low-rank LoRA computations (following ms-swift's
        convention).
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, in_features, out_features, dtype=dtype)
        )
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

    def forward(
        self, x: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        """Run grouped GEMM: ``x @ weight[i]`` per expert group.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(total_tokens, in_features)``.
            Tokens must be sorted by expert (same as the base
            ``GroupedMLP`` convention) so that ``tokens_per_expert``
            correctly partitions the input.
        tokens_per_expert : torch.Tensor
            1-D integer tensor of per-expert token counts.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(total_tokens, out_features)``.
        """
        return ops.grouped_matmul(x, self.weight, tokens_per_expert, trans_b=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_experts={self.num_experts}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features})"
        )


class GroupedMLPLoRA(nn.Module, BaseTunerLayer):
    """LoRA wrapper for :class:`GroupedMLP` using additive-residual grouped GEMM.

    This class inherits from both ``nn.Module`` and ``BaseTunerLayer`` so
    that it integrates with PEFT's adapter management (multi-adapter
    support, merge/unmerge, enable/disable, etc.).

    Each expert gets its own independent LoRA A/B matrices, stored as
    3-D parameters via :class:`GroupedLoRALinear`.  This is necessary
    because different EP ranks may hold different numbers of local
    experts, making a shared-LoRA configuration across experts unsound.

    Forward uses the additive-residual approach: the LoRA computation
    runs as separate grouped GEMM calls and the result is added to the
    base output, avoiding materialization of the full weight delta.

    Parameters
    ----------
    base_layer : GroupedMLP
        The grouped-MLP layer to wrap.
    adapter_name : str
        Name of the first adapter.
    r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability.  Unlike the previous merge-weight
        implementation, this is now supported because the additive-residual
        forward has per-token access to the LoRA input.
    init_lora_weights : bool or str
        Weight initialization strategy.
    use_rslora : bool
        Whether to use rank-stabilized LoRA.
    use_dora : bool
        Whether to use DoRA (not supported for GroupedMLP).
    lora_bias : bool
        Whether to add bias to LoRA-B (not supported for GroupedMLP).
    **kwargs
        Additional keyword arguments forwarded to ``LoraLayer.__init__``.
    """

    # PEFT uses these names to discover trainable adapter parameters.
    adapter_layer_names = ("lora_A", "lora_B", "lora_A2", "lora_B2")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(
        self,
        base_layer: GroupedMLP,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        # Validate unsupported features early.
        if use_dora:
            raise ValueError(
                "GroupedMLPLoRA does not support DoRA (use_dora=True)."
            )
        if lora_bias:
            raise ValueError(
                "GroupedMLPLoRA does not support lora_bias=True."
            )

        self.base_layer = base_layer

        # Extract per-expert dimensions from the base GroupedMLP.
        self.num_local_experts = base_layer.num_local_experts
        self.config = base_layer.config

        hidden_size = base_layer.config.hidden_size
        if base_layer.config.gated_linear_unit:
            fc1_output_size = base_layer.config.moe_ffn_hidden_size * 2
        else:
            fc1_output_size = base_layer.config.moe_ffn_hidden_size
        fc2_input_size = base_layer.config.moe_ffn_hidden_size

        # Per-expert dimensions for weight1 and weight2.
        # weight1 layout per expert: (hidden_size, fc1_output_size)  [in, out]
        # weight2 layout per expert: (fc2_input_size, hidden_size)   [in, out]
        self.in_features_weight1 = hidden_size
        self.out_features_weight1 = fc1_output_size
        self.in_features_weight2 = fc2_input_size
        self.out_features_weight2 = hidden_size

        # Set in_features/out_features for BaseTunerLayer compatibility
        # (used by _mark_only_adapters_as_trainable etc.)
        self.in_features = hidden_size
        self.out_features = hidden_size  # output dim of the full MLP

        # Adapter management dicts (same structure as LoraLayer).
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, int] = {}
        self.scaling: dict[str, float] = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})   # LoRA-A for weight1 (gate+up)
        self.lora_B = nn.ModuleDict({})   # LoRA-B for weight1 (gate+up)
        self.lora_A2 = nn.ModuleDict({})  # LoRA-A for weight2 (down)
        self.lora_B2 = nn.ModuleDict({})  # LoRA-B for weight2 (down)

        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = nn.ModuleDict()  # for DoRA (unused)
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = False
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    # ------------------------------------------------------------------
    # Legacy checkpoint conversion
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Override to auto-convert legacy 2-D LoRA weights to 3-D format.

        The old ``GroupedMLPLoRA`` stored LoRA weights as 2-D
        ``nn.Linear`` parameters (e.g. ``(r*E, in_features)``).  The
        refactored version uses 3-D ``GroupedLoRALinear`` parameters
        (e.g. ``(E, in_features, r)``).  This hook detects 2-D weights
        and converts them in-place so that old checkpoints load
        transparently.
        """
        self._convert_legacy_weights(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _convert_legacy_weights(
        self, state_dict: dict, prefix: str
    ) -> None:
        """Convert 2-D nn.Linear LoRA weights to 3-D GroupedLoRALinear format.

        Old format (per adapter):
            lora_A.{name}.weight  : (r * E, in_features)
            lora_B.{name}.weight  : (out_features, r * E)

        New format (per adapter):
            lora_A.{name}.weight  : (E, in_features, r)
            lora_B.{name}.weight  : (E, r, out_features)
        """
        if not self.r:
            return  # no adapters yet, nothing to convert

        n = self.num_local_experts
        # Use the first adapter's rank (all adapters should have the same r
        # in practice, but handle any).
        for adapter_name, r in self.r.items():
            conversions = [
                # (dict_key_suffix, old_transposed, in_feat, out_feat)
                (
                    f"lora_A.{adapter_name}.weight",
                    False,
                    self.in_features_weight1,
                    r,
                ),
                (
                    f"lora_B.{adapter_name}.weight",
                    True,
                    r,
                    self.out_features_weight1,
                ),
                (
                    f"lora_A2.{adapter_name}.weight",
                    False,
                    self.in_features_weight2,
                    r,
                ),
                (
                    f"lora_B2.{adapter_name}.weight",
                    True,
                    r,
                    self.out_features_weight2,
                ),
            ]
            for key_suffix, is_B, in_feat, out_feat in conversions:
                key = f"{prefix}{key_suffix}"
                if key not in state_dict:
                    continue
                old = state_dict[key]
                if old.dim() == 3:
                    continue  # already 3-D, skip
                if old.dim() != 2:
                    continue  # unexpected, skip

                if not is_B:
                    # A matrix: old (r*E, in) → new (E, in, r)
                    # old layout: row-major, [expert0_r0, expert0_r1, ..., expert1_r0, ...]
                    state_dict[key] = old.view(n, r, in_feat).permute(0, 2, 1).contiguous()
                else:
                    # B matrix: old (out, r*E) → new (E, r, out)
                    # old.T layout: (r*E, out) → view (E, r, out) → already correct after T
                    state_dict[key] = old.T.view(n, out_feat, r).permute(0, 2, 1).contiguous()

    # ------------------------------------------------------------------
    # BaseTunerLayer abstract methods / properties
    # ------------------------------------------------------------------

    @property
    def active_adapter(self) -> str:
        return self._active_adapter

    @active_adapter.setter
    def active_adapter(self, value: str) -> None:
        self._active_adapter = value

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self._active_adapter, str):
            return [self._active_adapter]
        return list(self._active_adapter)

    def set_adapter(self, adapter_names: Union[str, list[str]]) -> None:
        """Set the active adapter(s)."""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for adapter_name in adapter_names:
            if adapter_name not in self.r:
                raise ValueError(
                    f"Adapter {adapter_name!r} not found in GroupedMLPLoRA. "
                    f"Available: {list(self.r.keys())}"
                )
        self._active_adapter = adapter_names[0] if len(adapter_names) == 1 else adapter_names

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle adapters on/off."""
        self._disable_adapters = not enabled

    def get_base_layer(self) -> nn.Module:
        """Return the unwrapped base GroupedMLP layer."""
        return self.base_layer

    # ------------------------------------------------------------------
    # Layer update (adapter creation)
    # ------------------------------------------------------------------

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        use_rslora: bool = False,
        use_dora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        qalora_group_size: int = 32,
        **kwargs,
    ) -> None:
        """Create LoRA parameters for a new adapter."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer, got {r}")
        if use_dora:
            raise ValueError("GroupedMLPLoRA does not support DoRA.")
        if lora_bias:
            raise ValueError("GroupedMLPLoRA does not support lora_bias=True.")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # Dropout: now supported with the additive-residual forward.
        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        # Per-expert LoRA using 3-D GroupedLoRALinear.
        # Force float32 to prevent overflow (following ms-swift convention).
        n = self.num_local_experts
        self.lora_A[adapter_name] = GroupedLoRALinear(
            n, self.in_features_weight1, r, dtype=torch.float32
        )
        self.lora_B[adapter_name] = GroupedLoRALinear(
            n, r, self.out_features_weight1, dtype=torch.float32
        )
        self.lora_A2[adapter_name] = GroupedLoRALinear(
            n, self.in_features_weight2, r, dtype=torch.float32
        )
        self.lora_B2[adapter_name] = GroupedLoRALinear(
            n, r, self.out_features_weight2, dtype=torch.float32
        )

        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_dora[adapter_name] = False

        # Initialize weights.
        if isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / r)
            nn.init.normal_(self.lora_A2[adapter_name].weight, std=1 / r)
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            nn.init.zeros_(self.lora_B2[adapter_name].weight)
        elif init_lora_weights:
            # Kaiming uniform for A, zeros for B (standard LoRA init).
            # kaiming_uniform_ on 3-D (E, in, r) treats each (in, r) slice
            # as a matrix with fan_in=in, fan_out=r — correct for LoRA-A.
            nn.init.kaiming_uniform_(
                self.lora_A[adapter_name].weight, a=math.sqrt(5)
            )
            nn.init.kaiming_uniform_(
                self.lora_A2[adapter_name].weight, a=math.sqrt(5)
            )
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            nn.init.zeros_(self.lora_B2[adapter_name].weight)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _move_adapter_to_device_of_base_layer(
        self, adapter_name: str, device: Optional[torch.device] = None
    ) -> None:
        """Move adapter parameters to the device of the base layer."""
        if device is None:
            device = next(self.base_layer.parameters()).device

        for adapter_layer_name in self.adapter_layer_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, nn.ModuleDict):
                continue
            if adapter_name not in adapter_layer:
                continue
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    # ------------------------------------------------------------------
    # Delta weight computation (used by merge/unmerge)
    # ------------------------------------------------------------------

    def _compute_deltas(
        self, adapter_names: Optional[list[str]] = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the LoRA weight deltas for the given adapters.

        With the new 3-D weight layout ``(E, in, r)`` and ``(E, r, out)``,
        the delta is simply ``A @ B`` per expert (no transpose needed).

        Returns
        -------
        (delta_w1, delta_w2) : tuple of Tensors or None
            delta_w1: shape ``(num_local_experts, hidden_size, fc1_output_size)``
            delta_w2: shape ``(num_local_experts, fc2_input_size, hidden_size)``
        """
        if adapter_names is None:
            adapter_names = self.active_adapters

        delta_w1 = None
        delta_w2 = None

        for adapter_name in adapter_names:
            if adapter_name not in self.r:
                continue

            scaling = self.scaling[adapter_name]

            # lora_A.weight: (E, in, r),  lora_B.weight: (E, r, out)
            # delta = A @ B = (E, in, out)  — directly matches base weight reshape
            d_w1 = scaling * torch.matmul(
                self.lora_A[adapter_name].weight,
                self.lora_B[adapter_name].weight,
            )  # (E, hidden, fc1_out)

            d_w2 = scaling * torch.matmul(
                self.lora_A2[adapter_name].weight,
                self.lora_B2[adapter_name].weight,
            )  # (E, ffn, hidden)

            if delta_w1 is None:
                delta_w1 = d_w1
                delta_w2 = d_w2
            else:
                delta_w1 = delta_w1 + d_w1
                delta_w2 = delta_w2 + d_w2

        return delta_w1, delta_w2

    # ------------------------------------------------------------------
    # Merge / unmerge
    # ------------------------------------------------------------------

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge active adapter weights into the base weights."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for adapter_name in adapter_names:
            if adapter_name not in self.r:
                continue

            # Temporarily set active adapter to compute the delta.
            prev_active = self._active_adapter
            self._active_adapter = adapter_name
            delta_w1, delta_w2 = self._compute_deltas([adapter_name])
            self._active_adapter = prev_active

            base_layer = self.get_base_layer()

            if safe_merge:
                orig_w1 = base_layer.weight1.data.clone()
                orig_w2 = base_layer.weight2.data.clone()
                new_w1 = orig_w1 + delta_w1.view_as(base_layer.weight1)
                new_w2 = orig_w2 + delta_w2.view_as(base_layer.weight2)
                if not (torch.isfinite(new_w1).all() and torch.isfinite(new_w2).all()):
                    raise ValueError(
                        f"NaNs detected in merged weights for adapter {adapter_name!r}"
                    )
                base_layer.weight1.data = new_w1
                base_layer.weight2.data = new_w2
            else:
                base_layer.weight1.data += delta_w1.view_as(base_layer.weight1)
                base_layer.weight2.data += delta_w2.view_as(base_layer.weight2)

            self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        """Unmerge all merged adapters from the base weights."""
        if not self.merged_adapters:
            logger.warning_rank0_once("Already unmerged. Nothing to do.")
            return

        while self.merged_adapters:
            adapter_name = self.merged_adapters.pop()
            if adapter_name not in self.r:
                continue

            prev_active = self._active_adapter
            self._active_adapter = adapter_name
            delta_w1, delta_w2 = self._compute_deltas([adapter_name])
            self._active_adapter = prev_active

            base_layer = self.get_base_layer()
            base_layer.weight1.data -= delta_w1.view_as(base_layer.weight1)
            base_layer.weight2.data -= delta_w2.view_as(base_layer.weight2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass with additive-residual LoRA via grouped GEMM.

        When LoRA is active and not merged, the forward computes:

        1. Base fc1 via grouped GEMM
        2. LoRA fc1 via two grouped GEMM calls (A then B)
        3. Add LoRA fc1 result to base fc1 result
        4. Apply activation function and routing probabilities
        5. Base fc2 via grouped GEMM
        6. LoRA fc2 via two grouped GEMM calls (A then B)
        7. Add LoRA fc2 result to base fc2 result

        When LoRA is disabled or merged, delegates directly to
        ``self.base_layer.forward()``.
        """
        if self._disable_adapters:
            if self.merged_adapters:
                self.unmerge()
            return self.base_layer(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        if self.merged_adapters:
            # Weights already have the LoRA delta baked in.
            return self.base_layer(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        # Compute LoRA additive residual.
        active = [a for a in self.active_adapters if a in self.r]
        if not active:
            return self.base_layer(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        base = self.base_layer
        E = base.num_local_experts
        H = base.config.hidden_size
        F = base.config.moe_ffn_hidden_size

        # Handle moe_apply_probs_on_input (same as base GroupedMLP).
        if base.config.moe_apply_probs_on_input:
            assert (
                base.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(
                original_dtype
            )
            permuted_probs = torch.ones_like(permuted_probs)

        activation_probs = permuted_probs.unsqueeze(-1)
        x = permuted_local_hidden_states

        if x.nelement() != 0:
            # Reshape base weights to 3-D for grouped GEMM.
            w1 = base.weight1.view(E, H, -1)  # (E, H, fc1_out)
            w2 = base.weight2.view(E, -1, H)  # (E, ffn, H)

            # ===== FC1: base GEMM + LoRA GEMM =====
            # Base fc1.
            fc1_base_dtype = x.dtype
            fc1_output = ops.grouped_matmul(
                x, w1, tokens_per_expert, trans_b=False
            )

            # LoRA fc1: additive residual via grouped GEMM.
            for adapter_name in active:
                scaling = self.scaling[adapter_name]
                dropout = self.lora_dropout[adapter_name]
                lora_A = self.lora_A[adapter_name]
                lora_B = self.lora_B[adapter_name]

                # Apply dropout and cast to LoRA dtype (float32).
                x_drop = dropout(x).to(lora_A.weight.dtype)
                # LoRA-A: (E, H, r)  → output (tokens, r)
                a_result = ops.grouped_matmul(
                    x_drop, lora_A.weight, tokens_per_expert, trans_b=False
                )
                # LoRA-B: (E, r, fc1_out)  → output (tokens, fc1_out)
                b_result = ops.grouped_matmul(
                    a_result, lora_B.weight, tokens_per_expert, trans_b=False
                )
                fc1_output = fc1_output + b_result.to(fc1_base_dtype) * scaling

            # ===== Activation (SwiGLU / GeGLU) =====
            intermediate_parallel = base.activation_func_with_probs(
                fc1_output, activation_probs
            )

            # ===== FC2: base GEMM + LoRA GEMM =====
            fc2_base_dtype = intermediate_parallel.dtype
            fc2_output = ops.grouped_matmul(
                intermediate_parallel, w2, tokens_per_expert, trans_b=False
            )

            # LoRA fc2: additive residual via grouped GEMM.
            for adapter_name in active:
                scaling = self.scaling[adapter_name]
                dropout = self.lora_dropout[adapter_name]
                lora_A2 = self.lora_A2[adapter_name]
                lora_B2 = self.lora_B2[adapter_name]

                inter_drop = dropout(intermediate_parallel).to(lora_A2.weight.dtype)
                a2_result = ops.grouped_matmul(
                    inter_drop, lora_A2.weight, tokens_per_expert, trans_b=False
                )
                b2_result = ops.grouped_matmul(
                    a2_result, lora_B2.weight, tokens_per_expert, trans_b=False
                )
                fc2_output = fc2_output + b2_result.to(fc2_base_dtype) * scaling

        else:
            # No tokens allocated to local experts.
            # Preserve the output shape and keep local LoRA parameters in the
            # autograd graph so FSDP still receives zero gradients.
            assert torch.count_nonzero(tokens_per_expert) == 0
            # Empty expert ranks must also participate in the probability A2A backward.
            zero = permuted_local_hidden_states.sum() * 0 + permuted_probs.sum() * 0
            for adapter_name in active:
                zero = zero + self.lora_A[adapter_name].weight.sum() * 0
                zero = zero + self.lora_B[adapter_name].weight.sum() * 0
                zero = zero + self.lora_A2[adapter_name].weight.sum() * 0
                zero = zero + self.lora_B2[adapter_name].weight.sum() * 0
            fc2_output = torch.zeros_like(permuted_local_hidden_states) + zero

        return fc2_output, None

    def __repr__(self) -> str:
        rep = super().__repr__()
        return f"lora.GroupedMLPLoRA({rep})"
