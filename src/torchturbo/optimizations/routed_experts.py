# SPDX-License-Identifier: Apache-2.0
"""Bind routed expert execution within the original expert module hooks."""

from typing import Protocol, TYPE_CHECKING
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..models.modules.moe.experts import GroupedMLP
from ..models.modules.moe.moe_layer import MoELayer
from ..models.modules.moe.utils import RoutingResult

if TYPE_CHECKING:
    from ..engine.config import EngineConfig
    from ..models.modules.moe.moe_config import MoEConfig


class TokenCapacityArgs(Protocol):
    """Training inputs used to derive per-rank capacity without importing Swift."""

    packing: bool
    packing_length: int | None
    max_length: int | None
    per_device_train_batch_size: int


def check_shared_shmem_runtime(*, require_moonep: bool = False) -> None:
    """Reject a loaded MoonEP wheel linked against a different SHMEM image."""
    if require_moonep:
        try:
            import npu_patch  # noqa: F401
        except ModuleNotFoundError as error:
            if error.name == "npu_patch":
                raise ImportError(
                    "MoonEP is not installed; build the shared-runtime wheel with "
                    "scripts/runs/deps/moonep.sh after scripts/runs/deps/mega_moe.sh."
                ) from error
            raise
    if "npu_patch" in sys.modules and not vars(sys.modules["npu_patch"]).get(
        "EXTERNAL_SHMEM_RUNTIME", False
    ):
        raise RuntimeError(
            "MoonEP and hyper-multicore SHMEM are not symbol-isolated; build the shared-runtime "
            "MoonEP wheel with scripts/runs/deps/moonep.sh."
        )


def configure_token_bound(config: "EngineConfig", args: TokenCapacityArgs) -> None:
    """Derive a fixed per-rank token capacity from training arguments.

    Args:
        config: Engine configuration receiving the derived MoE capacity.
        args: Packing length, sequence length and micro-batch size.
    """
    length = args.packing_length if args.packing else args.max_length
    if length is None:
        length = args.max_length
    if length is None or length <= 0:
        raise ValueError(
            "The routed expert backend requires a positive packing_length or max_length."
        )
    tokens = length * args.per_device_train_batch_size
    sp_size = config.parallel_plan.sp_size
    local_tokens = (tokens + sp_size - 1) // sp_size
    required = ((local_tokens + 127) // 128) * 128
    capacity = config.moe.local_token_capacity
    if capacity is None:
        capacity = required
    if not isinstance(capacity, int) or capacity < required or capacity % 128:
        raise ValueError(
            "moe.local_token_capacity must be 128-aligned and at least "
            f"{required} for the configured sequence length, batch size and SP size."
        )
    config.moe.local_token_capacity = capacity


class RoutedExpertsBackend:
    """Adapt the existing router mask without changing expert parameter ownership."""

    def __init__(self, ep_group: dist.ProcessGroup, token_bound: int) -> None:
        if ep_group is None:
            raise ValueError(
                "Routed expert backends require an explicit EP group with ep_size > 1."
            )
        self.ep_group = ep_group
        self.token_bound = token_bound
        self._geometry_checked = False

    def _check_token_capacity(self, actual_tokens: int) -> None:
        """Fail before a backend touches capacity-bounded shared buffers."""
        if actual_tokens > self.token_bound:
            raise ValueError(
                f"Routed expert input actual_tokens={actual_tokens} exceeds token_bound={self.token_bound}."
            )

    def _check_ep_geometry(self, config: "MoEConfig", device: torch.device) -> None:
        """Agree on static capacity before any rank chooses an execution path."""
        if self._geometry_checked or not dist.is_initialized():
            return
        size = dist.get_world_size(self.ep_group)
        if size > 1:
            shape = (
                self.token_bound,
                config.num_moe_experts,
                config.moe_router_topk,
                config.hidden_size,
                config.moe_ffn_hidden_size,
            )
            local = torch.tensor(shape, device=device, dtype=torch.int64)
            gathered = local.new_empty(size * len(shape))
            dist.all_gather_into_tensor(gathered, local, group=self.ep_group)
            rows = gathered.reshape(size, -1).tolist()
            if any(row != rows[0] for row in rows[1:]):
                raise ValueError(
                    f"Routed expert geometry differs across EP ranks "
                    f"(capacity, experts, top_k, hidden, intermediate): {rows}"
                )
        self._geometry_checked = True

    def route(
        self,
        layer: MoELayer,
        hidden_states: torch.Tensor,
        routing: RoutingResult,
    ) -> torch.Tensor | None:
        """Preserve the router's selected zero weights and probability precision."""
        self._check_token_capacity(hidden_states.shape[0])
        self._check_ep_geometry(layer.config, hidden_states.device)
        output, _ = layer.experts(
            hidden_states,
            None,
            routing.selected_probs(),
            routed_backend=self,
            expert_indices=routing.expert_indices,
        )
        return output

    def execute(
        self,
        experts: GroupedMLP,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Execute with the model's current unsharded expert parameters."""
        raise NotImplementedError


def bind_expert_backends(
    model: nn.Module, config: "EngineConfig", ep_group: dist.ProcessGroup
) -> None:
    """Bind the selected expert backend before FSDP wraps the model."""
    if not (config.moe.enable_mega_moe or config.moe.enable_moonep):
        return
    if ep_group is None or dist.get_world_size(ep_group) <= 1:
        raise ValueError(
            "Routed expert backends require an explicit EP group with ep_size > 1."
        )
    token_bound = config.moe.local_token_capacity
    if not isinstance(token_bound, int) or token_bound <= 0 or token_bound % 128:
        raise ValueError(
            "The routed expert backend needs a positive 128-aligned per-rank token capacity. "
            "Set moe.local_token_capacity or --tt_moe_local_token_capacity; "
            "Swift derives it automatically when omitted."
        )
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError(
            "The routed expert backend requires a model using TorchTurbo MoELayer."
        )
    for layer in layers:
        experts = layer.experts
        if not isinstance(experts, GroupedMLP) or experts.use_te:
            raise ValueError(
                "The routed expert backend requires TorchTurbo GroupedMLP experts without TE."
            )
        if (
            layer.config.activation_func is not F.silu
            or not layer.config.gated_linear_unit
        ):
            raise ValueError("The routed expert backend requires SwiGLU experts.")
        if layer.config.moe_apply_probs_on_input:
            raise ValueError(
                "The routed expert backend requires routing probabilities after SwiGLU."
            )
    geometries = {
        (
            layer.config.hidden_size,
            layer.config.moe_ffn_hidden_size,
            layer.config.num_moe_experts,
            layer.config.moe_router_topk,
        )
        for layer in layers
    }
    if len(geometries) != 1:
        raise ValueError(
            "Shared expert workspaces require matching MoE layer geometry."
        )
    num_experts = layers[0].config.num_moe_experts
    ranks = dist.get_world_size(ep_group)
    if num_experts % ranks:
        raise ValueError(
            "The routed expert backend requires experts evenly divisible by EP size."
        )
    for layer in layers:
        if layer.experts.num_local_experts != num_experts // ranks:
            raise ValueError(
                f"Expected {num_experts // ranks} local experts for EP size {ranks}, "
                f"got {layer.experts.num_local_experts}."
            )
    grad_reduce_dtype = config.moe.moonep_grad_reduce_dtype
    if not config.moe.enable_mega_moe:
        try:
            from .moonep.native_backend import NativeMoonEPBackend
        except ModuleNotFoundError as error:
            if error.name == "hyper_multicore" or error.name == "npu_patch":
                raise ImportError(
                    "MoonEP requires matching hyper-multicore and MoonEP wheels; run "
                    "scripts/runs/deps/mega_moe.sh then scripts/runs/deps/moonep.sh."
                ) from error
            raise

        backend = NativeMoonEPBackend(
            ep_group, token_bound, grad_reduce_dtype=grad_reduce_dtype
        )
        for layer in layers:
            backend.register_dispatcher(layer.experts, layer.token_dispatcher)
            layer._routed_experts_backend = backend
        return
    layer_modes = []
    for layer in layers:
        limit = float(layer.experts.swiglu_limit or 0.0)
        mode = (
            (
                "post_activation"
                if layer.config.moe_swiglu_clamp_gate_after_activation
                else "pre_activation"
            )
            if limit
            else "none"
        )
        layer_modes.append((mode, limit))
    mode_manifest = tuple(sorted(set(layer_modes)))
    layer_manifest = tuple(layer_modes)
    if dist.is_initialized():
        manifests = [None] * ranks
        dist.all_gather_object(manifests, layer_manifest, group=ep_group)
        if any(manifest != layer_manifest for manifest in manifests):
            raise ValueError(
                f"MoE SwiGLU mode/limit manifests differ across EP ranks: {manifests}"
            )
    from .mega_moe.backend import MegaMoeBackend

    backend = MegaMoeBackend(
        ep_group,
        token_bound,
        mode_manifest,
        enable_moonep=config.moe.enable_moonep,
        grad_reduce_dtype=grad_reduce_dtype,
    )
    for layer in layers:
        layer._routed_experts_backend = backend
