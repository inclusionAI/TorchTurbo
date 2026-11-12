"""Install the fused MoonEP and MegaMoE backend inside expert FSDP hooks."""
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...models.modules.moe.experts import GroupedMLP
from ...models.modules.moe.moe_layer import MoELayer


def configure_token_bound(config: Any, args: Any) -> None:
    """Derive a fixed per-rank token capacity from Swift's training arguments."""
    length = args.packing_length if args.packing else args.max_length
    if length is None:
        length = args.max_length
    if length is None or length <= 0:
        raise ValueError("mega_moe requires a positive packing_length or max_length.")
    configure_token_capacity(config, length, args.per_device_train_batch_size)


def configure_token_capacity(config: Any, sequence_length: int, batch_size: int) -> None:
    """Set the fixed local capacity from an adapter's padded input bounds."""
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("mega_moe requires a positive sequence length.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("mega_moe requires a positive batch size.")
    sp_size = config.parallel_plan.sp_size
    if not isinstance(sp_size, int) or sp_size <= 0:
        raise ValueError("mega_moe requires a positive SP size.")
    # The backend pads each sequence to SP before splitting the batch.
    local_tokens = ((sequence_length + sp_size - 1) // sp_size) * batch_size
    config.extra["mega_moe_local_tokens"] = ((local_tokens + 127) // 128) * 128


class RoutedExpertsBackend:
    """Adapt compact router selections without changing expert parameter ownership."""

    def __init__(self, ep_group: dist.ProcessGroup, token_bound: int) -> None:
        self.ep_group = ep_group
        self.token_bound = token_bound

    def route(
        self,
        experts: GroupedMLP,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        *,
        expert_indices: torch.Tensor,
    ) -> tuple:
        """Gather selected probabilities, including selected zero-weight experts."""
        selected_probs = probs.gather(1, expert_indices).to(hidden_states.dtype)
        return experts(hidden_states, expert_indices, selected_probs)


def install(model: nn.Module, config: Any, ep_group: dist.ProcessGroup) -> None:
    """Attach one shared MoonEP and MegaMoE backend before FSDP wrapping."""
    token_bound = config.extra.get("mega_moe_local_tokens")
    if not isinstance(token_bound, int) or token_bound <= 0 or token_bound % 128:
        raise ValueError(
            "mega_moe needs a positive 128-aligned per-rank token capacity. "
            "LlamaFactory derives it automatically; engine callers set "
            "config.extra['mega_moe_local_tokens'] before building the model."
        )
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError("mega_moe requires a model using TorchTurbo MoELayer.")
    for layer in layers:
        experts = layer.experts
        if not isinstance(experts, GroupedMLP) or experts.use_te:
            raise ValueError("mega_moe requires TorchTurbo GroupedMLP experts without TE or expert LoRA wrappers.")
        if layer.config.activation_func is not F.silu or not layer.config.gated_linear_unit:
            raise ValueError("mega_moe requires SwiGLU experts.")
        if layer.config.moe_apply_probs_on_input:
            raise ValueError("mega_moe requires routing weights on expert outputs.")
    import importlib.util
    from pathlib import Path

    component = importlib.util.find_spec("tt_megamoe")
    if component is None or component.origin is None or not (
        Path(component.origin).parent / "moonep" / "__init__.py"
    ).is_file():
        raise RuntimeError("Install the MoonEP-enabled tt-megamoe component before enabling mega_moe.")
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
        raise ValueError("mega_moe shared workspaces require matching MoE layer geometry.")
    num_experts = layers[0].config.num_moe_experts
    ranks = dist.get_world_size(ep_group)
    if num_experts % ranks:
        raise ValueError("mega_moe requires experts evenly divisible by EP size.")
    if ranks > 1:
        # Fail before loading a large checkpoint; loading this library performs
        # no device computation and does not initialize SHMEM.
        try:
            from tt_megamoe.moonep import load_allocator

            load_allocator()
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "MegaMoE needs the compiled MoonEP allocator and matching native "
                "library paths. Source CANN 9.1, provision torch_npu 2.10.0.post4, "
                "run python -m tt_megamoe.build_allocator, then launch through "
                "scripts/runs/run_lf.py."
            ) from error
    modes = set()
    for layer in layers:
        limit = float(layer.experts.swiglu_limit or 0.0)
        mode = "pre_activation" if limit else "none"
        modes.add((mode, limit))
    from .moonep.backend import MoonEPMegaMoeBackend

    backend = MoonEPMegaMoeBackend(
        ep_group, token_bound, tuple(sorted(modes)),
        refetch_in_backward=not (
            config.activation_checkpointing.enabled
            and config.activation_checkpointing.use_reentrant
        ),
    )
    for layer in layers:
        layer._routed_experts_backend = backend
        layer.experts._execution_backend = backend
