"""Optional standalone MegaMoE execution with existing TorchTurbo parameters."""
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

_EXECUTORS: dict[tuple, Any] = {}
_PADDING_ROUTES: dict[tuple, torch.Tensor] = {}


def execute_mega_moe(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    probs: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    config: Any,
    ep_group: dist.ProcessGroup,
    swiglu_limit: float,
    padded_tokens: int,
) -> torch.Tensor:
    """Execute routed experts, padding only the kernel's fixed token dimension."""
    from tt_megamoe import MegaMoeExperts

    if config.activation_func is not F.silu or not config.gated_linear_unit:
        raise ValueError("MegaMoE requires SwiGLU experts.")
    if config.moe_apply_probs_on_input:
        raise ValueError("MegaMoE applies routing weights on expert outputs.")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.device.type != "npu":
        raise ValueError("MegaMoE requires BF16 NPU activations.")
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    tokens = flat.shape[0]
    if tokens > padded_tokens:
        raise ValueError(f"MoE input has {tokens} tokens, exceeds the configured {padded_tokens}")
    ep_size = dist.get_world_size(ep_group)
    prefix = (
        ep_group, hidden_states.device, padded_tokens, config.hidden_size,
        config.moe_ffn_hidden_size, config.num_moe_experts, config.moe_router_topk,
    )
    # Register all clipping variants before first execution so SHMEM reserves
    # their scratch space together. The executors never own model parameters.
    limits = set(float(v) for v in (config.moe_swiglu_limit or [0.0]))
    limits.add(swiglu_limit)
    clamp_mode = ("post_activation" if config.moe_swiglu_clamp_gate_after_activation
                  else "pre_activation")
    mode = clamp_mode if swiglu_limit else "none"
    if (*prefix, mode, swiglu_limit) not in _EXECUTORS:
        print(f"TT_MEGAMOE_BIND rank={dist.get_rank()} tokens={tokens} padded={padded_tokens} ep={ep_size} mode={mode}", flush=True)
    for limit in sorted(limits):
        limit_mode = clamp_mode if limit else "none"
        key = (*prefix, limit_mode, limit)
        if key not in _EXECUTORS:
            _EXECUTORS[key] = MegaMoeExperts(
                local_num_tokens=padded_tokens,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_ffn_hidden_size,
                num_experts=config.num_moe_experts,
                top_k=config.moe_router_topk,
                ep_size=ep_size,
                ep_group=ep_group,
                create_parameters=False,
                swiglu_limit=limit if limit else None,
                swiglu_mode=limit_mode,
            )
    indices = expert_indices.reshape(tokens, config.moe_router_topk)
    weights = probs.reshape_as(indices)
    padding = padded_tokens - tokens
    if padding:
        flat = F.pad(flat, (0, 0, 0, padding))
        # Distribute zero-weight padding across EP ranks to avoid artificial hotspots.
        route_key = (
            indices.device, indices.dtype, padded_tokens,
            config.moe_router_topk, config.num_moe_experts,
        )
        if route_key not in _PADDING_ROUTES:
            _PADDING_ROUTES[route_key] = torch.arange(
                padded_tokens * config.moe_router_topk,
                device=indices.device, dtype=indices.dtype,
            ).reshape(padded_tokens, config.moe_router_topk).remainder(
                config.num_moe_experts
            )
        dummy = _PADDING_ROUTES[route_key][tokens:padded_tokens]
        indices = torch.cat((indices, dummy), dim=0)
        weights = F.pad(weights, (0, 0, 0, padding))
    local_experts = config.num_moe_experts // ep_size
    output = _EXECUTORS[(*prefix, mode, swiglu_limit)](
        flat, indices, weights,
        expert_weights=(
            weight1.view(local_experts, config.hidden_size, -1),
            weight2.view(local_experts, -1, config.hidden_size),
        ),
    )
    return output[:tokens].reshape(shape)
