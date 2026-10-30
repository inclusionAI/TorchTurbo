"""PyTorch routed SwiGLU experts with differentiable token all-to-all."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(op_name="mega_moe", impl_name="torch_mega_moe", kind=OpImplKind.NATIVE,
             priority=0, devices=())
class TorchMegaMoe(OpImpl):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        *,
        experts: nn.Module,
        backend: Any,
    ) -> torch.Tensor:
        from torchturbo.models.modules.comm.comm import all_to_all

        config = experts.config
        if (config.activation_func is not F.silu or not config.gated_linear_unit
                or config.moe_apply_probs_on_input):
            raise ValueError("mega_moe requires SwiGLU experts with output-side routing weights.")
        group = backend.ep_group
        ranks = dist.get_world_size(group)
        num_local = experts.num_local_experts
        flat = hidden_states.reshape(-1, config.hidden_size)
        ids = expert_indices.reshape(-1).long()
        order = ids.argsort(stable=True)
        rows = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(
            expert_indices.shape[-1]
        )
        counts = torch.bincount(ids, minlength=config.num_moe_experts)
        received_counts = torch.empty_like(counts)
        dist.all_to_all_single(received_counts, counts, group=group)
        send_splits = counts.reshape(ranks, num_local).sum(1).tolist()
        receive_splits = received_counts.reshape(ranks, num_local).sum(1).tolist()
        received = all_to_all(group, flat[rows[order]], receive_splits, send_splits)
        received_ids = torch.repeat_interleave(
            torch.arange(num_local, device=flat.device).repeat(ranks), received_counts
        )
        expert_order = received_ids.argsort(stable=True)
        local_counts = received_counts.reshape(ranks, num_local).sum(0).tolist()
        weight1 = experts.weight1.to(flat.dtype).reshape(num_local, config.hidden_size, -1)
        weight2 = experts.weight2.to(flat.dtype).reshape(num_local, -1, config.hidden_size)
        outputs = []
        for index, tokens in enumerate(torch.split(received[expert_order], local_counts)):
            gate, up = (tokens @ weight1[index]).chunk(2, dim=-1)
            limit = experts.swiglu_limit
            if limit and not config.moe_swiglu_clamp_gate_after_activation:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            gate = F.silu(gate)
            if limit and config.moe_swiglu_clamp_gate_after_activation:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            outputs.append((gate * up) @ weight2[index])
        output = torch.cat(outputs, dim=0)[expert_order.argsort()]
        returned = all_to_all(group, output, send_splits, receive_splits)
        returned = returned[order.argsort()].reshape(*expert_indices.shape, config.hidden_size)
        output = (returned * probs.to(flat.dtype).unsqueeze(-1)).sum(-2)
        return output.reshape_as(hidden_states)
