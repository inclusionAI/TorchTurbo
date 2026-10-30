"""Autograd routing with a shared histogram and permutation kernel."""
from typing import Any
import torch
import torch_npu
from torchturbo.optimizations.mega_moe.runtime.torch import ops as multicore_ops


class LocalRouting(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, ids: torch.Tensor, experts: int) -> tuple:
        routed, mapping, counts, _ = torch_npu.npu_moe_init_routing_v2(
            hidden, ids, active_num=ids.numel(), expert_num=experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, experts], row_idx_type=0,
        )
        mapping = mapping.reshape(-1)
        ctx.save_for_backward(mapping)
        ctx.tokens, ctx.topk = ids.shape
        ctx.mark_non_differentiable(mapping, counts)
        return routed, mapping, counts

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor, _mapping: Any, _counts: Any) -> tuple:
        (mapping,) = ctx.saved_tensors
        dx = multicore_ops.moe_token_permute_grad(
            grad.contiguous(), mapping, ctx.tokens, ctx.topk
        )
        return dx, None, None
