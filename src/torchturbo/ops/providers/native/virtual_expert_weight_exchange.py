"""Native fallback for virtual-expert replica weight exchange."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...base import OpImpl, OpImplKind
from ...moe import ep_all_to_all
from ...registry import register_op


@dataclass(frozen=True, slots=True)
class _WeightExchangeContext:
    ep_group: dist.ProcessGroup | None
    send_indices: torch.Tensor
    send_splits: list[int]
    receive_splits: list[int]
    has_global_replicas: bool


def build_weight_exchange_context(
    copies: torch.Tensor, group: dist.ProcessGroup | None,
) -> _WeightExchangeContext:
    """Build native rank-major packet splits once during route planning."""
    copies_cpu = copies.to("cpu", dtype=torch.int64)
    num_ranks, num_local_experts = copies_cpu.shape
    rank = dist.get_rank(group) if num_ranks > 1 else 0
    copy_rows = copies_cpu.tolist()
    for row in copy_rows:
        active = [expert for expert in row if expert >= 0]
        owners = [expert // num_local_experts for expert in active]
        assert row[: len(active)] == active and owners == sorted(owners), (
            "Replica slots must be a contiguous prefix ordered by owner rank."
        )
        assert len(active) == len(set(active)), (
            "Replica slots on one rank must be distinct."
        )

    has_global_replicas = any(
        expert >= 0 for row in copy_rows for expert in row
    )
    send_indices: list[int] = []
    send_splits: list[int] = []
    for destination in range(num_ranks):
        local_indices = [
            expert % num_local_experts
            for expert in copy_rows[destination]
            if expert >= 0 and expert // num_local_experts == rank
        ]
        send_indices.extend(local_indices)
        send_splits.append(len(local_indices))

    local_replicas = [expert for expert in copy_rows[rank] if expert >= 0]
    receive_splits = [
        sum(
            1
            for expert in local_replicas
            if expert // num_local_experts == source
        )
        for source in range(num_ranks)
    ]

    return _WeightExchangeContext(
        group, torch.tensor(send_indices, dtype=torch.int64),
        send_splits, receive_splits, has_global_replicas,
    )


class _ExchangeExpertWeights(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packet: torch.Tensor,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        group: dist.ProcessGroup,
        receive_splits: list[int],
        send_splits: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.group = group
        ctx.receive_splits = receive_splits
        ctx.send_splits = send_splits
        received = ep_all_to_all(group, packet, receive_splits, send_splits)
        return received, hidden_states, probs

    @staticmethod
    def backward(
        ctx,
        grad_received: torch.Tensor,
        grad_hidden: torch.Tensor,
        grad_probs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None, None]:
        # The input/probability edges hold token-dispatch backward behind this
        # node so native and replica weight-gradient collectives run in one
        # consistent order across ranks.
        grad_packet = None
        if ctx.needs_input_grad[0]:
            grad_packet = ep_all_to_all(
                ctx.group, grad_received, ctx.send_splits, ctx.receive_splits,
            )
        return grad_packet, grad_hidden, grad_probs, None, None, None


@register_op(
    op_name="virtual_expert_weight_exchange",
    impl_name="torch_virtual_expert_weight_exchange",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchVirtualExpertWeightExchange(OpImpl):
    def __call__(
        self,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        context: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange selected native expert matrices in rank-major EP order."""
        ep_group = context.ep_group
        send_indices = context.send_indices
        send_splits = context.send_splits
        receive_splits = context.receive_splits
        if not context.has_global_replicas:
            return weight1[:0], weight2[:0], hidden_states, probs
        shape1 = weight1.shape[1:]
        shape2 = weight2.shape[1:]
        elements1 = shape1.numel()
        elements2 = shape2.numel()
        if send_indices.numel() == 0:
            packed1 = weight1[:0].reshape(0, elements1)
            packed2 = weight2[:0].reshape(0, elements2)
        else:
            indices = send_indices.to(device=weight1.device)
            packed1 = weight1.index_select(0, indices).reshape(
                indices.numel(), elements1,
            )
            packed2 = weight2.index_select(0, indices).reshape(
                indices.numel(), elements2,
            )
        packet = torch.cat((packed1, packed2), dim=1)
        if torch.is_grad_enabled() and (weight1.requires_grad or weight2.requires_grad):
            received, hidden_states, probs = _ExchangeExpertWeights.apply(
                packet,
                hidden_states,
                probs,
                ep_group,
                receive_splits,
                send_splits,
            )
        else:
            with torch.no_grad():
                received = ep_all_to_all(
                    ep_group, packet, receive_splits, send_splits,
                )
        num_replicas = received.shape[0]
        replica_weight1 = (
            received[:, :elements1].contiguous().reshape(num_replicas, *shape1)
        )
        replica_weight2 = (
            received[:, elements1:].contiguous().reshape(num_replicas, *shape2)
        )
        return replica_weight1, replica_weight2, hidden_states, probs
