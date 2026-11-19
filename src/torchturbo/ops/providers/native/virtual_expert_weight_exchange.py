"""Native fallback for virtual-expert replica weight exchange."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...base import OpImpl, OpImplKind
from ...moe import VirtualExpertPlan, ep_all_to_all
from ...registry import register_op


@dataclass(frozen=True, slots=True)
class _WeightExchangeContext:
    ep_group: dist.ProcessGroup | None
    send_indices: torch.Tensor
    send_splits: list[int]
    receive_splits: list[int]
    has_global_replicas: bool
    experts_to_copy: torch.Tensor


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
        torch.tensor(copy_rows, dtype=torch.int64),
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
        shape1: torch.Size,
        shape2: torch.Size,
        prefetched: tuple[torch.Tensor, torch.Tensor] | None,
        out: tuple[torch.Tensor, torch.Tensor] | None,
        grad_buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.group = group
        ctx.receive_splits = receive_splits
        ctx.send_splits = send_splits
        ctx.elements1 = shape1.numel()
        ctx.elements2 = shape2.numel()
        ctx.grad_buffer = grad_buffer
        if prefetched is None:
            received = ep_all_to_all(group, packet, receive_splits, send_splits)
            num_replicas = received.shape[0]
            if out is None:
                replica1 = received[:, :ctx.elements1].contiguous().reshape(num_replicas, *shape1)
                replica2 = received[:, ctx.elements1:].contiguous().reshape(num_replicas, *shape2)
            else:
                replica1, replica2 = (tensor[:num_replicas] for tensor in out)
                replica1.copy_(received[:, :ctx.elements1].reshape(num_replicas, *shape1))
                replica2.copy_(received[:, ctx.elements1:].reshape(num_replicas, *shape2))
        else:
            # Each graph needs its own version counter: the shared arena is
            # overwritten by other layers and restored before this graph reads
            # its saved weights. set_ aliases storage without sharing a counter.
            replica1, replica2 = (
                tensor.new_empty(0).set_(
                    tensor.untyped_storage(), tensor.storage_offset(),
                    tensor.size(), tensor.stride(),
                )
                for tensor in prefetched
            )
        return replica1, replica2, hidden_states, probs

    @staticmethod
    def backward(
        ctx,
        grad_replica1: torch.Tensor,
        grad_replica2: torch.Tensor,
        grad_hidden: torch.Tensor,
        grad_probs: torch.Tensor,
    ) -> tuple:
        # The input/probability edges hold token-dispatch backward behind this
        # node so native and replica weight-gradient collectives run in one
        # consistent order across ranks.
        grad_packet = None
        if ctx.needs_input_grad[0]:
            num_replicas = grad_replica1.shape[0]
            grad1 = grad_replica1.reshape(num_replicas, ctx.elements1)
            grad2 = grad_replica2.reshape(num_replicas, ctx.elements2)
            if ctx.grad_buffer is None:
                grad_received = torch.cat((grad1, grad2), dim=1)
            else:
                packet_size = ctx.elements1 + ctx.elements2
                grad_received = ctx.grad_buffer[:num_replicas * packet_size].view(
                    num_replicas, packet_size,
                )
                grad_received[:, :ctx.elements1].copy_(grad1)
                grad_received[:, ctx.elements1:].copy_(grad2)
            grad_packet = ep_all_to_all(
                ctx.group, grad_received, ctx.send_splits, ctx.receive_splits,
            )
        return grad_packet, grad_hidden, grad_probs, None, None, None, None, None, None, None, None


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
        hidden_states: torch.Tensor | None,
        probs: torch.Tensor | None,
        context: object,
        *,
        prefetched: tuple[torch.Tensor, torch.Tensor] | None = None,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
        grad_buffer: torch.Tensor | None = None,
        buffer_handles: tuple[object, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Exchange selected native expert matrices in rank-major EP order."""
        if isinstance(context, VirtualExpertPlan):
            context = context.wait().weight_context
        ep_group = context.ep_group
        send_indices = context.send_indices
        send_splits = context.send_splits
        receive_splits = context.receive_splits
        if not context.has_global_replicas:
            empty1, empty2 = weight1[:0], weight2[:0]
            if not torch.is_grad_enabled():
                # No-grad views can inherit requires_grad from their source.
                empty1, empty2 = empty1.detach(), empty2.detach()
            return empty1, empty2, hidden_states, probs
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
            replica_weight1, replica_weight2, hidden_states, probs = _ExchangeExpertWeights.apply(
                packet,
                hidden_states,
                probs,
                ep_group,
                receive_splits,
                send_splits,
                shape1,
                shape2,
                prefetched,
                out,
                grad_buffer,
            )
        else:
            with torch.no_grad():
                replica_weight1, replica_weight2, _, _ = _ExchangeExpertWeights.apply(
                    packet, hidden_states, probs, ep_group,
                    receive_splits, send_splits, shape1, shape2, prefetched, out, grad_buffer,
                )
        return replica_weight1, replica_weight2, hidden_states, probs
