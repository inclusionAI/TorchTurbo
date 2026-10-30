"""FlexMoE-style periodic expert re-placement for MoE expert parallelism.

Routing skew leaves some EP ranks with hotter expert sets than others, which
surfaces as all-to-all wait time. This module periodically re-places experts
across the fixed slot grid (``ep_size`` ranks x ``num_local_experts`` slots
per rank): hot and cold experts are packed onto the same ranks by an LPT
solver over an EMA of observed token counts. Slot counts never change, so
parameter shapes, optimizer bucket layouts, and the dispatcher's chunk tables
all stay valid; only the expert id -> slot mapping moves, and expert weights
(together with their optimizer momentum) follow their expert id.

The token dispatcher sees slot ids only: at dispatch entry the routing result
is remapped from expert-id space to slot space, so the entire downstream
pipeline (all-to-all splits, chunk sort, grouped GEMM, unpermutation) runs
unchanged on whichever placement is active.

Re-placement runs between optimizer steps, driven by the training loop's
step-end callback; forward and backward always see one stable placement.
"""

from __future__ import annotations

import heapq
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from torchturbo.logging import get_logger

logger = get_logger(__name__)


def _inverse_perm(placement: list[int]) -> list[int]:
    """Slot-of-expert inversion of a slot -> expert placement."""
    slot_of = [0] * len(placement)
    for slot, expert in enumerate(placement):
        slot_of[expert] = slot
    return slot_of


def solve_lpt(
    loads: list[float], ep_size: int, num_local_experts: int
) -> list[int]:
    """Pack experts into EP ranks by load, returning slot -> expert id.

    Longest-processing-time-first greedy with fixed per-rank capacity. Ties
    break toward lower expert ids and lower ranks so every rank reproduces
    the identical placement from identical loads.
    """
    num_experts = len(loads)
    if num_experts != ep_size * num_local_experts:
        raise ValueError(
            f"expert count {num_experts} does not fill the slot grid "
            f"{ep_size}x{num_local_experts}"
        )
    order = sorted(range(num_experts), key=lambda e: (-loads[e], e))
    heap: list[tuple[float, int]] = [(0.0, rank) for rank in range(ep_size)]
    heapq.heapify(heap)
    capacity = [num_local_experts] * ep_size
    slots: list[list[int]] = [[] for _ in range(ep_size)]
    for expert in order:
        load, rank = heapq.heappop(heap)
        while capacity[rank] == 0:
            load, rank = heapq.heappop(heap)
        slots[rank].append(expert)
        capacity[rank] -= 1
        heapq.heappush(heap, (load + loads[expert], rank))
    return [expert for rank_slots in slots for expert in rank_slots]


class ExpertPlacementManager:
    """Process-wide owner of the expert-to-slot placement for all MoE layers.

    Every layer's dispatcher shares the singleton: slot ids are stable within
    a step and identical across layers, so one placement update migrates all
    layers consistently.
    """

    _instance: Optional["ExpertPlacementManager"] = None

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        num_local_experts: int,
        ep_group: Any,
        interval_steps: int,
        ema_decay: float,
    ) -> None:
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.ep_group = ep_group
        self.interval_steps = interval_steps
        self.ema_decay = ema_decay
        self._device: Optional[torch.device] = None
        self._placement: Optional[torch.Tensor] = None  # slot -> expert id
        self._slot_of_expert: Optional[torch.Tensor] = None
        self._step_counts: Optional[torch.Tensor] = None
        self._ema_load: Optional[torch.Tensor] = None
        self._steps = 0
        self._migrations = 0
        # (ep_pos, edp_pos) -> global rank, built on first migration
        self._rank_grid: Optional[dict[tuple[int, int], int]] = None

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def activate(
        cls,
        num_experts: int,
        ep_size: int,
        num_local_experts: int,
        ep_group: Any,
        interval_steps: int,
        ema_decay: float,
    ) -> "ExpertPlacementManager":
        """Create or return the singleton, validating consistent topology."""
        if cls._instance is None:
            cls._instance = cls(
                num_experts,
                ep_size,
                num_local_experts,
                ep_group,
                interval_steps,
                ema_decay,
            )
            return cls._instance
        manager = cls._instance
        if (
            manager.num_experts != num_experts
            or manager.ep_size != ep_size
            or manager.num_local_experts != num_local_experts
        ):
            raise RuntimeError("inconsistent MoE expert topology across layers")
        return manager

    @classmethod
    def active(cls) -> Optional["ExpertPlacementManager"]:
        """Return the singleton when expert re-placement is enabled."""
        return cls._instance

    @classmethod
    def _reset_for_testing(cls) -> None:
        cls._instance = None

    def _ensure_device(self, device: torch.device) -> None:
        if self._device is not None:
            return
        self._device = device
        arange = torch.arange(self.num_experts, dtype=torch.long, device=device)
        self._placement = arange.clone()
        self._slot_of_expert = arange.clone()
        self._step_counts = torch.zeros(
            self.num_experts, dtype=torch.float32, device=device
        )
        self._ema_load = torch.zeros(
            self.num_experts, dtype=torch.float32, device=device
        )

    # --------------------------------------------------- dispatcher interface
    @property
    def is_identity(self) -> bool:
        """Whether the placement equals the compiled-in contiguous layout."""
        return self._migrations == 0

    @property
    def slot_of_expert(self) -> torch.Tensor:
        assert self._slot_of_expert is not None
        return self._slot_of_expert

    @property
    def expert_of_slot(self) -> torch.Tensor:
        assert self._placement is not None
        return self._placement

    def remap_routing(self, routing: Any) -> Any:
        """Express a routing result in slot space; identity placement is free."""
        if self.is_identity:
            return routing
        from .utils import RoutingResult

        placement = self._placement.to(routing.routing_map.device)
        probs = routing.probs
        if probs is not None:
            probs = probs.index_select(1, placement.to(probs.device))
        return RoutingResult(
            probs=probs,
            routing_map=routing.routing_map.index_select(1, placement),
            expert_indices=self._slot_of_expert.to(routing.expert_indices.device)[
                routing.expert_indices
            ],
        )

    def accumulate_counts(self, counts: torch.Tensor) -> None:
        """Add one layer's per-expert token counts in expert-id space."""
        self._ensure_device(counts.device)
        self._step_counts += counts.to(torch.float32)

    def remap_indices(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Map per-token expert ids to slot ids; identity placement is free."""
        if self.is_identity:
            return expert_indices
        return self._slot_of_expert.to(expert_indices.device)[expert_indices]

    def slot_counts(self, true_counts: torch.Tensor) -> torch.Tensor:
        """Reorder true-id expert counts into slot order."""
        if self.is_identity:
            return true_counts
        return true_counts.index_select(
            0, self._placement.to(true_counts.device)
        )

    # -------------------------------------------------------- decision + move
    def on_step_end(self, model: Any, optimizer: Any) -> None:
        """Accumulate a step, then solve and migrate on the interval."""
        if self._step_counts is None:
            return
        self._steps += 1
        if self._steps % self.interval_steps != 0:
            return
        with torch.no_grad():
            loads = self._step_counts.clone()
            dist.all_reduce(loads, op=dist.ReduceOp.SUM)
            self._step_counts.zero_()
            if self._migrations == 0 and self._steps == self.interval_steps:
                self._ema_load = loads
            else:
                self._ema_load.mul_(self.ema_decay).add_(
                    loads, alpha=1 - self.ema_decay
                )
            new_placement = solve_lpt(
                self._ema_load.tolist(), self.ep_size, self.num_local_experts
            )
            if new_placement == self._placement.tolist():
                return
            self._migrate(model, optimizer, new_placement)
            self._set_placement(new_placement)

    def _set_placement(self, placement: list[int]) -> None:
        assert self._placement is not None and self._slot_of_expert is not None
        new = torch.tensor(placement, dtype=torch.long, device=self._device)
        self._placement.copy_(new)
        self._slot_of_expert.scatter_(
            0, new, torch.arange(self.num_experts, dtype=torch.long,
                                 device=self._device)
        )
        self._migrations += 1

    # ------------------------------------------------------------ migration
    def _rank_grid_for(self, dt: DTensor) -> dict[tuple[int, int], int]:
        """Map (ep_position, edp_position) to global rank for expert tensors."""
        if self._rank_grid is not None:
            return self._rank_grid
        mesh_ranks = dt.device_mesh.mesh.flatten().tolist()
        my_ep_pos = dist.get_rank(self.ep_group)
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (my_ep_pos, mesh_ranks))
        grid: dict[tuple[int, int], int] = {}
        for ep_pos, ranks in gathered:
            for edp_pos, global_rank in enumerate(ranks):
                grid[(ep_pos, edp_pos)] = global_rank
        self._rank_grid = grid
        return grid

    def _migrate(self, model: Any, optimizer: Any, new_placement: list[int]) -> None:
        """Move expert weights and Muon momentum to their new owner ranks."""
        from .experts import GroupedMLP

        assert self._placement is not None
        old_placement = self._placement.tolist()
        old_slot_of = _inverse_perm(old_placement)
        new_slot_of = _inverse_perm(new_placement)
        moved = sum(
            old_slot_of[e] // self.num_local_experts
            != new_slot_of[e] // self.num_local_experts
            for e in range(self.num_experts)
        )
        layers = [m for m in model.modules() if isinstance(m, GroupedMLP)]
        optimizers = getattr(optimizer, "optimizers", [optimizer])
        for layer in layers:
            for param in (layer.weight1, layer.weight2):
                momentum = None
                for opt in optimizers:
                    state = opt.state.get(param)
                    if state is not None and "momentum_buffer" in state:
                        momentum = state["momentum_buffer"]
                        break
                for tensor in (param, momentum):
                    if tensor is None:
                        continue
                    self._migrate_dtensor(tensor, old_placement, new_placement)
        logger.info_rank0(
            "[expert-rebalance] step=%d experts_moved=%d/%d layers=%d",
            self._steps, moved, self.num_experts, len(layers),
        )

    def _migrate_dtensor(
        self,
        dt: Any,
        old_placement: list[int],
        new_placement: list[int],
    ) -> None:
        """Re-home one sharded expert tensor to the new placement via A2A.

        Both sides pack expert blocks in ascending expert id per peer rank,
        so sender and receiver agree on block order without extra metadata.
        """
        if not isinstance(dt, DTensor):
            raise RuntimeError(
                "expert re-placement requires DTensor expert state; "
                "disable it under CPU offload"
            )
        grid = self._rank_grid_for(dt)
        world = dist.get_world_size()
        global_rank = dist.get_rank()
        num_local = self.num_local_experts
        edp_size = dt.device_mesh.mesh.numel()
        if num_local % edp_size != 0:
            raise RuntimeError(
                f"expert slots per rank ({num_local}) must divide evenly over "
                f"the expert-DP mesh ({edp_size})"
            )
        slots_per_shard = num_local // edp_size
        local = dt.to_local()
        if dt.shape[0] % num_local != 0:
            raise RuntimeError(
                f"expert tensor rows ({dt.shape[0]}) must divide evenly over "
                f"local expert slots ({num_local})"
            )
        rows_per_expert = dt.shape[0] // num_local
        block_numel = rows_per_expert * local[0].numel()
        my_ep_pos = dist.get_rank(self.ep_group)
        my_mesh_ranks = dt.device_mesh.mesh.flatten().tolist()
        my_edp_pos = my_mesh_ranks.index(global_rank)
        shard_start = my_edp_pos * slots_per_shard

        old_slot_of = [0] * self.num_experts
        for slot, expert in enumerate(old_placement):
            old_slot_of[expert] = slot
        new_slot_of = [0] * self.num_experts
        for slot, expert in enumerate(new_placement):
            new_slot_of[expert] = slot

        def holder(slot: int) -> int:
            return grid[(slot // num_local, (slot % num_local) // slots_per_shard)]

        # My shard rows currently hold old slots [shard_start, +ps); after
        # migration the same rows must hold the new placement's experts.
        send_experts: dict[int, list[int]] = {}
        recv_experts: dict[int, list[int]] = {}
        for j in range(shard_start, shard_start + slots_per_shard):
            expert = old_placement[my_ep_pos * num_local + j]
            send_experts.setdefault(holder(new_slot_of[expert]), []).append(expert)
        for j in range(shard_start, shard_start + slots_per_shard):
            expert = new_placement[my_ep_pos * num_local + j]
            recv_experts.setdefault(holder(old_slot_of[expert]), []).append(expert)

        send_sizes = [
            len(send_experts.get(rank, ())) * block_numel for rank in range(world)
        ]
        recv_sizes = [
            len(recv_experts.get(rank, ())) * block_numel for rank in range(world)
        ]
        send_rows = [
            local[
                (old_slot_of[expert] % num_local - shard_start)
                * rows_per_expert:
                (old_slot_of[expert] % num_local - shard_start + 1)
                * rows_per_expert
            ].reshape(-1)
            for rank in range(world)
            for expert in sorted(send_experts.get(rank, ()))
        ]
        send_buf = (
            torch.cat(send_rows) if send_rows else local.new_empty(0)
        )
        recv_buf = local.new_empty(sum(recv_sizes))
        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes=recv_sizes, input_split_sizes=send_sizes,
        )

        offset = 0
        for rank in range(world):
            for expert in sorted(recv_experts.get(rank, ())):
                j = new_slot_of[expert] % num_local
                target = local[
                    (j - shard_start) * rows_per_expert:
                    (j - shard_start + 1) * rows_per_expert
                ]
                target.copy_(
                    recv_buf[offset:offset + block_numel].view_as(target)
                )
                offset += block_numel


def activate_from_config(config: Any, ep_group: Any) -> None:
    """Activate the singleton from a model MoEConfig; no-op when disabled."""
    if not config.moe_expert_rebalance:
        return
    ep_size = config.expert_model_parallel_size or 1
    num_local = config.num_moe_experts // ep_size
    ExpertPlacementManager.activate(
        config.num_moe_experts,
        ep_size,
        num_local,
        ep_group,
        config.moe_expert_rebalance_interval_steps,
        config.moe_expert_rebalance_ema_decay,
    )
