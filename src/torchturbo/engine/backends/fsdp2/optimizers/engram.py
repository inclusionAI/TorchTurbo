"""Owner-local ActiveRowAdam for CPU Engram tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

_BLOCK_ROWS = 4096


class ActiveRowAdam:
    """Allocate FP32 moments only for rows visited during training."""

    def __init__(self) -> None:
        self.step_count = 0
        self.slots: dict[int, int] = {}
        self.moments: list[tuple[torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def step(self, weight: torch.Tensor, ids: torch.Tensor, grads: torch.Tensor,
             lr: float, betas: tuple[float, float], eps: float) -> None:
        """Apply SparseAdam's active-row update with a per-table step counter."""
        self.step_count += 1
        beta1, beta2 = betas
        step_size = lr * (1 - beta2 ** self.step_count) ** 0.5 / (1 - beta1 ** self.step_count)
        batches = defaultdict(list)
        for index, row in enumerate(ids.tolist()):
            slot = self.slots.setdefault(row, len(self.slots))
            block, offset = divmod(slot, _BLOCK_ROWS)
            if block == len(self.moments):
                shape = (_BLOCK_ROWS, weight.size(1))
                self.moments.append(tuple(torch.zeros(shape, dtype=torch.float32, device="cpu") for _ in range(2)))
            batches[block].append((index, offset))
        for block, batch in batches.items():
            indices, offsets = torch.tensor(batch, dtype=torch.long, device="cpu").T
            rows = ids[indices]
            grad = grads[indices]
            first, second = self.moments[block]
            mean = first[offsets].mul_(beta1).add_(grad, alpha=1 - beta1)
            variance = second[offsets].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            first.index_copy_(0, offsets, mean)
            second.index_copy_(0, offsets, variance)
            updated = weight[rows].float().addcdiv_(mean, variance.sqrt().add_(eps), value=-step_size)
            weight.index_copy_(0, rows, updated.to(weight.dtype))


class EngramOptimizer:
    """Synchronize sparse rows and follow the dense optimizer's scheduled LR."""

    def __init__(self, tables: tuple, replica_group: dist.ProcessGroup, device: torch.device) -> None:
        self.tables = tables
        self.replica_group = replica_group
        self.replica_size = dist.get_world_size(replica_group)
        self.device = device
        self.states = [ActiveRowAdam() for _ in tables]
        self.gradients: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def prepare_gradients(self) -> None:
        """Reduce duplicate rows and average over all token-producing ranks."""
        if self.gradients is not None or not any(table.pending_gradients for table in self.tables):
            return
        gradients = []
        for table in self.tables:
            ids = torch.cat([ids for ids, _ in table.pending_gradients])
            values = torch.cat([values for _, values in table.pending_gradients])
            ids, values = self._merge(ids, values)
            if self.replica_size > 1:
                count = torch.tensor([ids.numel()], device=self.device, dtype=torch.long)
                counts = [torch.empty_like(count) for _ in range(self.replica_size)]
                dist.all_gather(counts, count, group=self.replica_group)
                lengths = [int(item.item()) for item in counts]
                width = max(lengths)
                if width:
                    padded_ids = torch.zeros(width, device=self.device, dtype=torch.long)
                    padded_values = torch.zeros(
                        (width, table.embedding_dim), dtype=torch.float32, device=self.device,
                    )
                    padded_ids[:ids.numel()].copy_(ids)
                    padded_values[:ids.numel()].copy_(values)
                    gathered_ids = [torch.empty_like(padded_ids) for _ in lengths]
                    gathered_values = [torch.empty_like(padded_values) for _ in lengths]
                    dist.all_gather(gathered_ids, padded_ids, group=self.replica_group)
                    dist.all_gather(gathered_values, padded_values, group=self.replica_group)
                    ids, values = self._merge(
                        torch.cat([part[:n].cpu() for part, n in zip(gathered_ids, lengths)]),
                        torch.cat([part[:n].cpu() for part, n in zip(gathered_values, lengths)]),
                    )
            values.div_(table.ep_size * self.replica_size)
            gradients.append((ids, values))
        self.gradients = gradients

    @staticmethod
    def _merge(ids: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, inverse = torch.unique(ids, return_inverse=True)
        merged = values.new_zeros((rows.numel(), values.size(1)))
        return rows, merged.index_add_(0, inverse, values)

    def norm(self) -> torch.Tensor:
        """Compute the Engram norm across EP owners exactly once."""
        squared = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.gradients is not None:
            for _, values in self.gradients:
                squared.add_(values.square().sum().to(self.device))
        dist.all_reduce(squared, group=self.tables[0].ep_group)
        return squared.sqrt()

    def scale(self, coefficient: float) -> None:
        """Apply the same clipping coefficient as dense parameters."""
        if self.gradients is not None:
            for _, values in self.gradients:
                values.mul_(coefficient)

    def after_step(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        """Update Engram once per dense optimizer step."""
        if self.gradients is None:
            return
        group = optimizer.param_groups[0]
        for table, state, (ids, values) in zip(self.tables, self.states, self.gradients):
            state.step(table.weight, ids, values, group["lr"], group["betas"], group["eps"])
            table.pending_gradients.clear()
        self.gradients = None

    def _geometry(self) -> list:
        return [(table.layer_id, table.ep_size, table.ep_rank, table.row_start,
                 tuple(table.weight.shape), table.hash_fingerprint) for table in self.tables]

    def save(self, output_dir: str) -> None:
        """Save FP32 moments once per EP owner, with shard identity."""
        if dist.get_rank(self.replica_group) == 0:
            path = Path(output_dir) / f"engram_optimizer_ep{self.tables[0].ep_rank}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save({"geometry": self._geometry(), "states": [vars(state) for state in self.states]}, temporary)
            temporary.replace(path)
        dist.barrier()

    def load(self, checkpoint_dir: str) -> None:
        """Restore active rows with the same EP geometry and hash layout."""
        path = Path(checkpoint_dir) / f"engram_optimizer_ep{self.tables[0].ep_rank}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["geometry"] != self._geometry() or len(payload["states"]) != len(self.states):
            raise ValueError("Engram optimizer checkpoint has incompatible EP shards or hash layout.")
        for state, saved in zip(self.states, payload["states"]):
            vars(state).update(saved)
