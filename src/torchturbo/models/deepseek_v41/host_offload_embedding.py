"""Host-offloaded engram embedding (port of FSDPTurbo PR100).

The authoritative BF16 table stays on the host, sharded over the EP group;
lookups are routed to the owning rank with all_to_all and only the requested
rows touch the accelerator. Training buffers FP32 gradients for those rows.

Only the pure-PyTorch variant (FSDPTurbo's ``host_offload`` backend, their
default) is ported; the CANN ElasticBuffer variant (``elastic_buffer``) is a
future option.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from ...logging import get_logger

logger = get_logger(__name__)

_TABLE_ROW_ALIGNMENT = 2048


def _torch_host_fetch(weight, row_ids, table, ctx=None):
    """EP-routed CPU lookup using only PyTorch collectives and copies."""
    flat_ids = row_ids.reshape(-1)
    if flat_ids.numel() and (
        bool((flat_ids < 0).any())
        or bool((flat_ids >= table.logical_num_embeddings).any())
    ):
        raise IndexError(
            f"Engram row IDs must be in [0, {table.logical_num_embeddings})."
        )

    unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
    owners = torch.div(unique_ids, table.local_rows, rounding_mode="floor")
    order = torch.argsort(owners)
    send_ids = unique_ids[order]
    send_counts = torch.bincount(owners, minlength=table.ep_size)
    recv_counts = torch.empty_like(send_counts)
    group = table.ep_group
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_sizes = send_counts.tolist()
    recv_sizes = recv_counts.tolist()
    recv_ids = torch.empty(sum(recv_sizes), device=row_ids.device, dtype=row_ids.dtype)
    dist.all_to_all_single(
        recv_ids,
        send_ids,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=group,
    )

    local_ids = (recv_ids - table.row_start).to(device="cpu", dtype=torch.long)
    if table.pin_memory and weight.is_pinned():
        values_cpu = torch.empty(
            local_ids.numel(),
            table.embedding_dim,
            dtype=weight.dtype,
            device="cpu",
            pin_memory=True,
        )
        torch.index_select(weight.detach(), 0, local_ids, out=values_cpu)
    else:
        values_cpu = weight.detach().index_select(0, local_ids)
    values = values_cpu.to(device=row_ids.device, non_blocking=table.pin_memory)
    selected_ordered = values.new_empty((sum(send_sizes), table.embedding_dim))
    dist.all_to_all_single(
        selected_ordered,
        values,
        output_split_sizes=send_sizes,
        input_split_sizes=recv_sizes,
        group=group,
    )
    selected = selected_ordered[torch.argsort(order)]
    if ctx is not None:
        ctx.table = table
        ctx.send_sizes = send_sizes
        ctx.recv_sizes = recv_sizes
        ctx.save_for_backward(inverse, order, local_ids)
    return selected[inverse]


class _TrainableHostFetch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, row_ids, table):
        return _torch_host_fetch(weight, row_ids, table, ctx)

    @staticmethod
    def backward(ctx, grad_output):
        inverse, order, local_ids = ctx.saved_tensors
        table = ctx.table
        unique_grads = grad_output.new_zeros((order.numel(), table.embedding_dim), dtype=torch.float32)
        unique_grads.index_add_(0, inverse, grad_output.reshape(-1, table.embedding_dim).float())
        owner_grads = unique_grads.new_empty((sum(ctx.recv_sizes), table.embedding_dim))
        dist.all_to_all_single(
            owner_grads, unique_grads[order],
            output_split_sizes=ctx.recv_sizes, input_split_sizes=ctx.send_sizes,
            group=table.ep_group,
        )
        table.pending_gradients.append((local_ids, owner_grads.cpu()))
        return None, None, None


class TorchHostOffloadEmbedding(nn.Module):
    """CPU BF16 engram table sharded over EP and served on demand.

    Built on the meta device; ``shard_`` materializes the owner-local CPU
    slice (call it from the model wrapper once the EP group is known), and
    ``load_rows`` fills it from the checkpoint. The nn.Parameter exists so
    checkpoint tooling sees the table shape; it must never be FSDP-managed
    (the parallel plan keeps the engram module no_shard).
    """

    uses_host_offload = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        layer_id: int,
        num_max_tokens_per_rank: int,
        pin_memory: bool = True,
    ):
        super().__init__()
        if num_max_tokens_per_rank <= 0:
            raise ValueError("Host-offload engram requires a positive lookup capacity.")
        physical_num_embeddings = (
            (num_embeddings + _TABLE_ROW_ALIGNMENT - 1) // _TABLE_ROW_ALIGNMENT
        ) * _TABLE_ROW_ALIGNMENT
        self.logical_num_embeddings = num_embeddings
        self.num_embeddings = physical_num_embeddings
        self.embedding_dim = embedding_dim
        self.layer_id = layer_id
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.pin_memory = pin_memory
        self.hash_fingerprint: Optional[str] = None
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, embedding_dim, dtype=torch.bfloat16, device="meta")
        )
        self.ep_group = None
        self.ep_size = 1
        self.ep_rank = 0
        self.row_start = 0
        self.local_rows = physical_num_embeddings
        self.pending_gradients: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shard_(self, ep_group):
        """Materialize the owner-local CPU slice on the EP group."""
        if self.ep_group is not None:
            if self.ep_group is not ep_group:
                raise ValueError("Host engram cannot change EP topology after materialization.")
            return self
        # A size-1 group stays a real group: the all_to_all routing inside
        # the fetch degenerates to an identity, which keeps single-rank A/B
        # runs on the same code path.
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        if self.num_embeddings % self.ep_size:
            # Pad rows are never looked up (logical bound is checked in the
            # fetch), but the even split keeps every shard the same size.
            raise ValueError(
                f"Host engram rows ({self.num_embeddings}) must be divisible by EP ({self.ep_size}); "
                "raise engram_num_embeddings to the next 2048-multiple."
            )
        self.local_rows = self.num_embeddings // self.ep_size
        self.row_start = self.ep_rank * self.local_rows
        try:
            local = torch.empty(
                self.local_rows,
                self.embedding_dim,
                dtype=self.weight.dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            if not self.pin_memory:
                raise
            logger.warning("Pinned host allocation unavailable; using pageable engram storage.")
            local = torch.empty(self.local_rows, self.embedding_dim, dtype=self.weight.dtype)
        self.weight = nn.Parameter(local, requires_grad=self.weight.requires_grad)
        self.weight._engram_host_offload = True
        return self

    def load_rows(self, model_path: str, checkpoint_key: str):
        """Fill the owner-local slice from the (full-table) checkpoint shard.

        Rows beyond the logical table (alignment padding) stay zero.
        """
        import json
        from pathlib import Path

        from safetensors.torch import safe_open

        index_path = Path(model_path) / "model.safetensors.index.json"
        with open(index_path) as f:
            filename = json.load(f)["weight_map"][checkpoint_key]
        row_end = min(self.row_start + self.local_rows, self.logical_num_embeddings)
        chunk_rows = max(1, (1 << 29) // (self.embedding_dim * 2))
        with safe_open(Path(model_path) / filename, framework="pt") as f:
            table_slice = f.get_slice(checkpoint_key)
            for start in range(self.row_start, row_end, chunk_rows):
                end = min(start + chunk_rows, row_end)
                rows = table_slice[start:end]
                self.weight.data[start - self.row_start : end - self.row_start] = rows.to(
                    dtype=self.weight.dtype
                )
        if row_end < self.row_start + self.local_rows:
            self.weight.data[row_end - self.row_start :].zero_()

    def init_rows_random(self):
        local = torch.empty(
            self.local_rows, self.embedding_dim, dtype=torch.float32, device="cpu"
        )
        nn.init.trunc_normal_(local, mean=0.0, std=0.02)
        self.weight.data.copy_(local.to(torch.bfloat16))

    def freeze(self):
        """Freeze the table (phase 1: read-only lookups, no sparse training)."""
        self.weight.requires_grad_(False)

    @torch.compiler.disable
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.ep_group is None or self.weight.is_meta:
            raise RuntimeError("Host engram must be EP-sharded before forward.")
        if indices.numel() > self.num_max_tokens_per_rank:
            raise ValueError(
                f"Engram lookup has {indices.numel()} rows but capacity is "
                f"{self.num_max_tokens_per_rank}."
            )
        if self.weight.requires_grad and torch.is_grad_enabled():
            fetched = _TrainableHostFetch.apply(self.weight, indices.reshape(-1), self)
        else:
            fetched = _torch_host_fetch(self.weight, indices.reshape(-1), self)
        return fetched.reshape(*indices.shape, self.embedding_dim)


__all__ = ["TorchHostOffloadEmbedding"]
