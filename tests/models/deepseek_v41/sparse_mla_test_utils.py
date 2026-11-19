"""Shared sparse indices and independent mathematical references."""

import torch
from torch.distributed import ProcessGroup


def make_compressed_indices(
    length: int,
    compress_ratio: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Select the most recent complete compressed blocks per query."""
    if length <= 0 or compress_ratio <= 0 or topk <= 0:
        raise ValueError("length, compress_ratio and topk must be positive.")

    positions = torch.arange(length, device=device)

    visible_counts = (positions + 1) // compress_ratio
    valid_counts = visible_counts.clamp_max(topk)

    first_indices = visible_counts - valid_counts

    slots = torch.arange(topk, device=device)

    candidates = first_indices.unsqueeze(1) + slots.unsqueeze(0)

    valid_slots = slots.unsqueeze(0) < valid_counts.unsqueeze(1)

    indices = torch.where(
        valid_slots,
        candidates,
        torch.full_like(candidates, -1),
    )

    return indices.to(torch.int32).unsqueeze(1).contiguous()


def make_far_compressed_indices(
    length: int,
    compress_ratio: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Select the first and latest visible compressed blocks."""
    if length <= 0 or compress_ratio <= 0:
        raise ValueError("length and compress_ratio must be positive.")
    if topk != 2:
        raise ValueError("This test pattern requires topk=2.")

    positions = torch.arange(length, device=device)
    visible_counts = (positions + 1) // compress_ratio

    indices = torch.full(
        (length, 2),
        -1,
        dtype=torch.int32,
        device=device,
    )

    indices[:, 0] = torch.where(
        visible_counts >= 1,
        torch.zeros_like(visible_counts),
        torch.full_like(visible_counts, -1),
    ).to(torch.int32)

    indices[:, 1] = torch.where(
        visible_counts >= 2,
        visible_counts - 1,
        torch.full_like(visible_counts, -1),
    ).to(torch.int32)

    return indices.unsqueeze(1).contiguous()


def reference(
    q: torch.Tensor,
    ori: torch.Tensor,
    cmp: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate joint SWA and explicit compressed attention in the input dtype."""
    length = q.shape[0]
    kv = torch.cat([ori[:, 0], cmp[:, 0]], dim=0)
    positions = torch.arange(length)
    distance = positions[:, None] - positions[None, :]
    window = (distance >= 0) & (distance < 128)
    compressed = torch.zeros(length, cmp.shape[0], dtype=torch.bool)
    rows, slots = torch.where(indices[:, 0] >= 0)
    compressed[rows, indices[rows, 0, slots].long()] = True
    allowed = torch.cat([window, compressed], dim=1)
    scores = torch.einsum("thd,sd->hts", q, kv) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed[None], -torch.inf)
    logits = torch.cat([scores, sink[:, None, None].expand(-1, length, 1)], dim=-1)
    probs = logits.softmax(-1)[..., :-1]
    return torch.einsum("hts,sd->thd", probs, kv)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report finite tensor errors without applying acceptance thresholds."""
    a, b = actual.detach().cpu().double(), expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite gradient.")
    delta = a - b
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / b.norm().clamp_min(1e-12)).item(),
        "elements": b.numel(),
    }


def run_sp_attention(
    query_local: torch.Tensor,
    ori_kv_local: torch.Tensor,
    sinks: torch.Tensor,
    *,
    sp_group: ProcessGroup,
    cmp_kv_global: torch.Tensor | None = None,
    cmp_indices_local: torch.Tensor | None = None,
    compress_ratio: int = 1,
    softmax_scale: float | None = None,
    cu_seqlens: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Supply model-shaped inputs and SP communication for operator tests."""
    import torch.distributed as dist

    from torchturbo.models.deepseek_v41.npu import run_cann_attention
    from torchturbo.models.modules.attention.ulysses import (
        all_gather_sp_with_grad_reduce,
    )

    size, rank = dist.get_world_size(sp_group), dist.get_rank(sp_group)
    length = query_local.shape[0]
    total = length * size
    if cu_seqlens is None:
        cu_seqlens = (0, total)
    mask = torch.zeros((1, total), dtype=torch.long, device=query_local.device)
    selected = (
        cmp_indices_local[:, 0].clone() if cmp_indices_local is not None else None
    )
    offset = 0
    for sample_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:]), 1):
        mask[:, start:end] = sample_id
        left, right = max(start, rank * length), min(end, (rank + 1) * length)
        if selected is not None and left < right:
            values = selected[left - rank * length : right - rank * length]
            selected[left - rank * length : right - rank * length] = torch.where(
                values >= 0, values + offset, values
            )
        offset += (end - start) // compress_ratio
    original = all_gather_sp_with_grad_reduce(
        ori_kv_local, dim=0, sp_group=sp_group, sp_size=size
    )
    output = run_cann_attention(
        query_local.unsqueeze(0),
        original[:, 0].unsqueeze(0),
        cmp_kv_global[:, 0].unsqueeze(0) if cmp_kv_global is not None else None,
        selected.unsqueeze(0) if selected is not None else None,
        sinks,
        softmax_scale if softmax_scale is not None else query_local.shape[-1] ** -0.5,
        compress_ratio,
        mask,
        sp_rank=rank,
    )
    return output.squeeze(0)
