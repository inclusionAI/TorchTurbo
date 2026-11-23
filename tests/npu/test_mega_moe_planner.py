"""NPU planner tests against an independent host integer reference."""
import os
import pytest
import importlib.util
from pathlib import Path
import runpy

_component_spec = importlib.util.find_spec("tt_megamoe")
if _component_spec is not None:
    runpy.run_path(str(Path(_component_spec.origin).parent / "bootstrap.py"))["activate"]()
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(os.environ.get("TT_MEGAMOE_TEST_NPU") != "1", reason="Explicit NPU test opt-in required")
from tt_megamoe.moonep import load_allocator


def _reference(counts, capacity):
    ranks, experts = counts.shape
    local = experts // ranks
    totals = counts.long().sum(0).tolist()
    allocation = [[0] * experts for _ in range(ranks)]
    transfers = [[0] * ranks for _ in range(ranks)]
    for e, total in enumerate(totals):
        allocation[e // local][e] = total
    balance = [sum(totals[r * local:(r + 1) * local]) - capacity for r in range(ranks)]
    for _ in range(2 * ranks):
        high = max(range(ranks), key=balance.__getitem__)
        low = min(range(ranks), key=balance.__getitem__)
        if balance[high] <= 0:
            break
        move = -balance[low]
        transfers[high][low] = move
        balance[high] -= move
        balance[low] = 0
    for home in range(ranks):
        remaining = totals[home * local:(home + 1) * local].copy()
        quota = transfers[home].copy()
        for _ in range(local + ranks):
            dest = max(range(ranks), key=quota.__getitem__)
            if quota[dest] <= 0:
                break
            j = max(range(local), key=remaining.__getitem__)
            expert = home * local + j
            take = min(remaining[j], quota[dest])
            allocation[dest][expert] += take
            allocation[home][expert] -= take
            remaining[j] -= take
            quota[dest] -= take
    offsets = torch.zeros((ranks, experts), dtype=torch.int32)
    boundaries = torch.empty((ranks, experts + local), dtype=torch.int32)
    copies = torch.full((ranks, local), -1, dtype=torch.int32)
    for dest in range(ranks):
        candidates = [e for e in range(experts) if e // local != dest and allocation[dest][e] > 0]
        chosen = sorted(candidates, key=lambda e: (allocation[dest][e], e), reverse=True)[:local]
        copies[dest, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        end = 0
        for group, expert in enumerate(list(range(experts)) + copies[dest].tolist()):
            count = 0 if expert < 0 or (group < experts and expert in chosen) else allocation[dest][expert]
            if count:
                offsets[dest, expert] = end
            end += count
            boundaries[dest, group] = end
    cumulative = torch.tensor(allocation, dtype=torch.int32).T.cumsum(1, dtype=torch.int32)
    return cumulative, offsets, boundaries, copies


def test_ds41_large_expert_routing():
    ranks, experts = 64, 384
    from tt_megamoe.modules.mega_moe.metadata import route_summary
    from torchturbo.optimizations.mega_moe.moonep.render import render_route
    torch.npu.set_device(0)
    tokens, topk = 257, min(6, experts // ranks)
    ids = torch.arange(topk, device="npu", dtype=torch.int32).expand(tokens, topk).contiguous()
    hidden = torch.zeros((tokens, 32), device="npu", dtype=torch.bfloat16)
    _, inverse, histogram, _ = torch_npu.npu_moe_init_routing_v2(
        hidden, ids, active_num=ids.numel(), expert_num=experts,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, experts], row_idx_type=0,
    )
    histogram = histogram.int()
    gathered = histogram.expand(ranks, experts).contiguous()
    summary = route_summary(gathered)
    assert summary[0] > tokens * topk
    expected = _reference(gathered.cpu(), tokens * topk)
    outputs = load_allocator().allocate(gathered, tokens * topk)
    for got, want in zip(outputs, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    cumulative, _, _, copies = outputs
    extra = int((copies >= 0).sum(1).max().item())
    width = experts // ranks + extra
    table = torch.cat((torch.arange(experts, device="npu").reshape(ranks, -1), copies[:, :extra]), dim=1)
    for rank in (0, ranks - 1):
        virtual, counts = render_route(ids, inverse, histogram, gathered, cumulative, copies, rank, extra)
        torch.testing.assert_close(table.flatten()[virtual.long()], ids.long(), rtol=0, atol=0)
        actual_histogram = torch.bincount(virtual.flatten().long(), minlength=ranks * width)
        torch.testing.assert_close(actual_histogram, counts[rank].long(), rtol=0, atol=0)
        assert bool((counts.sum(0).reshape(ranks, width).sum(1) <= tokens * topk).all())
